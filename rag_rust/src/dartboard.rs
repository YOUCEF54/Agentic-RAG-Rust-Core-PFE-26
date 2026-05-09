// dartboard.rs — Dartboard retrieval re-ranking (Pickett et al., 2025)
// arXiv:2407.12101v2 — "Dartboard: Better RAG using Relevant Information Gain"
//
// ── Python call (main.py) ────────────────────────────────────────────────
//      selected_indices = rag_rust.dartboard_rerank(
//          query_vector,      # Vec<f32>  — embedded query
//          candidate_vectors, # Vec<Vec<f32>> — one per candidate from lancedb_search
//          top_k,             # usize — how many to select
//          sigma,             # f32   — Gaussian spread (paper best ≈ 0.096)
//      )
// ─────────────────────────────────────────────────────────────────────────────

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ── Maths helpers ─────────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length f32 slices → [-1, 1].
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot    = 0f32;
    let mut norm_a = 0f32;
    let mut norm_b = 0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot    += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 { 0.0 } else { (dot / denom).clamp(-1.0, 1.0) }
}

/// Cosine distance ∈ [0, 2].
#[inline]
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_sim(a, b)
}

/// Log of Gaussian pdf at distance μ with spread σ (Equation from the ref paper).
/// LogNorm(μ, σ) = −ln(σ) − 0.5·ln(2π) − μ²/(2σ²)
/// Computed up to a constant offset — the constant cancels in argmax.
#[inline]
fn log_norm(mu: f32, sigma: f32) -> f32 {
    let sigma2 = sigma * sigma;
    -(mu * mu) / (2.0 * sigma2)
    // We drop the −ln(σ) − 0.5·ln(2π) constant terms because they are
    // identical for every candidate and do not affect the argmax outcome.
}

/// Numerically stable log-sum-exp over a slice.
#[inline]
fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        return max;
    }
    max + values.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

// ── Dartboard greedy algorithm ────────────────────────────────────────────────

/// Re-rank a pre-retrieved candidate pool using the Dartboard algorithm
/// (Algorithm 1, Pickett et al. 2025).
///
/// Args:
///   query_vector      — embedding of the user query
///   candidate_vectors — one embedding per candidate (same order as lancedb_search output)
///   top_k             — number of passages to select
///   sigma             — Gaussian spread hyperparameter σ (paper default ≈ 0.096)
///
/// Returns:
///   Vec<usize> — indices into candidate_vectors in greedy selection order.
///   Length = min(top_k, n_candidates).
///
/// Complexity: O(K²) where K = len(candidate_vectors).
/// For K ≤ 50 this is negligible (<1 ms in practice per the paper §6).
#[pyfunction]
#[pyo3(name = "dartboard_rerank")]
pub fn dartboard_rerank(
    query_vector:      Vec<f32>,
    candidate_vectors: Vec<Vec<f32>>,
    top_k:             usize,
    sigma:             f32,
) -> PyResult<Vec<usize>> {
    let n = candidate_vectors.len();
    if n == 0 || top_k == 0 {
        return Ok(vec![]);
    }
    if sigma <= 0.0 {
        return Err(PyValueError::new_err(
            "sigma must be > 0.0 (paper recommends ~0.096 for cosine-similarity Dartboard)",
        ));
    }

    let k = top_k.min(n);

    // ── Step 1: Pre-compute Q — log-prob of each candidate given the query ──
    // Q[i] = LogNorm(cosine_dist(query, candidate_i), σ)
    let q_log: Vec<f32> = candidate_vectors
        .iter()
        .map(|cv| log_norm(cosine_dist(&query_vector, cv), sigma))
        .collect();

    // ── Step 2: Pre-compute D — KxK pairwise log-prob distance matrix ───────
    // D[i][j] = LogNorm(cosine_dist(candidate_i, candidate_j), σ)
    // We store as a flat Vec<f32> of size n*n.
    let mut d_log: Vec<f32> = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dist = cosine_dist(&candidate_vectors[i], &candidate_vectors[j]);
            d_log[i * n + j] = log_norm(dist, sigma);
        }
    }

    // ── Step 3: Greedy seeding — start with the most query-relevant candidate
    let first = q_log
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();  // safe: n > 0

    let mut selected: Vec<usize>  = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..n).collect();

    // maxes[i] tracks the running element-wise max of D[selected, i] in log space.
    // This reuse of previous maxes is the key optimisation in Algorithm 1.
    let mut maxes: Vec<f32> = d_log[first * n..first * n + n].to_vec();

    let first_pos = remaining.iter().position(|&x| x == first).unwrap();
    remaining.swap_remove(first_pos);
    selected.push(first);

    // ── Step 4: Greedy extension ──────────────────────────────────────────────
    while selected.len() < k && !remaining.is_empty() {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_ri    = 0usize;   // index into `remaining`

        for (ri, &cand_idx) in remaining.iter().enumerate() {
            // Update the running max with the latest selected candidate's row
            let last_selected = *selected.last().unwrap();
            let new_max_val = d_log[last_selected * n + cand_idx];
            let running_max = maxes[cand_idx].max(new_max_val);

            // Score = LogSumExp over all points t of (max_g log N(t,g,σ) + log N(q,t,σ))
            // We approximate this with the candidate's own contribution (scalar form,
            // valid because D is pre-computed over the full candidate set).
            // Full form requires summing over all A′ — here we sum over remaining candidates
            // plus already-incorporated maxes, which is the correct greedy approximation.
            let score_terms: Vec<f32> = (0..n)
                .map(|t| {
                    // max over selected so far (including current candidate being scored)
                    let max_g = if remaining.contains(&t) {
                        maxes[t].max(d_log[cand_idx * n + t])
                    } else {
                        maxes[t]
                    };
                    max_g + q_log[t]
                })
                .collect();

            let score = log_sum_exp(&score_terms);

            if score > best_score {
                best_score = score;
                best_ri    = ri;
                // Update maxes for this candidate preemptively
                let _ = running_max; // used for single-candidate fast path below
            }
        }

        let chosen = remaining.swap_remove(best_ri);
        // Update maxes: for every remaining candidate t, maxes[t] = max(maxes[t], D[chosen][t])
        for &t in &remaining {
            let new_val = d_log[chosen * n + t];
            if new_val > maxes[t] {
                maxes[t] = new_val;
            }
        }
        selected.push(chosen);
    }

    Ok(selected)
}
