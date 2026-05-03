from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional
import os

# Bootstrap the pyo3 extension on Windows when the repo contains the built DLL
# but the Python importable .pyd isn't present yet.
_RAG_RUST_PYD = Path(__file__).with_name("rag_rust.pyd")
if not _RAG_RUST_PYD.exists():
  for _dll in (
    Path("rag_rust/target/release/rag_rust.dll"),
    Path("rag_rust/target/maturin/rag_rust.dll"),
  ):
    if _dll.exists():
      try:
        shutil.copyfile(_dll, _RAG_RUST_PYD)
      except Exception:
        pass
      break

import rag_rust

# ========================= CONFIG =========================
# Quick mode is now the default for API usage.
QUICK_NUM_CHUNKS_TO_TEST = 64
QUICK_BATCH_SIZES = [2, 4, 8, 16, 24, 32]
QUICK_REPEATS = 2
def _truthy_env(name: str) -> bool:
  """Parse env vars like '1', 'true', 'yes', 'on', 'zembed' as True."""
  val = os.getenv(name)
  if val is None:
    return False
  return val.strip().lower() in ("1", "true", "yes", "y", "on", "zembed")


EMBED_MODE = _truthy_env("EMBED_MODE")

# Full mode (slower, more exhaustive).
FULL_NUM_CHUNKS_TO_TEST = 256
FULL_BATCH_SIZES = [4, 8, 16, 24, 32, 48, 64, 128]
FULL_REPEATS = 3

# 800-1000 chars is typical for "Researcher" chunks (approx 150-200 tokens)
SYNTHETIC_CHUNK_LEN = 1000
WARMUP_CHUNKS = 16
WARMUP_TEXT_LEN = 256
CONFIG_PATH = "hardware_config.json"


def create_synthetic_data(n: int, length: int) -> list[str]:
  """Generate consistent technical-length strings for benchmarking."""
  base = "Scientific research requires precise data extraction and high-performance indexing. "
  repeat_factor = (length // len(base)) + 1
  sample_text = (base * repeat_factor)[:length]
  return [f"passage: {sample_text}" for _ in range(n)]


def load_hardware_config(config_path: str = CONFIG_PATH) -> Optional[Dict[str, Any]]:
  path = Path(config_path)
  if not path.exists():
    return None
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return None
  optimal = data.get("optimal_batch_size")
  if isinstance(optimal, int) and optimal > 0:
    return data
  return None


def run_hardware_calibration(
  save_to_file: bool = True,
  config_path: str = CONFIG_PATH,
  verbose: bool = True,
  early_stop_on_drop: bool = True,
  drop_tolerance_ratio: float = 0.0,
  quick_mode: bool = True,
  max_runtime_seconds: float = 120.0,
) -> Dict[str, Any]:
  num_chunks = QUICK_NUM_CHUNKS_TO_TEST if quick_mode else FULL_NUM_CHUNKS_TO_TEST
  batch_sizes = QUICK_BATCH_SIZES if quick_mode else FULL_BATCH_SIZES
  repeats = QUICK_REPEATS if quick_mode else FULL_REPEATS

  if verbose:
    print("=" * 60)
    print("RAG BACKEND: HARDWARE CALIBRATION")
    print(f"Processor: {platform.processor()}")
    print(f"Mode: {'quick' if quick_mode else 'full'}")
    print("=" * 60)

  test_data = create_synthetic_data(num_chunks, SYNTHETIC_CHUNK_LEN)
  warmup_data = create_synthetic_data(min(WARMUP_CHUNKS, num_chunks), WARMUP_TEXT_LEN)

  if verbose:
    print("Initializing embedding model...")
  if EMBED_MODE:
    rag_rust.load_embed_model_zembed()
    rag_rust.embed_texts_rust_zembed(warmup_data, batch_sizes[0])
  else:
    rag_rust.load_embed_model_local()
    rag_rust.embed_texts_rust_local(warmup_data, batch_sizes[0])

  results = []
  best_tps = 0.0
  stop_reason: Optional[str] = None
  benchmark_start = time.perf_counter()
  for b_size in batch_sizes:
    elapsed = time.perf_counter() - benchmark_start
    if max_runtime_seconds > 0 and elapsed >= max_runtime_seconds:
      stop_reason = (
        f"Stopped due to max_runtime_seconds={max_runtime_seconds:.1f}s "
        f"after testing {len(results)} batch sizes."
      )
      if verbose:
        print(stop_reason)
      break

    if verbose:
      print(f"Testing Batch Size: {b_size:3} ... ", end="", flush=True)
    durations = []
    try:
      # Small warmup per tested batch size to avoid cold-start bias.
      if EMBED_MODE:
        rag_rust.embed_texts_rust_zembed(warmup_data, b_size)
      else:
        rag_rust.embed_texts_rust_local(warmup_data, b_size)

      for _ in range(repeats):
        t0 = time.perf_counter()
        if EMBED_MODE:
          _ = rag_rust.embed_texts_rust_zembed(test_data, b_size)
        else:
          _ = rag_rust.embed_texts_rust_local(test_data, b_size)
        durations.append(time.perf_counter() - t0)

      avg_time = mean(durations)
      chunks_per_sec = num_chunks / avg_time
      results.append({"batch_size": b_size, "tps": chunks_per_sec})
      if verbose:
        print(f"{chunks_per_sec:8.2f} chunks/sec")
      if chunks_per_sec > best_tps:
        best_tps = chunks_per_sec
      elif early_stop_on_drop:
        # Stop early once throughput declines versus the best observed run.
        # drop_tolerance_ratio=0.0 means any decline triggers the stop.
        decline_ratio = (best_tps - chunks_per_sec) / best_tps if best_tps > 0 else 0.0
        if decline_ratio > drop_tolerance_ratio:
          stop_reason = (
            f"Stopped early at batch size {b_size}: throughput dropped "
            f"from best {best_tps:.2f} to {chunks_per_sec:.2f} chunks/sec."
          )
          if verbose:
            print(stop_reason)
          break
    except Exception as exc:
      if verbose:
        print(f"FAILED (likely OOM or hardware limit): {exc}")
      stop_reason = f"Stopped due to runtime error at batch size {b_size}: {exc}"
      break

  if not results:
    raise RuntimeError("Calibration failed to produce results.")

  best_config = max(results, key=lambda x: x["tps"])
  config_data = {
    "optimal_batch_size": best_config["batch_size"],
    "throughput_measured": round(best_config["tps"], 2),
    "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "cpu_info": platform.processor(),
    "quick_mode": quick_mode,
    "num_chunks_tested": num_chunks,
    "repeats": repeats,
    "max_runtime_seconds": max_runtime_seconds,
    "tested_batch_sizes": [r["batch_size"] for r in results],
    "stop_reason": stop_reason or "Completed full sweep.",
  }

  if save_to_file:
    path = Path(config_path)
    path.write_text(json.dumps(config_data, indent=4), encoding="utf-8")

  if verbose:
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print(f"Optimal Batch Size: {config_data['optimal_batch_size']}")
    if save_to_file:
      print(f"Configuration saved to: {config_path}")
    print("=" * 60)

  return config_data


if __name__ == "__main__":
  run_hardware_calibration(save_to_file=True, config_path=CONFIG_PATH, verbose=True)
