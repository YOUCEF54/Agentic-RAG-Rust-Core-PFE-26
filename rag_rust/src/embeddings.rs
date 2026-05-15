use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use once_cell::sync::OnceCell;
use reqwest::Client;
use serde_json::json;
use std::sync::Mutex;

use crate::runtime::get_runtime;

static RETRIEVAL_EMBEDDER: OnceCell<Mutex<TextEmbedding>> = OnceCell::new();
static CHUNKING_EMBEDDER: OnceCell<Mutex<TextEmbedding>> = OnceCell::new();

const ZE_EMBED_MODEL: &str = "zembed-1";
const ZE_API_BASE: &str = "https://api.zeroentropy.dev";

fn get_retrieval_embedder() -> Result<&'static Mutex<TextEmbedding>, String> {
    let _ = dotenvy::dotenv();
    let model_name_str = std::env::var("EMBED_MODEL").unwrap_or_else(|_| "BGESmallENV15".to_string());

    RETRIEVAL_EMBEDDER.get_or_try_init(|| {
        std::env::set_var("TOKENIZERS_PARALLELISM", "false");
        let model_enum = match model_name_str.as_str() {
            "BGELargeENV15" => EmbeddingModel::BGELargeENV15,
            "MultilingualE5Small" => EmbeddingModel::MultilingualE5Small,
            "AllMiniLML6V2" => EmbeddingModel::AllMiniLML6V2,
            _ => return Err(format!("Unsupported model name: {model_name_str}")),
        };
        let options = InitOptions::new(model_enum).with_show_download_progress(false);
        TextEmbedding::try_new(options)
            .map(Mutex::new)
            .map_err(|e| format!("Fastembed init failed: {e:?}"))
    })
}

pub fn get_chunking_embedder() -> Result<&'static Mutex<TextEmbedding>, String> {
    CHUNKING_EMBEDDER.get_or_try_init(|| {
        std::env::set_var("TOKENIZERS_PARALLELISM", "false");
        let mut options = InitOptions::default();
        options.model_name = EmbeddingModel::MultilingualE5Small;
        TextEmbedding::try_new(options)
            .map(Mutex::new)
            .map_err(|e| format!("Fastembed init failed: {e:?}"))
    })
}

pub fn load_embed_model_local_impl() -> Result<(), String> {
    get_retrieval_embedder().map(|_| ())
}

pub fn load_embed_model_zembed_impl() -> Result<(), String> {
    let _ = dotenvy::dotenv();
    std::env::var("ZEROENTROPY_API_KEY")
        .map(|_| ())
        .map_err(|_| "ZEROENTROPY_API_KEY environment variable is not set.".to_string())
}

async fn ze_embed(texts: Vec<String>, input_type: &str, batch_size: usize) -> Result<Vec<Vec<f32>>, String> {
    let _ = dotenvy::dotenv();
    let api_key =
        std::env::var("ZEROENTROPY_API_KEY").map_err(|_| "ZEROENTROPY_API_KEY env var not set".to_string())?;

    let client = Client::new();
    let url = format!("{ZE_API_BASE}/v1/models/embed");
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(batch_size.clamp(1, 256)) {
        let body = json!({
            "input": chunk,
            "input_type": input_type,
            "model": ZE_EMBED_MODEL,
            "encoding_format": "float"
        });

        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("ZeroEntropy API error {status}: {text}"));
        }

        let data: serde_json::Value = resp.json().await.map_err(|e| format!("JSON parse error: {e}"))?;
        let results = data["results"]
            .as_array()
            .ok_or_else(|| format!("Missing 'results' in response: {data}"))?;

        for result in results {
            let embedding = result["embedding"]
                .as_array()
                .ok_or_else(|| "Missing 'embedding' field in result".to_string())?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| "Non-numeric value in embedding".to_string())
                })
                .collect::<Result<Vec<f32>, _>>()?;
            all_embeddings.push(embedding);
        }
    }

    Ok(all_embeddings)
}

pub fn embed_texts_rust_zembed_impl(texts: Vec<String>, embed_batch_size: usize) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    get_runtime().block_on(ze_embed(texts, "document", embed_batch_size))
}

pub fn embed_texts_rust_local_impl(texts: Vec<String>, embed_batch_size: usize) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let batch_size = embed_batch_size.clamp(2, 256);
    let embedder = get_retrieval_embedder()?;
    let mut guard = embedder.lock().map_err(|_| "Embedder mutex poisoned".to_string())?;
    guard
        .embed(texts, Some(batch_size))
        .map_err(|e| format!("Fastembed failed: {e}"))
}

pub fn embed_query_rust_zembed_impl(query: String) -> Result<Vec<f32>, String> {
    get_runtime().block_on(async {
        let mut vecs = ze_embed(vec![query], "query", 1).await?;
        vecs.pop().ok_or_else(|| "Empty embedding response".to_string())
    })
}

