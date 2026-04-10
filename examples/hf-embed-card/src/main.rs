//! Port of the Hugging Face model card example:
//! <https://huggingface.co/nvidia/llama-nv-embed-reasoning-3b#example-usage>
//!
//! Same strings, prefixes (`query: ` / `passage: `), average pooling + L2 normalize in the engine,
//! then `scores = embeddings_queries @ embeddings_documents.T`.
//!
//! Input text is loaded from `data/input/` (via `include_str!`) so it stays identical to the files on disk.
//!
//! Throughout this file, `//` blocks quote the upstream **Example Usage** on the model card
//! (<https://huggingface.co/nvidia/llama-nv-embed-reasoning-3b#example-usage>) next to the Rust that mirrors it.

use anyhow::{anyhow, Result};
use clap::Parser;
use rvllm_config::{
    CacheConfigImpl, DeviceConfig, EngineConfig, ModelConfigImpl, ParallelConfigImpl,
    SchedulerConfigImpl, TelemetryConfig,
};
use rvllm_core::types::Dtype;
use rvllm_engine::GpuLLMEngine;
use rvllm_tokenizer::Tokenizer;

// --- Python (model card): raw `queries` / `documents` lists ------------------------------------
// queries = [
//     "how much protein should a female eat",
//     "summit define",
// ]
// documents = [
//     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
//     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
// ]

// --- Model card strings (must match `data/input/**` byte-for-byte) --------------------------------

const QUERY_Q01: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/input/query/q01.txt"
));
const QUERY_Q02: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/input/query/q02.txt"
));
const PASSAGE_P01: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/input/passage/p01.txt"
));
const PASSAGE_P02: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/input/passage/p02.txt"
));

// Python (model card):
// query_prefix = "query:"
// document_prefix = "passage:"
// queries = [f"{query_prefix} {query}" for query in queries]
// documents = [f"{document_prefix} {document}" for document in documents]

/// Prefixes match the card: `query_prefix = "query:"`, `document_prefix = "passage:"`, then `f"{prefix} {text}"`.
const QUERY_PREFIX: &str = "query: ";
const PASSAGE_PREFIX: &str = "passage: ";

// Python (model card) — reference matrix from `print(scores.tolist())` in that example:
// # Compute similarity scores
// scores = (embeddings_queries @ embeddings_documents.T)
// print("\nSimilarity scores:")
// print(scores.tolist())
// # Similarity scores:
// # [[0.6688634157180786, 0.23073062300682068], [0.24395054578781128, 0.5622682571411133]]

/// Reference printout from the model card (`# Compute similarity scores` → `scores.tolist()`).
/// This is **transformers + CUDA** as shown on the card, not rvllm.
const MODEL_CARD_TRANSFORMERS_SCORES: [[f64; 2]; 2] = [
    [0.6688634157180786_f64, 0.23073062300682068_f64],
    [0.24395054578781128_f64, 0.5622682571411133_f64],
];

const QUERY_IDS: [&str; 2] = ["q01", "q02"];
const PASSAGE_IDS: [&str; 2] = ["p01", "p02"];

// Python (model card) — top of Example Usage (after imports, before `queries` / `documents`):
// import torch
// import torch.nn.functional as F
// from transformers import AutoTokenizer, AutoModel
//
// def average_pool(last_hidden_states, attention_mask):
//     """Average pooling with attention mask."""
//     last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
//     embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
//     embedding = F.normalize(embedding, dim=-1)
//     return embedding

// --- CLI (mirrors `embed-in-process embed` tuning knobs; duplicated on purpose) --------------------

#[derive(Parser)]
#[command(
    name = "hf-embed-card",
    version,
    about = "Model-card embedding smoke test (rvllm vs card reference)"
)]
struct Cli {
    #[arg(long, default_value = "nvidia/llama-nv-embed-reasoning-3b")]
    model: String,
    #[arg(long)]
    tokenizer: Option<String>,
    #[arg(long, default_value = "auto")]
    dtype: Dtype,
    #[arg(long)]
    max_model_len: Option<usize>,
    #[arg(long, default_value_t = 0.90)]
    gpu_memory_utilization: f32,
    #[arg(long, default_value_t = 0.0)]
    gpu_memory_reserve_gb: f32,
    #[arg(long)]
    num_gpu_blocks: Option<usize>,
    #[arg(long)]
    num_cpu_blocks: Option<usize>,
    #[arg(long, default_value_t = 256)]
    max_num_seqs: usize,
    #[arg(long, default_value_t = 8192)]
    max_num_batched_tokens: usize,
    #[arg(long, default_value_t = 128)]
    max_prefill_chunk: usize,
    #[arg(long, default_value_t = 1)]
    tensor_parallel_size: usize,
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Python: `queries` / `documents` are the raw lists above; prefixed lists are built with:
    // queries = [f"{query_prefix} {query}" for query in queries]
    // documents = [f"{document_prefix} {document}" for document in documents]
    let queries_raw = [QUERY_Q01, QUERY_Q02];
    let passages_raw = [PASSAGE_P01, PASSAGE_P02];

    let queries_prefixed: Vec<String> = queries_raw
        .iter()
        .map(|t| format!("{QUERY_PREFIX}{t}"))
        .collect();
    let passages_prefixed: Vec<String> = passages_raw
        .iter()
        .map(|t| format!("{PASSAGE_PREFIX}{t}"))
        .collect();

    // Python (model card) — continues after `average_pool` and the query/passage lists:
    // model_name = "nvidia/llama-nv-embed-reasoning-3b"
    // tokenizer = AutoTokenizer.from_pretrained(model_name)
    // model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    // model = model.to("cuda:0")
    // model.eval()
    let config = build_engine_config(
        cli.model.clone(),
        cli.tokenizer.clone(),
        cli.dtype,
        cli.max_model_len,
        cli.gpu_memory_utilization,
        cli.gpu_memory_reserve_gb,
        cli.num_gpu_blocks,
        cli.num_cpu_blocks,
        cli.max_num_seqs,
        cli.max_num_batched_tokens,
        cli.max_prefill_chunk,
        cli.tensor_parallel_size,
        cli.log_level,
    )?;

    eprintln!("loading model (this may take a while)...");
    let engine = GpuLLMEngine::new(config).map_err(|e| anyhow!("GpuLLMEngine::new: {e}"))?;

    let tok_path = cli.tokenizer.as_deref().unwrap_or(cli.model.as_str());
    let tokenizer = Tokenizer::from_pretrained(tok_path).map_err(|e| anyhow!("tokenizer: {e}"))?;

    // Python (model card) — `average_pool` is defined above (with imports); rvllm applies pool+normalize inside `embed`.
    // batch_queries = tokenizer(queries, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
    // with torch.no_grad():
    //     outputs_queries = model(**batch_queries)
    // embeddings_queries = average_pool(outputs_queries.last_hidden_state, batch_queries["attention_mask"])
    let mut query_emb: Vec<Vec<f32>> = Vec::with_capacity(2);
    for (i, full) in queries_prefixed.iter().enumerate() {
        let token_ids = tokenizer
            .encode(full)
            .map_err(|e| anyhow!("encode query {}: {e}", QUERY_IDS[i]))?;
        let token_ids_u32: Vec<u32> = token_ids.iter().copied().collect();
        let v = engine
            .embed(&token_ids_u32)
            .map_err(|e| anyhow!("embed query {}: {e}", QUERY_IDS[i]))?;
        query_emb.push(v);
    }

    // Python (model card):
    // batch_documents = tokenizer(documents, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
    // with torch.no_grad():
    //     outputs_documents = model(**batch_documents)
    // embeddings_documents = average_pool(outputs_documents.last_hidden_state, batch_documents["attention_mask"])
    let mut passage_emb: Vec<Vec<f32>> = Vec::with_capacity(2);
    for (i, full) in passages_prefixed.iter().enumerate() {
        let token_ids = tokenizer
            .encode(full)
            .map_err(|e| anyhow!("encode passage {}: {e}", PASSAGE_IDS[i]))?;
        let token_ids_u32: Vec<u32> = token_ids.iter().copied().collect();
        let v = engine
            .embed(&token_ids_u32)
            .map_err(|e| anyhow!("embed passage {}: {e}", PASSAGE_IDS[i]))?;
        passage_emb.push(v);
    }

    // Python (model card):
    // scores = (embeddings_queries @ embeddings_documents.T)
    let mut rvllm_scores = [[0f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            rvllm_scores[i][j] = f64::from(dot_product(&query_emb[i], &passage_emb[j]));
        }
    }

    // Python only prints `scores.tolist()`; below we also compare to the frozen matrix on the card.
    println!("nvidia/llama-nv-embed-reasoning-3b — model card example (Rust / rvllm)\n");
    println!("Reference (transformers, printed on the model card):");
    println!("{:?}\n", MODEL_CARD_TRANSFORMERS_SCORES);

    println!("rvllm (this run):");
    println!("[");
    for i in 0..2 {
        print!("  [{:.16}, {:.16}]", rvllm_scores[i][0], rvllm_scores[i][1]);
        if i + 1 < 2 {
            println!(",");
        } else {
            println!();
        }
    }
    println!("]");

    println!("\nΔ matrix (rvllm − transformers, same shape as scores.tolist()):");
    println!("[");
    for i in 0..2 {
        let d0 = rvllm_scores[i][0] - MODEL_CARD_TRANSFORMERS_SCORES[i][0];
        let d1 = rvllm_scores[i][1] - MODEL_CARD_TRANSFORMERS_SCORES[i][1];
        print!("  [{:+.16e}, {:+.16e}]", d0, d1);
        if i + 1 < 2 {
            println!(",");
        } else {
            println!();
        }
    }
    println!("]\n");

    println!("Per-cell (row = query, col = passage):");
    println!("{}", "-".repeat(100));
    let mut abs_diffs = Vec::with_capacity(4);
    let mut signed_diffs = Vec::with_capacity(4);
    for i in 0..2 {
        for j in 0..2 {
            let r = rvllm_scores[i][j];
            let c = MODEL_CARD_TRANSFORMERS_SCORES[i][j];
            let d = r - c;
            abs_diffs.push(d.abs());
            signed_diffs.push(d);
            println!(
                "  [{} × {}]  rvllm = {:>22.16}   transformers (model card) = {:>22.16}   Δ (rvllm − card) = {:>+14.6e}",
                QUERY_IDS[i],
                PASSAGE_IDS[j],
                r,
                c,
                d,
            );
        }
    }
    println!("{}", "-".repeat(100));

    let max_abs = abs_diffs.iter().copied().fold(0.0_f64, f64::max);
    let mean_abs: f64 = abs_diffs.iter().sum::<f64>() / 4.0;
    let rmse: f64 = (signed_diffs.iter().map(|x| x * x).sum::<f64>() / 4.0_f64).sqrt();

    println!("\nSummary (n = 4 matrix elements):");
    println!("  max |Δ|     = {max_abs:.6e}");
    println!("  mean |Δ|   = {mean_abs:.6e}");
    println!("  RMSE(Δ)    = {rmse:.6e}");

    Ok(())
}

// Dot product on L2-normalized embeddings matches the matrix multiply in:
// scores = (embeddings_queries @ embeddings_documents.T)
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn build_engine_config(
    model: String,
    tokenizer: Option<String>,
    dtype: Dtype,
    max_model_len: Option<usize>,
    gpu_memory_utilization: f32,
    gpu_memory_reserve_gb: f32,
    num_gpu_blocks: Option<usize>,
    num_cpu_blocks: Option<usize>,
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    max_prefill_chunk: usize,
    tensor_parallel_size: usize,
    log_level: String,
) -> Result<EngineConfig> {
    let mut cache = CacheConfigImpl::builder()
        .gpu_memory_utilization(gpu_memory_utilization)
        .gpu_memory_reserve_gb(gpu_memory_reserve_gb);
    if let Some(n) = num_gpu_blocks {
        cache = cache.num_gpu_blocks(n);
    }
    if let Some(n) = num_cpu_blocks {
        cache = cache.num_cpu_blocks(n);
    }
    let mut model_b = ModelConfigImpl::builder()
        .model_path(model.clone())
        .dtype(dtype);
    if let Some(m) = max_model_len {
        model_b = model_b.max_model_len(m);
    }
    if let Some(ref t) = tokenizer {
        model_b = model_b.tokenizer_path(t);
    }
    let config = EngineConfig::builder()
        .model(model_b.build())
        .cache(cache.build())
        .scheduler(
            SchedulerConfigImpl::builder()
                .max_num_seqs(max_num_seqs)
                .max_num_batched_tokens(max_num_batched_tokens)
                .max_prefill_chunk(max_prefill_chunk)
                .build(),
        )
        .parallel(
            ParallelConfigImpl::builder()
                .tensor_parallel_size(tensor_parallel_size)
                .build(),
        )
        .device(DeviceConfig::builder().device("cuda").build())
        .telemetry(
            TelemetryConfig::builder()
                .enabled(false)
                .log_level(&log_level)
                .build(),
        )
        .build();
    Ok(config)
}
