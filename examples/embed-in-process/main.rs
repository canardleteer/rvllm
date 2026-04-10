//! Batch embedding + YAML parity compare (in-process `GpuLLMEngine`, no HTTP).
//!
//! ```text
//! embed-in-process embed --input-dir ... --output-yaml out.yaml
//! embed-in-process matrix --yaml out.yaml
//! embed-in-process compare --rust-yaml out.yaml --reference-yaml ref.yaml
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use rvllm_config::{
    CacheConfigImpl, DeviceConfig, EngineConfig, ModelConfigImpl, ParallelConfigImpl,
    SchedulerConfigImpl, TelemetryConfig,
};
use rvllm_core::types::Dtype;
use rvllm_engine::GpuLLMEngine;
use rvllm_tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// In-process embeddings vs YAML parity (see `README.md`).
#[derive(Parser)]
#[command(name = "embed-in-process", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed all files under `<input-dir>/query` and `<input-dir>/passage`.
    Embed {
        #[arg(long, default_value = "nvidia/llama-nv-embed-reasoning-3b")]
        model: String,
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        output_yaml: PathBuf,
        #[arg(long, default_value_t = 8192)]
        max_chunk_bytes: usize,
        #[arg(long, default_value = "query: ")]
        query_prefix: String,
        #[arg(long, default_value = "passage: ")]
        passage_prefix: String,
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
    },
    /// Compare two embedding YAML files (same schema as `embed` output).
    Compare {
        #[arg(long, alias = "a")]
        rust_yaml: PathBuf,
        #[arg(long, alias = "b")]
        reference_yaml: PathBuf,
        #[arg(long)]
        fail_below_cosine: Option<f32>,
    },
    /// Print query×passage dot-product scores from an `embed` YAML (L2-normalized vectors ⇒ cosine similarity).
    Matrix {
        #[arg(long)]
        yaml: PathBuf,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EmbedOutput {
    model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    engine: Option<String>,
    entries: Vec<EmbedEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EmbedEntry {
    path: String,
    task: String,
    byte_length: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
    embedding: Vec<f32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Embed {
            model,
            input_dir,
            output_yaml,
            max_chunk_bytes,
            query_prefix,
            passage_prefix,
            tokenizer,
            dtype,
            max_model_len,
            gpu_memory_utilization,
            gpu_memory_reserve_gb,
            num_gpu_blocks,
            num_cpu_blocks,
            max_num_seqs,
            max_num_batched_tokens,
            max_prefill_chunk,
            tensor_parallel_size,
            log_level,
        } => run_embed(
            model,
            input_dir,
            output_yaml,
            max_chunk_bytes,
            query_prefix,
            passage_prefix,
            tokenizer,
            dtype,
            max_model_len,
            gpu_memory_utilization,
            gpu_memory_reserve_gb,
            num_gpu_blocks,
            num_cpu_blocks,
            max_num_seqs,
            max_num_batched_tokens,
            max_prefill_chunk,
            tensor_parallel_size,
            log_level,
        ),
        Commands::Compare {
            rust_yaml,
            reference_yaml,
            fail_below_cosine,
        } => run_compare(rust_yaml, reference_yaml, fail_below_cosine),
        Commands::Matrix { yaml } => run_matrix(yaml),
    }
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

fn run_embed(
    model: String,
    input_dir: PathBuf,
    output_yaml: PathBuf,
    max_chunk_bytes: usize,
    query_prefix: String,
    passage_prefix: String,
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
) -> Result<()> {
    let input_dir = input_dir
        .canonicalize()
        .with_context(|| format!("input-dir {}", input_dir.display()))?;
    let query_dir = input_dir.join("query");
    let passage_dir = input_dir.join("passage");
    if !query_dir.is_dir() {
        anyhow::bail!("missing directory: {}", query_dir.display());
    }
    if !passage_dir.is_dir() {
        anyhow::bail!("missing directory: {}", passage_dir.display());
    }

    let config = build_engine_config(
        model.clone(),
        tokenizer.clone(),
        dtype,
        max_model_len,
        gpu_memory_utilization,
        gpu_memory_reserve_gb,
        num_gpu_blocks,
        num_cpu_blocks,
        max_num_seqs,
        max_num_batched_tokens,
        max_prefill_chunk,
        tensor_parallel_size,
        log_level,
    )?;

    eprintln!("loading model (this may take a while)...");
    let engine = GpuLLMEngine::new(config).map_err(|e| anyhow!("GpuLLMEngine::new: {e}"))?;

    let tok_path = tokenizer.as_deref().unwrap_or(model.as_str());
    let tokenizer = Tokenizer::from_pretrained(tok_path).map_err(|e| anyhow!("tokenizer: {e}"))?;

    let mut entries = Vec::new();

    for (task, dir, prefix) in [
        ("query", query_dir.as_path(), query_prefix.as_str()),
        ("passage", passage_dir.as_path(), passage_prefix.as_str()),
    ] {
        for path in list_files(dir)? {
            let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            if bytes.len() > max_chunk_bytes {
                anyhow::bail!(
                    "{} exceeds --max-chunk-bytes ({} > {})",
                    path.display(),
                    bytes.len(),
                    max_chunk_bytes
                );
            }
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let prefixed = format!("{prefix}{text}");
            let token_ids = tokenizer
                .encode(&prefixed)
                .map_err(|e| anyhow!("encode {}: {e}", path.display()))?;
            let token_ids_u32: Vec<u32> = token_ids.iter().copied().collect();
            let embedding = engine
                .embed(&token_ids_u32)
                .map_err(|e| anyhow!("embed {}: {e}", path.display()))?;

            let rel = path
                .strip_prefix(&input_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");

            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let sha256 = Some(hex::encode(hasher.finalize()));

            entries.push(EmbedEntry {
                path: rel,
                task: task.to_string(),
                byte_length: bytes.len(),
                sha256,
                embedding,
            });
        }
    }

    let out = EmbedOutput {
        model: model.clone(),
        engine: Some("rvllm_native".to_string()),
        entries,
    };

    let yaml = serde_yaml::to_string(&out).context("serialize yaml")?;
    if let Some(parent) = output_yaml.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output_yaml, yaml).with_context(|| output_yaml.display().to_string())?;
    eprintln!("wrote {}", output_yaml.display());
    Ok(())
}

fn list_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    v.sort();
    Ok(v)
}

fn run_matrix(yaml_path: PathBuf) -> Result<()> {
    let s = std::fs::read_to_string(&yaml_path).with_context(|| yaml_path.display().to_string())?;
    let out: EmbedOutput = serde_yaml::from_str(&s).context("parse yaml")?;

    let mut queries: Vec<&EmbedEntry> = out.entries.iter().filter(|e| e.task == "query").collect();
    queries.sort_by(|a, b| a.path.cmp(&b.path));
    let mut passages: Vec<&EmbedEntry> = out
        .entries
        .iter()
        .filter(|e| e.task == "passage")
        .collect();
    passages.sort_by(|a, b| a.path.cmp(&b.path));

    if queries.is_empty() || passages.is_empty() {
        anyhow::bail!("need at least one query and one passage entry");
    }
    let dim = queries[0].embedding.len();
    for q in &queries {
        if q.embedding.len() != dim {
            anyhow::bail!("query {}: dim {} != {}", q.path, q.embedding.len(), dim);
        }
    }
    for p in &passages {
        if p.embedding.len() != dim {
            anyhow::bail!("passage {}: dim {} != {}", p.path, p.embedding.len(), dim);
        }
    }

    println!("query paths (rows):  {:?}", queries.iter().map(|e| &e.path).collect::<Vec<_>>());
    println!(
        "passage paths (cols): {:?}",
        passages.iter().map(|e| &e.path).collect::<Vec<_>>()
    );
    println!("\nSimilarity scores (query @ passageᵀ):");
    print!("[");
    for (i, q) in queries.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("[");
        for (j, p) in passages.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            let s = dot_product(&q.embedding, &p.embedding);
            print!("{s:.16}");
        }
        print!("]");
    }
    println!("]");
    Ok(())
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn run_compare(
    rust_yaml: PathBuf,
    reference_yaml: PathBuf,
    fail_below_cosine: Option<f32>,
) -> Result<()> {
    let a_s =
        std::fs::read_to_string(&rust_yaml).with_context(|| rust_yaml.display().to_string())?;
    let b_s = std::fs::read_to_string(&reference_yaml)
        .with_context(|| reference_yaml.display().to_string())?;

    let a: EmbedOutput = serde_yaml::from_str(&a_s).context("parse rust yaml")?;
    let b: EmbedOutput = serde_yaml::from_str(&b_s).context("parse reference yaml")?;

    let mut map_a: HashMap<(String, String), Vec<f32>> = HashMap::new();
    for e in &a.entries {
        map_a.insert((e.path.clone(), e.task.clone()), e.embedding.clone());
    }
    let mut map_b: HashMap<(String, String), Vec<f32>> = HashMap::new();
    for e in &b.entries {
        map_b.insert((e.path.clone(), e.task.clone()), e.embedding.clone());
    }

    let keys_a: HashSet<_> = map_a.keys().cloned().collect();
    let keys_b: HashSet<_> = map_b.keys().cloned().collect();
    let only_a: Vec<_> = keys_a.difference(&keys_b).collect();
    let only_b: Vec<_> = keys_b.difference(&keys_a).collect();
    if !only_a.is_empty() || !only_b.is_empty() {
        eprintln!(
            "warning: key mismatch — only in rust: {only_a:?}, only in reference: {only_b:?}"
        );
    }

    let common: Vec<_> = keys_a.intersection(&keys_b).collect();
    if common.is_empty() {
        anyhow::bail!("no common (path, task) keys to compare");
    }

    let mut worst_cos = f32::INFINITY;
    let mut best_cos = f32::NEG_INFINITY;
    let mut sum_cos = 0.0f32;
    let mut worst_path = String::new();
    let mut fail = false;

    for key in &common {
        let va = map_a.get(*key).unwrap();
        let vb = map_b.get(*key).unwrap();
        if va.len() != vb.len() {
            eprintln!(
                "error: dim mismatch {:?}: {} vs {}",
                key,
                va.len(),
                vb.len()
            );
            fail = true;
            continue;
        }
        let cos = cosine_similarity(va, vb);
        let l2 = l2_distance(va, vb);
        let max_abs = max_abs_diff(va, vb);
        let mean_abs = mean_abs_diff(va, vb);
        sum_cos += cos;
        if cos < worst_cos {
            worst_cos = cos;
            worst_path = format!("{:?}", key);
        }
        best_cos = best_cos.max(cos);
        println!(
            "{:?}  cosine={cos:.6}  L2={l2:.6}  max_abs={max_abs:.6}  mean_abs={mean_abs:.6}",
            key
        );

        if let Some(th) = fail_below_cosine {
            if cos < th {
                eprintln!("fail: cosine {cos} < {th} for {key:?}");
                fail = true;
            }
        }
    }

    let n = common.len() as f32;
    let mean_cos = sum_cos / n;
    println!("\nsummary: n={}  mean_cosine={mean_cos:.6}  min_cosine={worst_cos:.6}  max_cosine={best_cos:.6}  worst={worst_path}", common.len());

    if fail {
        anyhow::bail!("compare failed (--fail-below-cosine or dimension mismatch)");
    }
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / d
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().max(1) as f32;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / n
}
