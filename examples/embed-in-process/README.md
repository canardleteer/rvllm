# embed-in-process

Batch **in-process** embeddings using [`GpuLLMEngine::embed`](../../crates/rvllm-engine/src/gpu_engine.rs) (no HTTP server), write fat YAML, and **compare** against a Hugging Face **transformers** reference on CUDA.

## Layout

Put raw text chunks (≤ 8192 bytes each) under:

- `<input-dir>/query/`
- `<input-dir>/passage/`

Defaults match the [NV embedding model card](https://huggingface.co/nvidia/llama-nv-embed-reasoning-3b): prefixes `query: ` and `passage: ` (override with flags).

## Local (CUDA host)

From the repository root:

```bash
cargo build --release -p embed-in-process
mkdir -p data/input/query data/input/passage
echo 'hello' > data/input/query/a.txt

./target/release/embed-in-process embed \
  --input-dir data/input \
  --output-yaml data/out/rust.yaml

./target/release/embed-in-process compare \
  --rust-yaml data/out/rust.yaml \
  --reference-yaml data/out/reference.yaml
```

Set `HF_TOKEN` if the Hub requires it. First run downloads weights.

## Docker + one-shot parity

[`compose.yaml`](compose.yaml) builds two images with **identical** mounts:

- `INPUT_DIR` → `/data/input` (read-only)
- `OUT_DIR` → `/data/out`
- shared Hugging Face cache volume `hf-cache`

[`run_parity.sh`](run_parity.sh) runs, in order:

1. `embed-rust` → `data/out/rust.yaml`
2. `embed-python` → `data/out/reference.yaml`
3. `embed-rust compare` on both YAMLs

```bash
cd examples/embed-in-process
export INPUT_DIR="$PWD/data/input"
export OUT_DIR="$PWD/data/out"
export MODEL=nvidia/llama-nv-embed-reasoning-3b
# optional: export FAIL_BELOW=0.99
./run_parity.sh
```

Outputs appear on the host under `$OUT_DIR`.

## YAML schema

Both pipelines emit:

```yaml
model: <id>
engine: <rvllm_native | transformers_cuda>
entries:
  - path: query/a.txt
    task: query
    byte_length: 5
    sha256: ...
    embedding: [f32, ...]
```

## Expectations

Exact **bitwise** equality between Rust and PyTorch is unlikely; use **cosine similarity** from `compare` (e.g. mean > 0.99). Tune `--fail-below-cosine` once you have baselines.

## See also

- [`python/README.md`](python/README.md) — reference container only
- [`../hf-embed-card/README.md`](../hf-embed-card/README.md) — standalone Rust port of the model card example (`hf-embed-card` binary; compares to the card’s printed transformers scores)
