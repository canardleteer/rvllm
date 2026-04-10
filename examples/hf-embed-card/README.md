# hf-embed-card

Standalone Rust port of the [Example Usage](https://huggingface.co/nvidia/llama-nv-embed-reasoning-3b#example-usage) block on `nvidia/llama-nv-embed-reasoning-3b`: same query/passage strings (under `data/input/`, also baked in with `include_str!`), same `query: ` / `passage: ` prefixes, then the 2×2 similarity matrix as `Q @ Dᵀ` on L2-normalized embeddings.

The program prints:

- **rvllm** scores from this run  
- **transformers** scores copied from the model card printout  
- **Δ = rvllm − card** per matrix element  
- a short **summary** (max/mean absolute Δ, RMSE)

Use this as a static check independent of the YAML parity flow in `embed-in-process`.

## Run

From the repository root (CUDA; set `HF_TOKEN` if the Hub requires it):

```bash
cargo run --release -p hf-embed-card
```

Or:

```bash
examples/hf-embed-card/run.sh
```

Optional flags match `embed-in-process embed` (e.g. `--model`, `--dtype`, `--tokenizer`, memory knobs).

## Layout

| Path | Role |
|------|------|
| `data/input/query/q01.txt`, `q02.txt` | Model card `queries` |
| `data/input/passage/p01.txt`, `p02.txt` | Model card `documents` |

`src/main.rs` loads these via `include_str!` so the binary fails to compile if files are missing or renamed without updating the code.
