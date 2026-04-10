#!/usr/bin/env bash
# Build and run the Hugging Face model-card embedding example (rvllm vs card reference).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

cargo run --release -p hf-embed-card -- "$@"
