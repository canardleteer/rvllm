#!/usr/bin/env bash
# Run Rust embed, Python reference, then Rust compare (same volume mounts as compose.yaml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

INPUT_DIR="${INPUT_DIR:-$ROOT/data/input}"
OUT_DIR="${OUT_DIR:-$ROOT/data/out}"
MODEL="${MODEL:-nvidia/llama-nv-embed-reasoning-3b}"
FAIL_BELOW="${FAIL_BELOW:-}"

export INPUT_DIR
export OUT_DIR

mkdir -p "$OUT_DIR" "$INPUT_DIR/query" "$INPUT_DIR/passage"

echo "Using INPUT_DIR=$INPUT_DIR OUT_DIR=$OUT_DIR MODEL=$MODEL"

docker compose -f "$ROOT/compose.yaml" build

docker compose -f "$ROOT/compose.yaml" run --rm embed-rust embed \
  --input-dir /data/input \
  --output-yaml /data/out/rust.yaml \
  --model "$MODEL"

docker compose -f "$ROOT/compose.yaml" run --rm embed-python \
  --input-dir /data/input \
  --output-yaml /data/out/reference.yaml \
  --model "$MODEL"

COMPARE_ARGS=(compare --rust-yaml /data/out/rust.yaml --reference-yaml /data/out/reference.yaml)
if [[ -n "$FAIL_BELOW" ]]; then
  COMPARE_ARGS+=(--fail-below-cosine "$FAIL_BELOW")
fi

docker compose -f "$ROOT/compose.yaml" run --rm embed-rust "${COMPARE_ARGS[@]}"

echo "Parity run finished. YAML outputs: $OUT_DIR/rust.yaml and $OUT_DIR/reference.yaml"
