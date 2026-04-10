# Python reference (`transformers` + CUDA)

Builds a small image that runs `embed_reference.py` with the same CLI shape as the Rust `embed` subcommand (input layout, YAML schema, prefixes).

- Requires NVIDIA Container Toolkit and a compatible GPU.
- Set `HF_TOKEN` if the model needs Hugging Face authentication.

For a **local** venv (not Docker), install CUDA-enabled PyTorch first, then the rest of `requirements.txt`:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch
pip install -r requirements.txt
```

```bash
docker build -t rvllm-embed-ref ./python
docker run --rm --gpus all \
  -v "$PWD/data/input:/data/input:ro" \
  -v "$PWD/data/out:/data/out" \
  -e HF_TOKEN \
  rvllm-embed-ref \
  --input-dir /data/input --output-yaml /data/out/reference.yaml
```

Prefer the parent [`../README.md`](../README.md) for full compose + `run_parity.sh` flow.
