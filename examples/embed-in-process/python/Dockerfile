# CUDA runtime + PyTorch reference embeddings (transformers + trust_remote_code).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CUDA-enabled PyTorch (matches CUDA 12.x runtime image)
RUN pip3 install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch \
  && pip3 install --no-cache-dir transformers PyYAML accelerate safetensors

COPY embed_reference.py .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "embed_reference.py"]
