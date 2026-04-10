# Multi-stage: build embed-in-process from workspace root; runtime + CUDA PTX for GpuLLMEngine.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .
RUN cargo build --release -p embed-in-process
RUN ./kernels/build.sh 2>&1 | tail -30

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y libssl3t64 ca-certificates && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/share/rvllm/kernels
COPY --from=builder /build/target/release/embed-in-process /usr/local/bin/
COPY --from=builder /build/kernels/*.ptx /usr/local/share/rvllm/kernels/

ENV RVLLM_KERNEL_DIR=/usr/local/share/rvllm/kernels
WORKDIR /work
ENTRYPOINT ["embed-in-process"]
