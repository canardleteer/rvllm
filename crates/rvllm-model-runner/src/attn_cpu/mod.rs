//! CPU attention backends for architectures that are not using the CUDA graph path.

mod bidirectional;

pub use bidirectional::CpuBidirectionalAttention;
