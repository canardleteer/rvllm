//! Full bidirectional multi-head attention with GQA (CPU reference implementation).

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, AttentionMetadata, GpuBuffer, Result};

/// Scaled dot-product attention over the full sequence (non-causal).
pub struct CpuBidirectionalAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl CpuBidirectionalAttention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        }
    }
}

impl AttentionBackend for CpuBidirectionalAttention {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key: &GpuBuffer<f16>,
        value: &GpuBuffer<f16>,
        _metadata: &AttentionMetadata,
        layer_idx: usize,
    ) -> Result<GpuBuffer<f16>> {
        trace!(layer = layer_idx, "cpu bidirectional attention forward");
        let num_tokens = query.shape.first().copied().unwrap_or(0);
        let q_features = self.num_heads * self.head_dim;
        let kv_features = self.num_kv_heads * self.head_dim;
        assert_eq!(query.data.len(), num_tokens * q_features);
        assert_eq!(key.data.len(), num_tokens * kv_features);
        assert_eq!(value.data.len(), num_tokens * kv_features);

        let h = self.num_heads;
        let d = self.head_dim;
        let group = h / self.num_kv_heads.max(1);

        let mut out = vec![f16::ZERO; num_tokens * h * d];

        for head in 0..h {
            let kv_head = head / group;
            for i in 0..num_tokens {
                // softmax over j
                let mut logits = vec![0.0f32; num_tokens];
                let mut max_logit = f32::NEG_INFINITY;
                for (j, logit) in logits.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for k in 0..d {
                        let q_idx = i * q_features + head * d + k;
                        let k_idx = j * kv_features + kv_head * d + k;
                        dot += query.data[q_idx].to_f32() * key.data[k_idx].to_f32();
                    }
                    *logit = dot * self.scale;
                    max_logit = max_logit.max(*logit);
                }
                let mut sum_exp = 0.0f32;
                for logit in &mut logits {
                    *logit = (*logit - max_logit).exp();
                    sum_exp += *logit;
                }
                let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                for logit in &mut logits {
                    *logit *= inv;
                }

                for k in 0..d {
                    let mut acc = 0.0f32;
                    for (j, &logit) in logits.iter().enumerate() {
                        let v_idx = j * kv_features + kv_head * d + k;
                        acc += logit * value.data[v_idx].to_f32();
                    }
                    let o_idx = i * q_features + head * d + k;
                    out[o_idx] = f16::from_f32(acc);
                }
            }
        }

        Ok(GpuBuffer::from_vec(out, vec![num_tokens, q_features]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::AttentionMetadata;

    fn metadata(n: usize) -> AttentionMetadata {
        AttentionMetadata {
            slot_mapping: vec![0; n],
            context_lens: vec![n as u32],
            block_tables: vec![vec![0u32]; n],
            query_lens: vec![1; n],
            max_context_len: n as u32,
        }
    }

    /// With a single token, softmax is trivially 1.0 on that token; output should match V.
    #[test]
    fn single_token_attention_matches_value() {
        let attn = CpuBidirectionalAttention::new(1, 1, 4);
        let d = 4usize;
        let q = GpuBuffer::from_vec(vec![f16::ONE; d], vec![1, d]);
        let k = GpuBuffer::from_vec(vec![f16::ONE; d], vec![1, d]);
        let v = GpuBuffer::from_vec(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![1, d],
        );
        let out = attn.forward(&q, &k, &v, &metadata(1), 0).unwrap();
        assert_eq!(out.shape, vec![1, d]);
        for i in 0..d {
            assert!(
                (out.data[i].to_f32() - v.data[i].to_f32()).abs() < 1e-4,
                "i={i} out={} v={}",
                out.data[i].to_f32(),
                v.data[i].to_f32()
            );
        }
    }

    /// GQA: K/V narrower than Q; output width is still `num_heads * head_dim`.
    #[test]
    fn gqa_output_shape() {
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;
        let num_tokens = 3usize;
        let attn = CpuBidirectionalAttention::new(num_heads, num_kv_heads, head_dim);
        let qf = num_heads * head_dim;
        let kvf = num_kv_heads * head_dim;
        let q = GpuBuffer::from_vec(
            vec![f16::from_f32(0.1); num_tokens * qf],
            vec![num_tokens, qf],
        );
        let k = GpuBuffer::from_vec(
            vec![f16::from_f32(0.2); num_tokens * kvf],
            vec![num_tokens, kvf],
        );
        let v = GpuBuffer::from_vec(
            vec![f16::from_f32(0.3); num_tokens * kvf],
            vec![num_tokens, kvf],
        );
        let out = attn.forward(&q, &k, &v, &metadata(num_tokens), 0).unwrap();
        assert_eq!(out.shape, vec![num_tokens, qf]);
        assert_eq!(out.data.len(), num_tokens * qf);
    }
}
