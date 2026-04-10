//! Bidirectional Llama encoder (embedding backbone): same tensor layout as
//! `LlamaForCausalLM` but full-sequence attention and no LM head.

use half::f16;
use tracing::trace;

use crate::architectures::llama::{add_inplace, embed_tokens, get_or_zeros};
use crate::architectures::Architecture;
use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::mlp::MLP;
use crate::layers::norm::RMSNorm;
use crate::rope_llama3::{compute_inv_freq_llama3, compute_inv_freq_standard, RopeCosSinCache};
use crate::runner::ModelRunnerConfig;

use super::llama::LlamaLayer;

/// Llama stack with bidirectional attention (weights match `LlamaModel`).
/// Forward returns last hidden states `[num_tokens, hidden_size]` as f32.
pub struct LlamaBidirectionalModel {
    config: LlamaBidirectionalConfig,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<LlamaLayer>,
    norm_weight: GpuBuffer<f16>,
    rope: RopeCosSinCache,
}

struct LlamaBidirectionalConfig {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
}

impl LlamaBidirectionalModel {
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let cfg = LlamaBidirectionalConfig {
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
        };

        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, cfg.hidden_size]));

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{}", i);
            layers.push(LlamaLayer {
                input_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.input_layernorm.weight"),
                    &[cfg.hidden_size],
                ),
                post_attention_layernorm: get_or_zeros(
                    &weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[cfg.hidden_size],
                ),
                q_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[cfg.num_heads * cfg.head_dim, cfg.hidden_size],
                ),
                k_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                ),
                v_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size],
                ),
                o_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[cfg.hidden_size, cfg.num_heads * cfg.head_dim],
                ),
                gate_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.gate_proj.weight"),
                    &[config.intermediate_size, cfg.hidden_size],
                ),
                up_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.up_proj.weight"),
                    &[config.intermediate_size, cfg.hidden_size],
                ),
                down_proj: get_or_zeros(
                    &weights,
                    &format!("{p}.mlp.down_proj.weight"),
                    &[cfg.hidden_size, config.intermediate_size],
                ),
            });
        }

        let norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[cfg.hidden_size]));

        let inv_freq = match &config.rope_scaling {
            Some(rs) if rs.rope_type == "llama3" => compute_inv_freq_llama3(
                config.head_dim,
                config.partial_rotary_factor,
                config.rope_theta,
                rs.factor,
                rs.low_freq_factor,
                rs.high_freq_factor,
                rs.original_max_position_embeddings as f32,
            ),
            _ => compute_inv_freq_standard(config.head_dim, config.rope_theta),
        };

        let max_pos = config.max_position.max(1);
        let rope = RopeCosSinCache::from_inv_freq(&inv_freq, max_pos, config.head_dim);

        Ok(Self {
            config: cfg,
            embed_tokens,
            layers,
            norm_weight,
            rope,
        })
    }
}

impl Architecture for LlamaBidirectionalModel {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let h = self.config.hidden_size;

        let mut hidden = embed_tokens(&self.embed_tokens, &input.token_ids, h);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "llama_bidirectional layer");

            let normed =
                RMSNorm::forward(&hidden, &layer.input_layernorm, self.config.rms_norm_eps)?;

            let q = LinearLayer::forward(&normed, &layer.q_proj, None)?;
            let k = LinearLayer::forward(&normed, &layer.k_proj, None)?;
            let v = LinearLayer::forward(&normed, &layer.v_proj, None)?;

            let mut q_data = q.data.clone();
            let mut k_data = k.data.clone();
            self.rope.apply(
                &mut q_data,
                &input.position_ids,
                num_tokens,
                self.config.num_heads,
                self.config.head_dim,
            );
            self.rope.apply(
                &mut k_data,
                &input.position_ids,
                num_tokens,
                self.config.num_kv_heads,
                self.config.head_dim,
            );
            let q_rot = GpuBuffer::from_vec(q_data, q.shape.clone());
            let k_rot = GpuBuffer::from_vec(k_data, k.shape.clone());

            let attn_out =
                attention.forward(&q_rot, &k_rot, &v, &input.attention_metadata, layer_idx)?;
            let attn_proj = LinearLayer::forward(&attn_out, &layer.o_proj, None)?;
            add_inplace(&mut hidden, &attn_proj);

            let normed2 = RMSNorm::forward(
                &hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            let mlp_out =
                MLP::forward(&normed2, &layer.gate_proj, &layer.up_proj, &layer.down_proj)?;
            add_inplace(&mut hidden, &mlp_out);
        }

        let normed_final = RMSNorm::forward(&hidden, &self.norm_weight, self.config.rms_norm_eps)?;
        let f32_data: Vec<f32> = normed_final.data.iter().map(|v| v.to_f32()).collect();
        Ok(GpuBuffer::from_vec(f32_data, vec![num_tokens, h]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{CacheEngine, MockAttentionBackend, ModelWeights};
    use crate::input::ModelInput;
    use crate::runner::{ModelRunnerConfig, RopeScalingConfig};
    use rvllm_core::types::Dtype;

    fn make_config(num_layers: usize, hidden_size: usize) -> ModelRunnerConfig {
        let head_dim = hidden_size / 2;
        ModelRunnerConfig {
            num_layers,
            hidden_size,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim,
            intermediate_size: hidden_size * 4,
            vocab_size: 32,
            max_position: 512,
            dtype: Dtype::Float16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rope_scaling: None,
            architecture: "LlamaBidirectionalModel".into(),
        }
    }

    fn make_input(token_ids: Vec<u32>) -> ModelInput {
        let n = token_ids.len();
        ModelInput {
            token_ids,
            position_ids: (0..n as u32).collect(),
            attention_metadata: crate::bridge::AttentionMetadata {
                slot_mapping: vec![],
                context_lens: vec![],
                block_tables: vec![],
                query_lens: vec![1],
                max_context_len: 0,
            },
            is_prefill: true,
        }
    }

    #[test]
    fn constructs_with_default_weights() {
        let config = make_config(1, 8);
        assert!(LlamaBidirectionalModel::new(ModelWeights::default(), &config).is_ok());
    }

    #[test]
    fn constructs_with_llama3_rope_scaling() {
        let mut config = make_config(1, 8);
        config.rope_scaling = Some(RopeScalingConfig {
            rope_type: "llama3".into(),
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        });
        assert!(LlamaBidirectionalModel::new(ModelWeights::default(), &config).is_ok());
    }

    #[test]
    fn forward_returns_hidden_shape() {
        let config = make_config(1, 8);
        let model = LlamaBidirectionalModel::new(ModelWeights::default(), &config).unwrap();
        let input = make_input(vec![1, 2, 3]);
        let cache = CacheEngine::new(1, 64);
        let attn = MockAttentionBackend;
        let out = model.forward(&input, &cache, &attn).unwrap();
        assert_eq!(out.shape, vec![3, 8]);
        assert_eq!(out.data.len(), 24);
    }
}
