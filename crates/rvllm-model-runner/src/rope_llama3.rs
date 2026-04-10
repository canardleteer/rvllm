//! Llama 3-style RoPE inverse-frequency adjustment (matches Hugging Face
//! `modeling_rope_utils._compute_llama3_parameters` for `rope_type == "llama3"`).

use half::f16;

/// Compute inverse frequencies for Llama 3 RoPE scaling.
pub fn compute_inv_freq_llama3(
    head_dim: usize,
    partial_rotary_factor: f32,
    rope_theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position_embeddings: f32,
) -> Vec<f32> {
    let dim = ((head_dim as f32) * partial_rotary_factor).round() as usize;
    let half = dim / 2;
    let mut inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            let idx = (2 * i) as f32;
            1.0 / rope_theta.powf(idx / dim as f32)
        })
        .collect();

    let low_freq_wavelen = original_max_position_embeddings / low_freq_factor;
    let high_freq_wavelen = original_max_position_embeddings / high_freq_factor;
    let wavelen_lo = high_freq_wavelen.min(low_freq_wavelen);
    let wavelen_hi = high_freq_wavelen.max(low_freq_wavelen);

    for slot in &mut inv_freq {
        let wavelen = 2.0 * std::f32::consts::PI / *slot;
        let inv_llama = if wavelen > low_freq_wavelen {
            *slot / factor
        } else {
            *slot
        };
        let smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor)
            / (high_freq_factor - low_freq_factor);
        let smoothed = (1.0 - smooth_factor) * inv_llama / factor + smooth_factor * inv_llama;
        let is_medium = wavelen >= wavelen_lo && wavelen <= wavelen_hi;
        *slot = if is_medium { smoothed } else { inv_llama };
    }

    inv_freq
}

/// Standard (non-scaled) RoPE inverse frequencies: 1 / theta^(2i/dim).
pub fn compute_inv_freq_standard(head_dim: usize, rope_theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            let idx = (2 * i) as f32;
            1.0 / rope_theta.powf(idx / head_dim as f32)
        })
        .collect()
}

/// Precomputed cos/sin tables for positions `0..max_pos` (per frequency pair).
pub struct RopeCosSinCache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half_dim: usize,
    pub max_seq_len: usize,
}

impl RopeCosSinCache {
    /// Build caches from inverse frequencies (length `head_dim/2`).
    pub fn from_inv_freq(inv_freq: &[f32], max_seq_len: usize, head_dim: usize) -> Self {
        let half_dim = head_dim / 2;
        assert_eq!(inv_freq.len(), half_dim);
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];
        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let theta = pos as f32 * inv_freq[i];
                cos[pos * half_dim + i] = theta.cos();
                sin[pos * half_dim + i] = theta.sin();
            }
        }
        Self {
            cos,
            sin,
            half_dim,
            max_seq_len,
        }
    }

    /// Apply RoPE to Q or K layout `[num_tokens, num_heads * head_dim]`.
    pub fn apply(
        &self,
        data: &mut [f16],
        positions: &[u32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let row_stride = num_heads * head_dim;
        for (t, &pos_u) in positions.iter().take(num_tokens).enumerate() {
            let pos = pos_u as usize;
            if pos >= self.max_seq_len {
                continue;
            }
            for h in 0..num_heads {
                let base = t * row_stride + h * head_dim;
                for i in 0..self.half_dim {
                    let cos_val = self.cos[pos * self.half_dim + i];
                    let sin_val = self.sin[pos * self.half_dim + i];
                    let x0 = data[base + 2 * i].to_f32();
                    let x1 = data[base + 2 * i + 1].to_f32();
                    data[base + 2 * i] = f16::from_f32(x0 * cos_val - x1 * sin_val);
                    data[base + 2 * i + 1] = f16::from_f32(x0 * sin_val + x1 * cos_val);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_inv_freq_len_matches_half_dim() {
        let v = compute_inv_freq_standard(128, 500_000.0);
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn llama3_inv_freq_smoke() {
        let v = compute_inv_freq_llama3(128, 1.0, 500_000.0, 32.0, 1.0, 4.0, 8192.0);
        assert_eq!(v.len(), 64);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
