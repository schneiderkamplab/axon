//! q4 weight packing — Rust port of `pack_q4_sm` from `kernel_qgemv.py`.
//!
//! Format (Q4_0-style, the project's own packing — no gguf interleave):
//! - `q` [n, k/2] u8: two 4-bit codes per byte, even k in the low nibble
//! - `sm` [n, k/group, 2] f16: per-group (scale, min) interleaved
//! - dequant: `w[i, j] = min[i, j/group] + q(i, j) * scale[i, j/group]`
//!
//! Numeric parity with the Python packer (which the kernels were validated
//! against) requires: f32 min/max/scale, round-ties-even, and f16
//! round-to-nearest-even for the stored scale/min. The `half` crate provides
//! the same RNE conversion numpy uses.

use half::f16;

use mlx::{Array, Dtype, Result, Stream};

/// Host-side packed weight (pure bytes; upload with [`PackedBytes::to_array`]).
pub struct PackedBytes {
    pub q: Vec<u8>,
    pub sm: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub group: usize,
}

/// Device-resident packed weight, ready to hand to the gemv kernels.
pub struct PackedWeight {
    /// [n, k/2] u8.
    pub q: Array,
    /// [n, k/group, 2] f16 (scale, min) interleaved.
    pub sm: Array,
    pub n: usize,
    pub k: usize,
    pub group: usize,
}

impl PackedBytes {
    /// Upload to device arrays (copies; the host bytes can be dropped after).
    pub fn to_array(&self, s: Stream) -> Result<PackedWeight> {
        Ok(PackedWeight {
            q: Array::from_slice(&self.q, &[self.n, self.k / 2], Dtype::U8, s)?,
            sm: Array::from_u16_bits(
                &self.sm,
                &[self.n, self.k / self.group, 2],
                Dtype::F16,
                s,
            )?,
            n: self.n,
            k: self.k,
            group: self.group,
        })
    }
}

/// Pack fp16 `[n, k]` weights (bit view) into (q_packed, sm), mirroring
/// `pack_q4_sm` exactly.
pub fn pack_q4_sm(w_f16: &[u16], n: usize, k: usize, group: usize) -> PackedBytes {
    assert_eq!(w_f16.len(), n * k, "pack_q4_sm: data/shape mismatch");
    assert!(k % group == 0, "K={k} not divisible by group={group}");
    assert!(group % 2 == 0, "pairs must not cross a group boundary");

    let ng = k / group;
    let mut q = vec![0u8; n * (k / 2)];
    let mut sm = vec![0u16; n * ng * 2];

    for row in 0..n {
        let w_row = &w_f16[row * k..(row + 1) * k];
        for g in 0..ng {
            let gv = &w_row[g * group..(g + 1) * group];

            // f32 min/max over the group
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &bits in gv {
                let v = f16::from_bits(bits).to_f32();
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            let mut scale = (mx - mn) / 15.0;
            if scale == 0.0 {
                scale = 1.0;
            }

            // codes: rint((v - mn) / scale).clip(0, 15); ties-to-even like
            // np.rint. Even k in the low nibble, odd k in the high nibble.
            let q_base = row * (k / 2) + g * (group / 2);
            for pair in 0..group / 2 {
                let code = |idx: usize| -> u8 {
                    let v = f16::from_bits(gv[pair * 2 + idx]).to_f32();
                    let t = ((v - mn) / scale).round_ties_even().clamp(0.0, 15.0);
                    t as u8
                };
                q[q_base + pair] = code(0) | (code(1) << 4);
            }

            sm[(row * ng + g) * 2] = f16::from_f32(scale).to_bits();
            sm[(row * ng + g) * 2 + 1] = f16::from_f32(mn).to_bits();
        }
    }

    PackedBytes {
        q,
        sm,
        n,
        k,
        group,
    }
}
