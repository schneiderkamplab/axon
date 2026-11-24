//! Custom Metal kernels for Bifrost v0.5 decode — the Rust counterparts of
//! the validated `make_*` factories in `log/bifrost-mlx/kernel_qgemv.py`.
//!
//! The MSL text lives in `msl.rs`, extracted verbatim from the Python module
//! by `log/bifrost-mlx/export_msl.py`; this module applies the same
//! placeholder substitutions as the Python factories and registers the
//! kernels through `mlx::MetalKernel`.
//!
//! Kernel set (validated configuration, `bench_fast.py --kern --attn fused`):
//! - `v6`  — plain two-rows-per-simdgroup q4 gemv (lm_head)
//! - `v6r` — v6 + residual add (o_proj; the gate is applied eagerly in-graph)
//! - `v7b` — v6 + rms_norm folded into the loop (gqkv, gate_up)
//! - `v8a` — down_proj with SwiGLU fused in (gu = raw gate_up output)
//! - `fattn` — single-kernel decode attention (rope + online softmax over the
//!   past window + self term), `r4` loop variant (flash-style 4-row merge,
//!   A/B'd faster than `base` at every window size), single [a|k] output

use mlx::kernel::{MetalKernel, OutputSpec};
use mlx::{Array, Dtype, Result, Stream, slice_ranges};

use crate::consts::{HD, NH};
use crate::msl;
use crate::pack::PackedWeight;

/// Assemble the fattn MSL exactly like `make_fattn("r4", single_output=True)`:
/// same substitutions, applied in the same order.
///
/// The `r4` loop (flash-style 4-row online-softmax merge) was A/B'd against
/// `base` same-window on 2026-11-23: −0.7 ms/step at KV window 128
/// (18.1 → 17.4 ms), growing to −1.4 ms at window 1024 — identical results
/// by construction (same online-softmax merge math, wider per-iteration
/// blocks). `t2` (two-pass) measured neutral-to-worse.
fn fattn_source() -> String {
    let mut source = msl::FATTN_SOURCE
        .replace("HD_CONST", "128")
        .replace("LOG2_THETA", "13.287712379549449f")
        .replace("%%FATTN_LOOP%%", msl::FATTN_LOOP_R4);
    // single output: one [a | k] buffer — a at head*HD, k at (NH + head)*HD
    source = source.replace("out_k[head * HD", "out[(NH + head) * HD");
    source = source.replace("out_a[", "out[");
    source
}

/// The registered kernel set (built once at model load).
pub struct Kernels {
    v6: MetalKernel,
    v6r: MetalKernel,
    v7b: MetalKernel,
    v8a: MetalKernel,
    fattn: MetalKernel,
}

impl Kernels {
    pub fn new(group: usize, eps: f32) -> Result<Kernels> {
        let g = group.to_string();
        let eps_s = format!("{eps:e}"); // "1e-6" — valid MSL float literal
        Ok(Kernels {
            v6: MetalKernel::new(
                "qgemv_v6",
                &["x", "q", "sm"],
                &["out"],
                &msl::V6.replace("G_PLACEHOLDER", &g),
            )?,
            v6r: MetalKernel::new(
                "qgemv_v6r",
                &["x", "q", "sm", "res"],
                &["out"],
                &msl::V6R.replace("G_PLACEHOLDER", &g),
            )?,
            v7b: MetalKernel::new(
                "qgemv_v7b",
                &["x", "q", "sm"],
                &["out"],
                &msl::V7B
                    .replace("G_PLACEHOLDER", &g)
                    .replace("EPS_PLACEHOLDER", &eps_s),
            )?,
            v8a: MetalKernel::new(
                "qgemv_v8a",
                &["gu", "q", "sm", "res"],
                &["out"],
                &msl::V8A.replace("G_PLACEHOLDER", &g),
            )?,
            fattn: MetalKernel::new(
                "fattn_r4_so",
                &["gqkv", "k", "vv", "pos"],
                &["out"],
                &fattn_source(),
            )?,
        })
    }

    fn one(out: Vec<Array>) -> Array {
        assert_eq!(out.len(), 1);
        out.into_iter().next().unwrap()
    }

    /// v6: `y = W @ x` (M=1 q4 gemv, plain). x: [k] f16 flat.
    pub fn qgemv_v6(&self, x: &Array, pw: &PackedWeight, s: Stream) -> Result<Array> {
        debug_assert_eq!(x.size(), pw.k);
        let grid = (pw.n as i32 / 2 * 32, 1, 1);
        let out = self.v6.apply(
            &[x, &pw.q, &pw.sm],
            &[OutputSpec::new(&[pw.n], Dtype::F16)],
            grid,
            (32, 1, 1),
            s,
        )?;
        Ok(Self::one(out))
    }

    /// v6r: `y = res + W @ x` (o_proj; gate applied eagerly in-graph).
    pub fn qgemv_v6r(
        &self,
        x: &Array,
        pw: &PackedWeight,
        res: &Array,
        s: Stream,
    ) -> Result<Array> {
        debug_assert_eq!(x.size(), pw.k);
        debug_assert_eq!(res.size(), pw.n);
        let grid = (pw.n as i32 / 2 * 32, 1, 1);
        let out = self.v6r.apply(
            &[x, &pw.q, &pw.sm, res],
            &[OutputSpec::new(&[pw.n], Dtype::F16)],
            grid,
            (32, 1, 1),
            s,
        )?;
        Ok(Self::one(out))
    }

    /// v7b: `y = rms_norm(x, eps) @ W^T` with the norm folded into the loop
    /// (gqkv, gate_up).
    pub fn qgemv_v7b(&self, x: &Array, pw: &PackedWeight, s: Stream) -> Result<Array> {
        debug_assert_eq!(x.size(), pw.k);
        let grid = (pw.n as i32 / 2 * 32, 1, 1);
        let out = self.v7b.apply(
            &[x, &pw.q, &pw.sm],
            &[OutputSpec::new(&[pw.n], Dtype::F16)],
            grid,
            (32, 1, 1),
            s,
        )?;
        Ok(Self::one(out))
    }

    /// v8a: `y = res + W @ swiglu(gu)` (down_proj; gu is the raw gate_up
    /// output of 2k elements).
    pub fn qgemv_v8a(
        &self,
        gu: &Array,
        pw: &PackedWeight,
        res: &Array,
        s: Stream,
    ) -> Result<Array> {
        debug_assert_eq!(gu.size(), 2 * pw.k);
        debug_assert_eq!(res.size(), pw.n);
        let grid = (pw.n as i32 / 2 * 32, 1, 1);
        let out = self.v8a.apply(
            &[gu, &pw.q, &pw.sm, res],
            &[OutputSpec::new(&[pw.n], Dtype::F16)],
            grid,
            (32, 1, 1),
            s,
        )?;
        Ok(Self::one(out))
    }

    /// fattn (base, single output): fused rope + decode attention + roped-k
    /// emission. `gqkv` is the WHOLE [4*D] row (the kernel reads q/k/v at
    /// offsets D/2D/3D), `k`/`v` are this slot's [NH, W, HD] window views,
    /// `pf` is pos as f32 [1]. Returns the [a | k] buffer of 2*NH*HD f16.
    pub fn fattn(
        &self,
        gqkv: &Array,
        k: &Array,
        v: &Array,
        pf: &Array,
        s: Stream,
    ) -> Result<Array> {
        let out = self.fattn.apply(
            &[gqkv, k, v, pf],
            &[OutputSpec::new(&[2 * NH * HD], Dtype::F16)],
            (NH as i32 * 128, 1, 1),
            (128, 1, 1),
            s,
        )?;
        Ok(Self::one(out))
    }
}

/// Slice the fattn [a | k] output into its two flat halves (views).
pub fn split_ak(ak: &Array, s: Stream) -> Result<(Array, Array)> {
    let a = slice_ranges(ak, &[0..(NH * HD) as i32], s)?;
    let k = slice_ranges(ak, &[((NH * HD) as i32)..(2 * NH * HD) as i32], s)?;
    Ok((a, k))
}
