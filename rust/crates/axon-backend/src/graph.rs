//! The compiled Bifrost v0.5 decode step — Rust port of
//! `bench_fast.py::build_step` (validated configuration: custom q4 kernels
//! everywhere + fused single-kernel attention).
//!
//! One `mx.compile`'d function per decode step:
//! - all 128 HRM layer-slot executions unrolled inside a single graph
//!   (2 passes × (3 L-stacks + 1 H-stack) × 16 blocks),
//! - K/V for all 128 slots in fixed [128, 12, W, 128] fp16 buffers; each step
//!   reads the past window through the fattn kernel's internal visibility
//!   (`min(pos, W)` rows) and writes the 128 new (k, v) rows with ONE
//!   slice_update per buffer at the end,
//! - everything stays flat 1-D through the whole stack (every kernel wrapper
//!   takes flat vectors),
//! - the current token's own attention term is added explicitly inside fattn
//!   (the buffer write is deferred),
//! - argmax happens inside the graph: one sync roundtrip per step.
//!
//! Closure contract (same as bench_fast):
//! inputs  = (ids [1,1] int32, pos int32 scalar, k_buf, v_buf)
//! outputs = (logits [VOCAB] fp16, k_buf', v_buf', next_id uint32 scalar)
//!
//! Difference vs bench_fast: the KV write start is `pos % WINDOW` instead of
//! `pos` — identical for the validated `pos < WINDOW` regime, and correct
//! sliding-window behavior beyond it (fattn reads only the first
//! `min(pos, W)` rows, which are exactly the in-window rows once the buffer
//! has rotated; the write then evicts the oldest row).

use std::sync::Arc;

use mlx::compile::{CompiledFn, GraphFn};
use mlx::{
    add, argmax, astype, multiply, remainder, reshape, rms_norm, sigmoid, slice_ranges,
    slice_update_dynamic, stack, take_along_axis, Array, Dtype, Result, Stream,
};

use crate::consts::{D, EPS, HC, HD, LC, NL, N_SLOTS, NH, VOCAB, WINDOW};
use crate::model::Model;

/// A compiled decode step plus the model it closes over.
pub struct Decoder {
    #[allow(dead_code)] // kept alive: the compiled closure borrows its graph
    model: Arc<Model>,
    compiled: CompiledFn,
}

/// One decode step's outputs.
pub struct DecodeOut {
    /// [VOCAB] fp16.
    pub logits: Array,
    /// Updated KV buffers (new arrays; inputs are not mutated).
    pub k_buf: Array,
    pub v_buf: Array,
    /// argmax(logits) — uint32, already evaluated (the per-step sync point).
    pub next_id: u32,
}

impl Decoder {
    /// The model this decoder closes over (KV allocation, introspection).
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Build and compile the decode graph (the expensive part; the lazy
    /// warmup apply happens on first use).
    pub fn new(model: Arc<Model>) -> Result<Decoder> {
        let compiled = CompiledFn::compile(StepGraph {
            model: model.clone(),
        })?;
        Ok(Decoder { model, compiled })
    }

    /// Run one decode step for token `tok` at position `pos`.
    pub fn step(
        &self,
        tok: i32,
        pos: i32,
        k_buf: &Array,
        v_buf: &Array,
        s: Stream,
    ) -> Result<DecodeOut> {
        let ids = Array::from_slice(&[tok], &[1, 1], Dtype::I32, s)?;
        let posa = Array::scalar_i32(pos, s)?;
        let outs = self.compiled.apply(&[&ids, &posa, k_buf, v_buf], s)?;
        assert_eq!(outs.len(), 4, "decode step must return 4 outputs");
        let mut it = outs.into_iter();
        let logits = it.next().unwrap();
        let k_buf = it.next().unwrap();
        let v_buf = it.next().unwrap();
        let next_id = it.next().unwrap().item_u32()?;
        Ok(DecodeOut {
            logits,
            k_buf,
            v_buf,
            next_id,
        })
    }
}

/// The graph builder (runs inside the compile closure on every apply).
struct StepGraph {
    model: Arc<Model>,
}

impl GraphFn for StepGraph {
    fn build(&self, inputs: &[Array], out: &mut Vec<Array>, s: Stream) {
        // A failure here surfaces through the trampoline as an MLX error with
        // this message (panics are caught before crossing back into C++).
        match self.build_inner(inputs, s) {
            Ok(mut outs) => out.append(&mut outs),
            Err(e) => panic!("decode graph build failed: {e}"),
        }
    }
}

impl StepGraph {
    #[allow(clippy::too_many_lines)] // the unrolled graph is the point
    fn build_inner(&self, inputs: &[Array], s: Stream) -> Result<Vec<Array>> {
        assert_eq!(inputs.len(), 4, "decode step takes (ids, pos, k_buf, v_buf)");
        let m = &*self.model;
        let k = &m.kernels;

        let ids = &inputs[0];
        let pos = &inputs[1];
        let k_buf = &inputs[2];
        let v_buf = &inputs[3];

        // the pos cast is once per step (not per slot): 2 graph nodes, not 256
        let pf = reshape(&astype(pos, Dtype::F32, s)?, &[1], s)?;

        // x = embed[ids] * EMB_SCALE, flat [D]
        let ids2 = reshape(ids, &[1, 1], s)?;
        let emb = take_along_axis(&m.embed, &ids2, 0, s)?; // [1, 1, D]
        let emb = multiply(&emb, &m.emb_scale, s)?;
        let x = reshape(&emb, &[D], s)?;

        let mut z_l = m.z_l_flat.clone();
        let mut z_h = x.clone();

        // KV write position (sliding window; see module docs)
        let w_const = Array::scalar_i32(WINDOW as i32, s)?;
        let slot_pos = remainder(pos, &w_const, s)?;
        let start = stack(&[&slot_pos], s)?; // [1]

        let mut ups: Vec<Array> = Vec::with_capacity(2 * N_SLOTS);
        let mut slot = 0usize;
        for _ in 0..HC {
            for _ in 0..LC {
                let inp = add(&z_l, &z_h, s)?;
                z_l = self.stack_run(&inp, slot, &pf, k_buf, v_buf, &mut ups, s)?;
                slot += NL;
            }
            let inp = add(&z_h, &z_l, s)?;
            z_h = self.stack_run(&inp, slot, &pf, k_buf, v_buf, &mut ups, s)?;
            slot += NL;
        }
        assert_eq!(slot, N_SLOTS);

        // one slice_update per buffer writes all 128 slots' new (k, v) rows
        let ks: Vec<&Array> = ups.iter().step_by(2).collect();
        let vs: Vec<&Array> = ups.iter().skip(1).step_by(2).collect();
        let ku = reshape(&stack(&ks, s)?, &[N_SLOTS, NH, 1, HD], s)?;
        let vu = reshape(&stack(&vs, s)?, &[N_SLOTS, NH, 1, HD], s)?;
        let k_buf2 = slice_update_dynamic(k_buf, &ku, &start, &[2], s)?;
        let v_buf2 = slice_update_dynamic(v_buf, &vu, &start, &[2], s)?;

        // lm_head (plain q4 gemv) + argmax inside the graph
        let logits = k.qgemv_v6(&z_h, &m.lm_head, s)?;
        debug_assert_eq!(logits.size(), VOCAB);
        let next_id = argmax(&logits, s)?;

        Ok(vec![logits, k_buf2, v_buf2, next_id])
    }

    fn stack_run(
        &self,
        x: &Array,
        slot: usize,
        pf: &Array,
        k_buf: &Array,
        v_buf: &Array,
        ups: &mut Vec<Array>,
        s: Stream,
    ) -> Result<Array> {
        let mut h = x.clone();
        for i in 0..NL {
            h = self.block(&h, slot + i, pf, k_buf, v_buf, ups, s)?;
        }
        rms_norm(&h, None, EPS, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        x: &Array,
        slot: usize,
        pf: &Array,
        k_buf: &Array,
        v_buf: &Array,
        ups: &mut Vec<Array>,
        s: Stream,
    ) -> Result<Array> {
        let layer = &self.model.layers[self.model.slot_layer[slot]];
        let k = &self.model.kernels;
        let d = D as i32;

        // gqkv with the rms_norm fused into the q4-gemv kernel (v7b)
        let gqkv = k.qgemv_v7b(x, &layer.gqkv, s)?; // [6144] = g|q|k|v
        let gl = slice_ranges(&gqkv, &[0..d], s)?; // gate block [1536]

        // fused attention over this slot's window + the always-visible self
        // term; the kernel reads q/k/v blocks of gqkv itself and emits [a|k].
        // The kernel reads NH/W from the k view's shape, so the view must be
        // exactly [NH, W, HD] (Python's k_buf[slot] indexing); the [1, NH, W,
        // HD] slice is contiguous, so the reshape is metadata-only.
        let k_view = slice_ranges(k_buf, &[slot as i32..(slot as i32 + 1)], s)?;
        let v_view = slice_ranges(v_buf, &[slot as i32..(slot as i32 + 1)], s)?;
        let k_view = reshape(&k_view, &[NH, WINDOW, HD], s)?;
        let v_view = reshape(&v_view, &[NH, WINDOW, HD], s)?;
        let ak = k.fattn(&gqkv, &k_view, &v_view, pf, s)?; // [3072]
        let (a, k_row) = crate::kernels::split_ak(&ak, s)?;
        let v_row = slice_ranges(&gqkv, &[(3 * d)..(4 * d)], s)?;

        // o_proj: sigmoid(gate) applied eagerly, gemv + residual in kernel
        let ga = sigmoid(&gl, s)?;
        let ga = multiply(&ga, &a, s)?;
        let h = k.qgemv_v6r(&ga, &layer.o, x, s)?;

        ups.push(k_row);
        ups.push(v_row);

        // MLP: gate_up (v7b) then down with SwiGLU + residual fused (v8a)
        let gu = k.qgemv_v7b(&h, &layer.gate_up, s)?; // [8192]
        k.qgemv_v8a(&gu, &layer.down, &h, s)
    }
}
