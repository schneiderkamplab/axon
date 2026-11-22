//! Session engine: owns the decode loop and the KV state, model stays pure.
//!
//! Rust port of `server/engine.py` + `server/sampling.py` +
//! `server/detokenize.py`, specialized to the Rust backend (the Python code is
//! backend-agnostic over MLX/torch/fleet; v1 here serves the normalized
//! checkpoint directly through [`axon_backend`], and the backend trait seam
//! arrives with `.axon` artifact support).
//!
//! Differences from the Python engine, all deliberate:
//! - Prefill rides the compiled decode step one token per step (the validated
//!   fast path; numerically identical attention semantics inside the window,
//!   zero parity risk vs `bench_fast`).
//! - Seeded sampling does not reproduce numpy's PCG64 bitstream — greedy
//!   decoding (temperature <= 0) is exact, seeded temperature sampling is
//!   distributionally equivalent.
//! - Greedy steps consume the decode graph's in-graph argmax directly (no
//!   host readback of the 262k logits).

mod detokenize;
mod engine;
mod sampling;

pub use detokenize::TextStream;
pub use engine::{Engine, GenerateParams, Stats};
pub use sampling::{sample_next_token, SamplerConfig};
