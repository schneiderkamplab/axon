//! Bifrost v0.5 (HRM-Text) model backend on MLX, driven from Rust.
//!
//! Ports the validated experiment path in `log/bifrost-mlx/`:
//! - `kernel_qgemv.py` → [`pack`] (q4 packing) and [`kernels`] (custom Metal
//!   kernels: v6/v6r/v7b/v8a gemv family + the fused decode-attention kernel).
//! - `bench_fast.py::build_step` → [`graph`] (the single compiled decode step:
//!   all 128 HRM slot executions unrolled, KV held in fixed window buffers).
//!
//! Semantics notes:
//! - The normalized checkpoint (`models/hrm_text/bifrost_v05/`) is BF16; the
//!   Python loader casts to fp16 at load, so does this backend (`astype`).
//! - The KV write position is `pos % window` (bench_fast writes at `pos`,
//!   which is only valid for `pos < window`); for `pos < window` the two are
//!   identical, and the modulo makes the graph correct for arbitrarily long
//!   generations (sliding window: the fattn kernel already reads only
//!   `min(pos, W)` rows, which are exactly the in-window rows once the buffer
//!   has rotated).

pub mod consts;
pub mod graph;
pub mod kernels;
pub mod model;
pub mod pack;

/// Generated MSL sources (from `log/bifrost-mlx/export_msl.py`).
#[path = "msl.rs"]
pub(crate) mod msl;

pub use graph::{DecodeOut, Decoder};
pub use model::Model;
