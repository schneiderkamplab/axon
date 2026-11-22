//! Bifrost v0.5 architecture constants (mirrors `bench_fast.py` / config.json).

/// Hidden size.
pub const D: usize = 1536;
/// Attention heads.
pub const NH: usize = 12;
/// Head dim.
pub const HD: usize = 128;
/// Blocks per stack pass.
pub const NL: usize = 16;
/// H-cycle count (outer).
pub const HC: usize = 2;
/// L-cycle count (inner).
pub const LC: usize = 3;
/// rms_norm epsilon.
pub const EPS: f32 = 1e-6;
/// RoPE theta.
pub const THETA: f32 = 10000.0;
/// Embedding scale.
pub const EMB_SCALE: f32 = 39.191835884530846;
/// Number of HRM layer slots executed per decode step (8 stacks x 16 blocks).
pub const N_SLOTS: usize = 128;
/// Attention / KV window.
pub const WINDOW: usize = 128;
/// q4 group size.
pub const GROUP: usize = 64;
/// MLP intermediate size.
pub const INTER: usize = 4096;
/// Vocab size.
pub const VOCAB: usize = 262144;
/// `<turn|>` end-of-turn id.
pub const EOS_ID: u32 = 106;

/// Stack module sequence: `["L","L","L","H"] * 2` (PASS_MODULES in
/// bench_fast.py); slot `s` runs module `PASS_MODULES[s / NL]`, layer
/// `s % NL`.
pub const PASS_MODULES: [&str; N_SLOTS / NL] = ["L", "L", "L", "H", "L", "L", "L", "H"];

/// log2(THETA), hardcoded into the fattn kernel (rope freqs folded in-kernel).
pub const LOG2_THETA: f32 = 13.287712379549449;
