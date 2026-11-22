//! Checkpoint loading and q4 packing for Bifrost v0.5.
//!
//! The normalized checkpoint (`models/hrm_text/bifrost_v05/model.safetensors`)
//! is BF16; the Python loader (`server/loading.py`) casts to fp16 at load, so
//! does this module — same `astype` op, same rounding. The big matmul weights
//! (gqkv, o, gate_up, down, lm_head) are packed host-side with
//! [`pack_q4_sm`](crate::pack::pack_q4_sm), mirroring `bench_fast.py::kern_pack_weights`
//! (note: o_proj must be packed too — its path contains no QUANT_PATHS
//! substring, a lesson banked in the Python experiment).

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use mlx::{astype, load_safetensors, reshape, Array, Dtype, Error, Result, Stream};

use crate::consts::{D, GROUP, N_SLOTS, NL, PASS_MODULES, WINDOW, NH};
use crate::kernels::Kernels;
use crate::pack::{pack_q4_sm, PackedWeight};

/// One transformer block's weights (q4-packed).
pub struct Layer {
    /// [6144, 1536] — fused g|q|k|v projection.
    pub gqkv: PackedWeight,
    /// [1536, 1536].
    pub o: PackedWeight,
    /// [8192, 1536].
    pub gate_up: PackedWeight,
    /// [1536, 4096].
    pub down: PackedWeight,
}

/// The loaded model: fp16 embedding row-lookup table, q4-packed weights for
/// all 32 distinct blocks (16 L-module + 16 H-module), and the kernel set.
pub struct Model {
    /// [VOCAB, D] fp16 (bf16 checkpoint cast at load).
    pub embed: Array,
    /// z_L_init, fp16, flat [D] (reshaped once at load; constant).
    pub z_l_flat: Array,
    /// EMB_SCALE as an f16 [1] array (weak-scalar parity with the Python
    /// `* EMB_SCALE` broadcast).
    pub emb_scale: Array,
    /// [262144, 1536] packed.
    pub lm_head: PackedWeight,
    /// 32 layers: indices 0..16 = L module, 16..32 = H module.
    pub layers: Vec<Layer>,
    /// slot → layer index (128 slots; mirrors `layer_weights` in bench_fast).
    pub slot_layer: Vec<usize>,
    pub kernels: Kernels,
}

impl Model {
    /// Load the normalized checkpoint from `dir` (expects `model.safetensors`).
    pub fn load(dir: &Path) -> Result<Model> {
        let s = Stream::default_stream();
        // The C-API load returns lazy `Load` nodes (CPU-eval only); the Python
        // loader likewise lands tensors on CPU and casts there. Do the bf16 ->
        // f16 cast on the CPU stream and read the host bits directly — same
        // RNE conversion as the GPU cast, no 2 GB GPU round-trip.
        let sc = Stream::cpu_stream();
        let tensors = load_safetensors(&dir.join("model.safetensors"), sc)?;
        let map: HashMap<String, Array> = tensors.into_iter().collect();
        let get = |name: &str| -> Result<&Array> {
            map.get(name)
                .ok_or_else(|| Error(format!("missing tensor {name}")))
        };

        // Pack one [n, k] bf16 tensor: cast to f16 (CPU), read back, pack,
        // upload. (One-time load cost.)
        let pack_one = |name: &str| -> Result<PackedWeight> {
            let a = get(name)?;
            let w = astype(a, Dtype::F16, sc)?;
            w.eval()?;
            let shape = w.shape();
            let (n, k) = (shape[0], shape[1]);
            assert!(k % GROUP == 0, "{name}: K={k} not divisible by group {GROUP}");
            assert!(
                (k / 2) % 128 == 0,
                "{name}: K/2={} not a multiple of 128 (kernel requirement)",
                k / 2
            );
            let packed = {
                let bits = w.data_u16_bits()?;
                pack_q4_sm(bits, n, k, GROUP)
            };
            packed.to_array(s)
        };

        let mut layers = Vec::with_capacity(32);
        for module in ["L", "H"] {
            for i in 0..NL {
                layers.push(Layer {
                    gqkv: pack_one(&format!("model.{module}_module.layers.{i}.attn.gqkv_proj.weight"))?,
                    o: pack_one(&format!("model.{module}_module.layers.{i}.attn.o_proj.weight"))?,
                    gate_up: pack_one(&format!(
                        "model.{module}_module.layers.{i}.mlp.gate_up_proj.weight"
                    ))?,
                    down: pack_one(&format!("model.{module}_module.layers.{i}.mlp.down_proj.weight"))?,
                });
            }
        }
        let lm_head = pack_one("lm_head.weight")?;

        // slot -> layer index, exactly `layer_weights` from bench_fast.py.
        let mut slot_layer = Vec::with_capacity(N_SLOTS);
        for module in PASS_MODULES {
            let base = if module == "H" { NL } else { 0 };
            for i in 0..NL {
                slot_layer.push(base + i);
            }
        }

        // embed/z_l materialize host-side like the packed weights; MLX migrates
        // them to the GPU on first use in the graph (as in Python).
        let embed = astype(get("model.embed_tokens.weight")?, Dtype::F16, sc)?;
        embed.eval()?;
        let z_l_flat = reshape(&astype(get("model.z_L_init")?, Dtype::F16, sc)?, &[D], sc)?;
        z_l_flat.eval()?;
        let emb_scale = Array::from_u16_bits(
            &[f16::from_f32(crate::consts::EMB_SCALE).to_bits()],
            &[1],
            Dtype::F16,
            s,
        )?;

        // The bf16 source tensors are dead after packing; release the map and
        // MLX's buffer cache so the load-time peak doesn't become the resident
        // footprint (frees ~3.3 GiB of retained load buffers).
        drop(map);
        mlx::clear_cache(sc)?;

        Ok(Model {
            embed,
            z_l_flat,
            emb_scale,
            lm_head,
            layers,
            slot_layer,
            kernels: Kernels::new(GROUP, crate::consts::EPS)?,
        })
    }

    /// Fresh zero KV window buffers, [N_SLOTS, NH, WINDOW, HD] fp16 each.
    pub fn new_kv(&self, s: Stream) -> Result<(Array, Array)> {
        let shape = [N_SLOTS, NH, WINDOW, crate::consts::HD];
        Ok((
            Array::zeros(&shape, Dtype::F16, s)?,
            Array::zeros(&shape, Dtype::F16, s)?,
        ))
    }
}
