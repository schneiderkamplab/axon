//! Full decode-step parity against the validated Python compiled step.
//!
//! `log/bifrost-mlx/export_refstep.py` runs `bench_fast.py --kern --attn fused`
//! on a seeded KV window and dumps inputs + outputs (including the updated KV
//! buffers, which validate the slot-interleaved write path). This test
//! rebuilds the same graph in Rust and compares everything. Skips when the
//! dump or the checkpoint is absent (both gitignored).

use std::path::PathBuf;
use std::sync::Arc;

use axon_backend::{Decoder, Model};
use mlx::{init, Array, Dtype, Stream};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn as_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[..4].try_into().unwrap())
}

fn as_i32(bytes: &[u8]) -> i32 {
    i32::from_le_bytes(bytes[..4].try_into().unwrap())
}

fn f16_val(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn max_abs(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (f16_val(*x) - f16_val(*y)).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn decode_step_matches_python() {
    let root = repo_root();
    let dump = root.join("log/bifrost-mlx/refstep");
    let model_dir = root.join("models/hrm_text/bifrost_v05");
    if !dump.is_dir() || !model_dir.is_dir() {
        eprintln!("skip: missing refstep dump or checkpoint (run log/bifrost-mlx/export_refstep.py)");
        return;
    }
    let _mlx = init();
    let s = Stream::default_stream();

    let read = |name: &str| -> Vec<u8> {
        std::fs::read(dump.join(format!("{name}.bin")))
            .unwrap_or_else(|e| panic!("missing {name}.bin ({e}); run log/bifrost-mlx/export_refstep.py"))
    };

    let tok = as_i32(&read("tok"));
    let pos0 = as_i32(&read("pos0"));
    let kbuf_bits = as_u16(&read("kbuf"));
    let vbuf_bits = as_u16(&read("vbuf"));
    assert_eq!(kbuf_bits.len(), 128 * 12 * 128 * 128, "kbuf shape");
    assert_eq!(vbuf_bits.len(), 128 * 12 * 128 * 128, "vbuf shape");
    let want_logits = as_u16(&read("logits"));
    let want_next = as_u32(&read("nextid"));
    let want_k2 = as_u16(&read("k2"));
    let want_v2 = as_u16(&read("v2"));
    let want_logits0 = as_u16(&read("logits_pos0"));
    let want_next0 = as_u32(&read("nextid_pos0"));

    // model load (bf16 -> f16 cast + q4 packing, ~11 s) + graph compile
    let model = Model::load(&model_dir).expect("model load");
    let decoder = Decoder::new(Arc::new(model)).expect("decode graph compile");

    let kbuf = Array::from_u16_bits(&kbuf_bits, &[128, 12, 128, 128], Dtype::F16, s).unwrap();
    let vbuf = Array::from_u16_bits(&vbuf_bits, &[128, 12, 128, 128], Dtype::F16, s).unwrap();

    // ---- main case: partially filled window (pos0 = 24) ----
    let out = decoder.step(tok, pos0, &kbuf, &vbuf, s).expect("step");
    assert_eq!(out.next_id, want_next, "argmax id mismatch (pos0={pos0})");

    let got_logits = out.logits.data_u16_bits().unwrap();
    let d = max_abs(got_logits, &want_logits);
    assert!(
        got_logits == &want_logits[..],
        "logits not bit-exact vs Python compiled step (max abs {d})"
    );

    let got_k2 = out.k_buf.data_u16_bits().unwrap();
    let dk = max_abs(got_k2, &want_k2);
    assert!(
        got_k2 == &want_k2[..],
        "k_buf update not bit-exact vs Python (max abs {dk})"
    );
    let got_v2 = out.v_buf.data_u16_bits().unwrap();
    let dv = max_abs(got_v2, &want_v2);
    assert!(
        got_v2 == &want_v2[..],
        "v_buf update not bit-exact vs Python (max abs {dv})"
    );

    // ---- edge case: empty window (pos = 0, self term only) ----
    let out0 = decoder.step(tok, 0, &kbuf, &vbuf, s).expect("step pos0");
    assert_eq!(out0.next_id, want_next0, "argmax id mismatch (pos=0)");
    let got_logits0 = out0.logits.data_u16_bits().unwrap();
    let d0 = max_abs(got_logits0, &want_logits0);
    assert!(
        got_logits0 == &want_logits0[..],
        "pos=0 logits not bit-exact vs Python (max abs {d0})"
    );

    println!("decode-step parity: bit-exact (pos0={pos0} next={want_next}, pos=0 next={want_next0})");
}
