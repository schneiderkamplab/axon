//! Kernel parity tests for the Bifrost v0.5 Rust port.
//!
//! Two layers of checking:
//! 1. Always-on tiny tests with an independent f32 reference implemented here
//!    (v7b algebra, fattn rope/online-softmax) — no external data needed.
//! 2. Bit-exact parity against the validated Python kernels: test vectors and
//!    Python outputs are dumped by `log/bifrost-mlx/export_ktest.py` into
//!    `log/bifrost-mlx/ktest/` (gitignored). If the dump is absent the
//!    dump-based tests skip, so a fresh checkout still runs layer 1.

use std::path::PathBuf;

use axon_backend::consts::{GROUP, LOG2_THETA};
use axon_backend::kernels::Kernels;
use axon_backend::pack::{pack_q4_sm, PackedBytes, PackedWeight};
use mlx::{init, Array, Dtype, Stream};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn ktest_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../log/bifrost-mlx/ktest");
    p.is_dir().then_some(p)
}

fn read_bytes(dir: &PathBuf, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("missing test vector {name}.bin ({e}); run log/bifrost-mlx/export_ktest.py"))
}

fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn as_u8(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic xorshift64* rng (no external crate).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let x = self.0;
        self.0 = x ^ (x >> 12);
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-lim, lim].
    fn normalish(&mut self, lim: f32) -> f32 {
        // sum of 3 uniforms approximates a normal enough for these tests
        let mut u = || (self.next_u64() >> 40) as f32 / 16_777_216.0 - 0.5;
        (u() + u() + u()) * 2.0 * lim
    }
}

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
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

fn packed_weight(qp: &[u8], sm: &[u16], n: usize, k: usize, s: Stream) -> PackedWeight {
    PackedBytes {
        q: qp.to_vec(),
        sm: sm.to_vec(),
        n,
        k,
        group: GROUP,
    }
    .to_array(s)
    .unwrap()
}

// ---------------------------------------------------------------------------
// layer 1: always-on tiny tests with independent references
// ---------------------------------------------------------------------------

/// Dequantize (q, sm) on the host — the reference meaning of the packed form.
fn dequant(qp: &[u8], sm: &[u16], n: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut w = vec![0f32; n * k];
    for row in 0..n {
        for g in 0..ng {
            let s = f16_val(sm[(row * ng + g) * 2]);
            let m = f16_val(sm[(row * ng + g) * 2 + 1]);
            for j in 0..group {
                let byte = qp[row * (k / 2) + g * (group / 2) + j / 2];
                let q = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
                w[row * k + g * group + j] = m + (q as f32) * s;
            }
        }
    }
    w
}

#[test]
fn v7b_tiny_reference() {
    let _mlx = init();
    let s = Stream::default_stream();
    let kernels = Kernels::new(GROUP, 1e-6).unwrap();

    let mut rng = Rng(0x1234_5678_9abc_def0);
    let (n, k) = (8usize, 256usize);
    let w: Vec<u16> = (0..n * k).map(|_| f16_bits(rng.normalish(0.05))).collect();
    let x: Vec<u16> = (0..k).map(|_| f16_bits(rng.normalish(1.0))).collect();

    let packed = pack_q4_sm(&w, n, k, GROUP);
    let pw = packed.to_array(s).unwrap();
    let xa = Array::from_u16_bits(&x, &[k], Dtype::F16, s).unwrap();
    let y = kernels.qgemv_v7b(&xa, &pw, s).unwrap();
    let y = y.data_u16_bits().unwrap();

    // reference: rms_norm(x) @ dequant(w)^T in f32
    let wdq = dequant(&packed.q, &packed.sm, n, k, GROUP);
    let xf: Vec<f32> = x.iter().map(|&b| f16_val(b)).collect();
    let ssq: f32 = xf.iter().map(|v| v * v).sum();
    let inv = (ssq / k as f32 + 1e-6).sqrt().recip();
    let mut worst = 0f32;
    for row in 0..n {
        let dot: f32 = (0..k).map(|i| xf[i] * wdq[row * k + i]).sum::<f32>() * inv;
        worst = worst.max((dot - f16_val(y[row])).abs());
    }
    assert!(
        worst < 0.02,
        "v7b tiny reference mismatch: max abs {worst}"
    );
}

/// Independent f32 reference for the fused decode-attention kernel (base
/// variant, single [a|k] output). nh/w are read from the k shape like the
/// kernel does; hd is fixed at 128 by the kernel.
fn fattn_reference(
    gqkv: &[u16],
    k_buf: &[u16],
    v_buf: &[u16],
    pos: f32,
    nh: usize,
    w: usize,
) -> Vec<u16> {
    let hd = 128usize;
    let d = nh * hd;
    let halfd = hd / 2;
    let mut out = vec![0u16; 2 * nh * hd];
    let theta_log2 = LOG2_THETA;
    for h in 0..nh {
        let mut qr = vec![0f32; hd];
        let mut kr = vec![0f32; hd];
        for i in 0..hd {
            let ii = if i < halfd { i } else { i - halfd };
            let ang = pos * (2f32).powf(-(2.0 * ii as f32 / hd as f32) * theta_log2);
            // std's sin_cos returns (sin, cos)
            let (sn, c) = ang.sin_cos();
            let q0 = f16_val(gqkv[d + h * hd + i]);
            let k0 = f16_val(gqkv[2 * d + h * hd + i]);
            let (q, k) = if i < halfd {
                (
                    q0 * c - f16_val(gqkv[d + h * hd + i + halfd]) * sn,
                    k0 * c - f16_val(gqkv[2 * d + h * hd + i + halfd]) * sn,
                )
            } else {
                (
                    q0 * c + f16_val(gqkv[d + h * hd + i - halfd]) * sn,
                    k0 * c + f16_val(gqkv[2 * d + h * hd + i - halfd]) * sn,
                )
            };
            qr[i] = q * (hd as f32).recip().sqrt();
            kr[i] = k;
            out[(nh + h) * hd + i] = half::f16::from_f32(kr[i]).to_bits();
        }
        // softmax over [past scores (j < min(pos, w)), self score]
        let vis = pos.min(w as f32) as usize;
        let mut scores = vec![0f32; vis + 1];
        for j in 0..vis {
            let mut sp = 0f32;
            for i in 0..hd {
                sp += qr[i] * f16_val(k_buf[(h * w + j) * hd + i]);
            }
            scores[j] = sp;
        }
        let mut s_self = 0f32;
        for i in 0..hd {
            s_self += qr[i] * kr[i];
        }
        scores[vis] = s_self;
        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ss: f32 = scores.iter().map(|&sc| (sc - mx).exp()).sum();
        for i in 0..hd {
            let mut a = 0f32;
            for j in 0..vis {
                a += (scores[j] - mx).exp() / ss * f16_val(v_buf[(h * w + j) * hd + i]);
            }
            a += (s_self - mx).exp() / ss * f16_val(gqkv[3 * d + h * hd + i]);
            out[h * hd + i] = half::f16::from_f32(a).to_bits();
        }
    }
    out
}

#[test]
fn fattn_tiny_reference() {
    let _mlx = init();
    let s = Stream::default_stream();
    let kernels = Kernels::new(GROUP, 1e-6).unwrap();

    let mut rng = Rng(0x0fed_cba9_8765_4321);
    // nh is fixed at 12 by the wrapper's output spec; shrink the window instead.
    let (nh, w) = (12usize, 16usize);
    let hd = 128usize;
    let d = nh * hd;
    let gqkv: Vec<u16> = (0..4 * d).map(|_| f16_bits(rng.normalish(1.0))).collect();
    let kb: Vec<u16> = (0..nh * w * hd).map(|_| f16_bits(rng.normalish(0.5))).collect();
    let vb: Vec<u16> = (0..nh * w * hd).map(|_| f16_bits(rng.normalish(0.5))).collect();
    let pos = 5.0f32;

    let gqkv_a = Array::from_u16_bits(&gqkv, &[4 * d], Dtype::F16, s).unwrap();
    let k_a = Array::from_u16_bits(&kb, &[nh, w, hd], Dtype::F16, s).unwrap();
    let v_a = Array::from_u16_bits(&vb, &[nh, w, hd], Dtype::F16, s).unwrap();
    let pf = Array::from_slice(&[pos], &[1], Dtype::F32, s).unwrap();

    let ak = kernels.fattn(&gqkv_a, &k_a, &v_a, &pf, s).unwrap();
    let got = ak.data_u16_bits().unwrap();

    let want = fattn_reference(&gqkv, &kb, &vb, pos, nh, w);
    let worst = max_abs(got, &want);
    assert!(
        worst < 0.02,
        "fattn tiny reference mismatch: max abs {worst}"
    );
}

// ---------------------------------------------------------------------------
// layer 2: bit-exact parity with the validated Python kernels (dumped)
// ---------------------------------------------------------------------------

/// Pack parity: Rust pack_q4_sm must reproduce Python's bytes exactly.
#[test]
fn pack_matches_python() {
    let Some(dir) = ktest_dir() else {
        eprintln!("skip: no ktest dump (run log/bifrost-mlx/export_ktest.py)");
        return;
    };
    for (wname, qpname, smname, n, k) in [
        ("w7", "qp7", "sm7", 6144usize, 1536usize),
        ("w8g", "qp8g", "sm8g", 1536, 1536),
        ("w8a", "qp8a", "sm8a", 1536, 4096),
        ("w6", "qp6", "sm6", 256, 1536),
    ] {
        let w = as_u16(&read_bytes(&dir, wname));
        let qp = as_u8(&read_bytes(&dir, qpname));
        let sm = as_u16(&read_bytes(&dir, smname));
        let packed = pack_q4_sm(&w, n, k, GROUP);
        assert_eq!(packed.q.len(), qp.len(), "{wname}: q length");
        assert_eq!(packed.sm.len(), sm.len(), "{wname}: sm length");
        let qdiff = packed
            .q
            .iter()
            .zip(&qp)
            .filter(|(a, b)| a != b)
            .count();
        let sdiff = packed
            .sm
            .iter()
            .zip(&sm)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(qdiff, 0, "{wname}: {qdiff} q bytes differ from Python");
        assert_eq!(sdiff, 0, "{wname}: {sdiff} sm values differ from Python");
    }
}

/// All five gemv kernels bit-exact vs the Python outputs on the same inputs.
#[test]
fn gemv_kernels_match_python() {
    let Some(dir) = ktest_dir() else {
        eprintln!("skip: no ktest dump (run log/bifrost-mlx/export_ktest.py)");
        return;
    };
    let _mlx = init();
    let s = Stream::default_stream();
    let kernels = Kernels::new(GROUP, 1e-6).unwrap();

    // v7b (gqkv-shaped)
    {
        let (n, k) = (6144usize, 1536usize);
        let pw = packed_weight(&read_bytes(&dir, "qp7"), &as_u16(&read_bytes(&dir, "sm7")), n, k, s);
        let x = as_u16(&read_bytes(&dir, "x7"));
        let xa = Array::from_u16_bits(&x, &[k], Dtype::F16, s).unwrap();
        let y = kernels.qgemv_v7b(&xa, &pw, s).unwrap();
        let want = as_u16(&read_bytes(&dir, "y7b"));
        let got = y.data_u16_bits().unwrap();
        let d = max_abs(got, &want);
        assert!(got == &want[..], "v7b not bit-exact (max abs {d})");
    }
    // v6 (lm-head-shaped, small)
    {
        let (n, k) = (256usize, 1536usize);
        let pw = packed_weight(&read_bytes(&dir, "qp6"), &as_u16(&read_bytes(&dir, "sm6")), n, k, s);
        let x = as_u16(&read_bytes(&dir, "x6"));
        let xa = Array::from_u16_bits(&x, &[k], Dtype::F16, s).unwrap();
        let y = kernels.qgemv_v6(&xa, &pw, s).unwrap();
        let want = as_u16(&read_bytes(&dir, "y6"));
        let got = y.data_u16_bits().unwrap();
        let d = max_abs(got, &want);
        assert!(got == &want[..], "v6 not bit-exact (max abs {d})");
    }
    // v6r (o_proj-shaped; inputs shared with the v8g case)
    {
        let (n, k) = (1536usize, 1536usize);
        let pw = packed_weight(&read_bytes(&dir, "qp8g"), &as_u16(&read_bytes(&dir, "sm8g")), n, k, s);
        let x = as_u16(&read_bytes(&dir, "a8g"));
        let res = as_u16(&read_bytes(&dir, "res8g"));
        let xa = Array::from_u16_bits(&x, &[k], Dtype::F16, s).unwrap();
        let ra = Array::from_u16_bits(&res, &[n], Dtype::F16, s).unwrap();
        let y = kernels.qgemv_v6r(&xa, &pw, &ra, s).unwrap();
        let want = as_u16(&read_bytes(&dir, "y6r"));
        let got = y.data_u16_bits().unwrap();
        let d = max_abs(got, &want);
        assert!(got == &want[..], "v6r not bit-exact (max abs {d})");
    }
    // v8a (down_proj-shaped)
    {
        let (n, k) = (1536usize, 4096usize);
        let pw = packed_weight(&read_bytes(&dir, "qp8a"), &as_u16(&read_bytes(&dir, "sm8a")), n, k, s);
        let gu = as_u16(&read_bytes(&dir, "gu8a"));
        let res = as_u16(&read_bytes(&dir, "res8a"));
        let ga = Array::from_u16_bits(&gu, &[2 * k], Dtype::F16, s).unwrap();
        let ra = Array::from_u16_bits(&res, &[n], Dtype::F16, s).unwrap();
        let y = kernels.qgemv_v8a(&ga, &pw, &ra, s).unwrap();
        let want = as_u16(&read_bytes(&dir, "y8a"));
        let got = y.data_u16_bits().unwrap();
        let d = max_abs(got, &want);
        assert!(got == &want[..], "v8a not bit-exact (max abs {d})");
    }
}

#[test]
fn fattn_matches_python() {
    let Some(dir) = ktest_dir() else {
        eprintln!("skip: no ktest dump (run log/bifrost-mlx/export_ktest.py)");
        return;
    };
    let _mlx = init();
    let s = Stream::default_stream();
    let kernels = Kernels::new(GROUP, 1e-6).unwrap();

    let gqkv = as_u16(&read_bytes(&dir, "gqkv_f"));
    let kb = as_u16(&read_bytes(&dir, "k_f"));
    let vb = as_u16(&read_bytes(&dir, "v_f"));
    let pf = as_f32(&read_bytes(&dir, "pf"));
    let gqkv_a = Array::from_u16_bits(&gqkv, &[gqkv.len()], Dtype::F16, s).unwrap();
    let k_a = Array::from_u16_bits(&kb, &[12, 128, 128], Dtype::F16, s).unwrap();
    let v_a = Array::from_u16_bits(&vb, &[12, 128, 128], Dtype::F16, s).unwrap();
    let pf_a = Array::from_slice(&pf, &[1], Dtype::F32, s).unwrap();

    let ak = kernels.fattn(&gqkv_a, &k_a, &v_a, &pf_a, s).unwrap();
    let got = ak.data_u16_bits().unwrap();
    let want = as_u16(&read_bytes(&dir, "ak_f"));
    let d = max_abs(got, &want);
    assert!(got == &want[..], "fattn not bit-exact (max abs {d})");
}
