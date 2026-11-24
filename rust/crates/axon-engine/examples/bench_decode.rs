//! Decode-throughput bench for the Rust engine — the mirror of
//! `log/bifrost-mlx/bench_fast.py --kern` (attn=fused, fattn base, window
//! 128, greedy).
//!
//! Methodology matches the llama.cpp-baseline comparisons: load + pack, one
//! warmup apply (kernel build + compiled-apply warmup), untimed 25-step
//! prefill, then N decode steps each timed around its compiled apply;
//! reports mean/median/min ms/step and tok/s.
//!
//!     cd rust && cargo run --release -p axon-engine --example bench_decode -- \
//!         [--tokens 64] [--model ../models/hrm_text/bifrost_v05]
//!
//! For a fair A/B, run the Python twin in the same window:
//!     python3 log/bifrost-mlx/bench_fast.py --kern --tokens 64

use std::path::PathBuf;
use std::time::Instant;

use mlx::Stream;

const EOS: i32 = 106;

fn main() {
    let mut tokens: usize = 64;
    let mut model = PathBuf::from("../models/hrm_text/bifrost_v05");
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--tokens" => tokens = argv.next().and_then(|v| v.parse().ok()).unwrap_or(64),
            "--model" => model = PathBuf::from(argv.next().unwrap_or_default()),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
    }

    let _mlx = mlx::init();
    let s = Stream::default_stream();

    let t = Instant::now();
    let engine = axon_engine::Engine::from_checkpoint(&model).unwrap();
    println!("load+pack+graph: {:.1} s", t.elapsed().as_secs_f64());

    // warmup: one greedy token builds the kernels and warms the compiled apply
    let warm = axon_engine::GenerateParams {
        max_new_tokens: 1,
        temperature: 0.0,
        ..Default::default()
    };
    engine.generate("<bos>Hi", &warm, |_| {}).unwrap();

    let prompt = "<|turn>user\nHej! Kan du fortælle mig kort, hvad Danmarks flag hedder?<turn|>\n<|turn>model\n";
    let ids = engine.encode(prompt).unwrap();
    let (mut k, mut v) = engine.new_kv(s).unwrap();

    // prefill (untimed): the compiled step rides one token per step
    let t = Instant::now();
    let mut next: i32 = 0;
    for (pos, &id) in ids.iter().enumerate() {
        let out = engine.step(id as i32, pos as i32, &k, &v, s).unwrap();
        k = out.k_buf;
        v = out.v_buf;
        next = out.next_id as i32;
    }
    println!(
        "prefill {} steps: {:.0} ms",
        ids.len(),
        t.elapsed().as_secs_f64() * 1e3
    );

    // steady-state decode, timed per step
    let mut step_ms: Vec<f64> = Vec::with_capacity(tokens);
    let mut pos = ids.len() as i32;
    for i in 0..tokens {
        let t = Instant::now();
        let out = engine.step(next, pos, &k, &v, s).unwrap();
        k = out.k_buf;
        v = out.v_buf;
        next = out.next_id as i32;
        pos += 1;
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        // EOS only ends generation, not the timing run: keep stepping past it
        // (feeding the EOS id back) so variant A/Bs compare equal step counts.
    }

    let first_id = engine.tokenizer().decode(&[next as u32], false).unwrap_or_default();
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = step_ms.len() as f64;
    let mean = step_ms.iter().sum::<f64>() / n;
    let median = step_ms[step_ms.len() / 2];
    let min = step_ms[0];
    let p90 = step_ms[(0.9 * (n - 1.0)).round() as usize];
    println!("last token id {next} ({first_id:?})");
    println!(
        "decode {n} steps: mean {mean:.2} ms/step ({:.1} tok/s), median {median:.2}, min {min:.2}, p90 {p90:.2}",
        1e3 / mean
    );
}
