//! Engine gate: greedy generation must match the Python fast path exactly.
//!
//! `log/bifrost-mlx/gen_ref.py` replicates the engine's flow with the Python
//! compiled step (step-wise prefill, in-graph argmax sampling); this test
//! asserts the Rust engine reproduces the same token ids bit-for-bit and the
//! same streamed text. Skips when the dump or checkpoint is absent.

use std::path::PathBuf;

use axon_engine::{Engine, GenerateParams};

const PROMPT: &str = "<|turn>user\nHej! Kan du fortælle mig kort, hvad Danmarks flag hedder?<turn|>\n<|turn>model\n";

#[test]
fn greedy_matches_python() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let gref = root.join("log/bifrost-mlx/gref");
    let model_dir = root.join("models/hrm_text/bifrost_v05");
    if !gref.is_dir() || !model_dir.is_dir() {
        eprintln!("skip: missing gref dump or checkpoint (run log/bifrost-mlx/gen_ref.py)");
        return;
    }
    let _mlx = mlx::init();

    let engine = Engine::from_checkpoint(&model_dir).expect("engine load");
    let params = GenerateParams {
        max_new_tokens: 40,
        temperature: 0.0, // greedy
        ..GenerateParams::default()
    };
    let mut text = String::new();
    let stats = engine
        .generate(PROMPT, &params, |delta| text.push_str(delta))
        .expect("generate");

    // reference ids; the engine stops at the first EOS while the bench twin
    // runs the full 40 steps
    let ids_bytes = std::fs::read(gref.join("ids.bin")).unwrap();
    let ids: Vec<u32> = ids_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let eos = axon_backend::consts::EOS_ID;
    let stop = ids
        .iter()
        .position(|&t| t == eos)
        .map_or(ids.len(), |i| i + 1);
    let want = &ids[..stop];

    assert_eq!(
        stats.token_ids, want,
        "greedy token ids diverge from the Python fast path"
    );
    assert_eq!(stats.completion_tokens, want.len(), "completion_tokens");
    assert_eq!(stats.finish_reason, "stop", "finish_reason");
    let prompt_len_bytes = std::fs::read(gref.join("prompt_len.bin")).unwrap();
    let prompt_len = u32::from_le_bytes(prompt_len_bytes[..4].try_into().unwrap()) as usize;
    assert_eq!(stats.prompt_tokens, prompt_len, "prompt_tokens");

    // streamed text: EOS is consumed but never emitted (the Python engine's
    // contract), so the reference text decodes the prefix without it
    let body = want.strip_suffix(&[eos][..]).unwrap_or(want);
    let want_text = engine.tokenizer().decode(body, false).unwrap();
    assert_eq!(text, want_text, "streamed text mismatch");

    println!("greedy parity: {} ids, text: {}", want.len(), want_text.trim());
}
