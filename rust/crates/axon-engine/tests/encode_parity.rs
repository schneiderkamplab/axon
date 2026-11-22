//! Tokenizer gate: prompt encoding must match the Python server's path.
//!
//! The Python server encodes through `transformers.AutoTokenizer`, which
//! applies the checkpoint's `fix_mistral_regex` patch on load (see
//! `Engine`'s `load_tokenizer`); the engine must reproduce it bit-for-bit.
//! `log/bifrost-mlx/export_tokref.py` dumps AutoTokenizer ids for a prompt
//! matrix; this test asserts the engine encodes the same ids. Skips when the
//! dump or checkpoint is absent.

use std::path::PathBuf;

use axon_engine::Engine;

#[test]
fn encode_matches_server_path() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let tokref = root.join("log/bifrost-mlx/tokref");
    let model_dir = root.join("models/hrm_text/bifrost_v05");
    if !tokref.is_dir() || !model_dir.is_dir() {
        eprintln!("skip: missing tokref dump or checkpoint (run log/bifrost-mlx/export_tokref.py)");
        return;
    }
    let _mlx = mlx::init();

    let engine = Engine::from_checkpoint(&model_dir).expect("engine load");

    let prompts: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(tokref.join("prompts.json")).unwrap())
            .unwrap();
    let lens_bytes = std::fs::read(tokref.join("lens.bin")).unwrap();
    let lens: Vec<usize> = lens_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    let ids_bytes = std::fs::read(tokref.join("ids.bin")).unwrap();
    let flat: Vec<u32> = ids_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(prompts.len(), lens.len(), "dump row count");
    let mut offset = 0;
    for (i, prompt) in prompts.iter().enumerate() {
        let want = &flat[offset..offset + lens[i]];
        offset += lens[i];
        let got = engine.encode(prompt).unwrap_or_else(|e| panic!("encode {i}: {e}"));
        assert_eq!(got, want, "prompt {i} encodes differently: {prompt:?}");
    }
    assert_eq!(offset, flat.len(), "consumed the whole dump");
}
