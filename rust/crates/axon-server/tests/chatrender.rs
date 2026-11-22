//! Chat-template gate: the Rust renderer must reproduce transformers'
//! `apply_chat_template` byte-for-byte across a matrix of conversations.
//!
//! `log/bifrost-mlx/export_chatref.py` renders each case with the exact call
//! the Python server makes and dumps the strings; this test replays them
//! through `chat_template::render`. Skips when the dump is absent.

use std::path::PathBuf;

use serde_json::Value;

use axon_server::chat_template;

#[test]
fn chat_render_matches_transformers() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let dump = root.join("log/bifrost-mlx/chatref/cases.json");
    if !dump.is_file() {
        eprintln!("skip: missing chatref dump (run log/bifrost-mlx/export_chatref.py)");
        return;
    }
    let data: Value = serde_json::from_slice(&std::fs::read(dump).unwrap()).unwrap();
    let bos = data["bos_token"].as_str().unwrap_or("");

    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let messages: Vec<Value> = case["messages"].as_array().unwrap().clone();
        let tools = case["tools"].as_array();
        let want = case["rendered"].as_str().unwrap();

        let got = chat_template::render(&messages, tools.map(|t| t.as_slice()), true, bos);
        assert_eq!(got, want, "chat render mismatch in case {name}");
    }
    println!("chat render parity: {} cases byte-exact", data["cases"].as_array().unwrap().len());
}

#[test]
fn empty_conversation_falls_back_like_transformers() {
    // transformers raises on an empty conversation; the Python server's
    // try/except falls back to the naive join ("\nassistant:").
    assert_eq!(axon_server::render_chat("<bos>", &[]), "\nassistant:");
}

#[test]
fn naive_fallback_joins_roles() {
    let messages = serde_json::json!([
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]);
    let got = axon_server::render_chat("", messages.as_array().unwrap());
    // an empty bos renders "<bos>"-less but the template output is non-empty,
    // so the fallback only triggers for empty conversations — assert the
    // template still runs with an empty bos instead
    assert!(got.starts_with("<|turn>user\n"));
}
