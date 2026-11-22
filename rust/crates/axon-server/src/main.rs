//! `axon-server` binary: load the engine, warm it, serve HTTP.
//!
//! Mirrors `python -m server`'s session engine (`server/server.py serve`):
//! same endpoints, same defaults, own port so it can run beside the Python
//! server. The model argument is a normalized checkpoint directory (e.g.
//! `models/hrm_text/bifrost_v05`) rather than a `.axon` artifact — that is
//! the one serving-surface difference in v1.

use std::path::PathBuf;
use std::sync::Arc;

use axon_server::{serve, ServeState, Slots};

struct Args {
    model: PathBuf,
    host: String,
    port: u16,
    max_concurrency: usize,
    warmup: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        model: PathBuf::from("models/hrm_text/bifrost_v05"),
        host: "127.0.0.1".into(),
        port: 8000,
        max_concurrency: 2,
        warmup: true,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || {
            argv.next()
                .unwrap_or_else(|| die(&format!("{flag} requires a value")))
        };
        match flag.as_str() {
            "--model" | "--checkpoint" => args.model = PathBuf::from(value()),
            "--host" => args.host = value(),
            "--port" => args.port = value().parse().unwrap_or_else(|_| die("bad --port")),
            "--max-concurrency" => {
                args.max_concurrency = value()
                    .parse()
                    .unwrap_or_else(|_| die("bad --max-concurrency"))
            }
            "--no-warmup" => args.warmup = false,
            other => die(&format!("unknown flag {other}")),
        }
    }
    args
}

fn die(message: &str) -> ! {
    eprintln!("axon-serve: {message}");
    eprintln!("usage: axon-server [--model DIR] [--host H] [--port P] [--max-concurrency N] [--no-warmup]");
    std::process::exit(2);
}

fn main() {
    let args = parse_args();
    let _mlx = mlx::init();

    println!("axon-serve: loading {} ...", args.model.display());
    let engine = Arc::new(
        axon_engine::Engine::from_checkpoint(&args.model)
            .unwrap_or_else(|e| die(&format!("engine load failed: {e}"))),
    );
    let model_name = args
        .model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    // The template's bos_token comes from tokenizer_config.json, like the
    // tokenizer object the Python server renders with.
    let bos_token = std::fs::read_to_string(args.model.join("tokenizer_config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|cfg| {
            cfg.get("bos_token")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default();

    if args.warmup {
        // One greedy token runs the first compiled apply (kernel build +
        // warmup) so the first request doesn't pay it.
        let params = axon_engine::GenerateParams {
            max_new_tokens: 1,
            temperature: 0.0,
            ..Default::default()
        };
        match engine.generate("<bos>Hi", &params, |_| {}) {
            Ok(_) => println!("axon-serve: warmup complete"),
            Err(e) => die(&format!("warmup failed: {e}")),
        }
    }

    let state = Arc::new(ServeState {
        engine,
        model_name,
        slots: Slots::new(args.max_concurrency.max(1)),
        bos_token,
    });
    if let Err(e) = serve::serve(state, &args.host, args.port, args.max_concurrency) {
        die(&format!("serve failed: {e}"));
    }
}
