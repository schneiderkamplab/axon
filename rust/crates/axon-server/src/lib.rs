//! Minimal OpenAI-compatible HTTP server over the Rust engine — the mirror of
//! `server/server.py`, serving the normalized checkpoint directly (the Python
//! server stays authoritative for `.axon` artifacts and the fleet engine).
//!
//! Endpoints (identical shapes to the Python server):
//! - `GET  /health`               liveness
//! - `GET  /v1/models`            minimal model listing
//! - `POST /v1/completions`       text completion (`stream` supported, SSE)
//! - `POST /v1/chat/completions`  chat completion (`stream` supported, SSE)
//!
//! Concurrency: thread-per-connection with a bounded-slot cap around every
//! generate (`BoundedSemaphore` semantics); decodes share one immutable
//! [`axon_engine::Engine`], each with its own KV state.

pub mod api;
pub mod chat_template;
pub mod http;
pub mod serve;

pub use api::{params_from_body, render_chat, ServeState};
pub use serve::{serve, serve_on, Slots};
