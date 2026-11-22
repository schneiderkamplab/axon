//! Endpoint logic — Rust mirror of `server/server.py`'s handler bodies
//! (`_params_from_body`, `_full_completion`, `_sse_chunks`, routing). The
//! layer is socket-free: requests come in as `(method, path, body)`, answers
//! go out as JSON values or an SSE emission closure, so the shape is
//! unit-testable without a network.

use std::sync::Arc;

use serde_json::{json, Value};

use axon_engine::{Engine, GenerateParams};

use crate::chat_template::{self, truthy};
use crate::serve::Slots;

/// Shared server state: one engine, a concurrency cap (Python's
/// `BoundedSemaphore`), request bookkeeping.
pub struct ServeState {
    pub engine: Arc<Engine>,
    pub model_name: String,
    pub slots: Slots,
    /// The template's `bos_token` (read from `tokenizer_config.json`).
    pub bos_token: String,
}

/// What routing decided. SSE carries the already-rendered prompt plus the
/// raw body (stream_options read at stream time).
pub enum Outcome {
    Json { status: u16, payload: Value },
    Sse { kind: Kind, prompt: String, body: Value },
}

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Chat,
    Completion,
}

impl Kind {
    fn object_chunk(self) -> &'static str {
        match self {
            Kind::Chat => "chat.completion.chunk",
            Kind::Completion => "text_completion.chunk",
        }
    }
}

/// Route one request (`do_GET` + `do_POST`).
pub fn handle(state: &ServeState, method: &str, path: &str, body: &[u8]) -> Outcome {
    match (method, path) {
        ("GET", "/health") | ("GET", "/healthz") => Outcome::Json {
            status: 200,
            payload: json!({"ok": true, "model": state.model_name}),
        },
        ("GET", "/v1/models") => Outcome::Json {
            status: 200,
            payload: json!({
                "object": "list",
                "data": [{"id": state.model_name, "object": "model", "owned_by": "axon"}]
            }),
        },
        ("POST", "/v1/completions") => post(state, body, Kind::Completion),
        ("POST", "/v1/chat/completions") => post(state, body, Kind::Chat),
        ("POST", _) => Outcome::Json {
            status: 404,
            payload: error_payload(&format!("not found: {path}"), 404),
        },
        ("GET", _) | (_, _) => Outcome::Json {
            status: 404,
            payload: error_payload(&format!("not found: {path}"), 404),
        },
    }
}

fn post(state: &ServeState, body: &[u8], kind: Kind) -> Outcome {
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("invalid JSON: {e}")),
    };
    if !parsed.is_object() {
        return bad_request("request body must be a JSON object".into());
    }
    match kind {
        Kind::Completion => {
            let prompt = match parsed.get("prompt").and_then(|p| p.as_str()) {
                Some(p) => p.to_string(),
                None => return bad_request("'prompt' must be a string".into()),
            };
            generate_response(state, prompt, &parsed, kind)
        }
        Kind::Chat => {
            let messages = match parsed.get("messages") {
                Some(m) if m.is_array() => m.clone(),
                _ => return bad_request("'messages' must be a list".into()),
            };
            let prompt = render_chat(&state.bos_token, messages.as_array().unwrap());
            generate_response(state, prompt, &parsed, kind)
        }
    }
}

fn bad_request(message: String) -> Outcome {
    Outcome::Json {
        status: 400,
        payload: error_payload(&message, 400),
    }
}

fn error_payload(message: &str, code: u16) -> Value {
    json!({"error": {"message": message, "code": code}})
}

/// `_render_chat`: the checkpoint's chat template (this crate's port), with
/// the same naive-join fallback the Python server keeps for template-less
/// checkpoints.
pub fn render_chat(bos_token: &str, messages: &[Value]) -> String {
    // transformers raises on an empty conversation (the Python server's
    // try/except then falls back); mirror that here.
    let rendered = if messages.is_empty() {
        String::new()
    } else {
        chat_template::render(messages, None, true, bos_token)
    };
    if !rendered.is_empty() {
        return rendered;
    }
    let parts: Vec<String> = messages
        .iter()
        .map(|m| {
            format!(
                "{}: {}",
                m.get("role").and_then(Value::as_str).unwrap_or("user"),
                m.get("content").and_then(Value::as_str).unwrap_or("")
            )
        })
        .collect();
    format!("{}\nassistant:", parts.join("\n"))
}

/// `_generate_response`: streaming goes out as SSE, otherwise a full
/// completion payload (chat wrapped in `chat.completion` shape).
fn generate_response(state: &ServeState, prompt: String, body: &Value, kind: Kind) -> Outcome {
    if body.get("stream").is_some_and(truthy) {
        return Outcome::Sse {
            kind,
            prompt,
            body: body.clone(),
        };
    }
    let params = match params_from_body(body) {
        Ok(p) => p,
        Err(e) => {
            return Outcome::Json {
                status: 500,
                payload: error_payload(&e, 500),
            }
        }
    };
    let _guard = state.slots.acquire();
    let mut pieces = Vec::new();
    let stats = match state
        .engine
        .generate(&prompt, &params, |delta| pieces.push(delta.to_string()))
    {
        Ok(s) => s,
        Err(e) => {
            return Outcome::Json {
                status: 500,
                payload: error_payload(&format!("mlx::Error: {e}"), 500),
            }
        }
    };
    let text = pieces.concat();
    let usage = usage_payload(stats.prompt_tokens, stats.completion_tokens);
    let finish_reason = Value::String(stats.finish_reason.clone());
    let payload = match kind {
        Kind::Completion => json!({
            "object": "text_completion",
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }),
        Kind::Chat => json!({
            "id": request_id("chatcmpl-"),
            "object": "chat.completion",
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }),
    };
    Outcome::Json {
        status: 200,
        payload,
    }
}

/// `_sse_chunks` + the slots guard: stream OpenAI-style SSE chunks through
/// `emit` (each chunk flushed by the caller's writer).
pub fn stream_sse(
    state: &ServeState,
    kind: Kind,
    prompt: &str,
    body: &Value,
    emit: &mut dyn FnMut(&str),
) {
    let request_id = request_id("cmpl-");
    let created = unix_time();
    let object_name = kind.object_chunk();
    let params = match params_from_body(body) {
        Ok(p) => p,
        Err(e) => {
            emit(&format!(
                "data: {}\n\n",
                error_payload(&e, 500)
            ));
            return;
        }
    };
    let include_usage = body
        .get("stream_options")
        .and_then(|o| o.as_object())
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let chunk = |delta: &Value, finish_reason: Option<&str>| -> String {
        let payload = json!({
            "id": request_id,
            "object": object_name,
            "created": created,
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason.map(Value::from).unwrap_or(Value::Null),
            }],
        });
        format!("data: {payload}\n\n")
    };

    let _guard = state.slots.acquire();
    let first = match kind {
        Kind::Chat => json!({"role": "assistant"}),
        Kind::Completion => json!({}),
    };
    emit(&chunk(&first, None));
    let mut stats = state.engine.generate(prompt, &params, |delta| {
        emit(&chunk(&json!({"content": delta}), None));
    });
    match &mut stats {
        Ok(stats) => {
            emit(&chunk(&json!({}), Some(&stats.finish_reason)));
            if include_usage {
                let usage = usage_payload(stats.prompt_tokens, stats.completion_tokens);
                let payload = json!({
                    "id": request_id,
                    "object": object_name,
                    "created": created,
                    "model": state.model_name,
                    "choices": [],
                    "usage": usage,
                });
                emit(&format!("data: {payload}\n\n"));
            }
        }
        Err(e) => {
            // headers are already sent; the connection simply ends (as in
            // Python, where the exception surfaces after the stream started)
            eprintln!("axon-serve: generate failed mid-stream: {e}");
        }
    }
    emit("data: [DONE]\n\n");
}

/// `GenerateParams` from a request body (`_params_from_body`), including the
/// Python falsy-fallthrough semantics (`max_tokens=0` falls back to
/// `max_new_tokens`, then 64). Type errors mirror Python's, which surface as
/// 500s.
pub fn params_from_body(body: &Value) -> Result<GenerateParams, String> {
    let max_tokens_field = || -> Result<u32, String> {
        let default_tokens = Value::from(64);
        let raw = body
            .get("max_tokens")
            .filter(|v| truthy(v))
            .or_else(|| body.get("max_new_tokens").filter(|v| truthy(v)))
            .unwrap_or(&default_tokens);
        int_of(raw).map(|v| v as u32)
    };
    let opt_int = |key: &str| -> Result<Option<usize>, String> {
        match body.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => int_of(v)
                .map(|n| Some(n as usize))
                .map_err(|e| format!("{key}: {e}")),
        }
    };
    let opt_float = |key: &str| -> Result<Option<f64>, String> {
        match body.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => float_of(v)
                .map(Some)
                .map_err(|e| format!("{key}: {e}")),
        }
    };
    let temperature = match body.get("temperature") {
        None => 1.0,
        Some(v) => float_of(v)?,
    };
    let stop = match body.get("stop") {
        Some(v) if truthy(v) => match v {
            Value::String(s) => vec![s.clone()],
            Value::Array(items) => items.iter().map(value_as_python_string).collect(),
            other => vec![value_as_python_string(other)],
        },
        _ => Vec::new(),
    };
    Ok(GenerateParams {
        max_new_tokens: max_tokens_field()?,
        temperature,
        top_k: opt_int("top_k")?,
        top_p: opt_float("top_p")?,
        seed: opt_int("seed")?.map(|v| v as u64),
        stop,
    })
}

/// `int(...)`: numbers (floats truncated), numeric strings, bools (1/0).
fn int_of(v: &Value) -> Result<i64, String> {
    match v {
        Value::Number(n) => Ok(n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64)),
        Value::Bool(b) => Ok(*b as i64),
        Value::String(s) => s
            .trim()
            .parse::<i64>()
            .map_err(|_| conv_err("cannot convert to int", s)),
        other => Err(conv_err("cannot convert to int", &other.to_string())),
    }
}

/// `float(...)`: numbers, numeric strings, bools.
fn float_of(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| conv_err("cannot convert to float", &n.to_string())),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| conv_err("cannot convert to float", s)),
        other => Err(conv_err("cannot convert to float", &other.to_string())),
    }
}

fn conv_err(what: &str, value: &str) -> String {
    format!("{what}: {value:?}")
}

/// Python's `str(value)` for stop entries (strings pass through).
fn value_as_python_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        other => other.to_string(),
    }
}

fn usage_payload(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

fn unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn max_tokens_falsy_fallthrough() {
        // Python: int(body.get("max_tokens") or body.get("max_new_tokens") or 64)
        let params = params_from_body(&json!({"max_tokens": 0, "max_new_tokens": 12})).unwrap();
        assert_eq!(params.max_new_tokens, 12);
        let params = params_from_body(&json!({"max_tokens": 0})).unwrap();
        assert_eq!(params.max_new_tokens, 64);
        let params = params_from_body(&json!({})).unwrap();
        assert_eq!(params.max_new_tokens, 64);
        let params = params_from_body(&json!({"max_tokens": 7})).unwrap();
        assert_eq!(params.max_new_tokens, 7);
    }

    #[test]
    fn defaults_and_optionals() {
        let params = params_from_body(&json!({
            "temperature": 0.0, "top_k": null, "seed": "42", "stop": "END"
        }))
        .unwrap();
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_k, None);
        assert_eq!(params.seed, Some(42));
        assert_eq!(params.stop, vec!["END".to_string()]);

        let params = params_from_body(&json!({"stop": ["a", 2], "top_p": 0.9})).unwrap();
        assert_eq!(params.stop, vec!["a".to_string(), "2".to_string()]);
        assert_eq!(params.top_p, Some(0.9));
    }

    #[test]
    fn type_errors_surface_like_python() {
        // Python: float(None) raises -> 500
        assert!(params_from_body(&json!({"temperature": null})).is_err());
        assert!(params_from_body(&json!({"top_k": ""})).is_err());
        // temperature missing -> 1.0
        assert_eq!(params_from_body(&json!({})).unwrap().temperature, 1.0);
    }

    #[test]
    fn request_ids_have_the_right_shape() {
        let id = request_id("cmpl-");
        assert!(id.starts_with("cmpl-"));
        assert_eq!(id.len(), "cmpl-".len() + 24);
    }
}

/// `cmpl-<24 hex>` / `chatcmpl-<24 hex>` (uuid4().hex[:24] shape).
fn request_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let mut x = nanos ^ (counter << 32) ^ (std::process::id() as u64);
    x ^= x >> 12;
    x = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 25;
    format!("{prefix}{:08x}{:016x}", x >> 32, x & 0xFFFF_FFFF)
}
