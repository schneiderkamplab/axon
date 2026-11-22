//! Chat-template rendering — a Rust port of the checkpoint's
//! `chat_template.jinja` (the Rust `tokenizers` crate has no template
//! support; the Python server renders via HF `transformers`).
//!
//! The port emits exactly the pieces the Jinja source emits after whitespace
//! stripping and is differential-tested byte-for-byte against
//! `transformers.apply_chat_template` (`tests/chatrender.rs`, dumps produced
//! by `log/bifrost-mlx/export_chatref.py`).
//!
//! Template variables: `bos_token` is passed in (the server reads it from
//! `tokenizer_config.json`); `enable_thinking` is never passed by the server,
//! so the corresponding term of the system-block condition is always false —
//! exactly as served today.

use serde_json::Value;

/// Truthiness following Jinja/Python: null, false, 0, "", [], {} are falsy.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Jinja `Undefined` stand-in: a missing key behaves as falsy Null in every
/// position the template uses it (tests, string rendering, iteration).
fn get<'v>(v: &'v Value, key: &str) -> &'v Value {
    v.get(key).unwrap_or(&Value::Null)
}

/// Jinja's `x | upper` for the string-comparison branches (missing → "").
fn upper(v: &Value) -> String {
    v.as_str().unwrap_or("").to_uppercase()
}

/// Jinja `| trim` on a value (missing → "").
fn str_trim(v: &Value) -> &str {
    v.as_str().map_or("", str::trim)
}

/// `x | dictsort`: (key, value) pairs sorted by key. Python's string ordering
/// is by code point, which matches UTF-8 byte order.
fn dictsort(v: &Value) -> Vec<(&String, &Value)> {
    let mut pairs: Vec<(&String, &Value)> = match v {
        Value::Object(map) => map.iter().collect(),
        _ => Vec::new(),
    };
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    pairs
}

// ── template macros ─────────────────────────────────────────────────────────

/// `format_argument(argument, escape_keys)`: strings wrapped in `<|"|>`,
/// booleans as true/false, mappings as `{key:...}` (keys escaped unless
/// `escape_keys` is false), sequences as `[...]`, anything else rendered
/// Python-style (`None` for null).
fn format_argument(v: &Value, escape_keys: bool, out: &mut String) {
    if let Value::String(s) = v {
        out.push_str("<|\"|>");
        out.push_str(s);
        out.push_str("<|\"|>");
    } else if let Value::Bool(b) = v {
        out.push_str(if *b { "true" } else { "false" });
    } else if v.is_object() {
        out.push('{');
        let pairs = dictsort(v);
        for (i, (key, value)) in pairs.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            if escape_keys {
                out.push_str("<|\"|>");
                out.push_str(key);
                out.push_str("<|\"|>");
            } else {
                out.push_str(key);
            }
            out.push(':');
            format_argument(value, escape_keys, out);
        }
        out.push('}');
    } else if let Value::Array(items) = v {
        out.push('[');
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            format_argument(item, escape_keys, out);
        }
        out.push(']');
    } else {
        // numbers render as JSON (matching str(int)/str(float) for sane
        // payloads); Python renders None as "None".
        match v {
            Value::Null => out.push_str("None"),
            other => out.push_str(&other.to_string()),
        }
    }
}

const STANDARD_KEYS: [&str; 5] = ["description", "type", "properties", "required", "nullable"];

/// `format_parameters(properties, required, filter_keys)` — the tool-schema
/// renderer (only reached when tools are passed; the server passes none, but
/// the template's full surface is ported and tested). The Jinja macro
/// declares a `required` parameter it never reads; the underscore name
/// mirrors that faithfully.
#[allow(unused_assignments)] // the template sets `add_comma` in spots nothing later reads
fn format_parameters(properties: &Value, _required: &Value, filter_keys: bool, out: &mut String) {
    let mut found_first = false;
    for (key, value) in dictsort(properties) {
        if filter_keys && STANDARD_KEYS.contains(&key.as_str()) {
            continue;
        }
        let mut add_comma = false;
        if found_first {
            out.push(',');
        }
        found_first = true;
        out.push_str(key);
        out.push_str(":{");
        if truthy(get(value, "description")) {
            out.push_str("description:<|\"|>");
            out.push_str(get(value, "description").as_str().unwrap_or(""));
            out.push_str("<|\"|>");
            add_comma = true;
        }
        // comma helper: each optional section below starts with
        // `{%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}`
        match upper(get(value, "type")).as_str() {
            "STRING" => {
                if truthy(get(value, "enum")) {
                    if add_comma {
                        out.push(',');
                    } else {
                        add_comma = true;
                    }
                    out.push_str("enum:");
                    format_argument(get(value, "enum"), true, out);
                }
            }
            "ARRAY" => {
                let items = get(value, "items");
                if items.is_object() && truthy(items) {
                    if add_comma {
                        out.push(',');
                    } else {
                        add_comma = true;
                    }
                    out.push_str("items:{");
                    let mut found_items = false;
                    for (item_key, item_value) in dictsort(items) {
                        if item_value.is_null() {
                            continue; // `item_value is not none`
                        }
                        if found_items {
                            out.push(',');
                        }
                        found_items = true;
                        match item_key.as_str() {
                            "properties" => {
                                out.push_str("properties:{");
                                if item_value.is_object() {
                                    format_parameters(item_value, get(items, "required"), false, out);
                                }
                                out.push('}');
                            }
                            "required" => {
                                out.push_str("required:[");
                                if let Value::Array(reqs) = item_value {
                                    for (i, req) in reqs.iter().enumerate() {
                                        if i > 0 {
                                            out.push(',');
                                        }
                                        out.push_str("<|\"|>");
                                        out.push_str(req.as_str().unwrap_or(""));
                                        out.push_str("<|\"|>");
                                    }
                                }
                                out.push(']');
                            }
                            "type" => {
                                out.push_str("type:");
                                if item_value.is_string() {
                                    format_argument(&Value::String(upper(item_value)), true, out);
                                } else {
                                    let mapped: Vec<Value> = match item_value {
                                        Value::Array(items) => {
                                            items.iter().map(|v| Value::String(upper(v))).collect()
                                        }
                                        other => vec![Value::String(upper(other))],
                                    };
                                    format_argument(&Value::Array(mapped), true, out);
                                }
                            }
                            _ => {
                                out.push_str(item_key);
                                out.push(':');
                                format_argument(item_value, true, out);
                            }
                        }
                    }
                    out.push('}');
                }
            }
            _ => {}
        }
        if truthy(get(value, "nullable")) {
            if add_comma {
                out.push(',');
            } else {
                add_comma = true;
            }
            out.push_str("nullable:true");
        }
        if upper(get(value, "type")) == "OBJECT" {
            let props = get(value, "properties");
            if !props.is_null() && props.is_object() {
                if add_comma {
                    out.push(',');
                } else {
                    add_comma = true;
                }
                out.push_str("properties:{");
                format_parameters(props, get(value, "required"), false, out);
                out.push('}');
            } else if value.is_object() {
                if add_comma {
                    out.push(',');
                } else {
                    add_comma = true;
                }
                out.push_str("properties:{");
                format_parameters(value, get(value, "required"), true, out);
                out.push('}');
            }
            if truthy(get(value, "required")) {
                if add_comma {
                    out.push(',');
                } else {
                    add_comma = true;
                }
                out.push_str("required:[");
                if let Value::Array(reqs) = get(value, "required") {
                    for (i, req) in reqs.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        out.push_str("<|\"|>");
                        out.push_str(req.as_str().unwrap_or(""));
                        out.push_str("<|\"|>");
                    }
                }
                out.push(']');
            }
        }
        if add_comma {
            out.push(',');
        } else {
            add_comma = true;
        }
        out.push_str("type:<|\"|>");
        out.push_str(&upper(get(value, "type")));
        out.push_str("<|\"|>}");
    }
}

/// `format_function_declaration(tool_data)`.
fn format_function_declaration(tool: &Value, out: &mut String) {
    let function = get(tool, "function");
    out.push_str("declaration:");
    out.push_str(get(function, "name").as_str().unwrap_or(""));
    out.push_str("{description:<|\"|>");
    out.push_str(get(function, "description").as_str().unwrap_or(""));
    out.push_str("<|\"|>");
    let params = get(function, "parameters");
    if truthy(params) {
        out.push_str(",parameters:{");
        if truthy(get(params, "properties")) {
            out.push_str("properties:{");
            format_parameters(get(params, "properties"), get(params, "required"), false, out);
            out.push_str("},");
        }
        if truthy(get(params, "required")) {
            out.push_str("required:[");
            if let Value::Array(reqs) = get(params, "required") {
                for (i, req) in reqs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str("<|\"|>");
                    out.push_str(req.as_str().unwrap_or(""));
                    out.push_str("<|\"|>");
                }
            }
            out.push_str("],");
        }
        if truthy(get(params, "type")) {
            out.push_str("type:<|\"|>");
            out.push_str(&upper(get(params, "type")));
            out.push_str("<|\"|>}");
        }
    }
    if function.as_object().is_some_and(|m| m.contains_key("response")) {
        let response = get(function, "response");
        out.push_str(",response:{");
        if truthy(get(response, "description")) {
            out.push_str("description:<|\"|>");
            out.push_str(get(response, "description").as_str().unwrap_or(""));
            out.push_str("<|\"|>,");
        }
        if upper(get(response, "type")) == "OBJECT" {
            out.push_str("type:<|\"|>");
            out.push_str(&upper(get(response, "type")));
            out.push_str("<|\"|>}");
        }
    }
    out.push('}');
}

/// `strip_thinking(text)`: drop `<|channel>…<channel|>` blocks, keeping the
/// text before each closing marker; the caller sites use the trimmed result.
fn strip_thinking(text: &str, out: &mut String) {
    let mut result = String::new();
    for part in text.split("<channel|>") {
        if part.contains("<|channel>") {
            // part.split('<|channel>')[0] — the text before the closing mark
            result.push_str(part.split("<|channel>").next().unwrap_or(""));
        } else {
            result.push_str(part);
        }
    }
    out.push_str(result.trim());
}

/// `format_tool_response_block(tool_name, response)`.
fn format_tool_response_block(tool_name: &str, response: &Value, out: &mut String) {
    out.push_str("<|tool_response>");
    if response.is_object() {
        out.push_str("response:");
        out.push_str(tool_name);
        out.push('{');
        let pairs = dictsort(response);
        for (i, (key, value)) in pairs.iter().enumerate() {
            out.push_str(key);
            out.push(':');
            format_argument(value, false, out);
            if i + 1 < pairs.len() {
                out.push(',');
            }
        }
        out.push('}');
    } else {
        out.push_str("response:");
        out.push_str(tool_name);
        out.push_str("{value:");
        format_argument(response, false, out);
        out.push('}');
    }
    out.push_str("<tool_response|>");
}

// ── the template body ───────────────────────────────────────────────────────

/// Render a conversation exactly as `apply_chat_template(messages,
/// tokenize=False, add_generation_prompt=add_generation_prompt)` does.
///
/// `tools` mirrors the template variable (never passed by the server; kept
/// for fidelity and tested). Messages are raw JSON objects.
#[allow(unused_assignments)] // the template sets `prev_message_type` in spots nothing later reads
pub fn render(
    messages: &[Value],
    tools: Option<&[Value]>,
    add_generation_prompt: bool,
    bos_token: &str,
) -> String {
    let mut out = String::new();
    out.push_str(bos_token);

    let mut prev_message_type: Option<&str> = None;
    let mut loop_messages: &[Value] = messages;

    // System / tool-definitions block: enable_thinking is never passed (so
    // that term is false), leaving `tools or messages[0]['role'] in
    // ['system', 'developer']`.
    let tools_truthy = tools.is_some_and(|t| truthy(&Value::Array(t.to_vec())));
    let first_role = messages.first().map_or("", |m| get(m, "role").as_str().unwrap_or(""));
    let first_is_sys = first_role == "system" || first_role == "developer";
    if tools_truthy || first_is_sys {
        out.push_str("<|turn>system\n");
        if first_is_sys {
            let content = get(&messages[0], "content");
            if let Some(text) = content.as_str() {
                out.push_str(text.trim());
            } else if let Value::Array(items) = content {
                for item in items {
                    out.push_str(str_trim(get(item, "text")));
                    out.push(' ');
                }
            }
            loop_messages = &messages[1..];
        }
        if let Some(tools) = tools {
            if !tools.is_empty() {
                for tool in tools {
                    out.push_str("<|tool>");
                    let mut decl = String::new();
                    format_function_declaration(tool, &mut decl);
                    out.push_str(decl.trim());
                    out.push_str("<tool|>");
                }
                prev_message_type = Some("tool");
            }
        }
        out.push_str("<turn|>\n");
    }

    // Pre-scan: last user message index (reasoning guard).
    let last_user_idx = loop_messages
        .iter()
        .rposition(|m| get(m, "role").as_str() == Some("user"));

    for (i, message) in loop_messages.iter().enumerate() {
        let role_raw = get(message, "role");
        if role_raw.as_str() == Some("tool") {
            continue;
        }
        prev_message_type = None;
        let role = match role_raw.as_str() {
            Some("assistant") => "model",
            Some(other) => other,
            None => "",
        };

        // Continuation: suppress a duplicate <|turn>model when the previous
        // non-tool message was also assistant.
        let mut prev_nt: Option<&str> = None;
        if i > 0 {
            for j in (0..i).rev() {
                let r = get(&loop_messages[j], "role").as_str();
                if r != Some("tool") {
                    prev_nt = r;
                    break;
                }
            }
        }
        let continue_same_model_turn = role == "model" && prev_nt == Some("assistant");
        if !continue_same_model_turn {
            out.push_str("<|turn>");
            out.push_str(role);
            out.push('\n');
        }

        // Reasoning rendered as a thinking channel (assistant continuation
        // with tool calls after the last user turn).
        let thinking = match get(message, "reasoning") {
            v if truthy(v) => v,
            _ => get(message, "reasoning_content"),
        };
        if truthy(thinking)
            && i > last_user_idx.unwrap_or(usize::MAX)
            && truthy(get(message, "tool_calls"))
        {
            out.push_str("<|channel>thought\n");
            out.push_str(thinking.as_str().unwrap_or(""));
            out.push_str("\n<channel|>");
        }

        let mut tr_flag = false;
        let tool_responses = get(message, "tool_responses");
        let tool_calls = get(message, "tool_calls");
        if truthy(tool_responses) {
            // Legacy: tool_responses embedded on the assistant message.
            if let Value::Array(resps) = tool_responses {
                for resp in resps {
                    let name = get(resp, "name").as_str().unwrap_or("unknown");
                    format_tool_response_block(name, get(resp, "response"), &mut out);
                    tr_flag = true;
                    prev_message_type = Some("tool_response");
                }
            }
        } else if truthy(tool_calls) {
            if let Value::Array(calls) = tool_calls {
                for call in calls {
                    let function = get(call, "function");
                    out.push_str("<|tool_call>call:");
                    out.push_str(get(function, "name").as_str().unwrap_or(""));
                    out.push('{');
                    let args = get(function, "arguments");
                    if args.is_object() {
                        let pairs = dictsort(args);
                        for (j, (key, value)) in pairs.iter().enumerate() {
                            if j > 0 {
                                out.push(',');
                            }
                            out.push_str(key);
                            out.push(':');
                            format_argument(value, false, &mut out);
                        }
                    } else if let Some(s) = args.as_str() {
                        out.push_str(s);
                    }
                    out.push_str("}<tool_call|>");
                }
            }
            prev_message_type = Some("tool_call");

            // OpenAI Chat Completions: forward-scan consecutive role:tool
            // messages following this assistant turn.
            let mut stopped = false;
            for k in (i + 1)..loop_messages.len() {
                if stopped {
                    continue;
                }
                let follow = &loop_messages[k];
                if get(follow, "role").as_str() != Some("tool") {
                    stopped = true;
                } else {
                    // Python's `.get('name') | default('unknown')` leaves a
                    // None name as None (and then fails); serving a missing
                    // name as "unknown" is the deliberate divergence here.
                    let mut tname = get(follow, "name").as_str().unwrap_or("unknown");
                    for call in tool_calls.as_array().map_or(&[][..], |c| c) {
                        let tc_id = get(call, "id").as_str();
                        if tc_id.is_some() && tc_id == get(follow, "tool_call_id").as_str() {
                            tname = get(get(call, "function"), "name").as_str().unwrap_or(tname);
                        }
                    }
                    let tool_body = get(follow, "content");
                    if tool_body.is_object() {
                        format_tool_response_block(tname, tool_body, &mut out);
                    } else if let Some(s) = tool_body.as_str() {
                        format_tool_response_block(tname, &Value::String(s.to_owned()), &mut out);
                    } else if tool_body.is_array() {
                        let mut text = String::new();
                        for part in tool_body.as_array().unwrap() {
                            if get(part, "type").as_str() == Some("text") {
                                text.push_str(get(part, "text").as_str().unwrap_or(""));
                            }
                        }
                        format_tool_response_block(tname, &Value::String(text), &mut out);
                    } else {
                        format_tool_response_block(tname, tool_body, &mut out);
                    }
                    tr_flag = true;
                    prev_message_type = Some("tool_response");
                }
            }
        }

        // Captured content (assistant content is thinking-stripped; other
        // roles are trimmed; parts arrays concatenate typed pieces).
        let mut captured = String::new();
        let content = get(message, "content");
        if let Some(text) = content.as_str() {
            if role == "model" {
                strip_thinking(text, &mut captured);
            } else {
                captured.push_str(text.trim());
            }
        } else if let Value::Array(parts) = content {
            for part in parts {
                match get(part, "type").as_str() {
                    Some("text") => {
                        let text = get(part, "text").as_str().unwrap_or("");
                        if role == "model" {
                            strip_thinking(text, &mut captured);
                        } else {
                            captured.push_str(text.trim());
                        }
                    }
                    Some("image") => {
                        captured.push_str("<|image|>");
                        prev_message_type = Some("image");
                    }
                    Some("audio") => {
                        captured.push_str("<|audio|>");
                        prev_message_type = Some("audio");
                    }
                    Some("video") => {
                        captured.push_str("<|video|>");
                        prev_message_type = Some("video");
                    }
                    _ => {}
                }
            }
        }
        let has_content = !captured.trim().is_empty();
        out.push_str(&captured);

        if prev_message_type == Some("tool_call") && !tr_flag {
            out.push_str("<|tool_response>");
        } else if !(tr_flag && !has_content) {
            out.push_str("<turn|>\n");
        }
    }

    if add_generation_prompt
        && prev_message_type != Some("tool_response")
        && prev_message_type != Some("tool_call")
    {
        out.push_str("<|turn>model\n");
    }
    out
}
