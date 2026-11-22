//! Minimal HTTP/1.1 plumbing over `std::net` — the Rust mirror of Python's
//! `BaseHTTPRequestHandler` for the narrow surface this server needs
//! (GET/POST with JSON bodies, JSON responses, SSE streaming). Like the
//! Python server, serving adds zero third-party dependencies.
//!
//! One request per connection: responses always carry `Connection: close`
//! (the SSE path sends it explicitly; keep-alive reconnects are cheap and
//! OpenAI clients handle it transparently).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Read one request. `Ok(None)` on a clean EOF before any bytes (client
/// closed the connection).
pub fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    if method.is_empty() || path.is_empty() {
        return Err(bad("malformed request line"));
    }

    // headers (names lowercased; the only one we act on is content-length)
    let mut headers: HashMap<String, String> = HashMap::new();
    let mut total = line.len();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            return Err(bad("eof in headers"));
        }
        total += header.len();
        if total > MAX_HEADER_BYTES {
            return Err(bad("headers too large"));
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let length: usize = headers
        .get("content-length")
        .map(|v| v.parse().map_err(|_| bad("bad content-length")))
        .transpose()?
        .unwrap_or(0);
    if length > MAX_BODY_BYTES {
        return Err(bad("body too large"));
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

fn bad(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

/// Send a complete JSON response (with `Connection: close`).
pub fn write_json(stream: &mut TcpStream, status: u16, payload: &str) -> std::io::Result<()> {
    let reason = reason_phrase(status);
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    )?;
    stream.write_all(payload.as_bytes())?;
    stream.flush()
}

/// Send the SSE response head (no Content-Length; stream terminates at EOF).
pub fn write_sse_head(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\
          Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "",
    }
}
