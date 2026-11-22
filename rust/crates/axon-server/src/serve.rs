//! Connection handling: thread-per-connection accept loop over the shared
//! [`ServeState`] — the Rust mirror of Python's `ThreadingHTTPServer` +
//! `BoundedSemaphore` model. Concurrent decodes share the immutable engine;
//! the slots cap bounds how many run at once.

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};

use crate::api::{self, Outcome, ServeState};
use crate::http;

/// A counting semaphore (`BoundedSemaphore`): N permits, blocking acquire.
pub struct Slots {
    state: Mutex<usize>,
    cond: Condvar,
}

impl Slots {
    pub fn new(permits: usize) -> Slots {
        Slots {
            state: Mutex::new(permits),
            cond: Condvar::new(),
        }
    }

    pub fn acquire(&self) -> SlotGuard<'_> {
        let mut free = self.state.lock().unwrap();
        while *free == 0 {
            free = self.cond.wait(free).unwrap();
        }
        *free -= 1;
        SlotGuard { slots: self }
    }
}

/// RAII permit (released on drop, like `with self.state.slots:`).
pub struct SlotGuard<'a> {
    slots: &'a Slots,
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        let mut free = self.slots.state.lock().unwrap();
        *free += 1;
        self.slots.cond.notify_one();
    }
}

/// Bind and serve forever (Ctrl+C kills the process, like the Python server).
pub fn serve(
    state: Arc<ServeState>,
    host: &str,
    port: u16,
    max_concurrency: usize,
) -> std::io::Result<()> {
    let listener = TcpListener::bind((host, port))?;
    serve_on(listener, state, max_concurrency)
}

/// Serve on an already-bound listener (port 0 gives an ephemeral port for
/// tests; `local_addr` reports the chosen port).
pub fn serve_on(
    listener: TcpListener,
    state: Arc<ServeState>,
    max_concurrency: usize,
) -> std::io::Result<()> {
    let addr = listener.local_addr()?;
    println!(
        "axon-serve: {} on http://{}:{} (max_concurrency={})",
        state.model_name,
        addr.ip(),
        addr.port(),
        max_concurrency
    );
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let state = state.clone();
        std::thread::spawn(move || {
            let _ = handle_connection(state, stream);
        });
    }
    Ok(())
}

fn handle_connection(state: Arc<ServeState>, stream: TcpStream) -> std::io::Result<()> {
    let mut writer = stream.try_clone()?;
    let mut reader = BufReader::new(stream);
    let Some(request) = http::read_request(&mut reader)? else {
        return Ok(());
    };

    match api::handle(&state, &request.method, &request.path, &request.body) {
        Outcome::Json { status, payload } => {
            http::write_json(&mut writer, status, &payload.to_string())?;
        }
        Outcome::Sse { kind, prompt, body } => {
            http::write_sse_head(&mut writer)?;
            let mut emit = |chunk: &str| {
                let _ = writer.write_all(chunk.as_bytes());
                let _ = writer.flush();
            };
            api::stream_sse(&state, kind, &prompt, &body, &mut emit);
        }
    }
    Ok(())
}
