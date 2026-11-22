//! Shared incremental detokenization — Rust port of `server/detokenize.py`.
//!
//! Same contract as the Python engine's streaming: full re-decode each token,
//! emit only the not-yet-emitted suffix (safe across multi-byte characters),
//! end generation at the first stop string with the truncated tail yielded.

use tokenizers::Tokenizer;

/// Truncate before the first stop sequence, or `None`.
///
/// Mirrors Python's list-order scan (not earliest occurrence position).
pub fn truncate_at_stop(text: &str, stops: &[String]) -> Option<String> {
    for stop in stops {
        if !stop.is_empty() {
            if let Some(idx) = text.find(stop.as_str()) {
                return Some(text[..idx].to_string());
            }
        }
    }
    None
}

/// Token ids in, safe text deltas out.
///
/// `feed` decodes everything generated so far and returns only the
/// not-yet-emitted suffix (empty when the tokenizer hasn't produced new safe
/// text yet, e.g. mid multi-byte character). `stopped()` signals a stop-string
/// hit: the delta was truncated and generation must end — the same semantics
/// Session's generate loop has always had.
pub struct TextStream {
    tokenizer: Tokenizer,
    stop: Vec<String>,
    generated: Vec<u32>,
    emitted: usize, // chars, like Python's `self._emitted`
    stopped: bool,
}

impl TextStream {
    pub fn new(tokenizer: Tokenizer, stop: &[String]) -> TextStream {
        TextStream {
            tokenizer,
            stop: stop.to_vec(),
            generated: Vec::new(),
            emitted: 0,
            stopped: false,
        }
    }

    pub fn feed(&mut self, token: u32) -> String {
        self.generated.push(token);
        // Full re-decode, skip_special_tokens=False (Python decode default).
        let text = self
            .tokenizer
            .decode(&self.generated, false)
            .unwrap_or_default();
        let total = text.chars().count();
        let delta: String = text.chars().skip(self.emitted).collect();
        self.emitted = total;
        if let Some(cut) = truncate_at_stop(&delta, &self.stop) {
            self.stopped = true;
            return cut;
        }
        delta
    }

    /// A stop string truncated the stream; the caller must end generation.
    pub fn stopped(&self) -> bool {
        self.stopped
    }
}
