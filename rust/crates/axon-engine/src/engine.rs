//! The decode loop — Rust port of `server/engine.py::Session`, specialized to
//! the Rust backend (`axon_backend::Decoder`).
//!
//! Concurrency model (unchanged from Python): per-call decode state (KV
//! buffers are per generation), one shared immutable engine; the serving layer
//! caps concurrent decodes explicitly.

use std::path::Path;
use std::sync::Arc;

use mlx::{Array, Error, Result, Stream};
use tokenizers::Tokenizer;

use axon_backend::{consts, Decoder, Model};

use crate::detokenize::TextStream;
use crate::sampling::{sample_next_token, Rng, SamplerConfig};

/// Sampling parameters for one generation (mirrors `engine.GenerateParams`).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_new_tokens: u32,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            max_new_tokens: 64,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: None,
            stop: Vec::new(),
        }
    }
}

impl GenerateParams {
    fn sampler(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
        }
    }
}

/// Generation bookkeeping (mirrors the Python `stats` dict fields).
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// `"stop"` on EOS/stop-string, `"length"` when the budget is exhausted.
    pub finish_reason: String,
    /// Every generated token id in order (the EOS terminator included), the
    /// `generate_ids` view of the run.
    pub token_ids: Vec<u32>,
}

/// One in-flight generation: KV buffers plus decode bookkeeping.
struct DecodeState {
    k_buf: Array,
    v_buf: Array,
    /// Logits from the prefill's last position; consumed by the first
    /// sampling decision so no extra forward is needed to produce token one.
    pending: Option<(Array, u32)>,
    /// Position the next forward runs at (the first generated token's slot).
    next_pos: i32,
    /// Last sampled token; the next forward's input.
    last_token: i32,
    finished: bool,
}

/// Drives the compiled decode step for streaming generation.
///
/// The engine is immutable and shared (`Arc`-friendly); every `generate` call
/// builds its own [`DecodeState`].
pub struct Engine {
    decoder: Decoder,
    tokenizer: Tokenizer,
}

impl Engine {
    /// Load the model + tokenizer from a normalized checkpoint directory
    /// (`model.safetensors` + `tokenizer.json`).
    pub fn from_checkpoint(dir: &Path) -> Result<Engine> {
        let model = Arc::new(Model::load(dir)?);
        let decoder = Decoder::new(model.clone())?;
        let tokenizer = load_tokenizer(dir)?;
        Ok(Engine { decoder, tokenizer })
    }

    /// The tokenizer (chat-template rendering and introspection in the
    /// serving layer).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Fresh zeroed KV buffers (one per generation), like `Session.prefill`'s
    /// state allocation.
    pub fn new_kv(&self, s: Stream) -> Result<(Array, Array)> {
        self.decoder.model().new_kv(s)
    }

    /// One compiled decode step: token in, logits + in-graph argmax and the
    /// updated KV buffers out. Public for callers that drive the loop
    /// themselves (benchmarks, harnesses).
    pub fn step(
        &self,
        tok: i32,
        pos: i32,
        k_buf: &Array,
        v_buf: &Array,
        s: Stream,
    ) -> Result<axon_backend::DecodeOut> {
        self.decoder.step(tok, pos, k_buf, v_buf, s)
    }

    /// Encode a prompt exactly like `Session._encode` (HF fast tokenizer,
    /// special tokens added — the checkpoint's post-processor prepends BOS).
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error(format!("encode failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Stream a completion, invoking `on_delta` with each decoded text chunk.
    ///
    /// Token accounting mirrors `generate_ids`: `completion_tokens` counts
    /// every generated token including a terminating EOS, which is consumed
    /// but never emitted; a stop-string truncation overrides the finish
    /// reason to `"stop"`.
    pub fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        mut on_delta: impl FnMut(&str),
    ) -> Result<Stats> {
        let s = Stream::default_stream();
        let prompt_ids = self.encode(prompt)?;
        let mut stats = Stats {
            prompt_tokens: prompt_ids.len(),
            ..Stats::default()
        };

        let (k_buf, v_buf) = {
            let model = self.decoder.model();
            model.new_kv(s)?
        };

        // Prefill rides the compiled decode step one token per step (see
        // crate docs); the last step's logits feed the first sample.
        let mut state = DecodeState {
            k_buf,
            v_buf,
            pending: None,
            next_pos: 0,
            last_token: 0,
            finished: false,
        };
        let mut last_out: Option<(Array, u32)> = None;
        for (pos, &tok) in prompt_ids.iter().enumerate() {
            let out = self
                .decoder
                .step(tok as i32, pos as i32, &state.k_buf, &state.v_buf, s)?;
            state.k_buf = out.k_buf;
            state.v_buf = out.v_buf;
            last_out = Some((out.logits, out.next_id));
        }
        state.pending = last_out;
        state.next_pos = prompt_ids.len() as i32;

        let mut sampler_rng = Rng::new(params.seed);
        let greedy = params.sampler().greedy();
        let mut stream = TextStream::new(self.tokenizer.clone(), &params.stop);
        let mut hit_eos = false;

        for _ in 0..params.max_new_tokens {
            if state.finished {
                break;
            }
            // one forward per generated token (the first rides the prefill)
            let token: i32 = if let Some((logits, next_id)) = state.pending.take() {
                if greedy {
                    next_id as i32
                } else {
                    let values = host_logits_f64(&logits)?;
                    sample_next_token(&values, &params.sampler(), &mut sampler_rng) as i32
                }
            } else {
                let out = self.decoder.step(
                    state.last_token,
                    state.next_pos,
                    &state.k_buf,
                    &state.v_buf,
                    s,
                )?;
                state.k_buf = out.k_buf;
                state.v_buf = out.v_buf;
                state.next_pos += 1;
                if greedy {
                    out.next_id as i32
                } else {
                    let values = host_logits_f64(&out.logits)?;
                    sample_next_token(&values, &params.sampler(), &mut sampler_rng) as i32
                }
            };

            state.last_token = token;
            stats.completion_tokens += 1;
            stats.token_ids.push(token as u32);

            if token == consts::EOS_ID as i32 {
                state.finished = true;
                hit_eos = true;
                break;
            }

            let delta = stream.feed(token as u32);
            if !delta.is_empty() {
                on_delta(&delta);
            }
            if stream.stopped() {
                state.finished = true;
                break;
            }
        }
        state.finished = true;

        stats.finish_reason = if hit_eos || stream.stopped() {
            "stop".to_string()
        } else {
            "length".to_string()
        };
        Ok(stats)
    }
}

/// Copy the model's f16 logits to host f64 (the sampling chain's input, like
/// `backend.to_numpy` in the Python engine).
fn host_logits_f64(logits: &Array) -> Result<Vec<f64>> {
    Ok(logits
        .data_u16_bits()?
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f64())
        .collect())
}

/// The Mistral/GPT-style split regex transformers prepends under
/// `fix_mistral_regex` (verbatim from transformers'
/// `TokenizersBackend._patch_mistral_regex` — never change it here without
/// changing it there).
const MISTRAL_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Model types that keep the mistral-regex patch even when config.json
/// records a pre-5.0 transformers version.
const MISTRAL_MODEL_TYPES: [&str; 5] = ["mistral", "mistral3", "voxtral", "ministral", "pixtral"];

/// Load the checkpoint tokenizer the way the Python server sees it.
///
/// The Python server encodes through `transformers.AutoTokenizer`, and for
/// this checkpoint transformers patches the tokenizer on load:
/// `config.json` records no `transformers_version`, the vocab exceeds 100k
/// and the tokenizer has a pre-tokenizer, so
/// `TokenizersBackend._patch_mistral_regex` honors `tokenizer_config.json`'s
/// `fix_mistral_regex: true` and prepends its Mistral regex `Split`
/// (behavior `Isolated`) ahead of the tokenizer.json pre-tokenizer
/// (`Split(" ", MergedWithPrevious)` here). The raw `tokenizers` crate does
/// none of that, and the two disagree on real text: under the patch
/// `"dragon, at"` splits `","` + `"▁"` + `"at"`, without it the space merges
/// into `"▁at"` — a different prompt token sequence, hence different
/// generations. This port reproduces the patch at the JSON level before
/// deserialization, including the decision tree (vocab size, pre-tokenizer
/// presence, config.json `transformers_version` / `model_type` gates and the
/// flag itself); the encode-parity gate (`tests/encode_parity`) pins it
/// against AutoTokenizer dumps.
fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
    use std::str::FromStr as _;
    let text = std::fs::read_to_string(dir.join("tokenizer.json"))
        .map_err(|e| Error(format!("tokenizer.json read failed: {e}")))?;
    let mut tok: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error(format!("tokenizer.json parse failed: {e}")))?;
    if wants_mistral_regex_fix(dir, &tok)? {
        apply_mistral_regex_fix(&mut tok);
    }
    let patched = serde_json::to_string(&tok).expect("tokenizer re-serialize");
    Tokenizer::from_str(&patched).map_err(|e| Error(format!("tokenizer load failed: {e}")))
}

/// Whether transformers would apply the mistral-regex patch for this local
/// checkpoint (the local-dir path of `_patch_mistral_regex`).
fn wants_mistral_regex_fix(dir: &Path, tok: &serde_json::Value) -> Result<bool> {
    // Vocab > 100k and an existing pre-tokenizer, or transformers skips.
    let vocab_size = tok["model"]["vocab"]
        .as_object()
        .map(|v| v.len())
        .or_else(|| tok["model"]["vocab"].as_array().map(|v| v.len()))
        .unwrap_or(0);
    if vocab_size <= 100_000 || tok["pre_tokenizer"].is_null() {
        return Ok(false);
    }
    // config.json gates: a recorded transformers_version ≥ 5 skips the patch
    // (fixed tokenizer writer); < 5 keeps it only for mistral-family model
    // types; no recorded version (this checkpoint) keeps it unconditionally.
    let cfg_text = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(text) => text,
        Err(e) => return Err(Error(format!("config.json read failed: {e}"))),
    };
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text)
        .map_err(|e| Error(format!("config.json parse failed: {e}")))?;
    if let Some(version) = cfg["transformers_version"].as_str() {
        match parse_version_prefix(version) {
            Some((major, minor)) if (major, minor) >= (5, 0) => return Ok(false),
            Some((major, _)) if major < 5 => {
                let model_type = cfg["model_type"].as_str().unwrap_or("");
                if !MISTRAL_MODEL_TYPES.contains(&model_type) {
                    return Ok(false);
                }
            }
            _ => {} // unparseable: transformers would raise; treat as recorded-absent
        }
    }
    // The flag itself, from tokenizer_config.json (the init_kwargs path).
    let tok_cfg_text = match std::fs::read_to_string(dir.join("tokenizer_config.json")) {
        Ok(text) => text,
        Err(e) => return Err(Error(format!("tokenizer_config.json read failed: {e}"))),
    };
    let tok_cfg: serde_json::Value = serde_json::from_str(&tok_cfg_text)
        .map_err(|e| Error(format!("tokenizer_config.json parse failed: {e}")))?;
    Ok(tok_cfg["fix_mistral_regex"].as_bool() == Some(true))
}

/// Prepend the mistral regex `Split` exactly like transformers: replace the
/// first element when the pre-tokenizer is already a `Sequence`, else wrap
/// the existing pre-tokenizer in one.
fn apply_mistral_regex_fix(tok: &mut serde_json::Value) {
    let split = serde_json::json!({
        "type": "Split",
        "pattern": { "Regex": MISTRAL_REGEX },
        "behavior": "Isolated",
        "invert": false,
    });
    let pre = &mut tok["pre_tokenizer"];
    if pre["type"] == "Sequence" {
        pre["pretokenizers"][0] = split;
    } else {
        let current = std::mem::take(pre);
        *pre = serde_json::json!({ "type": "Sequence", "pretokenizers": [split, current] });
    }
}

/// `major.minor` prefix of a version string, ignoring pre-release tags.
fn parse_version_prefix(version: &str) -> Option<(u64, u64)> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}
