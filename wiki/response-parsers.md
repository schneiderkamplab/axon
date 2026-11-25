# Cross-model response parsers (reasoning + tool calls)

Design for parsing model output into OpenAI-compatible `reasoning_content`
and `tool_calls`, working across all models the serving layer can host.
Status: design agreed 2026-11-25; Steps 1–3 landed the same day — Step 1
(think parser + vectors + chat surface wiring), Step 2 (tool parsers
`hermes` + `gemma4_calls`, reasoning parser `gemma4_thought`, combo wiring,
gemma-4 template round-trip gate), Step 3 (full-stream `harmony` protocol
parser, official gpt-oss template as ground truth). All in
`server/response_parsers.py`, vectors in `server/tests/parsers/vectors/`,
CLI `--reasoning-parser` / `--tool-call-parser` (default off; config.json
key resolution lands at Step 4).

## Decisions (user, 2026-11-25)

- **Spec source: explicit config keys.** `config.json` gains optional
  `"reasoning_parser"` and `"tool_call_parser"` keys selecting generic
  format implementations from a registry. Absent → raw text passthrough.
  No auto-detection, no template sniffing — same dispatch philosophy as
  `id2label` for classify. CLI overrides on serve for ad-hoc cases.
- **First pass: `think`, `hermes`, `harmony`** — plus, per user note, the
  **gemma-4 pair** (`gemma4_thought` + `gemma4_calls`): Bifrost and
  HRM-Mimir both use the gemma-4 chat template, so serving them with
  thinking/tools needs those formats from day one.

## Principles

1. **Formats, not models.** The ecosystem has a handful of textual
   output conventions; each is one generic implementation in a registry.
   No model- or family-name branches anywhere (repo policy).
2. **Parse decoded text, not token ids.** Verified: Bifrost declares zero
   special tokens — every marker (`<|turn>`, `<|channel>thought`,
   `<|tool_call>call:`) is plain text split arbitrarily across BPE tokens.
   Single-token specials (harmony's `<|channel|>` etc.) are handled by the
   same text matcher. One mechanism covers both.
3. **Parser sits downstream of the detokenizer.**
   `tokens → incremental detokenizer (marker-preserve) → response parser
   (stateful, per request) → event stream → OpenAI events`.
   The detokenizer gains a marker-preserve mode: it renders exactly the
   special tokens the checkpoint's parser spec declares (resolved to ids at
   load) and strips undeclared ones — so markers never leak into content
   and EOS stays engine-side. Existing stop-string machinery is unchanged.
4. **State is per request, per sequence** — fleet continuous batching gets
   one parser instance per running sequence, like session mode.

## Event model (the cross-model piece)

One state machine per parser combo emits an ordered event stream:

- `reasoning_delta(text)`
- `content_delta(text)`
- `tool_call_begin(index, name)`
- `tool_args_delta(index, text)`
- `tool_call_end(index)`

HTTP maps events mechanically: streaming deltas become
`delta.reasoning_content` / `delta.tool_calls[{index, function{...}}]`
fragments (OpenAI accumulate semantics); non-stream aggregates into
`message.reasoning_content` / `message.tool_calls`.
**Event order varies by format** (Qwen3: reasoning → content → tools;
harmony: analysis → commentary(tools) → final; gemma4: thought → calls →
content). Implemented as composition, not N×M machines: the tool parser
owns a marker segmenter on its tool-open/tool-close markers and routes all
body text to the reasoning parser in stream order — one coherent state
machine per combo, built from one reasoning parser + one tool machine.

## Streaming mechanics (the hard 10%)

- **Holdback**: never emit a suffix that could be the prefix of any marker
  declared by the active combo; buffer up to `maxlen(markers) − 1` chars,
  flush on match or when continuation becomes impossible. Handles
  `<|channel>thought` split across four tokens for free. Holdback set =
  union of the active reasoning + tool parser markers.
- **Block-boundary flush**: when the tool machine removes a block from the
  text stream, the reasoning parser's held marker-prefix is force-flushed
  first — the raw stream continues with a tool marker, which proves the
  held suffix is not a reasoning marker and must not join across the
  removed block. (Caught by the gemma4 vector: a held `\n` that could open
  `\n<channel|>` would otherwise merge with post-block text.)
- **Tool args**: stream the raw argument text between the call's braces as
  deltas; validate by parsing at `tool_call_end`. Malformed JSON still
  streams (client-visible), with the parse error surfaced at the end.
- **Non-stream path**: same machine fed the whole string, mapped once.

## First-pass format registry

| key | reasoning | tools | notes |
|---|---|---|---|
| `think` | HTML-style think open/close tags (&lt;think&gt; … &lt;/think&gt;) | — | the **open tag classifies reasoning**; no markers ⇒ all content (Qwen3 non-thinking); open-without-close at EOS ⇒ all reasoning (truncated think); close-without-open is stripped, not reclassified — bare-reasoning-then-close (DeepSeek-R1) is a separate `deepseek` format, deferred until such a checkpoint is hosted |
| `hermes` | — | XML-style tool-call tags (&lt;tool_call&gt;{json}&lt;/tool_call&gt;) | JSON body `{"name","arguments"}`; multiple blocks per turn |
| `harmony` | analysis channel | commentary channel w/ `to=functions.name` recipient | gpt-oss `<|start|>/<|channel|>/<|message|>/<|end|>/<|return|>` protocol; final channel = content; **full-stream parser** (owns channel classification, `tool_call_parser` key, no reasoning parser alongside) |
| `gemma4_thought` | `<|channel>thought ... <channel|>` | — | Bifrost/HRM-Mimir; plain-text markers |
| `gemma4_calls` | — | `<|tool_call>call:NAME{args}<tool_call|>` | gemma-style `key:<|"|>value<|"|>` serialization; template renders the matching history side (`<|tool_response>`) |

Exact marker strings are pinned by the vectors and the implementation
constants, not by this page's prose.

Harmony note: gpt-oss declares its markers as real special tokens, but the
parser still matches text after the marker-preserving detokenizer renders
them — uniform code path, and the detokenizer's resolve-at-load step maps
each marker to ids when they exist as single tokens.

## Implementation notes (Steps 2–3, 2026-11-25)

### Step 2: tool parsers + gemma-4 pair

- **gemma-4 thought framing.** The template renders history as
  `'<|channel>thought\n' + text + '\n<channel|>'` (chat_template.jinja:240).
  The parser keeps the open marker bare (robust to truncation) and consumes
  the framing via close marker `\n<channel|>` plus a one-shot
  leading-newline trim after the open marker, so `reasoning_content` is the
  bare thought text and re-renders byte-identically. A truncated bare
  `<|channel>thought` (no newline yet) still classifies as reasoning.
- **hermes machine.** Head buffered until the `"arguments"` key + colon
  complete, so the name is extractable before any args delta streams
  (`"name"` extracted from the pre-arguments region, JSON-unescape best
  effort). Args stream by JSON-aware brace counting (string/escape aware),
  ending at the brace closing the arguments value; text after it until the
  close marker is dropped. Non-object arguments fall back to streaming
  everything until the close marker. Limits: `name` must precede
  `arguments`; one object per block (no arrays); a literal close-marker
  string inside the arguments would split the block.
- **gemma4 calls machine.** Name = text between `<|tool_call>call:` and the
  first `{`; args stream raw **including the outer braces** (OpenAI-style
  `arguments` string). Termination by brace counting where the template's
  `<|"|>` quote token toggles string state, so braces inside quoted values
  are literal (partial-token buffering for the quote marker).
- **Args conversion (follow-up).** The template's history string-path
  expects the INNER args body (it adds the braces itself,
  chat_template.jinja:246/257); the mapping-path re-canonicalizes via
  `dictsort` + `format_argument` (strings quoted `<|"|>`, numbers bare).
  Proper multi-turn flow therefore converts parsed gemma args to a dict
  before re-render; until that converter exists, the string path is
  strip-outer-braces. OpenAI clients additionally want JSON arguments —
  same converter, queued after the parser steps.
- **Round-trip gate** (probe, not pytest — loads the real tokenizer):
  `log/parsers/probe_gemma4_roundtrip.py` parses a template-canonical
  gemma-4 output and re-renders the parsed message through
  `apply_chat_template` — byte-exact on both the string (inner body) and
  mapping (dict) paths, including the thought framing.

### Step 3: harmony (full-stream protocol)

- **Ground truth**: the official openai/gpt-oss chat template
  (`https://huggingface.co/openai/gpt-oss-20b/raw/main/chat_template.jinja`,
  fetched 2026-11-25; no gpt-oss checkpoint is hosted locally, so there is
  no harmony round-trip probe yet — the vectors pin the parse contract
  until one is hosted).
- **Segment shape**: `<|start|>{role}[ to={recipient}]<|channel|>{channel}[ header]<|message|>{payload}` terminated by `<|end|>` / `<|return|>` / `<|call|>`. The
  generation prompt supplies the leading `<|start|>assistant`, so the
  model's first segment usually starts at `<|channel|>` — both shapes
  parse. The template puts the tool recipient in the **start header**
  (`<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>`); the parser also accepts `to=` in the channel header
  (older harmony variants), start header first.
- **Full-stream registry**: harmony is one protocol, so it does not
  decompose into reasoning-parser + tool-machine composition — a third
  registry (`STREAM_PARSERS`) holds full-stream machines, resolved via the
  same `tool_call_parser` key; `from_spec` rejects a co-configured
  reasoning parser (the route callbacks are never called).
- **Classification**: `analysis` → reasoning, `final` → content,
  `commentary` with recipient → tool call (Begin at `<|message|>` time —
  the name is known from the header; args stream raw, already JSON, so no
  gemma4-style converter), `commentary` without recipient → content
  (preamble), unknown channels → content (defensive). `<|return|>` ends
  generation but is not a hard stop (a malformed trailing call still
  parses). Pre/post-segment text outside any segment is content — plain
  non-harmony output degrades to passthrough.
- **Why it is simpler than hermes/gemma4 tools**: terminators are explicit
  markers, so args need no brace counting — the segmenter finds the end.
  Same text-level limit as the other machines: a payload containing a
  literal terminator token splits the segment.

## Multi-turn contract (discovered from the Bifrost template)

The gemma-4 template strips thinking from assistant history
(`strip_thinking`) and re-renders reasoning from
`message.get('reasoning') or message.get('reasoning_content')` — but only
after the last user message and only when the message carries tool_calls.
So the parser must keep `reasoning_content` and `content` cleanly apart in
the API surface, and the server passes client-supplied `reasoning_content`
back into the template unchanged. Round-trip consistency (parse output →
template input) is verified by
`log/parsers/probe_gemma4_roundtrip.py`: byte-exact re-render on both the
args-string and args-mapping paths (see implementation notes for the
conversion rule).

## Gating (same discipline as kernels)

1. **Vectors as the contract** — `server/tests/parsers/vectors/*.json`:
   per format, token-split inputs (markers deliberately cut at every
   boundary), multi tool calls, unclosed think, nested JSON braces, empty
   reasoning, round-trip cases. Vectors are plain JSON: input → expected
   event stream.
2. **Python server first** (`server/response_parsers.py`, reference
   implementation), wired into `server.py` SSE + non-stream paths.
3. **Rust port second**, parity-gated by running both implementations
   over the shared vectors (tokref/chatref pattern applied to text).
4. No generation-numerics or compiler changes — parsing is post-hoc, the
   bit-exact gates are untouched. No model-family names in code
   (policy-guarded like the rest).

## Relation to the Rust generalization ladder

- Chat templating: the current `rust/crates/axon-server/src/chat_template.rs`
  is a hand-port of exactly the gemma-4 template above (the Rust
  `tokenizers` crate has no template support). At **R1** templating moves
  to a real Jinja engine (`minijinja`); the gemma-4 family is its first
  proof target, per user note. Template rendering and response parsing are
  separate layers but share the marker tables.
- Parser specs ride the same config.json that R1's geometry comes from —
  one metadata-driven load path.

## Non-goals

- No auto-detection / template sniffing (decided against).
- No server-side tool execution; the server stops at emitting tool_calls.
- No vision/audio content-part parsing in the first pass.
- Bifrost `<|tool_response>` parsing (client→model direction) stays
  template-side; the parser only handles model output.
