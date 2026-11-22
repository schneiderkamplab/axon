//! Backend-agnostic sampler chain — Rust port of `server/sampling.py`.
//!
//! Operates on an owned f64 logits vector (the host copy of the model's f16
//! logits; `[vocab]` is tiny next to the model forward). Greedy decoding
//! (temperature <= 0) is the behavior parity harnesses rely on.
//!
//! RNG note: numpy's `default_rng` is PCG64; this port uses xorshift64*, so
//! seeded temperature sampling follows a different bitstream (same
//! distribution). Greedy decoding is unaffected.

/// Sampling parameters for one generation.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// `<= 0` selects greedy decoding.
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed: None,
        }
    }
}

impl SamplerConfig {
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// xorshift64* with a SplitMix64 seed mix; uniforms at 53-bit resolution like
/// numpy's `random_sample`.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: Option<u64>) -> Rng {
        let s = seed.unwrap_or_else(new_entropy_seed);
        // SplitMix64 warmup so poor seeds spread (numpy does equivalent work
        // in its SeedSequence).
        let mut z = s;
        let mut mix = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        let _ = mix();
        Rng(mix())
    }

    fn next_u64(&mut self) -> u64 {
        let x = self.0;
        self.0 = x ^ (x >> 12);
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

fn new_entropy_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    nanos ^ ((std::process::id() as u64) << 32)
}

fn apply_temperature(logits: &mut [f64], temperature: f64) {
    if temperature != 1.0 {
        let t = temperature.max(1e-8);
        for v in logits.iter_mut() {
            *v /= t;
        }
    }
}

fn apply_top_k(logits: &mut [f64], top_k: usize) {
    let n = logits.len();
    if top_k == 0 || top_k >= n {
        return;
    }
    // np.partition(logits, -top_k)[-top_k]: the top_k-th largest value.
    let mut scratch = logits.to_vec();
    let kth = *scratch
        .select_nth_unstable_by(n - top_k, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .1;
    for v in logits.iter_mut() {
        if *v < kth {
            *v = f64::NEG_INFINITY;
        }
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_value = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let uniform = |count: usize| vec![1.0 / count as f64; logits.len()];
    if !max_value.is_finite() {
        // Degenerate logits (e.g. all -inf): uniform over finite entries.
        let finite: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, _)| i)
            .collect();
        if finite.is_empty() {
            return uniform(logits.len());
        }
        let mut out = vec![0.0; logits.len()];
        let u = 1.0 / finite.len() as f64;
        for &i in &finite {
            out[i] = u;
        }
        return out;
    }
    let exponents: Vec<f64> = logits.iter().map(|v| (v - max_value).exp()).collect();
    let total: f64 = exponents.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        let finite: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, _)| i)
            .collect();
        if finite.is_empty() {
            return uniform(logits.len());
        }
        let mut out = vec![0.0; logits.len()];
        let u = 1.0 / finite.len() as f64;
        for &i in &finite {
            out[i] = u;
        }
        return out;
    }
    exponents.into_iter().map(|e| e / total).collect()
}

fn apply_top_p(logits: &mut Vec<f64>, top_p: f64) {
    if !(0.0 < top_p && top_p < 1.0) {
        return;
    }
    let n = logits.len();
    // argsort descending
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_logits: Vec<f64> = sorted_indices.iter().map(|&i| logits[i]).collect();
    let probabilities = softmax(&sorted_logits);
    let mut cumulative: Vec<f64> = Vec::with_capacity(n);
    let mut acc = 0.0;
    for p in probabilities {
        acc += p;
        cumulative.push(acc);
    }
    // Keep the smallest prefix whose cumulative probability >= top_p
    // (always keep at least one token).
    let cutoff_index = cumulative.partition_point(|c| *c < top_p) + 1;
    let mut filtered = vec![f64::NEG_INFINITY; n];
    for &i in &sorted_indices[..cutoff_index] {
        filtered[i] = logits[i];
    }
    *logits = filtered;
}

/// Sample one token id from a 1-D logits vector.
///
/// Greedy when `config.temperature <= 0`; otherwise temperature scaling,
/// top-k filtering, nucleus (top-p) filtering, then multinomial sampling.
pub fn sample_next_token(logits: &[f64], config: &SamplerConfig, rng: &mut Rng) -> u32 {
    if config.greedy() {
        // np.argmax = first occurrence of the max
        let mut best = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        return best as u32;
    }
    let mut values = logits.to_vec();
    apply_temperature(&mut values, config.temperature);
    if let Some(top_k) = config.top_k {
        apply_top_k(&mut values, top_k);
    }
    if let Some(top_p) = config.top_p {
        apply_top_p(&mut values, top_p);
    }
    let probabilities = softmax(&values);
    // np.random.Generator.choice(n, p=...): inverse CDF over the cumulative.
    let u = rng.uniform();
    let mut acc = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as u32;
        }
    }
    (probabilities.len() - 1) as u32
}
