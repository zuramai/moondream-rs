//! Sampling and streaming detokenization (`moondream.py::_generate_text`).

use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};

use crate::Result;

/// Zeroes tokens outside the top-p nucleus (exclusive cumulative mass `> top_p`)
/// and renormalizes in place.
pub(crate) fn apply_top_p(probs: &mut [f32], top_p: f32) {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let mut cum = 0f32;
    let mut kept = 0f32;
    for &i in &idx {
        let p = probs[i];
        if cum > top_p {
            probs[i] = 0.0;
        } else {
            kept += p;
        }
        cum += p;
    }
    if kept > 0.0 {
        for p in probs.iter_mut() {
            *p /= kept;
        }
    }
}

/// Index of the first maximum.
pub(crate) fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// Greedy when `temperature == 0`, otherwise temperature + top-p sampling.
pub(crate) fn sample(logits: &[f32], temperature: f32, top_p: f32, rng: &mut impl Rng) -> u32 {
    if temperature == 0.0 {
        return argmax(logits) as u32;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| ((l - max) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    apply_top_p(&mut probs, top_p);
    match WeightedIndex::new(&probs) {
        Ok(dist) => dist.sample(rng) as u32,
        Err(_) => argmax(logits) as u32,
    }
}

fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// Byte offset of char index `n` in `s` (clamped to the end).
fn char_offset(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

/// Emits decoded text only at word boundaries, newlines, or CJK characters, so
/// multi-token words and UTF-8 sequences are never split.
pub(crate) struct StreamDetokenizer<F> {
    decode: F,
    token_cache: Vec<u32>,
    /// Characters of the decoded cache already emitted.
    print_len: usize,
}

impl<F: FnMut(&[u32]) -> Result<String>> StreamDetokenizer<F> {
    pub(crate) fn new(decode: F) -> Self {
        Self {
            decode,
            token_cache: Vec::new(),
            print_len: 0,
        }
    }

    /// Adds a token; returns newly printable text, if any.
    pub(crate) fn push(&mut self, token: u32) -> Result<Option<String>> {
        self.token_cache.push(token);
        let text = (self.decode)(&self.token_cache)?;
        let start = char_offset(&text, self.print_len);
        let printable = if text.ends_with('\n') {
            self.token_cache.clear();
            self.print_len = 0;
            &text[start..]
        } else if text.chars().next_back().is_some_and(is_cjk_char) {
            self.print_len += text[start..].chars().count();
            &text[start..]
        } else if let Some(i) = text[start..].rfind(' ') {
            let p = &text[start..start + i + 1];
            self.print_len += p.chars().count();
            p
        } else {
            ""
        };
        Ok((!printable.is_empty()).then(|| printable.to_string()))
    }

    /// Flushes the remaining undecoded text.
    pub(crate) fn finish(&mut self) -> Result<Option<String>> {
        if self.token_cache.is_empty() {
            return Ok(None);
        }
        let text = (self.decode)(&self.token_cache)?;
        let rest = &text[char_offset(&text, self.print_len)..];
        self.token_cache.clear();
        self.print_len = 0;
        Ok((!rest.is_empty()).then(|| rest.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_top_p_should_keep_nucleus_and_renormalize() {
        let mut p = vec![0.5, 0.3, 0.2];
        apply_top_p(&mut p, 0.3);
        assert_eq!(p, vec![1.0, 0.0, 0.0]);
        let mut p = vec![0.5, 0.3, 0.2];
        apply_top_p(&mut p, 0.6);
        assert_eq!(p, vec![0.625, 0.375, 0.0]);
    }

    const VOCAB: [&str; 6] = ["Hello", " wor", "ld", "\n", "你", " end"];

    fn detok() -> StreamDetokenizer<impl FnMut(&[u32]) -> Result<String>> {
        StreamDetokenizer::new(|ids: &[u32]| Ok(ids.iter().map(|&i| VOCAB[i as usize]).collect()))
    }

    #[test]
    fn detokenizer_should_emit_at_spaces_and_newlines() -> Result<()> {
        let mut d = detok();
        assert_eq!(d.push(0)?, None);
        assert_eq!(d.push(1)?.as_deref(), Some("Hello "));
        assert_eq!(d.push(2)?, None);
        assert_eq!(d.push(3)?.as_deref(), Some("world\n"));
        assert_eq!(d.finish()?, None);
        Ok(())
    }

    #[test]
    fn detokenizer_should_emit_cjk_immediately_and_flush_tail() -> Result<()> {
        let mut d = detok();
        assert_eq!(d.push(4)?.as_deref(), Some("你"));
        assert_eq!(d.push(5)?.as_deref(), Some(" "));
        assert_eq!(d.push(2)?, None);
        assert_eq!(d.finish()?.as_deref(), Some("endld"));
        Ok(())
    }
}
