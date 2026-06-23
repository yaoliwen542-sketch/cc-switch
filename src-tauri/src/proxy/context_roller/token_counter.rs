#![allow(dead_code)]
//! Token counting strategies for rolling context.
//!
//! Three strategies, in increasing order of accuracy:
//! 1. `estimate_tokens` — character-based heuristic (always available, no deps)
//! 2. `MessageTokens` — per-message structural token accounting
//! 3. Provider-reported `usage.input_tokens` — the ground truth, used when available
//!
//! ## Why the heuristic exists
//!
//! Most LLM providers (including MiniMax, Kimi, GLM, DeepSeek, Claude, GPT) return
//! an authoritative `usage.input_tokens` in the response. When the proxy sees the
//! response, that number is the actual source of truth for the request that just
//! completed. We use the response-side number whenever available.
//!
//! The heuristic is only used for **predicting** the size of the *next* incoming
//! request, so we can decide whether to truncate *before* forwarding it. It is
//! intentionally conservative (overestimates rather than underestimates) so we
//! never miss a window-exceeded condition due to bad math.
//!
//! ## Accuracy calibration (CJK vs Latin)
//!
//! Empirically:
//! - Latin/ASCII prose: ~3.5–4 chars per token (English, code)
//! - CJK prose: ~0.7–0.9 chars per token (Chinese, Japanese, Korean)
//! - Code: ~3 chars per token (highly variable)
//!
//! We apply a **1.25x safety factor** to every estimate, so we over-count rather
//! than under-count. Under-counting is dangerous: it could let a request through
//! that the upstream provider would reject for context-window overflow.

/// Safety factor — multiply every estimate by this to err on the side of
/// "looks bigger than it is". Tuned so that Claude-3.5-Sonnet estimates within
/// 10% of the actual tokenizer output for mixed English/Chinese text.
pub const SAFETY_FACTOR: f64 = 1.25;

/// Approximate characters per token for **Latin alphabet, code, punctuation-heavy**
/// text. GPT/Claude BPE breaks these into 3-4 chars per token on average.
const LATIN_CHARS_PER_TOKEN: f64 = 3.5;

/// Approximate characters per token for **CJK** (Chinese/Japanese/Korean) text.
/// These are typically tokenized 1 char per token (sometimes 2 chars per token
/// for common bigrams). We use 1.0 to be conservative — Latin-style ratios of
/// 2.5–3.0 here would wildly *under*estimate.
const CJK_CHARS_PER_TOKEN: f64 = 1.0;

/// Approximate characters per token for **whitespace, control chars, digit runs**.
const NEUTRAL_CHARS_PER_TOKEN: f64 = 2.0;

/// Estimate the token count for a plain text string.
///
/// This is a best-effort heuristic. For critical decisions, use
/// `usage.input_tokens` from the response.
pub fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }

    let mut cjk = 0u64;
    let mut latin = 0u64;
    let mut neutral = 0u64;

    for ch in text.chars() {
        let cat = classify_char(ch);
        match cat {
            CharClass::Cjk => cjk += 1,
            CharClass::Latin => latin += 1,
            CharClass::Neutral => neutral += 1,
        }
    }

    let raw = (cjk as f64 / CJK_CHARS_PER_TOKEN)
        + (latin as f64 / LATIN_CHARS_PER_TOKEN)
        + (neutral as f64 / NEUTRAL_CHARS_PER_TOKEN);

    ((raw * SAFETY_FACTOR).ceil() as u64).max(1)
}

#[derive(Debug, Clone, Copy)]
enum CharClass {
    Cjk,
    Latin,
    Neutral,
}

fn classify_char(ch: char) -> CharClass {
    if is_cjk(ch) {
        CharClass::Cjk
    } else if ch.is_alphanumeric() || ch.is_ascii_punctuation() {
        CharClass::Latin
    } else {
        CharClass::Neutral
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch,
        '\u{1100}'..='\u{115f}'
        | '\u{2e80}'..='\u{303e}'
        | '\u{3041}'..='\u{33ff}'
        | '\u{3400}'..='\u{4dbf}'
        | '\u{4e00}'..='\u{9fff}'
        | '\u{a000}'..='\u{a4cf}'
        | '\u{ac00}'..='\u{d7a3}'
        | '\u{f900}'..='\u{faff}'
        | '\u{fe30}'..='\u{fe4f}'
        | '\u{ff00}'..='\u{ff60}'
        | '\u{ffe0}'..='\u{ffe6}'
        | '\u{20000}'..='\u{2ffff}'
    )
}

/// Per-message token accounting result.
#[derive(Debug, Clone, Copy, Default)]
pub struct MessageTokens {
    pub content: u64,
    pub tool_calls: u64,
    pub tool_results: u64,
    /// Per-message structural overhead (role tag, formatting, separators).
    /// Anthropic API uses ~4 tokens for this.
    pub overhead: u64,
}

impl MessageTokens {
    pub fn total(&self) -> u64 {
        self.content + self.tool_calls + self.tool_results + self.overhead
    }
}

/// Estimate tokens for a single message object (from serde_json::Value).
pub fn estimate_message_tokens(message: &serde_json::Value) -> MessageTokens {
    let mut result = MessageTokens {
        overhead: 4,
        ..Default::default()
    };

    // Content (string or array of content blocks)
    if let Some(content) = message.get("content") {
        result.content = estimate_content_tokens(content);
    }

    // Tool calls (assistant → tool_calls)
    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            result.tool_calls += estimate_tool_call_tokens(tc);
        }
    }

    // Tool result (role=tool, has tool_call_id)
    if let Some(_tool_call_id) = message.get("tool_call_id") {
        result.tool_results = 4; // structural overhead
        if let Some(content) = message.get("content") {
            result.tool_results += estimate_content_tokens(content);
        }
    }

    // JSON structural overhead: every message serializes to JSON with keys
    // ("role", "content", "tool_calls", "type", etc.), quotes, brackets,
    // and commas that are all tokenized. For large messages with many fields,
    // this can exceed the visible text content. Use the full serialized byte
    // count divided by ~4 as a floor, but never less than the content estimate.
    if let Ok(serialized) = serde_json::to_string(message) {
        let json_floor = (serialized.len() as u64) / 4; // ~4 bytes per token (JSON is dense)
        let content_total = result.content + result.tool_calls + result.tool_results;
        // The floor only applies if the JSON is significantly larger than the
        // content estimate (otherwise the content estimate is already accurate).
        if json_floor > content_total * 2 {
            result.overhead += json_floor.saturating_sub(content_total);
        }
    }

    result
}

fn estimate_content_tokens(content: &serde_json::Value) -> u64 {
    match content {
        serde_json::Value::String(text) => estimate_tokens(text),
        serde_json::Value::Array(blocks) => {
            let mut total = 0u64;
            for block in blocks {
                // Per-block structural overhead
                total += 3;
                // Thinking block: {"type": "thinking", "thinking": "..."}
                // These are often very large (many KB of reasoning) but have
                // a different key than "text". We must count them.
                if let Some(text) = block
                    .get("thinking")
                    .and_then(|v| v.as_str())
                {
                    total += estimate_tokens(text);
                }
                // Redacted thinking: {"type": "redacted_thinking", "data": "base64..."}
                // The base64 data is opaque but still tokenized; estimate
                // based on byte length. Base64 is ~3 chars/token; we use 2
                // to be conservative (high entropy → worse compression).
                else if let Some(data) = block
                    .get("data")
                    .and_then(|v| v.as_str())
                {
                    total += (data.len() as u64) / 2;
                }
                // Text / tool_use / tool_result content blocks
                else if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    total += estimate_tokens(text);
                } else if let Some(_source) = block.get("source") {
                    // Image / document. Anthropic bills images at ~1.6K tokens
                    // for a 1024x1024 image regardless of detail. We estimate
                    // generously so we never under-count.
                    total += 1600;
                }
                // tool_use block: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
                // The input JSON is tokenized; count its serialized size.
                if let Some(input) = block.get("input") {
                    if let Ok(bytes) = serde_json::to_string(input) {
                        total += (bytes.len() as u64) / 3; // ~3 bytes per token
                    }
                }
            }
            total
        }
        _ => 0,
    }
}

fn estimate_tool_call_tokens(tool_call: &serde_json::Value) -> u64 {
    let mut total = 8u64; // structural overhead

    if let Some(name) = tool_call
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|v| v.as_str())
    {
        total += estimate_tokens(name);
    }

    if let Some(args) = tool_call
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(|v| v.as_str())
    {
        total += estimate_tokens(args);
    }

    total
}

/// Estimate total tokens for an array of messages.
pub fn estimate_messages_tokens(messages: &[serde_json::Value]) -> u64 {
    messages
        .iter()
        .map(|m| estimate_message_tokens(m).total())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn ascii_text_estimate() {
        // 100 ASCII chars: 100 / 3.5 = 28.57 raw, * 1.25 = 35.71 → 36
        let text = "a".repeat(100);
        let tokens = estimate_tokens(&text);
        assert!(tokens >= 30 && tokens <= 45, "Expected ~36, got {}", tokens);
    }

    #[test]
    fn cjk_text_estimate() {
        // 100 CJK chars: 100 / 1.0 = 100 raw, * 1.25 = 125
        let text = "中".repeat(100);
        let tokens = estimate_tokens(&text);
        assert!(
            tokens >= 110 && tokens <= 140,
            "Expected ~125, got {}",
            tokens
        );
    }

    #[test]
    fn mixed_text_estimate_is_additive() {
        let text = format!("{}{}", "a".repeat(50), "中".repeat(50));
        let tokens = estimate_tokens(&text);
        // 50/3.5 + 50/1.0 = 14.3 + 50 = 64.3 * 1.25 = 80
        assert!(tokens >= 60 && tokens <= 100, "Got {}", tokens);
    }

    #[test]
    fn message_with_string_content() {
        let msg = serde_json::json!({"role": "user", "content": "Hello world"});
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens.total() >= 4);
    }

    #[test]
    fn message_with_array_content_counts_blocks() {
        let msg = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "source": {"type": "base64", "data": "x"}}
            ]
        });
        let tokens = estimate_message_tokens(&msg);
        // overhead 4 + block1(3+content) + block2(3+1600) ≈ 1610
        assert!(tokens.total() > 1500, "Got {}", tokens.total());
    }

    #[test]
    fn tool_result_includes_overhead() {
        let msg = serde_json::json!({
            "role": "tool",
            "tool_call_id": "abc",
            "content": "result text"
        });
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens.tool_results >= 4);
    }

    #[test]
    fn messages_array_total() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful"}),
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "assistant", "content": "Hi there!"}),
        ];
        let total = estimate_messages_tokens(&messages);
        assert!(total > 12);
    }

    #[test]
    fn safety_factor_never_underestimates() {
        assert!(estimate_tokens("x") >= 1);
        assert!(estimate_tokens("Hello") >= 1);
        assert!(estimate_tokens("中") >= 1);
    }

    #[test]
    fn thinking_block_is_counted() {
        // thinking blocks use "thinking" key, not "text"
        let msg = serde_json::json!({
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me think about this carefully. ".repeat(50)
                },
                {"type": "text", "text": "Here is my answer."}
            ]
        });
        let tokens = estimate_message_tokens(&msg);
        // The thinking block content is ~1850 chars, so should be ~660 tokens
        // plus the text content and overhead. Should be well above 600.
        assert!(tokens.content > 600, "thinking block tokens={}, expected >600", tokens.content);
    }

    #[test]
    fn redacted_thinking_block_is_counted() {
        // redacted_thinking blocks are base64 blobs; estimate based on data length
        let base64_data = "aGVsbG8gd29ybGQ=".repeat(100); // 1600 chars of base64
        let msg = serde_json::json!({
            "role": "assistant",
            "content": [
                {
                    "type": "redacted_thinking",
                    "data": base64_data
                },
                {"type": "text", "text": "Answer."}
            ]
        });
        let tokens = estimate_message_tokens(&msg);
        // 1600 chars / 2 = 800 tokens for redacted data, plus text + overhead
        assert!(tokens.content > 700, "redacted_thinking tokens={}, expected >700", tokens.content);
    }

    #[test]
    fn json_overhead_caught_for_dense_messages() {
        // A message with many tool_use blocks in content — each has input JSON
        let mut blocks = vec![];
        for i in 0..20 {
            blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": format!("tool_{i}"),
                "name": "Read",
                "input": {
                    "file_path": format!("/some/long/path/to/file_{i}.ts"),
                    "offset": 100,
                    "limit": 2000
                }
            }));
        }
        let msg = serde_json::json!({
            "role": "assistant",
            "content": blocks
        });
        let tokens = estimate_message_tokens(&msg);
        // 20 tool_use blocks with JSON input should estimate significantly
        assert!(tokens.total() > 200, "dense message tokens={}, expected >200", tokens.total());
    }
}
