//! Token counting strategies for rolling context
//!
//! Phase 1: Character-based heuristic with model-aware adjustments.
//! Phase 2: Can add tiktoken-rs for precise GPT token counting.

/// Safety factor applied to all estimates to avoid underestimating.
const SAFETY_FACTOR: f64 = 1.2;

/// Characters per token ratio for general text (conservative estimate).
const CHARS_PER_TOKEN: f64 = 2.5;

/// Characters per token for CJK text (Chinese, Japanese, Korean).
const CJK_CHARS_PER_TOKEN: f64 = 1.5;

/// Estimate token count for a given text string.
///
/// Uses a simple heuristic: count characters and divide by chars-per-token ratio.
/// CJK characters are counted separately with a lower ratio (more tokens per char).
pub fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }

    let mut cjk_chars = 0u64;
    let mut other_chars = 0u64;

    for ch in text.chars() {
        if is_cjk(ch) {
            cjk_chars += 1;
        } else {
            other_chars += 1;
        }
    }

    let cjk_tokens = (cjk_chars as f64 / CJK_CHARS_PER_TOKEN).ceil() as u64;
    let other_tokens = (other_chars as f64 / CHARS_PER_TOKEN).ceil() as u64;
    let raw_estimate = cjk_tokens + other_tokens;

    // Apply safety factor
    ((raw_estimate as f64) * SAFETY_FACTOR).ceil() as u64
}

/// Estimate tokens for a single message object (from serde_json::Value).
///
/// Handles both string content and array content blocks.
pub fn estimate_message_tokens(message: &serde_json::Value) -> u64 {
    let mut total = 0u64;

    // Add base overhead per message (role + formatting)
    total += 4;

    // Count content tokens
    if let Some(content) = message.get("content") {
        total += estimate_content_tokens(content);
    }

    // Count tool_calls tokens
    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            total += estimate_tool_call_tokens(tc);
        }
    }

    // Count tool_call_id overhead
    if message.get("tool_call_id").is_some() {
        total += 4;
    }

    total
}

/// Estimate tokens for content field (can be string or array of blocks).
fn estimate_content_tokens(content: &serde_json::Value) -> u64 {
    match content {
        serde_json::Value::String(text) => estimate_tokens(text),
        serde_json::Value::Array(blocks) => {
            let mut total = 0u64;
            for block in blocks {
                // Each block has overhead
                total += 3;
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    total += estimate_tokens(text);
                } else if let Some(source) = block.get("source") {
                    // Image/document block - estimate based on source data size
                    total += estimate_media_block_tokens(source);
                }
            }
            total
        }
        _ => 0,
    }
}

/// Estimate tokens for a tool call object.
fn estimate_tool_call_tokens(tool_call: &serde_json::Value) -> u64 {
    let mut total = 8u64; // base overhead

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

/// Estimate tokens for a media block (image, document).
fn estimate_media_block_tokens(source: &serde_json::Value) -> u64 {
    // Rough estimate: count the JSON representation length
    let json_text = source.to_string();
    estimate_tokens(&json_text)
}

/// Check if a character is CJK (Chinese, Japanese, Korean).
fn is_cjk(ch: char) -> bool {
    matches!(
        ch,
        '\u{4e00}'..='\u{9fff}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4dbf}'  // CJK Extension A
        | '\u{3000}'..='\u{303f}'  // CJK Symbols and Punctuation
        | '\u{3040}'..='\u{309f}'  // Hiragana
        | '\u{30a0}'..='\u{30ff}'  // Katakana
        | '\u{ac00}'..='\u{d7af}'  // Hangul Syllables
        | '\u{ff00}'..='\u{ffef}'  // Full-width forms
    )
}

/// Estimate total tokens for an array of messages.
pub fn estimate_messages_tokens(messages: &[serde_json::Value]) -> u64 {
    let mut total = 0u64;
    for msg in messages {
        total += estimate_message_tokens(msg);
    }
    total
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
        // 100 ASCII chars ≈ 40 tokens raw, * 1.2 safety = 48
        let text = "a".repeat(100);
        let tokens = estimate_tokens(&text);
        assert!(tokens >= 40 && tokens <= 60, "Expected ~48, got {}", tokens);
    }

    #[test]
    fn cjk_text_estimate() {
        // 100 CJK chars ≈ 67 tokens raw, * 1.2 safety = 80
        let text = "中".repeat(100);
        let tokens = estimate_tokens(&text);
        assert!(tokens >= 60 && tokens <= 100, "Expected ~80, got {}", tokens);
    }

    #[test]
    fn mixed_text_estimate() {
        let text = "Hello 世界, this is a test 测试文本".to_string();
        let tokens = estimate_tokens(&text);
        assert!(tokens > 0);
    }

    #[test]
    fn message_with_string_content() {
        let msg = serde_json::json!({
            "role": "user",
            "content": "Hello world"
        });
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens >= 4, "Message should have base overhead + content");
    }

    #[test]
    fn message_with_array_content() {
        let msg = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"}
            ]
        });
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens > 6, "Array content should include block overhead");
    }

    #[test]
    fn messages_array_total() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful"}),
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "assistant", "content": "Hi there!"}),
        ];
        let total = estimate_messages_tokens(&messages);
        assert!(total > 12, "3 messages should have base overhead * 3 + content");
    }
}
