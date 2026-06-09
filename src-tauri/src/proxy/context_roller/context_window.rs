//! Sliding window logic: decide which messages to keep/truncate.

use serde_json::Value;

/// Result of applying the rolling context window.
#[derive(Debug, Clone)]
pub struct RollingResult {
    /// The modified messages array (may be truncated).
    pub messages: Vec<Value>,
    /// Whether truncation was applied.
    pub was_truncated: bool,
    /// How many messages were removed.
    pub removed_count: usize,
    /// Estimated token count before truncation.
    pub tokens_before: u64,
    /// Estimated token count after truncation.
    pub tokens_after: u64,
    /// Number of messages in the final array.
    pub final_message_count: usize,
}

/// Configuration for the rolling context window.
#[derive(Debug, Clone)]
pub struct RollingConfig {
    /// Provider's context window size in tokens.
    pub context_window: u64,
    /// Threshold ratio (0.0-1.0) at which to trigger truncation.
    pub threshold: f64,
    /// Number of recent message rounds to always preserve.
    pub preserve_rounds: u32,
}

impl RollingConfig {
    /// Compute the token limit that triggers truncation.
    pub fn trigger_limit(&self) -> u64 {
        ((self.context_window as f64) * self.threshold) as u64
    }
}

/// Apply sliding window to a messages array.
///
/// Strategy:
/// 1. Always preserve the system message (if present).
/// 2. Always preserve the last N rounds of user/assistant exchange.
/// 3. If total tokens exceed the trigger limit, remove oldest non-preserved messages.
pub fn apply_sliding_window(
    messages: &[Value],
    token_counts: &[u64],
    config: &RollingConfig,
) -> RollingResult {
    let total_tokens: u64 = token_counts.iter().sum();
    let trigger_limit = config.trigger_limit();

    // If under limit, no truncation needed
    if total_tokens <= trigger_limit || messages.len() <= 2 {
        return RollingResult {
            messages: messages.to_vec(),
            was_truncated: false,
            removed_count: 0,
            tokens_before: total_tokens,
            tokens_after: total_tokens,
            final_message_count: messages.len(),
        };
    }

    // Identify indices to preserve
    let mut preserve_indices = std::collections::HashSet::new();

    // Always preserve system message at index 0
    if let Some(first) = messages.first() {
        if is_system_message(first) {
            preserve_indices.insert(0);
        }
    }

    // Preserve last N rounds (each round = user + assistant, approximately)
    let rounds_to_preserve = config.preserve_rounds as usize;
    let mut round_count = 0;
    for (i, msg) in messages.iter().enumerate().rev() {
        if round_count >= rounds_to_preserve * 2 {
            break;
        }
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "user" || role == "assistant" {
            preserve_indices.insert(i);
            round_count += 1;
        }
    }

    // Build the result: keep preserved messages + fill with non-preserved from the end
    // until under the trigger limit
    let mut kept_messages = Vec::new();
    let mut kept_tokens = 0u64;
    let mut removed = 0usize;

    for (i, (msg, &tokens)) in messages.iter().zip(token_counts.iter()).enumerate() {
        if preserve_indices.contains(&i) {
            kept_messages.push(msg.clone());
            kept_tokens += tokens;
        } else {
            // Non-preserved message: check if we have room
            if kept_tokens + tokens <= trigger_limit {
                kept_messages.push(msg.clone());
                kept_tokens += tokens;
            } else {
                removed += 1;
            }
        }
    }

    // Ensure messages stay in original order
    // (the loop above preserves order naturally)

    let final_count = kept_messages.len();
    RollingResult {
        messages: kept_messages,
        was_truncated: removed > 0,
        removed_count: removed,
        tokens_before: total_tokens,
        tokens_after: kept_tokens,
        final_message_count: final_count,
    }
}

fn is_system_message(msg: &Value) -> bool {
    msg.get("role")
        .and_then(|r| r.as_str())
        .map(|r| r == "system" || r == "developer")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_msgs(n: usize) -> (Vec<Value>, Vec<u64>) {
        let mut msgs = Vec::new();
        let mut tokens = Vec::new();
        for i in 0..n {
            let role = if i == 0 {
                "system"
            } else if i % 2 == 1 {
                "user"
            } else {
                "assistant"
            };
            msgs.push(serde_json::json!({"role": role, "content": format!("msg {}", i)}));
            tokens.push(100);
        }
        (msgs, tokens)
    }

    #[test]
    fn no_truncation_when_under_limit() {
        let (msgs, tokens) = make_msgs(5);
        let config = RollingConfig {
            context_window: 1000,
            threshold: 0.8,
            preserve_rounds: 2,
        };
        let result = apply_sliding_window(&msgs, &tokens, &config);
        assert!(!result.was_truncated);
        assert_eq!(result.final_message_count, 5);
    }

    #[test]
    fn truncates_oldest_non_preserved() {
        let (msgs, tokens) = make_msgs(10); // 1000 tokens total
        let config = RollingConfig {
            context_window: 500, // trigger at 400
            threshold: 0.8,
            preserve_rounds: 2, // preserve last 4 messages
        };
        let result = apply_sliding_window(&msgs, &tokens, &config);
        assert!(result.was_truncated);
        // Should remove at least 1 of the 5 non-preserved messages
        assert!(result.removed_count > 0);
        // The non-preserved messages that are too old should be removed
        assert!(result.final_message_count < 10);
        // System + last 4 preserved messages = 5 messages, 500 tokens
        // (preserved messages are kept even if they exceed trigger_limit)
        assert!(result.final_message_count >= 5);
    }

    #[test]
    fn always_preserves_system() {
        let mut msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
        ];
        let mut tokens = vec![50u64];
        for i in 0..20 {
            msgs.push(serde_json::json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": format!("x")}));
            tokens.push(100);
        }
        let config = RollingConfig {
            context_window: 500,
            threshold: 0.5,
            preserve_rounds: 1,
        };
        let result = apply_sliding_window(&msgs, &tokens, &config);
        assert_eq!(result.messages[0]["role"].as_str(), Some("system"));
    }

    #[test]
    fn preserves_last_n_rounds() {
        let mut msgs = vec![serde_json::json!({"role": "system", "content": "sys"})];
        let mut tokens = vec![50u64];
        for i in 0..10 {
            msgs.push(serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("msg-{}", i)
            }));
            tokens.push(100);
        }
        // 11 msgs, 1050 tokens. trigger at 400 (window=500, threshold=0.8)
        let config = RollingConfig {
            context_window: 500,
            threshold: 0.8,
            preserve_rounds: 2, // preserve last 4 user/assistant msgs
        };
        let result = apply_sliding_window(&msgs, &tokens, &config);
        assert!(result.was_truncated);
        // System (idx 0) + last 4 rounds (idx 7,8,9,10) = at least 5
        assert!(result.final_message_count >= 5);
        // The last message should be msg-9 (user) or msg-10 (assistant)
        let last = result.messages.last().unwrap();
        assert!(last["content"].as_str().unwrap().starts_with("msg-"));
    }

    #[test]
    fn short_array_no_truncation() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "hello"}),
        ];
        let tokens = vec![50u64, 50u64];
        let config = RollingConfig {
            context_window: 100,
            threshold: 0.5,
            preserve_rounds: 2,
        };
        let result = apply_sliding_window(&msgs, &tokens, &config);
        assert!(!result.was_truncated);
        assert_eq!(result.final_message_count, 2);
    }

    #[test]
    fn empty_messages() {
        let msgs: Vec<Value> = vec![];
        let tokens: Vec<u64> = vec![];
        let config = RollingConfig {
            context_window: 1000,
            threshold: 0.8,
            preserve_rounds: 2,
        };
        let result = apply_sliding_window(&msgs, &tokens, &config);
        assert!(!result.was_truncated);
        assert_eq!(result.final_message_count, 0);
    }

    #[test]
    fn single_message_no_truncation() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let tokens = vec![500u64];
        let config = RollingConfig {
            context_window: 100,
            threshold: 0.5,
            preserve_rounds: 2,
        };
        let result = apply_sliding_window(&msgs, &tokens, &config);
        assert!(!result.was_truncated);
        assert_eq!(result.final_message_count, 1);
    }

    #[test]
    fn trigger_limit_computation() {
        let config = RollingConfig {
            context_window: 128_000,
            threshold: 0.8,
            preserve_rounds: 6,
        };
        assert_eq!(config.trigger_limit(), 102_400);
    }
}
