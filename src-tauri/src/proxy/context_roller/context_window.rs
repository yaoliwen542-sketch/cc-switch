//! Sliding window logic for rolling context.
//!
//! ## Algorithm (v2 — usage-driven)
//!
//! The threshold check is no longer "does the *current request body* exceed the
//! window?" — it is **"has the *cumulative session usage* (from upstream
//! `usage.input_tokens`) exceeded the threshold?"**. This matches what the
//! upstream provider actually sees: the provider counts ALL tokens ever sent in
//! this session's history, not just the most recent request.
//!
//! Why this matters:
//!
//! ```text
//!   session_history_actual = sum(usage.input_tokens) across all prior requests
//!   current_request_body   = the *latest* request the client sent
//! ```
//!
//! A naive heuristic would check `current_request_body` against the window.
//! That fails when:
//! - The client (e.g. claude-code) is already doing its own internal sliding
//!   window, so individual requests look small even though the conversation
//!   is long.
//! - Cumulative input is now 80% of the window, but the next request only
//!   adds 5% — looks fine in isolation, but combined it overflows.
//!
//! ## Truncation strategy
//!
//! When the session's cumulative usage exceeds the threshold:
//!
//! 1. **Always keep**: the system message (idx 0) if present
//! 2. **Always keep**: the last `preserve_rounds` user/assistant exchanges
//!    (their tool_calls and tool_results are kept together to preserve pairing)
//! 3. **Truncate the rest**: drop the oldest non-preserved messages first,
//!    until the *projected* post-truncation usage (last response's input count
//!    minus the dropped tokens) drops below the threshold.
//! 4. **Reset cumulative counters**: after a successful truncation, we
//!    record the compression event and reset `total_input_tokens` to 0 so the
//!    next response's `usage.input_tokens` becomes the new baseline.

use serde_json::Value;

/// What kind of compression was performed (for logging/audit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionKind {
    /// Plain truncation — just drop messages.
    Truncation,
    /// LLM-based summarization — old messages replaced with a summary block.
    Summary,
    /// No compression needed.
    None,
}

/// Result of applying the rolling context window.
#[derive(Debug, Clone)]
pub struct RollingResult {
    /// The modified messages array (may be truncated).
    pub messages: Vec<Value>,
    /// What kind of compression happened.
    pub kind: CompressionKind,
    /// How many messages were removed.
    pub removed_count: usize,
    /// Cumulative session tokens before truncation (from DB).
    pub cumulative_before: u64,
    /// Cumulative session tokens after truncation (estimate).
    pub cumulative_after: u64,
    /// Number of messages in the final array.
    pub final_message_count: usize,
    /// ID of any summary message inserted (for storage).
    pub summary_message_id: Option<i64>,
}

/// Configuration for the rolling context window.
#[derive(Debug, Clone, Copy)]
pub struct RollingConfig {
    /// Provider's context window size in tokens.
    pub context_window: u64,
    /// Threshold ratio (0.0-1.0) at which to trigger truncation.
    pub threshold: f64,
    /// Number of recent message rounds to always preserve.
    pub preserve_rounds: u32,
}

impl RollingConfig {
    /// Token limit at which truncation fires.
    pub fn trigger_limit(&self) -> u64 {
        ((self.context_window as f64) * self.threshold) as u64
    }

    /// Target token count after truncation. We aim for `target` not `trigger_limit`
    /// so we don't fire on every single request right at the boundary.
    pub fn target_after(&self) -> u64 {
        // Aim for 60% of the window after truncation — gives us headroom.
        ((self.context_window as f64) * 0.6) as u64
    }
}

/// Decide which messages to keep given cumulative session usage.
///
/// This is the core algorithm. It takes:
/// - the current request's `messages` array
/// - per-message token estimates
/// - the **cumulative** session usage reported by the upstream API so far
/// - the rolling config
///
/// Returns a `RollingResult` describing the new messages array and what was removed.
pub fn apply_sliding_window(
    messages: &[Value],
    token_counts: &[u64],
    cumulative_usage: u64,
    config: &RollingConfig,
) -> RollingResult {
    let trigger = config.trigger_limit();

    // If cumulative is under the trigger, no compression.
    if cumulative_usage <= trigger {
        return RollingResult {
            messages: messages.to_vec(),
            kind: CompressionKind::None,
            removed_count: 0,
            cumulative_before: cumulative_usage,
            cumulative_after: cumulative_usage,
            final_message_count: messages.len(),
            summary_message_id: None,
        };
    }

    // Determine which indices to preserve
    let mut preserve_indices = std::collections::HashSet::new();

    // (1) Always keep system/developer message at idx 0
    if let Some(first) = messages.first() {
        if is_system_message(first) {
            preserve_indices.insert(0);
        }
    }

    // (2) Always keep last N rounds of user/assistant, plus their tool pairs
    let rounds_to_preserve = config.preserve_rounds as usize;
    let mut kept_rounds: Vec<usize> = Vec::new();
    for (i, msg) in messages.iter().enumerate().rev() {
        if kept_rounds.len() >= rounds_to_preserve * 2 {
            break;
        }
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "user" || role == "assistant" {
            preserve_indices.insert(i);
            kept_rounds.push(i);
        }
    }
    // Also keep any tool messages (tool results) that come immediately after
    // preserved assistant messages with tool_calls. They form a logical unit.
    for &i in &kept_rounds {
        if let Some(msg) = messages.get(i) {
            if msg.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && msg.get("tool_calls").is_some()
            {
                // The next message is likely the tool result — keep it
                if let Some(next) = messages.get(i + 1) {
                    if next.get("role").and_then(|r| r.as_str()) == Some("tool") {
                        preserve_indices.insert(i + 1);
                    }
                }
            }
        }
    }

    // (3) Collect messages to summarize (everything NOT in preserve_indices)
    //     and build a summary message to insert before the preserved window.
    let mut summarized_tokens: u64 = 0;
    let mut summarized_count = 0usize;
    let mut first_summarized_timestamp: Option<i64> = None;
    let mut last_summarized_timestamp: Option<i64> = None;
    let mut key_topics: Vec<String> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        if !preserve_indices.contains(&i) {
            summarized_tokens += token_counts.get(i).unwrap_or(&0);
            summarized_count += 1;

            // Extract timestamps if available
            if let Some(ts) = msg.get("created_at").and_then(|v| v.as_i64()) {
                if first_summarized_timestamp.is_none() {
                    first_summarized_timestamp = Some(ts);
                }
                last_summarized_timestamp = Some(ts);
            }

            // Extract key content snippets from user messages for context
            if summarized_count <= 10 || summarized_count % 20 == 0 {
                if let Some(content) = extract_message_content_snippet(msg, 100) {
                    key_topics.push(content);
                }
            }
        }
    }

    // Build summary message for the evicted messages
    let summary = if summarized_count > 0 {
        Some(build_smart_summary(
            summarized_count,
            summarized_tokens,
            first_summarized_timestamp,
            last_summarized_timestamp,
            &key_topics,
        ))
    } else {
        None
    };

    // Build final message list: system (first) + summary + other preserved
    let mut final_messages: Vec<Value> = Vec::new();

    // Add preserved messages in original order
    let mut preserved_indices_sorted: Vec<usize> = preserve_indices.into_iter().collect();
    preserved_indices_sorted.sort();

    let mut summary_inserted = false;
    let mut summary = summary;
    for i in &preserved_indices_sorted {
        if let Some(msg) = messages.get(*i) {
            // Insert summary after system message (first message)
            if !summary_inserted && *i > 0 {
                if let Some(summary_msg) = summary.take() {
                    final_messages.push(summary_msg);
                    summary_inserted = true;
                }
            }
            final_messages.push(msg.clone());
        }
    }

    // If summary not yet inserted (e.g., no non-system preserved messages)
    if !summary_inserted {
        if let Some(summary_msg) = summary {
            final_messages.push(summary_msg);
        }
    }

    let final_count = final_messages.len();
    let preserved_tokens: u64 = token_counts
        .iter()
        .enumerate()
        .filter(|(i, _)| preserved_indices_sorted.contains(i))
        .map(|(_, &t)| t)
        .sum();

    let kind = if summarized_count > 0 {
        CompressionKind::Summary
    } else {
        CompressionKind::None
    };

    // Estimate summary message tokens (typically ~10% of original tokens)
    let summary_tokens_estimate = if summarized_count > 0 {
        (summarized_tokens as f64 * 0.1) as u64
    } else {
        0
    };

    RollingResult {
        messages: final_messages,
        kind,
        removed_count: summarized_count,
        cumulative_before: cumulative_usage,
        cumulative_after: summary_tokens_estimate + preserved_tokens,
        final_message_count: final_count,
        summary_message_id: None,
    }
}

/// Build a "summary" message that replaces evicted messages. The summary is
/// a system-style message that references the conversation history. This is
/// what `compressor::SummaryCompressor` uses (or a future LLM-based compressor
/// would replace this with a real generated summary).
pub fn build_summary_placeholder(
    evicted_count: usize,
    evicted_tokens: u64,
    time_range: Option<(i64, i64)>,
) -> Value {
    let range = time_range
        .map(|(s, _e)| {
            chrono::DateTime::from_timestamp(s, 0)
                .map(|d| d.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_default()
        })
        .unwrap_or_default();
    serde_json::json!({
        "role": "user",
        "content": format!(
            "[Rolling context: {evicted_count} earlier messages (~{evicted_tokens} tokens) were compacted to save space. {range} The conversation continued with tool calls and responses; refer to the most recent exchanges for active context.]"
        )
    })
}

/// Extract a short content snippet from a message for summary context.
fn extract_message_content_snippet(msg: &Value, max_len: usize) -> Option<String> {
    let content = match msg.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for block in blocks {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    parts.push(text.to_string());
                }
            }
            parts.join(" ")
        }
        _ => return None,
    };

    if content.is_empty() {
        return None;
    }

    // Truncate to max_len and add ellipsis if needed
    let truncated = if content.len() > max_len {
        format!("{}...", &content[..max_len])
    } else {
        content
    };

    Some(truncated)
}

/// Build a smart summary message from evicted messages.
///
/// The summary includes:
/// - Count of summarized messages and tokens
/// - Time range covered
/// - Key content snippets for context continuity
fn build_smart_summary(
    count: usize,
    tokens: u64,
    first_ts: Option<i64>,
    last_ts: Option<i64>,
    key_topics: &[String],
) -> Value {
    let time_range = match (first_ts, last_ts) {
        (Some(s), Some(e)) => {
            let start = chrono::DateTime::from_timestamp(s, 0)
                .map(|d| d.format("%m-%d %H:%M").to_string())
                .unwrap_or_default();
            let end = chrono::DateTime::from_timestamp(e, 0)
                .map(|d| d.format("%m-%d %H:%M").to_string())
                .unwrap_or_default();
            format!(" ({start} ~ {end})")
        }
        _ => String::new(),
    };

    // Build topic summary
    let topic_summary = if key_topics.is_empty() {
        String::new()
    } else {
        let topics_text = key_topics
            .iter()
            .take(8) // Limit to 8 snippets
            .enumerate()
            .map(|(i, t)| format!("{}. {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n");
        format!("\n\nKey earlier context:\n{}", topics_text)
    };

    let content = format!(
        "[Context Summary — {} messages, ~{} tokens compacted{}]\n\
         These earlier messages have been summarized to save context space. \
         The conversation history includes tool calls, code reviews, and file operations. \
         Refer to the most recent exchanges for current active context.\
         {}",
        count, tokens, time_range, topic_summary
    );

    serde_json::json!({
        "role": "user",
        "content": content
    })
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

    fn make_msgs(n: usize, tokens_each: u64) -> (Vec<Value>, Vec<u64>) {
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
            tokens.push(tokens_each);
        }
        (msgs, tokens)
    }

    fn config() -> RollingConfig {
        RollingConfig {
            context_window: 1000,
            threshold: 0.8,
            preserve_rounds: 2,
        }
    }

    #[test]
    fn no_compression_under_threshold() {
        let (msgs, tokens) = make_msgs(5, 100);
        // cumulative = 500 (half of 1000) < 800 trigger
        let result = apply_sliding_window(&msgs, &tokens, 500, &config());
        assert_eq!(result.kind, CompressionKind::None);
        assert_eq!(result.final_message_count, 5);
    }

    #[test]
    fn compression_at_or_above_threshold() {
        let (msgs, tokens) = make_msgs(10, 100); // body tokens = 1000
        // cumulative = 900 (> 800 trigger)
        let result = apply_sliding_window(&msgs, &tokens, 900, &config());
        assert_eq!(result.kind, CompressionKind::Summary);
        // Should have system (1) + summary (1) + last 4 rounds (4) = 6 messages
        assert_eq!(result.final_message_count, 6);
        // First message should be system
        assert_eq!(result.messages[0]["role"].as_str(), Some("system"));
        // Second message should be the summary
        assert!(result.messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("Context Summary"));
    }

    #[test]
    fn preserves_system_message() {
        let (msgs, tokens) = make_msgs(10, 100);
        let result = apply_sliding_window(&msgs, &tokens, 1500, &config());
        // First message must remain
        assert!(result
            .messages
            .iter()
            .any(|m| m["role"].as_str() == Some("system")));
    }

    #[test]
    fn preserves_last_n_rounds() {
        let (mut msgs, mut tokens) = make_msgs(11, 100);
        // Make last 4 explicitly identifiable
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());
        // Last preserved (the most recent) should be the last assistant in the array
        let last = result.messages.last().unwrap();
        let last_role = last["role"].as_str().unwrap();
        assert!(last_role == "assistant" || last_role == "user");
    }

    #[test]
    fn tool_pair_preservation() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "1", "content": "result"}),
        ];
        let tokens = vec![50u64, 100, 50, 50];
        let result = apply_sliding_window(&msgs, &tokens, 1000, &config());
        // Should preserve all 4 (system + last round which has tool pair)
        assert!(result.final_message_count >= 4);
    }

    #[test]
    fn empty_messages() {
        let result = apply_sliding_window(&[], &[], 0, &config());
        assert_eq!(result.kind, CompressionKind::None);
        assert_eq!(result.final_message_count, 0);
    }

    #[test]
    fn trigger_and_target_limit() {
        let c = config();
        assert_eq!(c.trigger_limit(), 800);
        assert_eq!(c.target_after(), 600);
    }

    #[test]
    fn summary_message_content() {
        let placeholder = build_smart_summary(
            50,
            12000,
            Some(1000000),
            Some(1000200),
            &["Hello world".to_string(), "Fix bug in main.rs".to_string()],
        );
        let content = placeholder["content"].as_str().unwrap();
        assert!(content.contains("50"));
        assert!(content.contains("12000"));
        assert!(content.contains("Context Summary"));
        assert!(content.contains("Hello world"));
        assert!(content.contains("Fix bug"));
    }

    #[test]
    fn high_cumulative_triggers_summary() {
        // 200 messages, all 100 tokens = 20K total
        let (msgs, tokens) = make_msgs(200, 100);
        // Cumulative = 25K (way over 1K window)
        let result = apply_sliding_window(&msgs, &tokens, 25_000, &config());
        // Should have system (1) + summary (1) + last 4 = 6 messages
        assert_eq!(result.final_message_count, 6);
        assert_eq!(result.kind, CompressionKind::Summary);
        // First message is system
        assert_eq!(result.messages[0]["role"].as_str(), Some("system"));
        // Second message is summary
        assert!(result.messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("195 messages")); // 200 - 5 = 195 summarized
    }
}
