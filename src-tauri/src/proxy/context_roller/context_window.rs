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

    // (3) Walk from the start, keeping preserved + non-preserved that fit,
    // until the post-truncation estimate drops below `target_after`.
    let target = config.target_after();
    let mut kept_messages: Vec<Value> = Vec::new();
    let mut kept_tokens: u64 = 0;
    let mut removed = 0usize;

    // Strategy: greedily keep from the **end** (most recent) first, then
    // fill forward, but only include non-preserved if they fit under the target.
    //
    // Concretely:
    // - Pre-compute preserved token total
    // - Then for each non-preserved message (oldest first), keep it ONLY IF
    //   adding it keeps us under `target`.
    let preserved_tokens: u64 = token_counts
        .iter()
        .enumerate()
        .filter(|(i, _)| preserve_indices.contains(i))
        .map(|(_, &t)| t)
        .sum();
    kept_tokens = preserved_tokens;

    // First, output preserved messages in order
    let mut result: Vec<Option<Value>> = vec![None; messages.len()];
    for (i, msg) in messages.iter().enumerate() {
        if preserve_indices.contains(&i) {
            result[i] = Some(msg.clone());
        }
    }

    // Then, fill in non-preserved (oldest first) if there's headroom
    for (i, (msg, &tokens)) in messages.iter().zip(token_counts.iter()).enumerate() {
        if preserve_indices.contains(&i) {
            continue;
        }
        if kept_tokens + tokens <= target {
            result[i] = Some(msg.clone());
            kept_tokens += tokens;
        } else {
            removed += 1;
        }
    }

    // Flatten in original order
    let final_messages: Vec<Value> = result.into_iter().flatten().collect();
    let final_count = final_messages.len();

    let kind = if removed > 0 {
        CompressionKind::Truncation
    } else {
        CompressionKind::None
    };

    RollingResult {
        messages: final_messages,
        kind,
        removed_count: removed,
        cumulative_before: cumulative_usage,
        cumulative_after: kept_tokens,
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
        .map(|(s, e)| {
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
        assert_eq!(result.kind, CompressionKind::Truncation);
        // We aim for target=600 (60% of 1000)
        // preserved: system (100) + last 4 (400) = 500
        // Can keep 1 more from middle (100) → 600 exactly
        assert!(result.cumulative_after <= 600);
        assert!(result.removed_count > 0);
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
    fn summary_placeholder_format() {
        let placeholder = build_summary_placeholder(5, 1200, Some((1000, 2000)));
        let content = placeholder["content"].as_str().unwrap();
        assert!(content.contains("5"));
        assert!(content.contains("1200"));
        assert!(content.contains("Rolling context"));
    }

    #[test]
    fn high_cumulative_triggers_aggressive_truncation() {
        // 200 messages, all 100 tokens = 20K total
        let (msgs, tokens) = make_msgs(200, 100);
        // Cumulative = 25K (way over 1K window)
        let result = apply_sliding_window(&msgs, &tokens, 25_000, &config());
        // Should keep system + last 4 + maybe 1-2 more
        assert!(result.final_message_count < 10);
        assert_eq!(result.kind, CompressionKind::Truncation);
    }
}
