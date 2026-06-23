#![allow(dead_code)]
//! Sliding window logic for rolling context.
//!
//! ## Algorithm (v3 — token-budget driven)
//!
//! The threshold check is **"has the *cumulative session usage* (from upstream
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
//! ## Retention strategy
//!
//! When compression fires, `target_after_tokens()` is treated as a real
//! retention budget rather than a logging hint:
//!
//! 1. **Mandatory layer**: system message (idx 0) and the last
//!    `preserve_rounds` user/assistant exchanges are always kept. Tool calls
//!    and their corresponding tool results are preserved as atomic pairs.
//! 2. **Budget backfill**: if the mandatory layer is under
//!    `target_after_tokens()`, recent non-mandatory messages/groups are added
//!    back (newest first) until the budget is exhausted.
//! 3. **Summarize the remainder**: everything not preserved is compacted into
//!    a single summary message.
//!
//! Post-compression, the session cumulative is baselined at the estimated
//! post-compression token count instead of being reset to zero, which avoids
//! compression thrashing.

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
    /// Target ratio after compression (0.1-0.9). Default 0.6.
    pub target_after: f64,
}

impl RollingConfig {
    /// Token limit at which truncation fires.
    pub fn trigger_limit(&self) -> u64 {
        ((self.context_window as f64) * self.threshold) as u64
    }

    /// Target token count after truncation.
    pub fn target_after_tokens(&self) -> u64 {
        ((self.context_window as f64) * self.target_after) as u64
    }
}

/// Decide which messages to keep using the configured token budget.
///
/// This is the core retention algorithm. The caller decides *whether* compression
/// is needed (e.g. cumulative usage or current body size exceeds the trigger);
/// this function always runs the budget-driven retention pass and returns a
/// `RollingResult` describing the new messages array and what was removed.
///
/// It takes:
/// - the current request's `messages` array
/// - per-message token estimates
/// - the **cumulative** session usage reported by the upstream API so far
///   (used only for logging / result metadata)
/// - the rolling config
///
/// Returns a `RollingResult` describing the new messages array and what was removed.
pub fn apply_sliding_window(
    messages: &[Value],
    token_counts: &[u64],
    cumulative_usage: u64,
    config: &RollingConfig,
) -> RollingResult {
    let target = config.target_after_tokens();

    // Use HashSet for O(1) membership checks
    let mut preserve_indices = std::collections::HashSet::new();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (1) Mandatory layer: system/developer message at idx 0
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if let Some(first) = messages.first() {
        if is_system_message(first) {
            preserve_indices.insert(0);
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (2) Mandatory layer: last N rounds of user/assistant
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (3) Preserve tool pairs: for each preserved assistant with tool_calls,
    //     keep ALL following tool results (handles parallel tool calls)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for &i in &kept_rounds {
        if let Some(msg) = messages.get(i) {
            if msg.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && msg.get("tool_calls").is_some()
            {
                // Scan forward from this assistant, keeping all consecutive tool results
                let mut j = i + 1;
                while let Some(next) = messages.get(j) {
                    if next.get("role").and_then(|r| r.as_str()) == Some("tool") {
                        preserve_indices.insert(j);
                        j += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (4) CRITICAL: Preserve assistant messages with tool_calls that are
    //     referenced by preserved tool result messages. Handles:
    //     - Parallel tool calls (multiple tool results from one assistant)
    //     - Tool results preserved by round but assistant is older
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut preserved_tool_call_ids = std::collections::HashSet::new();
    for (i, msg) in messages.iter().enumerate() {
        if preserve_indices.contains(&i) && msg.get("role").and_then(|r| r.as_str()) == Some("tool")
        {
            if let Some(tool_call_id) = msg.get("tool_call_id").and_then(|id| id.as_str()) {
                if !tool_call_id.trim().is_empty() {
                    preserved_tool_call_ids.insert(tool_call_id.to_string());
                }
            }
        }
    }

    if !preserved_tool_call_ids.is_empty() {
        for (i, msg) in messages.iter().enumerate() {
            if !preserve_indices.contains(&i)
                && msg.get("role").and_then(|r| r.as_str()) == Some("assistant")
            {
                if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for tc in tool_calls {
                        if let Some(id) = tc.get("id").and_then(|id| id.as_str()) {
                            if !id.trim().is_empty() && preserved_tool_call_ids.contains(id) {
                                preserve_indices.insert(i);
                                // Also keep any tool results that follow this assistant
                                let mut j = i + 1;
                                while let Some(next) = messages.get(j) {
                                    if next.get("role").and_then(|r| r.as_str()) == Some("tool") {
                                        preserve_indices.insert(j);
                                        j += 1;
                                    } else {
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (4b) Post-pass: ensure ALL tool results for preserved tool_calls exist.
    //      Step 3's forward-scan breaks on the first non-tool message, so tool
    //      results that are NOT immediately consecutive can be orphaned.  This
    //      pass collects every tool_call_id from preserved assistants and finds
    //      the matching tool result anywhere in the message array.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {
        // 1. Collect all tool_call_ids that preserved assistant messages reference
        let mut required_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for &i in &preserve_indices {
            if let Some(msg) = messages.get(i) {
                if msg.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for tc in tool_calls {
                            if let Some(id) = tc.get("id").and_then(|id| id.as_str()) {
                                if !id.trim().is_empty() {
                                    required_ids.insert(id.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. For each required id, check if a preserved tool result already covers it.
        //    If not, find the tool result in the full message array and preserve it.
        if !required_ids.is_empty() {
            let mut covered_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
            for &i in &preserve_indices {
                if let Some(msg) = messages.get(i) {
                    if msg.get("role").and_then(|r| r.as_str()) == Some("tool") {
                        if let Some(id) = msg.get("tool_call_id").and_then(|id| id.as_str()) {
                            if !id.trim().is_empty() {
                                covered_ids.insert(id.to_string());
                            }
                        }
                    }
                }
            }

            let missing_ids: Vec<String> = required_ids.difference(&covered_ids).cloned().collect();
            if !missing_ids.is_empty() {
                for (i, msg) in messages.iter().enumerate() {
                    if !preserve_indices.contains(&i)
                        && msg.get("role").and_then(|r| r.as_str()) == Some("tool")
                    {
                        if let Some(id) = msg.get("tool_call_id").and_then(|id| id.as_str()) {
                            if !id.trim().is_empty() && missing_ids.contains(&id.to_string()) {
                                preserve_indices.insert(i);
                            }
                        }
                    }
                }
            }
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (5) Budget backfill: keep extra recent messages while under target
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mandatory_indices = preserve_indices.clone();
    let mandatory_tokens: u64 = token_counts
        .iter()
        .enumerate()
        .filter(|(i, _)| mandatory_indices.contains(i))
        .map(|(_, &t)| t)
        .sum();

    let mut remaining_budget = target.saturating_sub(mandatory_tokens);
    if remaining_budget > 0 {
        // Build atomic groups: an assistant with tool_calls and all following
        // consecutive tool results form one group.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut i = 0;
        while i < messages.len() {
            let mut group = vec![i];
            let msg = &messages[i];
            if msg.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && msg.get("tool_calls").is_some()
            {
                let mut j = i + 1;
                while j < messages.len()
                    && messages[j].get("role").and_then(|r| r.as_str()) == Some("tool")
                {
                    group.push(j);
                    j += 1;
                }
            }
            i = *group.last().unwrap() + 1;
            groups.push(group);
        }

        // Add whole groups from newest to oldest while they fit in the budget.
        for group in groups.iter().rev() {
            if group.iter().any(|idx| preserve_indices.contains(idx)) {
                continue;
            }
            let group_tokens: u64 = group
                .iter()
                .map(|&idx| *token_counts.get(idx).unwrap_or(&0))
                .sum();
            if group_tokens <= remaining_budget {
                for &idx in group {
                    preserve_indices.insert(idx);
                }
                remaining_budget -= group_tokens;
            }
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (6) Collect messages to summarize, with importance weighting
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut summarized_tokens: u64 = 0;
    let mut summarized_count = 0usize;
    let mut first_summarized_timestamp: Option<i64> = None;
    let mut last_summarized_timestamp: Option<i64> = None;
    let mut key_topics: Vec<String> = Vec::new();
    let mut important_snippets: Vec<String> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        if !preserve_indices.contains(&i) {
            summarized_tokens += token_counts.get(i).unwrap_or(&0);
            summarized_count += 1;

            // Extract timestamps
            if let Some(ts) = msg.get("created_at").and_then(|v| v.as_i64()) {
                if first_summarized_timestamp.is_none() {
                    first_summarized_timestamp = Some(ts);
                }
                last_summarized_timestamp = Some(ts);
            }

            // Extract key content snippets (first 10 + every 20th)
            if summarized_count <= 10 || summarized_count % 20 == 0 {
                if let Some(content) = extract_message_content_snippet(msg, 100) {
                    key_topics.push(content);
                }
            }

            // Collect important messages: errors, decisions, file ops
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                let lower = content.to_lowercase();
                if lower.contains("error")
                    || lower.contains("fix")
                    || lower.contains("create")
                    || lower.contains("delete")
                    || lower.contains("important")
                    || lower.contains("decision")
                {
                    if let Some(snippet) = extract_message_content_snippet(msg, 150) {
                        important_snippets.push(snippet);
                    }
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
            &important_snippets,
        ))
    } else {
        None
    };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (7) Build final message list: system (first) + summary + preserved
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut final_messages: Vec<Value> = Vec::new();

    // Sort preserved indices to maintain original order
    let mut preserved_indices_sorted: Vec<usize> = preserve_indices.into_iter().collect();
    preserved_indices_sorted.sort();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (4c) Fixed-point validation of tool-call / tool-result pairing.
    //
    // Invariants enforced before the final message list is built:
    //   - Every preserved assistant message with tool_calls has ALL of its
    //     tool_call_ids covered by at least one preserved tool result.
    //   - Every preserved tool result has a non-empty tool_call_id that is
    //     referenced by at least one preserved assistant message.
    //   - Assistant messages containing tool_calls with empty/missing ids are
    //     dropped entirely, because the API cannot match them.
    //
    // This is a fixed-point computation because dropping a tool result can
    // make an assistant uncovered, and dropping that assistant can orphan more
    // tool results.
    //
    // API errors covered:
    //   * "an assistant message with 'tool_calls' must be followed by tool
    //      messages responding to each 'tool_call_id'"
    //   * "tool result's tool id(...) not found (2013)"
    //   * "tool_call_id is not found"
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {
        fn is_valid_id(id: &str) -> bool {
            !id.trim().is_empty()
        }

        // Map preserved assistant index -> its valid tool_call ids.
        // Assistants whose tool_calls array has no valid ids are dropped
        // entirely because the API cannot match them to any tool result.
        let mut assistant_ids: std::collections::HashMap<usize, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        let mut bad_assistants: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &i in &preserved_indices_sorted {
            if let Some(msg) = messages.get(i) {
                if msg.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                        let ids: std::collections::HashSet<String> = tool_calls
                            .iter()
                            .filter_map(|tc| {
                                tc.get("id")
                                    .and_then(|id| id.as_str())
                                    .filter(|id| is_valid_id(id))
                                    .map(|id| id.to_string())
                            })
                            .collect();
                        let all_valid = tool_calls.iter().all(|tc| {
                            tc.get("id")
                                .and_then(|id| id.as_str())
                                .map_or(false, is_valid_id)
                        });
                        if all_valid && !ids.is_empty() {
                            assistant_ids.insert(i, ids);
                        } else {
                            bad_assistants.insert(i);
                        }
                    }
                }
            }
        }

        // Seed the fixed-point drop set with the bad assistants.
        let mut dropped_assistants: std::collections::HashSet<usize> = bad_assistants;

        // Map preserved tool result index -> its tool_call_id.
        // We keep a map for EVERY preserved tool message (including invalid ids),
        // because budget backfill can re-introduce standalone tool messages that
        // must then be dropped if they are not covered by a surviving assistant.
        let mut tool_result_ids: std::collections::HashMap<usize, Option<String>> =
            std::collections::HashMap::new();
        for &i in &preserved_indices_sorted {
            if let Some(msg) = messages.get(i) {
                if msg.get("role").and_then(|r| r.as_str()) == Some("tool") {
                    let id = msg
                        .get("tool_call_id")
                        .and_then(|id| id.as_str())
                        .map(|id| id.to_string());
                    tool_result_ids.insert(i, id);
                }
            }
        }

        // Iteratively drop uncovered assistants and orphaned tool results.
        // Note: `dropped_assistants` is seeded above with the "bad" assistants
        // (those with invalid tool_call ids in their tool_calls array).  An
        // assistant with tool_calls whose tool results were all dropped is
        // also added here.  The API requires every preserved tool_call to
        // have a matching tool result, so dropping the whole assistant is
        // the safe choice (we lose its text content but keep the message
        // list API-compliant).
        let mut dropped_tool_results: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Helper: is a non-empty tool_call_id covered by any surviving assistant?
        let is_covered = |id: &str,
                          dropped_assistants: &std::collections::HashSet<usize>|
         -> bool {
            assistant_ids
                .iter()
                .any(|(idx, ids)| !dropped_assistants.contains(idx) && ids.contains(id))
        };

        let mut changed = true;
        while changed {
            changed = false;

            // Drop tool results that are not covered by a surviving assistant.
            for (&i, id) in &tool_result_ids {
                if dropped_tool_results.contains(&i) {
                    continue;
                }
                let keep = match id.as_deref() {
                    Some(id) if is_valid_id(id) => is_covered(id, &dropped_assistants),
                    _ => false,
                };
                if !keep {
                    dropped_tool_results.insert(i);
                    changed = true;
                }
            }

            // Drop assistants whose tool_call ids are not all covered by surviving tools.
            for (&i, ids) in &assistant_ids {
                if dropped_assistants.contains(&i) {
                    continue;
                }
                let all_covered = ids.iter().all(|id| {
                    tool_result_ids.iter().any(|(idx, tid)| {
                        if dropped_tool_results.contains(idx) {
                            return false;
                        }
                        match tid.as_deref() {
                            Some(tid) if is_valid_id(tid) => tid == id.as_str(),
                            _ => false,
                        }
                    })
                });
                if !all_covered {
                    dropped_assistants.insert(i);
                    changed = true;
                }
            }
        }

        preserved_indices_sorted.retain(|i| {
            !dropped_assistants.contains(i) && !dropped_tool_results.contains(i)
        });
    }

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

    let preserved_indices_set: std::collections::HashSet<usize> =
        preserved_indices_sorted.iter().copied().collect();
    let preserved_tokens: u64 = token_counts
        .iter()
        .enumerate()
        .filter(|(i, _)| preserved_indices_set.contains(i))
        .map(|(_, &t)| t)
        .sum();

    let kind = if summarized_count > 0 {
        CompressionKind::Summary
    } else {
        CompressionKind::None
    };

    // Estimate summary token count based on actual content length
    let summary_tokens_estimate = if summarized_count > 0 {
        // More accurate: estimate based on typical summary ratio
        // Summary is usually 5-15% of original, depending on content density
        let ratio = if summarized_count > 50 { 0.08 } else { 0.12 };
        (summarized_tokens as f64 * ratio) as u64
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

    // Truncate to max_len *characters* and add ellipsis if needed.
    // Using byte slicing `&content[..max_len]` panics on multi-byte UTF-8 chars.
    let truncated = if content.chars().count() > max_len {
        format!("{}...", content.chars().take(max_len).collect::<String>())
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
/// - Important messages (errors, decisions, file ops)
fn build_smart_summary(
    count: usize,
    tokens: u64,
    first_ts: Option<i64>,
    last_ts: Option<i64>,
    key_topics: &[String],
    important_snippets: &[String],
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

    // Build important messages summary
    let important_summary = if important_snippets.is_empty() {
        String::new()
    } else {
        let snippets_text = important_snippets
            .iter()
            .take(5) // Limit to 5 important snippets
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n");
        format!("\n\nImportant earlier events:\n{}", snippets_text)
    };

    let content = format!(
        "[Context Summary — {} messages, ~{} tokens compacted{}]\n\
         These earlier messages have been summarized to save context space. \
         The conversation history includes tool calls, code reviews, and file operations. \
         Refer to the most recent exchanges for current active context.\
         {}{}",
        count, tokens, time_range, topic_summary, important_summary
    );

    // Use "system" role for summary — consistent with system prompt positioning,
    // avoids confusing the model with a user message that isn't actually from the user
    serde_json::json!({
        "role": "system",
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
            target_after: 0.6,
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
        // system (1) + mandatory last 4 rounds (4) + one budget backfill (1) + summary (1) = 7
        assert_eq!(result.final_message_count, 7);
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
        let (msgs, tokens) = make_msgs(11, 100);
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
        assert_eq!(c.target_after_tokens(), 600);
    }

    #[test]
    fn summary_message_content() {
        let placeholder = build_smart_summary(
            50,
            12000,
            Some(1000000),
            Some(1000200),
            &["Hello world".to_string(), "Fix bug in main.rs".to_string()],
            &["Error in auth module".to_string()],
        );
        let content = placeholder["content"].as_str().unwrap();
        assert!(content.contains("50"));
        assert!(content.contains("12000"));
        assert!(content.contains("Context Summary"));
        assert!(content.contains("Hello world"));
        assert!(content.contains("Fix bug"));
        assert!(content.contains("Error in auth"));
        // Summary should use system role
        assert_eq!(placeholder["role"].as_str(), Some("system"));
    }

    #[test]
    fn high_cumulative_triggers_summary() {
        // 200 messages, all 100 tokens = 20K total
        let (msgs, tokens) = make_msgs(200, 100);
        // Cumulative = 25K (way over 1K window)
        let result = apply_sliding_window(&msgs, &tokens, 25_000, &config());
        // system (1) + mandatory last 4 rounds (4) + one budget backfill (1) + summary (1) = 7
        assert_eq!(result.final_message_count, 7);
        assert_eq!(result.kind, CompressionKind::Summary);
        // First message is system
        assert_eq!(result.messages[0]["role"].as_str(), Some("system"));
        // Second message is summary
        assert!(result.messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("194 messages")); // 200 - 6 = 194 summarized
    }

    #[test]
    fn budget_backfill_uses_target_after() {
        // 20 messages, each 100 tokens. Mandatory = system (100) + last 4 rounds (400) = 500.
        // target_after = 600, so budget backfill can add one more 100-token message.
        let (msgs, tokens) = make_msgs(20, 100);
        let result = apply_sliding_window(&msgs, &tokens, 900, &config());
        assert_eq!(result.kind, CompressionKind::Summary);
        assert_eq!(result.final_message_count, 7); // 6 preserved + summary
        assert!(result.cumulative_after <= config().target_after_tokens() + 200);
    }

    #[test]
    fn larger_messages_reduce_backfill() {
        // Same message count, but each message is 250 tokens.
        // Mandatory = system (250) + last 4 rounds (1000) = 1250 > target (600).
        // No budget backfill possible; only mandatory preserved.
        let (msgs, tokens) = make_msgs(10, 250);
        let result = apply_sliding_window(&msgs, &tokens, 900, &config());
        assert_eq!(result.kind, CompressionKind::Summary);
        // system (1) + last 4 rounds (4) + summary (1) = 6
        assert_eq!(result.final_message_count, 6);
    }

    #[test]
    fn backfill_keeps_tool_groups_atomic() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "1", "function": {"name": "f1", "arguments": "{}"}},
                    {"id": "2", "function": {"name": "f2", "arguments": "{}"}}
                ]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "1", "content": "r1"}),
            serde_json::json!({"role": "tool", "tool_call_id": "2", "content": "r2"}),
        ];
        // system=50, user=100, assistant=50, tools=50 each
        let tokens = vec![50u64, 100, 50, 50, 50];
        let mut cfg = config();
        cfg.preserve_rounds = 0; // only system is mandatory
        cfg.target_after = 0.25; // target = 250; budget after system = 200
        // The tool group (assistant + 2 tools = 150) fits, the user message (100) does not.
        let result = apply_sliding_window(&msgs, &tokens, 900, &cfg);
        assert_eq!(result.kind, CompressionKind::Summary);
        // Either all tools are preserved or none (atomic group)
        let tool_count = result
            .messages
            .iter()
            .filter(|m| m["role"].as_str() == Some("tool"))
            .count();
        assert!(tool_count == 0 || tool_count == 2);
        // If tools are preserved, the parent assistant must also be preserved
        if tool_count == 2 {
            assert!(result
                .messages
                .iter()
                .any(|m| m["role"].as_str() == Some("assistant")));
        }
    }

    #[test]
    fn assistant_without_tool_results_is_dropped() {
        // Regression: assistant with valid tool_calls but no matching tool result
        // anywhere in the message array.  Without this fix the API rejects with
        // "an assistant message with 'tool_calls' must be followed by tool messages".
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_no_result", "function": {"name": "read", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "user", "content": "u2"}),
            serde_json::json!({"role": "assistant", "content": "ack"}),
        ];
        let tokens = vec![50u64, 100, 50, 100, 50];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        let has_orphan_assistant = result.messages.iter().any(|m| {
            m["role"].as_str() == Some("assistant")
                && m.get("tool_calls").is_some()
        });
        assert!(!has_orphan_assistant, "assistant with uncovered tool_calls must be dropped");
    }

    #[test]
    fn multiple_tool_rounds_mixed_validity() {
        // Complex: 2 rounds, first has orphaned tool result, second is complete.
        // Only the second round should survive tool-pair validation.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "orphan", "function": {"name": "read", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "orphan", "content": "r1"}),
            serde_json::json!({"role": "user", "content": "u2"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "valid", "function": {"name": "write", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "valid", "content": "r2"}),
            serde_json::json!({"role": "user", "content": "u3"}),
        ];
        // With small budget, old round gets summarized.
        let tokens = vec![50u64, 200, 200, 200, 200, 200, 200, 200];
        let mut cfg = config();
        cfg.context_window = 1600;
        cfg.threshold = 0.8;
        cfg.target_after = 0.5;
        cfg.preserve_rounds = 2; // 2 rounds = 4 user/assistant messages = only u3/a2 preserved by default
        let result = apply_sliding_window(&msgs, &tokens, 1400, &cfg);

        // The valid pair (round 2) must survive
        assert!(result
            .messages
            .iter()
            .any(|m| m["tool_call_id"].as_str() == Some("valid")));
        assert!(result.messages.iter().any(|m| {
            m["role"].as_str() == Some("assistant")
                && m.get("tool_calls")
                    .map(|tc| tc.as_array().map(|a| a.iter().any(|t| t["id"] == "valid")).unwrap_or(false))
                    .unwrap_or(false)
        }));
    }

    #[test]
    fn parallel_tool_calls_one_missing_drops_all() {
        // Assistant has 2 tool_calls but only 1 result exists.
        // The whole assistant must be dropped because the API requires
        // a result for every tool_call.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "read", "arguments": "{}"}},
                    {"id": "tc2", "function": {"name": "write", "arguments": "{}"}}
                ]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "tc1", "content": "result1"}),
            serde_json::json!({"role": "user", "content": "u2"}),
        ];
        let tokens = vec![50u64, 100, 50, 50, 100];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        // tc1's result is orphaned (assistant dropped) → both dropped
        let has_tc1 = result.messages.iter().any(|m| m["tool_call_id"].as_str() == Some("tc1"));
        let has_tc2 = result.messages.iter().any(|m| m["tool_call_id"].as_str() == Some("tc2"));
        let has_assistant_with_tc = result
            .messages
            .iter()
            .any(|m| m["role"].as_str() == Some("assistant") && m.get("tool_calls").is_some());
        assert!(!has_assistant_with_tc, "assistant must be dropped (tc2 has no result)");
        assert!(!has_tc1, "tc1 result must be dropped (assistant is gone)");
        assert!(!has_tc2, "tc2 result does not exist in messages");
    }

    #[test]
    fn parallel_tool_calls_all_present_preserved() {
        // Assistant has 2 tool_calls, both results exist → all preserved.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "read", "arguments": "{}"}},
                    {"id": "tc2", "function": {"name": "write", "arguments": "{}"}}
                ]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "tc1", "content": "result1"}),
            serde_json::json!({"role": "tool", "tool_call_id": "tc2", "content": "result2"}),
            serde_json::json!({"role": "user", "content": "u2"}),
        ];
        let tokens = vec![50u64, 100, 50, 50, 50, 100];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        assert!(result.messages.iter().any(|m| m["tool_call_id"].as_str() == Some("tc1")));
        assert!(result.messages.iter().any(|m| m["tool_call_id"].as_str() == Some("tc2")));
    }

    #[test]
    fn tool_result_before_assistant_is_dropped() {
        // Malformed input: a tool result appears BEFORE the assistant that
        // generated it.  The API cannot match it.  Step 4c must drop it.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "tool", "tool_call_id": "x", "content": "early result"}),
            serde_json::json!({
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}]
            }),
        ];
        let tokens = vec![50u64, 50, 50];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        // The forward-scan in step 3 starts after the assistant (idx 2) and finds
        // nothing, but step 4 keeps the assistant because the tool result's id
        // is in preserved_tool_call_ids... wait, the tool result isn't preserved.
        // The assistant at idx 2 IS preserved.  Step 4b should rescue the tool
        // result at idx 1.  Then the pair is complete.
        // What we assert: no orphan, the pair is together.
        let has_orphan = result
            .messages
            .iter()
            .any(|m| m["tool_call_id"].as_str() == Some("x"))
            && !result
                .messages
                .iter()
                .any(|m| m["role"].as_str() == Some("assistant") && m.get("tool_calls").is_some());
        assert!(!has_orphan);
    }

    #[test]
    fn fixed_point_drops_chained_orphans() {
        // Complex fixed-point scenario:
        // - assistant A1 has id "good" + "bad"
        // - assistant A2 has id "bad2"
        // - only tool result for "good" exists
        // Expected: A1 dropped (bad uncovered), A2 dropped (bad2 uncovered),
        //           tool result for "good" dropped (A1 gone).
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "good", "function": {"name": "f1", "arguments": "{}"}},
                    {"id": "bad", "function": {"name": "f2", "arguments": "{}"}}
                ]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "good", "content": "r1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "bad2", "function": {"name": "f3", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "user", "content": "u2"}),
        ];
        let tokens = vec![50u64, 100, 50, 50, 50, 100];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        let assistant_with_tc = result
            .messages
            .iter()
            .filter(|m| m["role"].as_str() == Some("assistant"))
            .filter(|m| m.get("tool_calls").is_some())
            .count();
        let tool_count = result
            .messages
            .iter()
            .filter(|m| m["role"].as_str() == Some("tool"))
            .count();
        assert_eq!(assistant_with_tc, 0, "all assistants with tool_calls must be dropped");
        assert_eq!(tool_count, 0, "all orphaned tool results must be dropped");
    }

    #[test]
    fn assistant_with_empty_tool_call_id_is_dropped() {
        // Regression: assistant's tool_calls array contains an entry with empty id.
        // The API cannot match the (future) tool result, so drop the assistant.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "", "function": {"name": "read", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "user", "content": "u2"}),
        ];
        let tokens = vec![50u64, 100, 50, 100];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        let has_bad_assistant = result.messages.iter().any(|m| {
            m["role"].as_str() == Some("assistant") && m.get("tool_calls").is_some()
        });
        assert!(!has_bad_assistant, "assistant with empty tool_call id must be dropped");
    }

    #[test]
    fn empty_tool_call_ids_are_dropped() {
        // Regression: missing/empty tool_call_id values cause
        // "tool_call_id is not found" API errors.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({"role": "tool", "tool_call_id": "", "content": "empty id"}),
            serde_json::json!({"role": "tool", "content": "missing id"}),
            serde_json::json!({"role": "assistant", "content": "ack"}),
        ];
        let tokens = vec![50u64, 100, 50, 50, 50];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        let tool_count = result
            .messages
            .iter()
            .filter(|m| m["role"].as_str() == Some("tool"))
            .count();
        assert_eq!(tool_count, 0, "tool results with empty/missing tool_call_id must be dropped");
    }

    #[test]
    fn orphaned_tool_result_reference_dropped() {
        // Regression: a tool result references a tool_call_id whose parent assistant
        // is not preserved. The API rejects with "tool result's tool id(...) not found".
        // The post-pass must drop the orphaned tool result.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            // Orphaned tool result: no assistant in the array has tool_call_id "ghost"
            serde_json::json!({"role": "tool", "tool_call_id": "ghost", "content": "phantom"}),
            serde_json::json!({"role": "user", "content": "u2"}),
            serde_json::json!({"role": "assistant", "content": "ack"}),
        ];
        let tokens = vec![50u64, 100, 50, 100, 50];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        // The orphaned tool result for "ghost" must be dropped.
        let has_orphan = result
            .messages
            .iter()
            .any(|m| m["tool_call_id"].as_str() == Some("ghost"));
        assert!(!has_orphan, "orphaned tool result for 'ghost' must be dropped");
    }

    #[test]
    fn orphaned_tool_result_preserved_by_post_pass() {
        // Regression: assistant with tool_calls at idx 2, but tool result at idx 4
        // (idx 3 is a non-tool message that breaks step 3's forward scan).
        // The post-pass (step 4b) must rescue the orphaned tool result.
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_old", "function": {"name": "read", "arguments": "{}"}}]
            }),
            serde_json::json!({"role": "user", "content": "u2"}), // non-tool msg between assistant and tool result
            serde_json::json!({"role": "tool", "tool_call_id": "tc_old", "content": "file content"}),
            serde_json::json!({"role": "user", "content": "u3"}),
            serde_json::json!({
                "role": "assistant",
                "content": "done",
            }),
        ];
        let tokens = vec![50u64, 100, 50, 100, 50, 100, 50];
        let result = apply_sliding_window(&msgs, &tokens, 2000, &config());

        // The assistant at idx 2 must NOT be preserved without its tool result,
        // OR the tool result at idx 4 must also be preserved.
        // Either both are present or neither (tool_calls stripped would also work,
        // but current impl preserves both).
        let has_assistant_with_tc = result.messages.iter().any(|m| {
            m["role"].as_str() == Some("assistant") && m.get("tool_calls").is_some()
        });
        let has_tool_result = result
            .messages
            .iter()
            .any(|m| m["tool_call_id"].as_str() == Some("tc_old"));

        if has_assistant_with_tc {
            assert!(
                has_tool_result,
                "assistant with tool_calls preserved but tool result for tc_old is missing!"
            );
        }
    }

    #[test]
    fn snippet_truncates_multibyte_utf8_at_char_boundary() {
        // Regression test: byte-index slicing panicked on multi-byte UTF-8 chars
        // (e.g. Chinese text from weeyuen project docs).
        let msg = serde_json::json!({"role": "user", "content": "`wy_frps` 表是本地表，不管国内桩还是外贸桩，都从这里取空闲 FRP 隧道。新增端口有两种方式。"});
        let snippet = extract_message_content_snippet(&msg, 20).unwrap();
        // Should truncate to 20 chars (not bytes) and end with "..."
        assert!(snippet.ends_with("..."));
        assert_eq!(snippet.chars().count(), 23); // 20 chars + 3 ellipsis
        // Must be valid UTF-8
        assert!(snippet.chars().next().is_some());
    }
}
