//! Rolling Context module for cc-switch proxy.
//!
//! ## Lifecycle
//!
//! ```text
//!                            REQUEST PATH                          RESPONSE PATH
//!                                                                ┌─────────────────────┐
//!                                                                │  upstream returns    │
//!                                                                │  response with       │
//!                                                                │  usage.input_tokens  │
//!                                                                └──────────┬──────────┘
//!                                                                           │
//!   client sends request with                                                │
//!   session_id + messages                                                    │
//!        │                                                                   │
//!        ▼                                                                   ▼
//!   ┌─────────────────┐                                            ┌──────────────────┐
//!   │ pre_send:       │                                            │ post_response:   │
//!   │ apply() checks  │                                            │ record_response_ │
//!   │ session.cumul.  │                                            │ usage() updates  │
//!   │ tokens > thresh?│                                            │ session.cumul.   │
//!   └────────┬────────┘                                            └────────┬─────────┘
//!            │                                                              │
//!            ▼ if yes                                                       ▼
//!   ┌─────────────────┐                                            ┌──────────────────┐
//!   │ truncate body,  │                                            │ store message in │
//!   │ record event,   │                                            │ history (best-   │
//!   │ reset cumul.    │                                            │ effort)          │
//!   └────────┬────────┘                                            └──────────────────┘
//!            │
//!            ▼
//!       forward to upstream
//! ```
//!
//! ## What changed in v2
//!
//! - **Pre-send threshold check** uses `session.total_input_tokens` (cumulative
//!   from upstream responses), not a per-request heuristic.
//! - **Post-response accumulator** writes `usage.input_tokens` into
//!   `rolling_context_sessions.total_input_tokens`.
//! - **Compression** is recorded in `rolling_context_compressions` for audit.
//! - **Eviction** keeps the per-session message log bounded to 500 rows.

pub mod compressor;
pub mod context_window;
pub mod message_store;
pub mod token_counter;

use crate::provider::{Provider, ProviderMeta};
use compressor::CompressionStrategy;
use context_window::{apply_sliding_window, RollingConfig, RollingResult};
use message_store::{CompressionEvent, MessageRecord, MessageStore};
use serde_json::Value;
use token_counter::estimate_message_tokens;

/// Statistics about a rolling-context operation, for logging/observability.
#[derive(Debug, Clone, Default)]
pub struct RollingStats {
    pub was_truncated: bool,
    pub messages_before: usize,
    pub messages_after: usize,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub cumulative_before: u64,
    pub compression_index: u32, // session.compression_count at the time of truncation
}

/// Apply rolling context to a request body **before** forwarding to upstream.
///
/// This is the pre-send entry point. It:
/// 1. Checks if rolling-context is enabled for this provider
/// 2. Reads the session's cumulative input tokens from the DB
/// 3. If cumulative > threshold, truncates the messages array
/// 4. Records the compression event
/// 5. Returns the modified body and statistics
///
/// Returns `Ok(None)` if rolling-context is disabled or no work was done.
pub fn apply(
    body: &mut Value,
    session_id: &str,
    provider: &Provider,
    store: &MessageStore,
) -> Result<Option<RollingStats>, String> {
    // (1) Gate: feature enabled?
    let meta = match provider.meta.as_ref() {
        Some(m) if m.rolling_context_active() => m,
        _ => return Ok(None),
    };

    log::info!(
        "[RollingContext] apply() entered: session={} provider={} rolling_active=true",
        session_id,
        provider.id,
    );

    // (2) Extract model name before mutably borrowing messages
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string());

    // (3) Extract messages array
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(msgs) if !msgs.is_empty() => msgs,
        _ => {
            log::info!(
                "[RollingContext] apply() early return: no messages array or empty (session={})",
                session_id,
            );
            return Ok(None);
        }
    };

    log::info!(
        "[RollingContext] apply() messages count={} (session={})",
        messages.len(),
        session_id,
    );

    // (4) Build rolling config
    let context_window = meta.context_window_or_default();
    let config = RollingConfig {
        context_window,
        threshold: meta.rolling_threshold(),
        preserve_rounds: meta.preserve_rounds(),
    };

    // (5) Get or create session, read cumulative usage
    let session = store.get_or_create_session(
        session_id,
        &provider.id,
        model.as_deref(),
        Some(context_window),
    )?;
    let cumulative_before = session.total_input_tokens;
    let trigger_limit = (context_window as f64 * config.threshold) as u64;

    log::info!(
        "[RollingContext] apply() session loaded: cumulative={} trigger={} (window={} threshold={}) session_id={} provider={}",
        cumulative_before,
        trigger_limit,
        context_window,
        config.threshold,
        session_id,
        provider.id,
    );

    // (6) Estimate per-message tokens (for the truncate decision)
    let token_counts: Vec<u64> = messages.iter().map(|m| estimate_message_tokens(m).total()).collect();
    let body_tokens: u64 = token_counts.iter().sum();

    // (7) Persist the current request's messages to history (best-effort)
    let records: Vec<MessageRecord> = messages
        .iter()
        .zip(token_counts.iter())
        .map(|(msg, &tokens)| {
            let role = msg
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("unknown")
                .to_string();
            let content = extract_content_text(msg);
            MessageRecord {
                id: None,
                session_id: session_id.to_string(),
                role,
                content,
                token_count: Some(tokens),
                is_summary: false,
                summary_source_ids: Vec::new(),
                created_at: None,
            }
        })
        .collect();
    if let Err(e) = store.insert_messages(session_id, &records) {
        log::debug!("[RollingContext] insert_messages best-effort failed: {e}");
    }

    // (8) Decide whether to truncate
    let current_messages: Vec<Value> = messages.clone();
    let result = apply_sliding_window(&current_messages, &token_counts, cumulative_before, &config);

    let stats = RollingStats {
        was_truncated: result.kind != context_window::CompressionKind::None,
        messages_before: current_messages.len(),
        messages_after: result.final_message_count,
        tokens_before: body_tokens,
        tokens_after: result.cumulative_after,
        cumulative_before,
        compression_index: session.compression_count,
    };

    // (9) Apply truncation if any
    if !stats.was_truncated {
        return Ok(Some(stats));
    }

    log::info!(
        "[RollingContext] session={} provider={} truncated {} → {} messages (cumulative {} → {} tokens, ratio {:.1}%)",
        session_id,
        provider.id,
        stats.messages_before,
        stats.messages_after,
        cumulative_before,
        result.cumulative_after,
        100.0 * cumulative_before as f64 / context_window as f64,
    );

    // Replace messages in body
    *body.get_mut("messages").unwrap() = Value::Array(result.messages);

    // Record compression event
    let event = CompressionEvent {
        id: None,
        session_id: session_id.to_string(),
        trigger: "threshold".to_string(),
        tokens_before: cumulative_before,
        tokens_after: result.cumulative_after,
        messages_removed: result.removed_count as i64,
        messages_summarized: 0,
        summary_text: None,
        created_at: None,
    };
    if let Err(e) = store.record_compression(&event) {
        log::warn!("[RollingContext] failed to record compression event: {e}");
    }

    // Reset cumulative so the next response's usage becomes the new baseline
    if let Err(e) = store.reset_cumulative_tokens(session_id) {
        log::warn!("[RollingContext] failed to reset cumulative tokens: {e}");
    }

    Ok(Some(stats))
}

/// Record token usage from an upstream response.
///
/// This is the **post-response** entry point. Should be called from
/// `response_processor` after parsing the response body for `usage`.
///
/// `delta_*` fields represent *new* usage from this request (not cumulative).
/// The function adds them to the session's running totals.
pub fn record_response_usage(
    session_id: &str,
    store: &MessageStore,
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: u32,
    cache_creation_tokens: u32,
) -> Result<(), String> {
    if input_tokens == 0 && output_tokens == 0 {
        // Nothing to record
        return Ok(());
    }
    store.record_response_usage(
        session_id,
        input_tokens as u64,
        output_tokens as u64,
        cache_read_tokens as u64,
        cache_creation_tokens as u64,
    )?;
    log::debug!(
        "[RollingContext] session={} recorded +{} in / +{} out (cache_read={} cache_creation={})",
        session_id,
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
    );
    Ok(())
}

/// Extract text content from a message for storage.
fn extract_content_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for block in blocks {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    parts.push(text.to_string());
                }
            }
            parts.join("\n")
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn in_memory_store() -> MessageStore {
        let conn = rusqlite::Connection::open_in_memory().expect("open memory db");
        conn.execute(
            "CREATE TABLE rolling_context_sessions (
                session_id TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                model TEXT,
                context_window INTEGER,
                total_input_tokens INTEGER NOT NULL DEFAULT 0,
                total_output_tokens INTEGER NOT NULL DEFAULT 0,
                total_cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                total_cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                compression_count INTEGER NOT NULL DEFAULT 0,
                tokens_saved INTEGER NOT NULL DEFAULT 0,
                last_active_at INTEGER,
                created_at INTEGER
            );",
            [],
        )
        .unwrap();
        conn.execute(
            "CREATE TABLE rolling_context_messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER,
                is_summary INTEGER DEFAULT 0,
                summary_source_ids TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER
            );",
            [],
        )
        .unwrap();
        conn.execute(
            "CREATE TABLE rolling_context_compressions (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                trigger TEXT NOT NULL,
                tokens_before INTEGER NOT NULL,
                tokens_after INTEGER NOT NULL,
                messages_removed INTEGER NOT NULL DEFAULT 0,
                messages_summarized INTEGER NOT NULL DEFAULT 0,
                summary_text TEXT,
                created_at INTEGER
            );",
            [],
        )
        .unwrap();
        MessageStore::new(Arc::new(Mutex::new(conn)))
    }

    fn make_provider(context_window: u64, enabled: bool) -> Provider {
        Provider {
            id: "test-prov".to_string(),
            name: "Test".to_string(),
            settings_config: serde_json::json!({}),
            website_url: None,
            category: None,
            created_at: None,
            sort_index: None,
            notes: None,
            meta: Some(ProviderMeta {
                context_window: Some(context_window),
                rolling_context_enabled: Some(enabled),
                rolling_context_threshold: Some(0.8),
                rolling_context_preserve_rounds: Some(2),
                ..Default::default()
            }),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        }
    }

    #[test]
    fn disabled_returns_none() {
        let store = in_memory_store();
        let provider = make_provider(1000, false);
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
        });
        let stats = apply(&mut body, "sess-1", &provider, &store).unwrap();
        assert!(stats.is_none());
    }

    #[test]
    fn enabled_under_threshold_passes_through() {
        let store = in_memory_store();
        let provider = make_provider(10000, true);
        // Pre-populate session with low cumulative usage
        store
            .get_or_create_session("sess-1", "test-prov", None, Some(10000))
            .unwrap();
        store
            .record_response_usage("sess-1", 100, 50, 0, 0)
            .unwrap();
        let mut body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        });
        let stats = apply(&mut body, "sess-1", &provider, &store).unwrap();
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert!(!s.was_truncated);
        assert_eq!(s.messages_before, 3);
    }

    #[test]
    fn enabled_over_cumulative_threshold_truncates() {
        let store = in_memory_store();
        let provider = make_provider(10000, true); // trigger at 8000
        // Pre-populate: cumulative = 9000 (over 8000)
        store
            .get_or_create_session("sess-2", "test-prov", None, Some(10000))
            .unwrap();
        store
            .record_response_usage("sess-2", 9000, 0, 0, 0)
            .unwrap();
        // Many rounds with large content so target (60% of 10K = 6K) is exceeded
        let mut msgs = vec![serde_json::json!({"role": "system", "content": "sys"})];
        for i in 0..20 {
            msgs.push(serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("round {} {}", i, "x".repeat(2000)),
            }));
        }
        let mut body = serde_json::json!({
            "model": "gpt-4",
            "messages": msgs
        });
        let stats = apply(&mut body, "sess-2", &provider, &store).unwrap().unwrap();
        assert!(stats.was_truncated);
        // Body should be modified
        let final_msgs = body["messages"].as_array().unwrap();
        assert!(final_msgs.len() < 21);
        // System message preserved
        assert_eq!(final_msgs[0]["role"].as_str(), Some("system"));
    }

    #[test]
    fn record_response_usage_updates_cumulative() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "test-prov", None, Some(1000))
            .unwrap();
        record_response_usage("sess-1", &store, 100, 50, 0, 0).unwrap();
        record_response_usage("sess-1", &store, 200, 100, 10, 5).unwrap();
        let session = store
            .get_or_create_session("sess-1", "test-prov", None, Some(1000))
            .unwrap();
        assert_eq!(session.total_input_tokens, 300);
        assert_eq!(session.total_output_tokens, 150);
        assert_eq!(session.total_cache_read_tokens, 10);
        assert_eq!(session.total_cache_creation_tokens, 5);
    }

    #[test]
    fn record_response_usage_with_zero_is_noop() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "test-prov", None, Some(1000))
            .unwrap();
        record_response_usage("sess-1", &store, 0, 0, 0, 0).unwrap();
        let session = store
            .get_or_create_session("sess-1", "test-prov", None, Some(1000))
            .unwrap();
        assert_eq!(session.total_input_tokens, 0);
    }

    #[test]
    fn compression_event_recorded() {
        let store = in_memory_store();
        let provider = make_provider(10000, true); // trigger at 8000
        store
            .get_or_create_session("sess-3", "test-prov", None, Some(10000))
            .unwrap();
        // Cumulative > trigger; need body large enough to actually have to drop
        store.record_response_usage("sess-3", 9000, 0, 0, 0).unwrap();
        // Many rounds + large content so target (60% of 10K = 6K) is exceeded
        // by preserved alone, forcing non-preserved to be dropped.
        let mut msgs = vec![serde_json::json!({"role": "system", "content": "sys"})];
        for i in 0..20 {
            msgs.push(serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("round {} {}", i, "x".repeat(2000)),
            }));
        }
        let mut body = serde_json::json!({"messages": msgs});
        let stats = apply(&mut body, "sess-3", &provider, &store).unwrap().unwrap();
        assert!(stats.was_truncated, "Expected truncation, got stats: {stats:?}");
        let history = store.get_compression_history("sess-3", 10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].trigger, "threshold");
        assert!(history[0].tokens_before > history[0].tokens_after);
    }

    #[test]
    fn cumulative_resets_after_compression() {
        let store = in_memory_store();
        let provider = make_provider(10000, true);
        store
            .get_or_create_session("sess-4", "test-prov", None, Some(10000))
            .unwrap();
        store.record_response_usage("sess-4", 9000, 0, 0, 0).unwrap();
        let mut msgs = vec![serde_json::json!({"role": "system", "content": "sys"})];
        for i in 0..20 {
            msgs.push(serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("round {} {}", i, "x".repeat(2000)),
            }));
        }
        let mut body = serde_json::json!({"messages": msgs});
        apply(&mut body, "sess-4", &provider, &store).unwrap();
        // After compression, cumulative should be reset
        let session = store
            .get_or_create_session("sess-4", "test-prov", None, Some(10000))
            .unwrap();
        assert_eq!(session.total_input_tokens, 0);
        assert_eq!(session.compression_count, 1);
        assert!(session.tokens_saved > 0);
    }

    #[test]
    fn no_meta_returns_none() {
        let store = in_memory_store();
        let mut provider = make_provider(1000, false);
        provider.meta = None;
        let mut body = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        let stats = apply(&mut body, "sess-1", &provider, &store).unwrap();
        assert!(stats.is_none());
    }

    #[test]
    fn empty_messages_array_returns_none() {
        let store = in_memory_store();
        let provider = make_provider(1000, true);
        let mut body = serde_json::json!({"messages": []});
        let stats = apply(&mut body, "sess-empty", &provider, &store).unwrap();
        assert!(stats.is_none());
    }
}
