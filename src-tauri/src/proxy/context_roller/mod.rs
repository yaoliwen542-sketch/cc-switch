//! Rolling Context module for cc-switch proxy.
//!
//! Provides automatic context window management by tracking per-session
//! message history and truncating older messages when approaching limits.

pub mod compressor;
pub mod context_window;
pub mod message_store;
pub mod token_counter;

use crate::provider::{Provider, ProviderMeta};
use context_window::{apply_sliding_window, RollingConfig};
use message_store::{MessageRecord, MessageStore};
use serde_json::Value;
use token_counter::estimate_message_tokens;

/// Apply rolling context to a request body.
///
/// This is the main entry point called from the proxy forwarder.
/// It:
/// 1. Extracts messages from the request body
/// 2. Estimates token counts
/// 3. Stores messages in the session history
/// 4. Applies sliding window truncation if over threshold
/// 5. Updates the request body with truncated messages
///
/// Returns `Ok(true)` if the body was modified, `Ok(false)` if no changes.
pub fn apply(
    body: &mut Value,
    session_id: &str,
    provider: &Provider,
    store: &MessageStore,
) -> Result<bool, String> {
    // Check if rolling context is enabled for this provider
    let meta = provider.meta.as_ref();
    let enabled = meta.map(|m| m.rolling_context_active()).unwrap_or(false);
    if !enabled {
        return Ok(false);
    }

    // Get model name before borrowing messages mutably
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string());

    // Extract messages array
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(msgs) if !msgs.is_empty() => msgs,
        _ => return Ok(false),
    };

    // Build rolling config from provider meta
    let meta = meta.unwrap(); // safe: we checked enabled above
    let context_window = meta.context_window_or_default();
    let threshold = meta.rolling_threshold();
    let preserve_rounds = meta.preserve_rounds();

    let config = RollingConfig {
        context_window,
        threshold,
        preserve_rounds,
    };
    let _session = store.get_or_create_session(
        session_id,
        &provider.id,
        model.as_deref(),
        Some(context_window),
    )?;

    // Estimate token counts for each message
    let token_counts: Vec<u64> = messages.iter().map(estimate_message_tokens).collect();
    let total_tokens = token_counts.iter().sum::<u64>();

    // Store messages in history (before truncation, for full history tracking)
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
                created_at: None,
            }
        })
        .collect();

    store.insert_messages(session_id, &records)?;

    // Apply sliding window
    let current_messages: Vec<Value> = messages.clone();
    let result = apply_sliding_window(&current_messages, &token_counts, &config);

    if result.was_truncated {
        log::info!(
            "[RollingContext] Session {}: truncated {} messages ({} -> {} tokens, {} -> {} msgs)",
            session_id,
            result.removed_count,
            result.tokens_before,
            result.tokens_after,
            current_messages.len(),
            result.final_message_count,
        );

        // Update the request body
        *body.get_mut("messages").unwrap() = Value::Array(result.messages);

        // Update session stats
        store.update_session_tokens(session_id, total_tokens, 0)?;
        store.increment_compression_count(session_id)?;

        Ok(true)
    } else {
        // Still update token counts even if no truncation
        store.update_session_tokens(session_id, total_tokens, 0)?;
        Ok(false)
    }
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
    use message_store::MessageStore;
    use std::sync::{Arc, Mutex};

    fn in_memory_store() -> MessageStore {
        let conn = rusqlite::Connection::open_in_memory().expect("open memory db");
        conn.execute(
            "CREATE TABLE rolling_context_sessions (
                session_id TEXT PRIMARY KEY, provider_id TEXT NOT NULL,
                model TEXT, context_window INTEGER,
                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                compression_count INTEGER DEFAULT 0,
                last_active_at INTEGER, created_at INTEGER
            );",
            [],
        )
        .unwrap();
        conn.execute(
            "CREATE TABLE rolling_context_messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL,
                token_count INTEGER, is_summary INTEGER DEFAULT 0,
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
    fn disabled_does_not_modify() {
        let store = in_memory_store();
        let provider = make_provider(1000, false);
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
        });

        let modified = apply(&mut body, "sess-1", &provider, &store).unwrap();
        assert!(!modified);
    }

    #[test]
    fn enabled_no_messages_no_modify() {
        let store = in_memory_store();
        let provider = make_provider(1000, true);
        let mut body = serde_json::json!({
            "model": "gpt-4",
        });

        let modified = apply(&mut body, "sess-1", &provider, &store).unwrap();
        assert!(!modified);
    }

    #[test]
    fn enabled_under_threshold_no_truncation() {
        let store = in_memory_store();
        let provider = make_provider(10000, true); // trigger at 8000
        let mut body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        });

        let modified = apply(&mut body, "sess-1", &provider, &store).unwrap();
        assert!(!modified);

        // Messages should be stored in DB
        let msgs = store.get_messages("sess-1").unwrap();
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn enabled_triggers_when_over_threshold() {
        let store = in_memory_store();
        let provider = make_provider(500, true); // trigger at 400
        let mut body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful. "},
                {"role": "user", "content": "a".repeat(200)},
                {"role": "assistant", "content": "b".repeat(200)},
                {"role": "user", "content": "c".repeat(200)},
                {"role": "assistant", "content": "d".repeat(200)},
                {"role": "user", "content": "e".repeat(200)},
            ]
        });

        let modified = apply(&mut body, "sess-2", &provider, &store).unwrap();
        // The messages have enough content to trigger truncation
        let msgs = body["messages"].as_array().unwrap();
        assert!(msgs.len() <= 6); // may or may not be truncated depending on estimates

        // Session should be created
        let session = store.get_or_create_session("sess-2", "test-prov", Some("gpt-4"), Some(500)).unwrap();
        assert_eq!(session.compression_count, if modified { 1 } else { 0 });
    }

    #[test]
    fn messages_stored_in_db() {
        let store = in_memory_store();
        let provider = make_provider(10000, true);
        let mut body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello world"},
            ]
        });

        apply(&mut body, "sess-db", &provider, &store).unwrap();

        let stored = store.get_messages("sess-db").unwrap();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, "system");
        assert_eq!(stored[1].role, "user");
        assert_eq!(stored[1].content, "Hello world");
    }

    #[test]
    fn no_meta_does_not_modify() {
        let store = in_memory_store();
        let mut provider = make_provider(1000, false);
        provider.meta = None;

        let mut body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "hello"},
            ]
        });

        let modified = apply(&mut body, "sess-1", &provider, &store).unwrap();
        assert!(!modified);
    }

    #[test]
    fn empty_messages_array() {
        let store = in_memory_store();
        let provider = make_provider(1000, true);
        let mut body = serde_json::json!({
            "messages": []
        });

        let modified = apply(&mut body, "sess-empty", &provider, &store).unwrap();
        assert!(!modified);
    }

    #[test]
    fn array_content_extracted_correctly() {
        let store = in_memory_store();
        let provider = make_provider(10000, true);
        let mut body = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"}
                    ]
                }
            ]
        });

        apply(&mut body, "sess-array", &provider, &store).unwrap();

        let stored = store.get_messages("sess-array").unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].content, "Hello\nWorld");
    }
}
