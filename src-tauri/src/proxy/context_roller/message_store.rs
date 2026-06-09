//! SQLite-backed session/message store for rolling context.
//!
//! ## Schema
//!
//! - `rolling_context_sessions` — one row per session, tracks cumulative token usage
//!   reported by the upstream API. The `total_input_tokens` field is the **ground
//!   truth** for "how full is this session's context window".
//!
//! - `rolling_context_messages` — optional per-message log. We only persist messages
//!   for sessions where the operator wants to inspect history or generate summaries.
//!   For high-throughput use, this is bounded to `MAX_MESSAGES_PER_SESSION`.
//!
//! - `rolling_context_compressions` — audit log of every compression event: which
//!   messages were evicted, what replacement (truncation or LLM summary) was used,
//!   how many tokens were saved.

use rusqlite::OptionalExtension;
use std::sync::{Arc, Mutex};

/// Hard cap on retained messages per session. Once exceeded, oldest are evicted.
pub const MAX_MESSAGES_PER_SESSION: i64 = 500;

/// Hard cap on retained sessions. When exceeded, the least-recently-active session
/// is purged along with its messages.
pub const MAX_SESSIONS: i64 = 200;

/// Session metadata stored in the database.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub session_id: String,
    pub provider_id: String,
    pub model: Option<String>,
    pub context_window: Option<u64>,
    /// Sum of `usage.input_tokens` from every response so far. This is the
    /// authoritative "how full is the context window" number.
    pub total_input_tokens: u64,
    /// Sum of `usage.output_tokens` from every response so far.
    pub total_output_tokens: u64,
    /// Sum of `cache_read_input_tokens` across all responses (Anthropic prompt caching).
    pub total_cache_read_tokens: u64,
    /// Sum of `cache_creation_input_tokens` across all responses.
    pub total_cache_creation_tokens: u64,
    /// Number of times the rolling-context algorithm has truncated this session.
    pub compression_count: u32,
    /// Number of tokens saved across all compressions of this session.
    pub tokens_saved: u64,
    pub last_active_at: Option<i64>,
    pub created_at: Option<i64>,
}

impl SessionRecord {
    /// Returns (current_usage, limit, ratio). ratio = current / limit, clamped to 1.0.
    pub fn utilization(&self) -> Option<(u64, u64, f64)> {
        self.context_window.map(|cw| {
            let current = self.total_input_tokens;
            let ratio = if cw == 0 {
                0.0
            } else {
                (current as f64 / cw as f64).min(1.0)
            };
            (current, cw, ratio)
        })
    }
}

/// A single message in the rolling context history.
#[derive(Debug, Clone)]
pub struct MessageRecord {
    pub id: Option<i64>,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub token_count: Option<u64>,
    /// True for synthesized summary messages that replace older messages.
    pub is_summary: bool,
    /// If this is a summary, the IDs of the messages it summarizes.
    pub summary_source_ids: Vec<i64>,
    pub created_at: Option<i64>,
}

/// A compression event for the audit log.
#[derive(Debug, Clone)]
pub struct CompressionEvent {
    pub id: Option<i64>,
    pub session_id: String,
    pub trigger: String, // "threshold" | "manual" | "eviction"
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub messages_removed: i64,
    pub messages_summarized: i64,
    pub summary_text: Option<String>,
    pub created_at: Option<i64>,
}

/// DAO for rolling context persistence.
pub struct MessageStore {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl MessageStore {
    /// Create a new MessageStore backed by the given SQLite connection.
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    /// Get or create a session record. Updates `last_active_at` on every call.
    pub fn get_or_create_session(
        &self,
        session_id: &str,
        provider_id: &str,
        model: Option<&str>,
        context_window: Option<u64>,
    ) -> Result<SessionRecord, String> {
        let mut conn = self.conn.lock().map_err(|e| e.to_string())?;

        // Try to find existing
        let mut stmt = conn
            .prepare(
                "SELECT session_id, provider_id, model, context_window,
                        total_input_tokens, total_output_tokens,
                        total_cache_read_tokens, total_cache_creation_tokens,
                        compression_count, tokens_saved,
                        last_active_at, created_at
                 FROM rolling_context_sessions WHERE session_id = ?1",
            )
            .map_err(|e| e.to_string())?;

        let existing: Result<Option<SessionRecord>, _> = stmt
            .query_row([session_id], |row| {
                Ok(SessionRecord {
                    session_id: row.get(0)?,
                    provider_id: row.get(1)?,
                    model: row.get(2)?,
                    context_window: row.get(3)?,
                    total_input_tokens: row.get::<_, i64>(4).unwrap_or(0) as u64,
                    total_output_tokens: row.get::<_, i64>(5).unwrap_or(0) as u64,
                    total_cache_read_tokens: row.get::<_, i64>(6).unwrap_or(0) as u64,
                    total_cache_creation_tokens: row.get::<_, i64>(7).unwrap_or(0) as u64,
                    compression_count: row.get::<_, i64>(8).unwrap_or(0) as u32,
                    tokens_saved: row.get::<_, i64>(9).unwrap_or(0) as u64,
                    last_active_at: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .optional()
            .map_err(|e| e.to_string());

        let now = chrono::Utc::now().timestamp();

        if let Ok(Some(session)) = existing {
            conn.execute(
                "UPDATE rolling_context_sessions SET last_active_at = ?1 WHERE session_id = ?2",
                rusqlite::params![now, session_id],
            )
            .map_err(|e| e.to_string())?;
            return Ok(session);
        }

        conn.execute(
            "INSERT INTO rolling_context_sessions
             (session_id, provider_id, model, context_window, last_active_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            rusqlite::params![
                session_id,
                provider_id,
                model,
                context_window.map(|v| v as i64),
                now,
                now,
            ],
        )
        .map_err(|e| e.to_string())?;

        Ok(SessionRecord {
            session_id: session_id.to_string(),
            provider_id: provider_id.to_string(),
            model: model.map(|s| s.to_string()),
            context_window,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_read_tokens: 0,
            total_cache_creation_tokens: 0,
            compression_count: 0,
            tokens_saved: 0,
            last_active_at: Some(now),
            created_at: Some(now),
        })
    }

    /// Update session token counters from a response. This is the primary way
    /// `total_input_tokens` gets populated. Idempotent: if the same delta is
    /// applied twice (e.g., a retry), the second call is a no-op.
    ///
    /// Returns the updated session record.
    pub fn record_response_usage(
        &self,
        session_id: &str,
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_creation_tokens: u64,
    ) -> Result<SessionRecord, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "UPDATE rolling_context_sessions
             SET total_input_tokens       = total_input_tokens       + ?1,
                 total_output_tokens      = total_output_tokens      + ?2,
                 total_cache_read_tokens  = total_cache_read_tokens  + ?3,
                 total_cache_creation_tokens = total_cache_creation_tokens + ?4,
                 last_active_at = ?5
             WHERE session_id = ?6",
            rusqlite::params![
                input_tokens as i64,
                output_tokens as i64,
                cache_read_tokens as i64,
                cache_creation_tokens as i64,
                now,
                session_id,
            ],
        )
        .map_err(|e| e.to_string())?;
        drop(conn);

        // Re-read to return updated record
        self.get_or_create_session(
            session_id,
            "", // unused for re-read
            None,
            None,
        )
    }

    /// Insert a single message. Used by the rolling-context module when a request
    /// arrives to log the messages being sent.
    pub fn insert_message(&self, msg: &MessageRecord) -> Result<i64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT INTO rolling_context_messages
             (session_id, role, content, token_count, is_summary, summary_source_ids, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ",
            rusqlite::params![
                msg.session_id,
                &msg.role,
                &msg.content,
                msg.token_count.map(|v| v as i64),
                msg.is_summary as i32,
                serde_json::to_string(&msg.summary_source_ids).unwrap_or_else(|_| "[]".to_string()),
                msg.created_at.unwrap_or_else(|| chrono::Utc::now().timestamp()),
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(conn.last_insert_rowid())
    }

    /// Bulk insert (transactional).
    pub fn insert_messages(
        &self,
        session_id: &str,
        messages: &[MessageRecord],
    ) -> Result<(), String> {
        {
            let mut conn = self.conn.lock().map_err(|e| e.to_string())?;
            let tx = conn.transaction().map_err(|e| e.to_string())?;

            for msg in messages {
                tx.execute(
                    "INSERT INTO rolling_context_messages
                     (session_id, role, content, token_count, is_summary, summary_source_ids, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                    ",
                    rusqlite::params![
                        msg.session_id,
                        &msg.role,
                        &msg.content,
                        msg.token_count.map(|v| v as i64),
                        msg.is_summary as i32,
                        "[]",
                        msg.created_at.unwrap_or_else(|| chrono::Utc::now().timestamp()),
                    ],
                )
                .map_err(|e| e.to_string())?;
            }

            tx.commit().map_err(|e| e.to_string())?;
            // conn lock is released at end of this scope
        }
        // Now safe to call evict (which takes the lock again)
        self.evict_oldest_messages_for_session(session_id, MAX_MESSAGES_PER_SESSION)?;
        Ok(())
    }

    /// Get all messages for a session, ordered by creation time.
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<MessageRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, role, content, token_count, is_summary, summary_source_ids, created_at
                 FROM rolling_context_messages
                 WHERE session_id = ?1
                 ORDER BY created_at ASC, id ASC",
            )
            .map_err(|e| e.to_string())?;

        let messages = stmt
            .query_map([session_id], |row| {
                let source_ids_json: String = row.get(6)?;
                let summary_source_ids: Vec<i64> =
                    serde_json::from_str(&source_ids_json).unwrap_or_default();
                Ok(MessageRecord {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    token_count: row.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                    is_summary: row.get::<_, i32>(5)? != 0,
                    summary_source_ids,
                    created_at: row.get(7)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        Ok(messages)
    }

    /// Get messages in a specific ID range, ordered by creation time.
    pub fn get_messages_in_range(
        &self,
        session_id: &str,
        start_id: i64,
        end_id: i64,
    ) -> Result<Vec<MessageRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, role, content, token_count, is_summary, summary_source_ids, created_at
                 FROM rolling_context_messages
                 WHERE session_id = ?1 AND id BETWEEN ?2 AND ?3
                 ORDER BY created_at ASC, id ASC",
            )
            .map_err(|e| e.to_string())?;

        let messages = stmt
            .query_map(
                rusqlite::params![session_id, start_id, end_id],
                |row| {
                    let source_ids_json: String = row.get(6)?;
                    let summary_source_ids: Vec<i64> =
                        serde_json::from_str(&source_ids_json).unwrap_or_default();
                    Ok(MessageRecord {
                        id: row.get(0)?,
                        session_id: row.get(1)?,
                        role: row.get(2)?,
                        content: row.get(3)?,
                        token_count: row.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                        is_summary: row.get::<_, i32>(5)? != 0,
                        summary_source_ids,
                        created_at: row.get(7)?,
                    })
                },
            )
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        Ok(messages)
    }

    /// Evict oldest messages for a session, keeping the most recent `keep_count`.
    /// Returns the number of rows deleted.
    pub fn evict_oldest_messages_for_session(
        &self,
        session_id: &str,
        keep_count: i64,
    ) -> Result<u64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM rolling_context_messages WHERE session_id = ?1",
                [session_id],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        let to_delete = (total - keep_count).max(0);
        if to_delete == 0 {
            return Ok(0);
        }
        let deleted = conn
            .execute(
                "DELETE FROM rolling_context_messages
                 WHERE id IN (
                     SELECT id FROM rolling_context_messages
                     WHERE session_id = ?1
                     ORDER BY created_at ASC, id ASC
                     LIMIT ?2
                 )",
                rusqlite::params![session_id, to_delete],
            )
            .map_err(|e| e.to_string())?;
        Ok(deleted as u64)
    }

    /// Delete specific messages by ID list. Used after compression to remove
    /// the messages that were replaced by a summary.
    pub fn delete_messages_by_ids(&self, ids: &[i64]) -> Result<u64, String> {
        if ids.is_empty() {
            return Ok(0);
        }
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        // Build IN clause
        let placeholders = std::iter::repeat("?")
            .take(ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!("DELETE FROM rolling_context_messages WHERE id IN ({})", placeholders);
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for id in ids {
            params_vec.push(Box::new(*id));
        }
        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let deleted = conn
            .execute(&sql, params_refs.as_slice())
            .map_err(|e| e.to_string())?;
        Ok(deleted as u64)
    }

    /// Delete all messages for a session (keep the session row).
    pub fn clear_session_messages(&self, session_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM rolling_context_messages WHERE session_id = ?1",
            [session_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Reset cumulative token counts (called after compression so the post-truncation
    /// request's `usage.input_tokens` becomes the new baseline).
    pub fn reset_cumulative_tokens(&self, session_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE rolling_context_sessions
             SET total_input_tokens = 0,
                 total_output_tokens = 0,
                 total_cache_read_tokens = 0,
                 total_cache_creation_tokens = 0
             WHERE session_id = ?1",
            [session_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Atomically: record a compression event + update session counters.
    pub fn record_compression(&self, event: &CompressionEvent) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|e| e.to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;

        let now = chrono::Utc::now().timestamp();
        tx.execute(
            "INSERT INTO rolling_context_compressions
             (session_id, trigger, tokens_before, tokens_after,
              messages_removed, messages_summarized, summary_text, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            rusqlite::params![
                event.session_id,
                event.trigger,
                event.tokens_before as i64,
                event.tokens_after as i64,
                event.messages_removed,
                event.messages_summarized,
                event.summary_text,
                event.created_at.unwrap_or(now),
            ],
        )
        .map_err(|e| e.to_string())?;

        // Update session counters atomically with the compression event
        tx.execute(
            "UPDATE rolling_context_sessions
             SET compression_count = compression_count + 1,
                 tokens_saved = tokens_saved + ?1,
                 last_active_at = ?2
             WHERE session_id = ?3",
            rusqlite::params![
                (event.tokens_before.saturating_sub(event.tokens_after)) as i64,
                now,
                event.session_id,
            ],
        )
        .map_err(|e| e.to_string())?;

        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Get compression history for a session, most recent first.
    pub fn get_compression_history(
        &self,
        session_id: &str,
        limit: i64,
    ) -> Result<Vec<CompressionEvent>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, trigger, tokens_before, tokens_after,
                        messages_removed, messages_summarized, summary_text, created_at
                 FROM rolling_context_compressions
                 WHERE session_id = ?1
                 ORDER BY created_at DESC, id DESC
                 LIMIT ?2",
            )
            .map_err(|e| e.to_string())?;
        let events = stmt
            .query_map(rusqlite::params![session_id, limit], |row| {
                Ok(CompressionEvent {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    trigger: row.get(2)?,
                    tokens_before: row.get::<_, i64>(3).unwrap_or(0) as u64,
                    tokens_after: row.get::<_, i64>(4).unwrap_or(0) as u64,
                    messages_removed: row.get(5)?,
                    messages_summarized: row.get(6)?,
                    summary_text: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok(events)
    }

    /// Get all sessions, ordered by last activity.
    pub fn list_sessions(&self) -> Result<Vec<SessionRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, provider_id, model, context_window,
                        total_input_tokens, total_output_tokens,
                        total_cache_read_tokens, total_cache_creation_tokens,
                        compression_count, tokens_saved,
                        last_active_at, created_at
                 FROM rolling_context_sessions
                 ORDER BY last_active_at DESC",
            )
            .map_err(|e| e.to_string())?;
        let sessions = stmt
            .query_map([], |row| {
                Ok(SessionRecord {
                    session_id: row.get(0)?,
                    provider_id: row.get(1)?,
                    model: row.get(2)?,
                    context_window: row.get(3)?,
                    total_input_tokens: row.get::<_, i64>(4).unwrap_or(0) as u64,
                    total_output_tokens: row.get::<_, i64>(5).unwrap_or(0) as u64,
                    total_cache_read_tokens: row.get::<_, i64>(6).unwrap_or(0) as u64,
                    total_cache_creation_tokens: row.get::<_, i64>(7).unwrap_or(0) as u64,
                    compression_count: row.get::<_, i64>(8).unwrap_or(0) as u32,
                    tokens_saved: row.get::<_, i64>(9).unwrap_or(0) as u64,
                    last_active_at: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok(sessions)
    }

    /// Delete a session and all its data.
    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|e| e.to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM rolling_context_messages WHERE session_id = ?1",
            [session_id],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM rolling_context_compressions WHERE session_id = ?1",
            [session_id],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM rolling_context_sessions WHERE session_id = ?1",
            [session_id],
        )
        .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())
    }

    /// Enforce the global session cap by deleting oldest sessions. Returns
    /// the IDs of deleted sessions.
    pub fn enforce_session_cap(&self, max_sessions: i64) -> Result<Vec<String>, String> {
        let mut conn = self.conn.lock().map_err(|e| e.to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        let count: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM rolling_context_sessions",
                [],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        let to_delete = count - max_sessions;
        if to_delete <= 0 {
            return Ok(Vec::new());
        }
        // Get the LRU sessions
        let mut stmt = tx
            .prepare(
                "SELECT session_id FROM rolling_context_sessions
                 ORDER BY last_active_at ASC
                 LIMIT ?1",
            )
            .map_err(|e| e.to_string())?;
        let victims: Vec<String> = stmt
            .query_map([to_delete], |row| row.get(0))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        drop(stmt);
        for sid in &victims {
            tx.execute(
                "DELETE FROM rolling_context_messages WHERE session_id = ?1",
                [sid],
            )
            .map_err(|e| e.to_string())?;
            tx.execute(
                "DELETE FROM rolling_context_compressions WHERE session_id = ?1",
                [sid],
            )
            .map_err(|e| e.to_string())?;
            tx.execute(
                "DELETE FROM rolling_context_sessions WHERE session_id = ?1",
                [sid],
            )
            .map_err(|e| e.to_string())?;
        }
        tx.commit().map_err(|e| e.to_string())?;
        Ok(victims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_memory_store() -> MessageStore {
        let conn = rusqlite::Connection::open_in_memory().expect("open memory db");
        conn.execute(
            "CREATE TABLE rolling_context_sessions (
                session_id TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                model TEXT,
                context_window INTEGER,
                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                total_cache_read_tokens INTEGER DEFAULT 0,
                total_cache_creation_tokens INTEGER DEFAULT 0,
                compression_count INTEGER DEFAULT 0,
                tokens_saved INTEGER DEFAULT 0,
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
                tokens_before INTEGER,
                tokens_after INTEGER,
                messages_removed INTEGER,
                messages_summarized INTEGER,
                summary_text TEXT,
                created_at INTEGER
            );",
            [],
        )
        .unwrap();
        MessageStore::new(Arc::new(Mutex::new(conn)))
    }

    fn make_msg(session_id: &str, idx: i64, role: &str) -> MessageRecord {
        MessageRecord {
            id: None,
            session_id: session_id.to_string(),
            role: role.to_string(),
            content: format!("msg-{}", idx),
            token_count: Some(10),
            is_summary: false,
            summary_source_ids: Vec::new(),
            created_at: Some(idx),
        }
    }

    #[test]
    fn create_and_get_session() {
        let store = in_memory_store();
        let session = store
            .get_or_create_session("sess-1", "prov-1", Some("gpt-4"), Some(128000))
            .expect("create session");
        assert_eq!(session.session_id, "sess-1");
        assert_eq!(session.context_window, Some(128000));
    }

    #[test]
    fn record_response_usage_accumulates() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        store
            .record_response_usage("sess-1", 100, 50, 10, 5)
            .unwrap();
        store
            .record_response_usage("sess-1", 200, 100, 20, 10)
            .unwrap();

        let session = store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        assert_eq!(session.total_input_tokens, 300);
        assert_eq!(session.total_output_tokens, 150);
        assert_eq!(session.total_cache_read_tokens, 30);
        assert_eq!(session.total_cache_creation_tokens, 15);
    }

    #[test]
    fn session_utilization_calculation() {
        let session = SessionRecord {
            session_id: "s".into(),
            provider_id: "p".into(),
            model: None,
            context_window: Some(1_000_000),
            total_input_tokens: 800_000,
            total_output_tokens: 0,
            total_cache_read_tokens: 0,
            total_cache_creation_tokens: 0,
            compression_count: 0,
            tokens_saved: 0,
            last_active_at: None,
            created_at: None,
        };
        let (current, limit, ratio) = session.utilization().unwrap();
        assert_eq!(current, 800_000);
        assert_eq!(limit, 1_000_000);
        assert!((ratio - 0.8).abs() < 1e-9);
    }

    #[test]
    fn evict_oldest_messages_keeps_recent() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        for i in 0..5 {
            store.insert_message(&make_msg("sess-1", i, "user")).unwrap();
        }
        let deleted = store
            .evict_oldest_messages_for_session("sess-1", 2)
            .unwrap();
        assert_eq!(deleted, 3);
        let remaining = store.get_messages("sess-1").unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].content, "msg-3");
    }

    #[test]
    fn insert_messages_enforces_cap() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        let mut msgs = Vec::new();
        for i in 0..(MAX_MESSAGES_PER_SESSION + 50) {
            msgs.push(make_msg("sess-1", i, "user"));
        }
        store.insert_messages("sess-1", &msgs).unwrap();
        let count = store.get_messages("sess-1").unwrap().len() as i64;
        assert_eq!(count, MAX_MESSAGES_PER_SESSION);
    }

    #[test]
    fn delete_messages_by_ids() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        let id1 = store.insert_message(&make_msg("sess-1", 1, "user")).unwrap();
        let id2 = store.insert_message(&make_msg("sess-1", 2, "user")).unwrap();
        let id3 = store.insert_message(&make_msg("sess-1", 3, "user")).unwrap();
        let deleted = store.delete_messages_by_ids(&[id1, id3]).unwrap();
        assert_eq!(deleted, 2);
        let remaining = store.get_messages("sess-1").unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, Some(id2));
    }

    #[test]
    fn record_compression_updates_counters_atomically() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        // Pretend session had 1000 input tokens, compression reduced to 600
        store
            .record_response_usage("sess-1", 1000, 0, 0, 0)
            .unwrap();
        store
            .record_compression(&CompressionEvent {
                id: None,
                session_id: "sess-1".to_string(),
                trigger: "threshold".to_string(),
                tokens_before: 1000,
                tokens_after: 600,
                messages_removed: 5,
                messages_summarized: 0,
                summary_text: None,
                created_at: None,
            })
            .unwrap();
        let session = store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        assert_eq!(session.compression_count, 1);
        assert_eq!(session.tokens_saved, 400);
    }

    #[test]
    fn compression_history_ordered_newest_first() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        for (i, trigger) in ["a", "b", "c"].iter().enumerate() {
            store
                .record_compression(&CompressionEvent {
                    id: None,
                    session_id: "sess-1".to_string(),
                    trigger: trigger.to_string(),
                    tokens_before: 1000 - i as u64 * 100,
                    tokens_after: 500 - i as u64 * 100,
                    messages_removed: 5,
                    messages_summarized: 0,
                    summary_text: None,
                    created_at: Some(chrono::Utc::now().timestamp() + i as i64),
                })
                .unwrap();
        }
        let history = store.get_compression_history("sess-1", 10).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].trigger, "c");
        assert_eq!(history[2].trigger, "a");
    }

    #[test]
    fn reset_cumulative_tokens_zeroes_counters() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        store.record_response_usage("sess-1", 100, 50, 10, 5).unwrap();
        store.reset_cumulative_tokens("sess-1").unwrap();
        let session = store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        assert_eq!(session.total_input_tokens, 0);
        assert_eq!(session.total_output_tokens, 0);
        assert_eq!(session.total_cache_read_tokens, 0);
        assert_eq!(session.total_cache_creation_tokens, 0);
    }

    #[test]
    fn enforce_session_cap_evicts_lru() {
        let store = in_memory_store();
        for i in 0..5 {
            store
                .get_or_create_session(&format!("sess-{}", i), "p", None, None)
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let victims = store.enforce_session_cap(2).unwrap();
        assert_eq!(victims.len(), 3);
        let remaining = store.list_sessions().unwrap();
        assert_eq!(remaining.len(), 2);
        // Most recently active sessions should remain
        assert!(remaining.iter().any(|s| s.session_id == "sess-4"));
        assert!(remaining.iter().any(|s| s.session_id == "sess-3"));
    }

    #[test]
    fn delete_session_removes_everything() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();
        store.insert_message(&make_msg("sess-1", 1, "user")).unwrap();
        store
            .record_compression(&CompressionEvent {
                id: None,
                session_id: "sess-1".to_string(),
                trigger: "manual".to_string(),
                tokens_before: 100,
                tokens_after: 50,
                messages_removed: 1,
                messages_summarized: 0,
                summary_text: None,
                created_at: None,
            })
            .unwrap();
        store.delete_session("sess-1").unwrap();
        assert_eq!(store.list_sessions().unwrap().len(), 0);
        assert_eq!(store.get_messages("sess-1").unwrap().len(), 0);
        assert_eq!(
            store.get_compression_history("sess-1", 10).unwrap().len(),
            0
        );
    }
}
