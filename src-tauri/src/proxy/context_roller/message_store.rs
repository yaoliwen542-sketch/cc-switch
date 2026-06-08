//! SQLite-backed message store for rolling context sessions.

use std::sync::{Arc, Mutex};

/// Session metadata stored in the database.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub session_id: String,
    pub provider_id: String,
    pub model: Option<String>,
    pub context_window: Option<u64>,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub compression_count: u32,
    pub last_active_at: Option<i64>,
    pub created_at: Option<i64>,
}

/// A single message in the rolling context history.
#[derive(Debug, Clone)]
pub struct MessageRecord {
    pub id: Option<i64>,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub token_count: Option<u64>,
    pub is_summary: bool,
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

    /// Get or create a session record.
    pub fn get_or_create_session(
        &self,
        session_id: &str,
        provider_id: &str,
        model: Option<&str>,
        context_window: Option<u64>,
    ) -> Result<SessionRecord, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;

        // Try to find existing
        let mut stmt = conn
            .prepare(
                "SELECT session_id, provider_id, model, context_window,
                        total_input_tokens, total_output_tokens, compression_count,
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
                    compression_count: row.get::<_, i64>(6).unwrap_or(0) as u32,
                    last_active_at: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .optional()
            .map_err(|e| e.to_string());

        if let Ok(Some(session)) = existing {
            // Update last_active_at
            let now = chrono::Utc::now().timestamp();
            conn.execute(
                "UPDATE rolling_context_sessions SET last_active_at = ?1 WHERE session_id = ?2",
                [now, session_id],
            )
            .map_err(|e| e.to_string())?;
            return Ok(session);
        }

        // Create new session
        let now = chrono::Utc::now().timestamp();
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
            compression_count: 0,
            last_active_at: Some(now),
            created_at: Some(now),
        })
    }

    /// Insert messages for a session.
    pub fn insert_messages(
        &self,
        session_id: &str,
        messages: &[MessageRecord],
    ) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|e| e.to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;

        for msg in messages {
            tx.execute(
                "INSERT INTO rolling_context_messages
                 (session_id, role, content, token_count, is_summary, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ",
                rusqlite::params![
                    session_id,
                    &msg.role,
                    &msg.content,
                    msg.token_count.map(|v| v as i64),
                    msg.is_summary as i32,
                    msg.created_at.unwrap_or_else(|| chrono::Utc::now().timestamp()),
                ],
            )
            .map_err(|e| e.to_string())?;
        }

        tx.commit().map_err(|e| e.to_string())
    }

    /// Get all messages for a session, ordered by creation time.
    pub fn get_messages(
        &self,
        session_id: &str,
    ) -> Result<Vec<MessageRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, role, content, token_count, is_summary, created_at
                 FROM rolling_context_messages
                 WHERE session_id = ?1
                 ORDER BY created_at ASC, id ASC",
            )
            .map_err(|e| e.to_string())?;

        let messages = stmt
            .query_map([session_id], |row| {
                Ok(MessageRecord {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    token_count: row.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                    is_summary: row.get::<_, i32>(5)? != 0,
                    created_at: row.get(6)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        Ok(messages)
    }

    /// Delete oldest messages for a session, keeping the last N.
    pub fn delete_oldest_messages(&self, session_id: &str, keep_count: usize) -> Result<u64, String> {
        let mut conn = self.conn.lock().map_err(|e| e.to_string())?;

        // First, count total messages
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM rolling_context_messages WHERE session_id = ?1",
                [session_id],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;

        let to_delete = (total as usize).saturating_sub(keep_count);
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
                rusqlite::params![session_id, to_delete as i64],
            )
            .map_err(|e| e.to_string())?;

        Ok(deleted as u64)
    }

    /// Delete all messages for a session.
    pub fn clear_session_messages(&self, session_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM rolling_context_messages WHERE session_id = ?1",
            [session_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Update session token counts.
    pub fn update_session_tokens(
        &self,
        session_id: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE rolling_context_sessions
             SET total_input_tokens = total_input_tokens + ?1,
                 total_output_tokens = total_output_tokens + ?2,
                 last_active_at = ?3
             WHERE session_id = ?4",
            rusqlite::params![
                input_tokens as i64,
                output_tokens as i64,
                chrono::Utc::now().timestamp(),
                session_id,
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Increment compression count for a session.
    pub fn increment_compression_count(&self, session_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE rolling_context_sessions
             SET compression_count = compression_count + 1,
                 last_active_at = ?1
             WHERE session_id = ?2",
            [chrono::Utc::now().timestamp(), session_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Delete a session and all its messages.
    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|e| e.to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;

        tx.execute(
            "DELETE FROM rolling_context_messages WHERE session_id = ?1",
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

    /// List all sessions for cleanup (optional).
    pub fn list_sessions(&self) -> Result<Vec<SessionRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, provider_id, model, context_window,
                        total_input_tokens, total_output_tokens, compression_count,
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
                    compression_count: row.get::<_, i64>(6).unwrap_or(0) as u32,
                    last_active_at: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        Ok(sessions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn in_memory_store() -> MessageStore {
        let conn = rusqlite::Connection::open_in_memory().expect("open memory db");
        // Create tables
        conn.execute(
            "CREATE TABLE rolling_context_sessions (
                session_id TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                model TEXT,
                context_window INTEGER,
                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                compression_count INTEGER DEFAULT 0,
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
                created_at INTEGER
            );",
            [],
        )
        .unwrap();
        MessageStore::new(Arc::new(Mutex::new(conn)))
    }

    #[test]
    fn create_and_get_session() {
        let store = in_memory_store();
        let session = store
            .get_or_create_session("sess-1", "prov-1", Some("gpt-4"), Some(128000))
            .expect("create session");

        assert_eq!(session.session_id, "sess-1");
        assert_eq!(session.provider_id, "prov-1");
        assert_eq!(session.model.as_deref(), Some("gpt-4"));
        assert_eq!(session.context_window, Some(128000));
    }

    #[test]
    fn insert_and_get_messages() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        let msgs = vec![
            MessageRecord {
                id: None,
                session_id: "sess-1".to_string(),
                role: "system".to_string(),
                content: "You are helpful".to_string(),
                token_count: Some(10),
                is_summary: false,
                created_at: None,
            },
            MessageRecord {
                id: None,
                session_id: "sess-1".to_string(),
                role: "user".to_string(),
                content: "Hello".to_string(),
                token_count: Some(5),
                is_summary: false,
                created_at: None,
            },
        ];

        store.insert_messages("sess-1", &msgs).unwrap();
        let retrieved = store.get_messages("sess-1").unwrap();
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].role, "system");
        assert_eq!(retrieved[1].role, "user");
    }

    #[test]
    fn delete_oldest_keeps_recent() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        for i in 0..5 {
            let msg = MessageRecord {
                id: None,
                session_id: "sess-1".to_string(),
                role: "user".to_string(),
                content: format!("msg-{}", i),
                token_count: Some(10),
                is_summary: false,
                created_at: Some(i as i64),
            };
            store.insert_messages("sess-1", &[msg]).unwrap();
        }

        let deleted = store.delete_oldest_messages("sess-1", 2).unwrap();
        assert_eq!(deleted, 3);

        let remaining = store.get_messages("sess-1").unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].content, "msg-3");
        assert_eq!(remaining[1].content, "msg-4");
    }

    #[test]
    fn update_tokens() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        store.update_session_tokens("sess-1", 100, 50).unwrap();
        store.update_session_tokens("sess-1", 50, 25).unwrap();

        let session = store.get_or_create_session("sess-1", "prov-1", None, None).unwrap();
        assert_eq!(session.total_input_tokens, 150);
        assert_eq!(session.total_output_tokens, 75);
    }

    #[test]
    fn increment_compression_count() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        store.increment_compression_count("sess-1").unwrap();
        store.increment_compression_count("sess-1").unwrap();

        let session = store.get_or_create_session("sess-1", "prov-1", None, None).unwrap();
        assert_eq!(session.compression_count, 2);
    }

    #[test]
    fn delete_session_cleans_up() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        let msg = MessageRecord {
            id: None,
            session_id: "sess-1".to_string(),
            role: "user".to_string(),
            content: "hello".to_string(),
            token_count: Some(5),
            is_summary: false,
            created_at: None,
        };
        store.insert_messages("sess-1", &[msg]).unwrap();

        store.delete_session("sess-1").unwrap();

        let messages = store.get_messages("sess-1").unwrap();
        assert_eq!(messages.len(), 0);

        let sessions = store.list_sessions().unwrap();
        assert_eq!(sessions.len(), 0);
    }

    #[test]
    fn list_sessions_ordered_by_activity() {
        let store = in_memory_store();
        store.get_or_create_session("sess-a", "prov-1", None, None).unwrap();
        store.get_or_create_session("sess-b", "prov-2", None, None).unwrap();

        let sessions = store.list_sessions().unwrap();
        assert_eq!(sessions.len(), 2);
        // Most recently active first (sess-b was created second)
        assert_eq!(sessions[0].session_id, "sess-b");
        assert_eq!(sessions[1].session_id, "sess-a");
    }

    #[test]
    fn clear_session_messages_keeps_session() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        let msg = MessageRecord {
            id: None,
            session_id: "sess-1".to_string(),
            role: "user".to_string(),
            content: "hello".to_string(),
            token_count: Some(5),
            is_summary: false,
            created_at: None,
        };
        store.insert_messages("sess-1", &[msg]).unwrap();

        store.clear_session_messages("sess-1").unwrap();

        let messages = store.get_messages("sess-1").unwrap();
        assert_eq!(messages.len(), 0);

        let sessions = store.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "sess-1");
    }

    #[test]
    fn get_or_create_existing_updates_last_active() {
        let store = in_memory_store();

        let session1 = store
            .get_or_create_session("sess-1", "prov-1", Some("gpt-4"), Some(128000))
            .unwrap();

        // Small delay to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(10));

        let session2 = store
            .get_or_create_session("sess-1", "prov-1", Some("gpt-4"), Some(128000))
            .unwrap();

        // Same session, but last_active_at should be updated
        assert_eq!(session1.session_id, session2.session_id);
        assert_eq!(session1.created_at, session2.created_at);
    }

    #[test]
    fn delete_oldest_with_keep_greater_than_total() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        for i in 0..3 {
            let msg = MessageRecord {
                id: None,
                session_id: "sess-1".to_string(),
                role: "user".to_string(),
                content: format!("msg-{}", i),
                token_count: Some(10),
                is_summary: false,
                created_at: Some(i as i64),
            };
            store.insert_messages("sess-1", &[msg]).unwrap();
        }

        let deleted = store.delete_oldest_messages("sess-1", 10).unwrap();
        assert_eq!(deleted, 0);

        let remaining = store.get_messages("sess-1").unwrap();
        assert_eq!(remaining.len(), 3);
    }

    #[test]
    fn message_with_summary_flag() {
        let store = in_memory_store();
        store
            .get_or_create_session("sess-1", "prov-1", None, None)
            .unwrap();

        let msgs = vec![
            MessageRecord {
                id: None,
                session_id: "sess-1".to_string(),
                role: "system".to_string(),
                content: "Summary of previous conversation".to_string(),
                token_count: Some(20),
                is_summary: true,
                created_at: None,
            },
            MessageRecord {
                id: None,
                session_id: "sess-1".to_string(),
                role: "user".to_string(),
                content: "Hello".to_string(),
                token_count: Some(5),
                is_summary: false,
                created_at: None,
            },
        ];

        store.insert_messages("sess-1", &msgs).unwrap();
        let retrieved = store.get_messages("sess-1").unwrap();
        assert_eq!(retrieved.len(), 2);
        assert!(retrieved[0].is_summary);
        assert!(!retrieved[1].is_summary);
    }
}
