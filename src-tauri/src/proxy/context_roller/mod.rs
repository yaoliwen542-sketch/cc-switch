//! Rolling Context module for cc-switch proxy.
//!
//! Provides automatic context window management by tracking per-session
//! message history and truncating older messages when approaching limits.

pub mod message_store;
