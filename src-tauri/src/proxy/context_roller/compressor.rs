//! Compression strategies for rolling context.
//!
//! Phase 1: Simple truncation (remove oldest messages).
//! Phase 2 (future): LLM-based summarization.

use serde_json::Value;

/// A compression strategy that reduces message history size.
pub trait CompressionStrategy: Send + Sync {
    /// Compress a list of messages, returning the compressed list.
    fn compress(&self, messages: &[Value], target_tokens: u64) -> Vec<Value>;
}

/// Phase 1: Simple truncation - just drop oldest messages.
pub struct TruncationCompressor;

impl CompressionStrategy for TruncationCompressor {
    fn compress(&self, messages: &[Value], _target_tokens: u64) -> Vec<Value> {
        // The actual truncation logic is in context_window.rs
        // This is a placeholder for future strategies
        messages.to_vec()
    }
}

/// Phase 2 stub: LLM-based summarization (not yet implemented).
pub struct SummaryCompressor;

impl CompressionStrategy for SummaryCompressor {
    fn compress(&self, messages: &[Value], _target_tokens: u64) -> Vec<Value> {
        // TODO: Call a cheap LLM to summarize older messages
        // For now, fall back to truncation
        messages.to_vec()
    }
}

/// Get the default compression strategy.
pub fn default_compressor() -> Box<dyn CompressionStrategy> {
    Box::new(TruncationCompressor)
}
