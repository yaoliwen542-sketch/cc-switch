//! LLM-based summarization for rolling context.
//!
//! When cumulative tokens exceed the threshold, instead of simple truncation,
//! this module calls the provider's own API to generate a semantic summary
//! of the conversation history. This preserves context continuity much better
//! than template-based summaries.

use serde_json::{json, Value};

/// Prompt template for summarizing conversation history.
/// The {conversation} placeholder will be replaced with the actual messages.
const SUMMARIZE_PROMPT: &str = r#"You are a conversation summarizer. Summarize the following conversation history concisely, preserving:
1. Key decisions and their rationale
2. Files created/modified and their purposes
3. Current task progress and next steps
4. Any important context that would be needed to continue the conversation

Focus on actionable information, not pleasantries or redundant details. Keep the summary under 500 tokens.

Conversation to summarize:
{conversation}"#;

/// Maximum number of tokens to send for summarization (avoid overwhelming the model).
const MAX_SUMMARY_INPUT_TOKENS: usize = 50_000;

/// Result of LLM summarization.
#[derive(Debug, Clone)]
pub struct SummaryResult {
    /// The generated summary text.
    pub summary: String,
    /// Number of messages that were summarized.
    pub messages_summarized: usize,
    /// Estimated tokens in the summary.
    pub summary_tokens: usize,
}

/// Summarize a list of messages using the provider's own API.
///
/// # Arguments
/// * `messages` - The messages to summarize (typically the old/non-preserved ones)
/// * `provider_endpoint` - The provider's API endpoint (e.g., "https://api.openai.com/v1/chat/completions")
/// * `api_key` - The provider's API key
/// * `model` - The model to use for summarization (typically the same model or a cheaper one)
///
/// # Returns
/// `Ok(SummaryResult)` on success, `Err(String)` on failure.
pub async fn summarize_messages(
    messages: &[Value],
    provider_endpoint: &str,
    api_key: &str,
    model: &str,
) -> Result<SummaryResult, String> {
    if messages.is_empty() {
        return Err("No messages to summarize".to_string());
    }

    // Build conversation text from messages, respecting token limit
    let conversation = build_conversation_text(messages, MAX_SUMMARY_INPUT_TOKENS);

    // Build the summarization prompt
    let prompt = SUMMARIZE_PROMPT.replace("{conversation}", &conversation);

    // Build the request body (OpenAI-compatible format)
    let request_body = json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.3,
    });

    log::info!(
        "[RollingContext] Calling LLM for summarization: endpoint={}, model={}, messages={}",
        provider_endpoint,
        model,
        messages.len()
    );

    // Make the API call
    let client = crate::proxy::http_client::get();
    let response = client
        .post(provider_endpoint)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Failed to send summarization request: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Summarization API error {}: {}", status, body));
    }

    let response_json: Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse summarization response: {}", e))?;

    // Extract the summary from the response
    let summary = extract_summary_from_response(&response_json)?;

    Ok(SummaryResult {
        summary: summary.clone(),
        messages_summarized: messages.len(),
        summary_tokens: estimate_tokens(&summary),
    })
}

/// Build conversation text from messages, respecting token limit.
fn build_conversation_text(messages: &[Value], max_tokens: usize) -> String {
    let mut parts = Vec::new();
    let mut estimated_tokens = 0;

    // Start from the end to preserve recent context
    for msg in messages.iter().rev() {
        let role = msg
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");
        let content = extract_content_text(msg);

        if content.is_empty() {
            continue;
        }

        let entry = format!("{}: {}", role, content);
        let entry_tokens = estimate_tokens(&entry);

        if estimated_tokens + entry_tokens > max_tokens {
            // Add truncation marker if we have room for a few more tokens
            if estimated_tokens + 20 <= max_tokens {
                parts.push("... (earlier messages truncated)".to_string());
            }
            break;
        }

        estimated_tokens += entry_tokens;
        parts.push(entry);
    }

    // Reverse to restore chronological order
    parts.reverse();
    parts.join("\n\n")
}

/// Extract summary text from the API response (OpenAI-compatible format).
fn extract_summary_from_response(response: &Value) -> Result<String, String> {
    // Try OpenAI format first
    if let Some(choices) = response.get("choices").and_then(|c| c.as_array()) {
        if let Some(choice) = choices.first() {
            if let Some(message) = choice.get("message") {
                if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                    return Ok(content.to_string());
                }
            }
        }
    }

    // Try Claude format
    if let Some(content) = response.get("content").and_then(|c| c.as_array()) {
        if let Some(block) = content.first() {
            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                return Ok(text.to_string());
            }
        }
    }

    Err(format!(
        "Could not extract summary from response: {}",
        response
    ))
}

/// Extract text content from a message.
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

/// Rough token estimate (4 chars ≈ 1 token).
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_conversation_text_respects_limit() {
        let messages = vec![
            json!({"role": "user", "content": "a".repeat(1000)}),
            json!({"role": "assistant", "content": "b".repeat(1000)}),
            json!({"role": "user", "content": "c".repeat(1000)}),
        ];

        let text = build_conversation_text(&messages, 500);
        // Should truncate to fit within ~500 tokens
        assert!(estimate_tokens(&text) <= 600); // Some slack for formatting
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }
}
