//! Tool-Call Orphan Rectifier
//!
//! Reactive recovery for upstream 400 errors caused by mismatched tool calls:
//!
//!   * "an assistant message with 'tool_calls' must be followed by tool messages
//!      responding to each 'tool_call_id'. The following tool_call_ids did not
//!      have response messages: ..."
//!   * "tool result's tool id(...) not found (2013)"
//!   * "tool call result does not follow tool call (2013)"
//!   * "tool_call_id is not found"
//!
//! When the universal pre-flight sanitizer misses an edge case (e.g. a
//! client-side corrupted session, or a format-conversion corner case), this
//! rectifier parses the offending ids from the upstream error body and strips
//! the corresponding tool_calls / tool_use blocks before retrying the same
//! provider.  This gives users a recovery path for `/compact` and normal chat
//! requests instead of leaving the session permanently broken.

use serde_json::Value;

/// Result of a tool-call rectification pass.
#[derive(Debug, Clone, Default)]
pub struct ToolCallRectifyResult {
    /// Whether any repair was applied to the body.
    pub applied: bool,
    /// Number of assistant messages that lost tool_calls / tool_use blocks.
    pub stripped_assistants: usize,
    /// Number of OpenAI `role: "tool"` messages dropped.
    pub dropped_tool_messages: usize,
    /// Number of Anthropic `tool_result` content blocks converted to text.
    pub converted_tool_results: usize,
    /// Ids that were targeted for removal.
    pub removed_ids: Vec<String>,
}

/// Returns true when the upstream error message indicates a tool-call/tool-result
/// pairing problem that we can attempt to repair.
pub fn should_rectify_tool_call_orphan(error_message: Option<&str>) -> bool {
    let Some(msg) = error_message else {
        return false;
    };
    let lower = msg.to_ascii_lowercase();

    // OpenAI / Azure / relay variants.
    if lower.contains("tool_call_ids did not have response messages") {
        return true;
    }
    if lower.contains("tool_call_id is not found") {
        return true;
    }
    if lower.contains("must be followed by tool messages") && lower.contains("tool_call_id") {
        return true;
    }

    // Anthropic / Claude-compatible variants.
    if lower.contains("tool result's tool id") && lower.contains("not found") {
        return true;
    }
    if lower.contains("tool result") && lower.contains("not found") && lower.contains("2013") {
        return true;
    }
    if lower.contains("tool call result does not follow tool call") {
        return true;
    }
    if lower.contains("tool result does not follow tool use") {
        return true;
    }

    false
}

/// Parse the literal tool_call ids listed in an OpenAI-style error message.
///
/// Example input:
///   "... The following tool_call_ids did not have response messages: id1, id2"
/// Returns ["id1", "id2"].
fn parse_openai_orphan_ids(error_message: &str) -> Vec<String> {
    let lower = error_message.to_ascii_lowercase();
    let marker = "the following tool_call_ids did not have response messages:";
    let Some(idx) = lower.find(marker) else {
        return Vec::new();
    };
    let tail = &error_message[idx + marker.len()..];
    // Stop at the first sentence boundary.
    let end = tail
        .find(|c: char| c == '.' || c == '\n')
        .unwrap_or(tail.len());
    let list = &tail[..end];
    list.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse a single tool id from an Anthropic-style error message.
///
/// Example inputs:
///   "tool result's tool id(call_abc123) not found (2013)"
///   "tool result's tool id(mcp__plugin:2) not found"
fn parse_anthropic_orphan_id(error_message: &str) -> Option<String> {
    // Find the first occurrence of "tool id(" and extract the matching ')'
    let lower = error_message.to_ascii_lowercase();
    let marker = "tool id(";
    let idx = lower.find(marker)?;
    let start = idx + marker.len();
    let rest = &error_message[start..];
    let end = rest.find(')')?;
    let id = &rest[..end];
    if id.trim().is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

/// Collect all offending ids we can extract from the upstream error message.
fn extract_orphan_ids(error_message: &str) -> Vec<String> {
    let mut ids = parse_openai_orphan_ids(error_message);
    if let Some(id) = parse_anthropic_orphan_id(error_message) {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

/// Repair the request body by removing references to the given tool ids.
///
/// Handles both OpenAI format (`assistant.tool_calls[]` + `role: "tool"`) and
/// Anthropic format (`assistant.content[].tool_use` + `user.content[].tool_result`).
pub fn rectify_tool_call_orphans(body: &mut Value, ids: &[String]) -> ToolCallRectifyResult {
    let mut result = ToolCallRectifyResult::default();
    if ids.is_empty() {
        return result;
    }
    result.removed_ids = ids.to_vec();

    let id_set: std::collections::HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();

    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(m) => m,
        None => return result,
    };

    for msg in messages.iter_mut() {
        let is_assistant = msg.get("role").and_then(|r| r.as_str()) == Some("assistant");
        let is_tool = msg.get("role").and_then(|r| r.as_str()) == Some("tool");

        // OpenAI: strip tool_calls entries.
        if is_assistant {
            if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                let before = tool_calls.len();
                tool_calls.retain(|tc| {
                    tc.get("id")
                        .and_then(|id| id.as_str())
                        .map_or(true, |id| !id_set.contains(id))
                });
                if tool_calls.len() != before {
                    result.applied = true;
                    result.stripped_assistants += 1;
                }
                if tool_calls.is_empty() {
                    if let Some(obj) = msg.as_object_mut() {
                        obj.remove("tool_calls");
                    }
                }
            }
        }

        // OpenAI: drop tool messages whose tool_call_id is in the id set.
        if is_tool {
            if let Some(tid) = msg.get("tool_call_id").and_then(|id| id.as_str()) {
                if id_set.contains(tid) {
                    result.applied = true;
                    result.dropped_tool_messages += 1;
                    // Mark for removal by clearing role; we'll filter below.
                    if let Some(obj) = msg.as_object_mut() {
                        obj.insert("role".to_string(), Value::Null);
                    }
                }
            }
        }

        // Anthropic: strip tool_use content blocks from assistant messages and
        // convert orphan tool_result blocks to text in user messages.
        if let Some(content) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
            let mut modified = false;
            for block in content.iter_mut() {
                let block_type = block.get("type").and_then(|t| t.as_str());
                match block_type {
                    Some("tool_use") => {
                        if let Some(id) = block.get("id").and_then(|id| id.as_str()) {
                            if id_set.contains(id) {
                                // Convert to a text placeholder so the message
                                // structure remains valid; pass3-like cleanup can
                                // drop it later if it becomes empty.
                                *block = serde_json::json!({
                                    "type": "text",
                                    "text": format!("[removed orphan tool_use: {id}]")
                                });
                                modified = true;
                                result.stripped_assistants += 1;
                            }
                        }
                    }
                    Some("tool_result") => {
                        if let Some(tuid) = block.get("tool_use_id").and_then(|id| id.as_str()) {
                            if id_set.contains(tuid) {
                                let content_text = match block.get("content") {
                                    Some(Value::String(text)) => text.clone(),
                                    Some(Value::Array(blocks)) => blocks
                                        .iter()
                                        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                                        .collect::<Vec<_>>()
                                        .join("\n"),
                                    _ => String::new(),
                                };
                                *block = serde_json::json!({
                                    "type": "text",
                                    "text": format!("[Tool result for {tuid}]: {content_text}")
                                });
                                modified = true;
                                result.converted_tool_results += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
            if modified {
                result.applied = true;
            }
        }
    }

    // Remove marked OpenAI tool messages.
    messages.retain(|msg| msg.get("role") != Some(&Value::Null));

    result
}

/// Convenience entry used by the forwarder: detects whether the error is a
/// tool-call orphan error and, if so, repairs the body in place.
pub fn rectify_if_needed(
    body: &mut Value,
    error_message: Option<&str>,
) -> Option<ToolCallRectifyResult> {
    if !should_rectify_tool_call_orphan(error_message) {
        return None;
    }
    let ids = error_message.map(extract_orphan_ids).unwrap_or_default();
    if ids.is_empty() {
        // Error matched but we couldn't extract ids (e.g. generic 2013).
        // Fall back to stripping any assistant whose tool_calls are not all
        // covered by immediately following tool messages.
        let extra = strip_uncovered_tool_calls(body);
        if extra.applied {
            return Some(extra);
        }
        return None;
    }
    Some(rectify_tool_call_orphans(body, &ids))
}

/// Fallback repair when the upstream error does not name specific ids.
///
/// For every assistant with tool_calls / tool_use blocks, verify that the
/// immediately following messages cover every id.  If not, strip the uncovered
/// entries.  This mirrors the universal sanitizer's pass2 but runs reactively
/// on the already-failed body.
fn strip_uncovered_tool_calls(body: &mut Value) -> ToolCallRectifyResult {
    let mut result = ToolCallRectifyResult::default();
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(m) => m,
        None => return result,
    };

    let len = messages.len();
    for i in 0..len {
        if messages[i].get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }

        // Collect ids of immediately following OpenAI tool messages.
        let mut next_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut k = i + 1;
        while k < len && messages[k].get("role").and_then(|r| r.as_str()) == Some("tool") {
            if let Some(id) = messages[k]
                .get("tool_call_id")
                .and_then(|id| id.as_str())
                .filter(|id| !id.is_empty())
            {
                next_ids.insert(id.to_string());
            }
            k += 1;
        }

        // OpenAI tool_calls.
        if let Some(tool_calls) = messages[i].get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
            let before = tool_calls.len();
            tool_calls.retain(|tc| {
                tc.get("id")
                    .and_then(|id| id.as_str())
                    .map_or(false, |id| next_ids.contains(id))
            });
            if tool_calls.len() != before {
                result.applied = true;
                result.stripped_assistants += 1;
            }
            if tool_calls.is_empty() {
                if let Some(obj) = messages[i].as_object_mut() {
                    obj.remove("tool_calls");
                }
            }
        }

        // Anthropic tool_use blocks.
        if let Some(content) = messages[i].get_mut("content").and_then(|c| c.as_array_mut()) {
            let before = content.len();
            content.retain(|b| {
                if b.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                    b.get("id")
                        .and_then(|id| id.as_str())
                        .map_or(false, |id| next_ids.contains(id))
                } else {
                    true
                }
            });
            if content.len() != before {
                result.applied = true;
                result.stripped_assistants += 1;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_openai_orphan_error() {
        let msg = "an assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: id1, id2";
        assert!(should_rectify_tool_call_orphan(Some(msg)));
        let ids = extract_orphan_ids(msg);
        assert_eq!(ids, vec!["id1".to_string(), "id2".to_string()]);
    }

    #[test]
    fn detects_anthropic_orphan_error() {
        let msg = "API Error: 400 invalid params, tool result's tool id(call_abc123) not found (2013)";
        assert!(should_rectify_tool_call_orphan(Some(msg)));
        let ids = extract_orphan_ids(msg);
        assert_eq!(ids, vec!["call_abc123".to_string()]);
    }

    #[test]
    fn ignores_unrelated_400() {
        let msg = "Invalid API key";
        assert!(!should_rectify_tool_call_orphan(Some(msg)));
    }

    #[test]
    fn strips_openai_orphan_tool_calls() {
        let mut body = json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "good", "type": "function", "function": {"name": "ok", "arguments": "{}"}},
                    {"id": "bad", "type": "function", "function": {"name": "missing", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "good", "content": "ok result"}
            ]
        });
        let result = rectify_tool_call_orphans(&mut body, &["bad".to_string()]);
        assert!(result.applied);
        let tool_calls = body["messages"][0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "good");
    }

    #[test]
    fn converts_anthropic_orphan_tool_result_to_text() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "orphan", "content": "result"},
                    {"type": "text", "text": "hello"}
                ]}
            ]
        });
        let result = rectify_tool_call_orphans(&mut body, &["orphan".to_string()]);
        assert!(result.applied);
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert!(content[0]["text"].as_str().unwrap().contains("orphan"));
        assert_eq!(content[1]["text"], "hello");
    }

    #[test]
    fn fallback_strips_uncovered_openai_tool_calls() {
        let mut body = json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}
                ]},
                {"role": "user", "content": "no tool result here"}
            ]
        });
        let result = strip_uncovered_tool_calls(&mut body);
        assert!(result.applied);
        assert!(body["messages"][0].get("tool_calls").is_none());
    }
}
