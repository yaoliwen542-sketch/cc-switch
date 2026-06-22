use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("服务器已在运行")]
    AlreadyRunning,

    #[error("服务器未运行")]
    NotRunning,

    #[error("地址绑定失败: {0}")]
    BindFailed(String),

    #[error("停止超时")]
    StopTimeout,

    #[error("停止失败: {0}")]
    StopFailed(String),

    #[error("请求转发失败: {0}")]
    ForwardFailed(String),

    #[error("无可用的Provider")]
    NoAvailableProvider,

    #[error("所有供应商已熔断，无可用渠道")]
    AllProvidersCircuitOpen,

    #[error("未配置供应商")]
    NoProvidersConfigured,

    #[allow(dead_code)]
    #[error("Provider不健康: {0}")]
    ProviderUnhealthy(String),

    #[error("上游错误 (状态码 {status}): {body:?}")]
    UpstreamError { status: u16, body: Option<String> },

    #[error("超过最大重试次数")]
    MaxRetriesExceeded,

    #[error("数据库错误: {0}")]
    DatabaseError(String),

    #[error("配置错误: {0}")]
    ConfigError(String),

    #[allow(dead_code)]
    #[error("格式转换错误: {0}")]
    TransformError(String),

    #[allow(dead_code)]
    #[error("无效的请求: {0}")]
    InvalidRequest(String),

    #[error("超时: {0}")]
    Timeout(String),

    /// 流式响应空闲超时
    #[allow(dead_code)]
    #[error("流式响应空闲超时: {0}秒无数据")]
    StreamIdleTimeout(u64),

    /// 认证错误
    #[error("认证失败: {0}")]
    AuthError(String),

    #[allow(dead_code)]
    #[error("内部错误: {0}")]
    Internal(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, body) = match &self {
            ProxyError::UpstreamError {
                status: upstream_status,
                body: upstream_body,
            } => {
                let http_status =
                    StatusCode::from_u16(*upstream_status).unwrap_or(StatusCode::BAD_GATEWAY);

                // 尝试解析上游响应体为 JSON，如果失败则包装为字符串
                let error_body = if let Some(body_str) = upstream_body {
                    if let Ok(json_body) = serde_json::from_str::<serde_json::Value>(body_str) {
                        // 上游返回的是 JSON，直接透传
                        json_body
                    } else {
                        // 上游返回的不是 JSON，包装为错误消息
                        json!({
                            "error": {
                                "message": body_str,
                                "type": "upstream_error",
                            }
                        })
                    }
                } else {
                    json!({
                        "error": {
                            "message": format!("Upstream error (status {})", upstream_status),
                            "type": "upstream_error",
                        }
                    })
                };

                (http_status, error_body)
            }
            _ => {
                let (http_status, message) = match &self {
                    ProxyError::AlreadyRunning => (StatusCode::CONFLICT, self.to_string()),
                    ProxyError::NotRunning => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
                    ProxyError::BindFailed(_) => {
                        (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
                    }
                    ProxyError::StopTimeout => {
                        (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
                    }
                    ProxyError::StopFailed(_) => {
                        (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
                    }
                    ProxyError::ForwardFailed(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
                    ProxyError::NoAvailableProvider => {
                        (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
                    }
                    ProxyError::AllProvidersCircuitOpen => {
                        (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
                    }
                    ProxyError::NoProvidersConfigured => {
                        (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
                    }
                    ProxyError::ProviderUnhealthy(_) => {
                        (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
                    }
                    ProxyError::MaxRetriesExceeded => {
                        (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
                    }
                    ProxyError::DatabaseError(_) => {
                        (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
                    }
                    ProxyError::ConfigError(_) => (StatusCode::BAD_REQUEST, self.to_string()),
                    ProxyError::TransformError(_) => {
                        (StatusCode::UNPROCESSABLE_ENTITY, self.to_string())
                    }
                    ProxyError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
                    ProxyError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, self.to_string()),
                    ProxyError::StreamIdleTimeout(_) => {
                        (StatusCode::GATEWAY_TIMEOUT, self.to_string())
                    }
                    ProxyError::AuthError(_) => (StatusCode::UNAUTHORIZED, self.to_string()),
                    ProxyError::Internal(_) => {
                        (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
                    }
                    ProxyError::UpstreamError { .. } => unreachable!(),
                };

                let error_body = json!({
                    "error": {
                        "message": message,
                        "type": "proxy_error",
                    }
                });

                (http_status, error_body)
            }
        };

        (status, Json(body)).into_response()
    }
}

/// 错误分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// 可重试错误（网络问题、5xx）
    Retryable, // 网络超时、5xx 错误
    /// 不可重试错误（4xx、认证失败）
    NonRetryable, // 认证失败、参数错误、4xx 错误
    #[allow(dead_code)]
    ClientAbort, // 客户端主动中断
}

/// 判断错误是否可重试
#[allow(dead_code)]
pub fn categorize_error(error: &reqwest::Error) -> ErrorCategory {
    if error.is_timeout() || error.is_connect() {
        return ErrorCategory::Retryable;
    }

    if let Some(status) = error.status() {
        if status.is_server_error() {
            ErrorCategory::Retryable
        } else if status.is_client_error() {
            ErrorCategory::NonRetryable
        } else {
            ErrorCategory::Retryable
        }
    } else {
        ErrorCategory::Retryable
    }
}

/// 常见模型供应商的"上下文/Token 长度超限"错误文案片段。
///
/// 命中后会把对应的 400/413 错误重新归类为可故障转移的错误，
/// 因为这意味着当前供应商的模型窗口不够大，换一家配置了更大窗口或
/// 不同模型的供应商可以成功响应。
const CONTEXT_LENGTH_OVERFLOW_PATTERNS: &[&str] = &[
    // 通用 / 英文（你提供的实际错误）
    "exceeded model token limit",
    "exceeds the maximum context length",
    "exceeds the maximum number of tokens",
    "exceeds the maximum token limit",
    "exceeds the model token limit",
    // OpenAI / OpenAI 兼容 (Chat Completions + Responses + Codex)
    "context_length_exceeded",
    "maximum context length",
    "maximum number of tokens",
    "string too long",
    "prompt is too long",
    "input is too long",
    "too many input tokens",
    "too many tokens",
    "context length exceeded",
    "reduce the length",
    // Anthropic / Claude 兼容
    "prompt is too long for",
    "input is too long for",
    "too long for requested model",
    "request_too_large",
    // Google Gemini
    "exceeds the maximum",
    "context length",
    "RESOURCE_EXHAUSTED",
    // Azure OpenAI / Azure AI Inference
    "max_tokens",
    "context_length",
    // 常见中转/反代（透传上游 + 中文化）
    "上下文长度超出限制",
    "上下文超过最大",
    "上下文超出",
    "超过模型最大",
    "超过最大 token",
    "超过上下文",
    "上下文窗口超出",
    "提示过长",
    "输入过长",
    "请求过长",
    // 部分反代/三方服务透传的中文短句
    "请求超限",
    "token 超限",
    "上下文超限",
    // 兼容 HuggingFace TGI / vLLM / Ollama / LM Studio 等本地服务
    "context length exceeded",
    "max sequence length",
    "maximum sequence length",
    "input tokens exceed",
    "context window exceeded",
    "context size exceeded",
];

/// 判断上游错误是否是"上下文/Token 长度超限"类错误。
///
/// 命中后：
/// - `categorize_proxy_error` 会把 400/413/414 重新归类为 `Retryable`，
///   让故障转移继续尝试下一个供应商。
/// - 转发循环会用 `release_permit_neutral` 释放被超限供应商的熔断器名额，
///   不计入它的健康度（容量不足 ≠ 供应商本身不健康）。
pub fn is_context_length_overflow(status: u16, body: Option<&str>) -> bool {
    // 上下文长度超限一般出现在 400 / 413 / 414；其他状态码（5xx、401 等）不看文案。
    if !matches!(status, 400 | 413 | 414) {
        return false;
    }
    let Some(body) = body else {
        return false;
    };
    let lower = body.to_ascii_lowercase();
    CONTEXT_LENGTH_OVERFLOW_PATTERNS
        .iter()
        .any(|p| lower.contains(&p.to_ascii_lowercase()))
}

#[cfg(test)]
mod overflow_tests {
    use super::*;

    #[test]
    fn detects_user_reported_anthropic_compatible_error() {
        let body = r#"{"error":{"message":"Your request exceeded model token limit: 262144 (requested: 263669)","type":"invalid_request_error"}}"#;
        assert!(is_context_length_overflow(400, Some(body)));
    }

    #[test]
    fn detects_openai_context_length_exceeded_code() {
        let body = r#"{"error":{"message":"This model's maximum context length is 8192 tokens. However, you requested 12000 tokens.","type":"invalid_request_error","code":"context_length_exceeded","param":"messages"}}"#;
        assert!(is_context_length_overflow(400, Some(body)));
    }

    #[test]
    fn detects_anthropic_prompt_too_long() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 215006 tokens > 200000 maximum"}}"#;
        assert!(is_context_length_overflow(400, Some(body)));
    }

    #[test]
    fn detects_gemini_resource_exhausted() {
        // Gemini 的 429 用的是 RESOURCE_EXHAUSTED + "exceeds the maximum"。
        // 429 已经被归为 Retryable，文案匹配只是兜底；
        // 我们的辅助函数只认 400/413/414，所以这里用 400 模拟文案场景。
        let body = r#"{"error":{"code":400,"message":"The request exceeds the maximum number of tokens (200000).","status":"INVALID_ARGUMENT"}}"#;
        assert!(is_context_length_overflow(400, Some(body)));
    }

    #[test]
    fn detects_chinese_relay_text() {
        let body = r#"{"error":{"message":"请求失败：上下文长度超出限制 200000 tokens (requested 263669)","type":"invalid_request_error"}}"#;
        assert!(is_context_length_overflow(400, Some(body)));
    }

    #[test]
    fn detects_ollama_context_length_exceeded() {
        let body = r#"{"error":"model \"qwen2\" has a context length of 32768 tokens, but the prompt is 40000 tokens long. Try reducing the length of the prompt."}"#;
        // Ollama 通常返回 400 + "context length" 文案。
        assert!(is_context_length_overflow(400, Some(body)));
    }

    #[test]
    fn ignores_unrelated_400() {
        let body = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error"}}"#;
        assert!(!is_context_length_overflow(400, Some(body)));
    }

    #[test]
    fn ignores_missing_body() {
        assert!(!is_context_length_overflow(400, None));
    }

    #[test]
    fn ignores_non_overflow_status() {
        // 500 服务端错误，即使带 token 文本也不归为 overflow（已经是 Retryable）。
        let body = "internal error: context length check failed";
        assert!(!is_context_length_overflow(500, Some(body)));
    }
}
