use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Result, ToolCall, ToolSpec};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Host-owned durable transcript identity; never sent to the model endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_id: Option<String>,
    /// Host-only compiled memory snapshot. Providers must never serialize this as language text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiled_memory: Option<serde_json::Value>,
    pub role: MessageRole,
    pub content: String,
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub reasoning: Option<String>,
}

impl Message {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            record_id: None,
            compiled_memory: None,
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }

    pub fn with_compiled_memory(mut self, snapshot: serde_json::Value) -> Self {
        self.compiled_memory = Some(snapshot);
        self
    }

    pub fn assistant(
        content: impl Into<String>,
        reasoning: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            record_id: None,
            compiled_memory: None,
            role: MessageRole::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls,
            reasoning,
        }
    }

    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            record_id: None,
            compiled_memory: None,
            role: MessageRole::Tool,
            content: content.into(),
            tool_call_id: Some(call_id.into()),
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelProgress {
    pub content: String,
    pub reasoning: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CompletionRequest {
    pub progress: Option<tokio::sync::watch::Sender<ModelProgress>>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub allow_tool_calls: bool,
    /// Per-request sampling override selected by host policy.
    pub temperature: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenUsage {
    pub prompt: u64,
    pub completion: u64,
}

impl TokenUsage {
    pub fn add(&mut self, other: Self) {
        self.prompt = self.prompt.saturating_add(other.prompt);
        self.completion = self.completion.saturating_add(other.completion);
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelTurn {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelTurn>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compiled_memory_is_backward_compatible_and_survives_transcript_roundtrips() {
        let old = json!({"role":"System","content":"old","tool_call_id":null});
        let message: Message = serde_json::from_value(old).unwrap();
        assert_eq!(message.compiled_memory, None);
        assert!(
            serde_json::to_value(message)
                .unwrap()
                .get("compiled_memory")
                .is_none()
        );
        let message = Message::new(MessageRole::System, "<spine-compiled-memory-context/>")
            .with_compiled_memory(json!({"schema":1,"sources":[{"event_id":"event-a"}]}));
        let restored: Message =
            serde_json::from_slice(&serde_json::to_vec(&message).unwrap()).unwrap();
        assert_eq!(restored, message);
        assert_eq!(restored.clone().compiled_memory, message.compiled_memory);
        assert_eq!(
            Message::assistant("answer", None, vec![]).compiled_memory,
            None
        );
        assert_eq!(Message::tool("call", "result").compiled_memory, None);
    }
}
