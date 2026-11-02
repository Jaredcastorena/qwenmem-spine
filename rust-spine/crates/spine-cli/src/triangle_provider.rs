//! Full Spine provider: Rust owns memory, tools and durable transcript; Python owns tensors.
use crate::triangle_backend::Backend;
use async_trait::async_trait;
use clap::Args;
use serde_json::{Value, json};
use spine_runtime::{
    CompletionRequest, MessageRole, ModelProvider, ModelTurn, RuntimeError, TokenUsage, ToolCall,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use tokio::sync::Mutex;

#[derive(Args, Clone, Default)]
pub struct Options {
    #[arg(long, conflicts_with_all = ["llamacpp", "server_url", "api_key", "server_model"],
        help = "Local triangle backend executable for the normal terminal/web agent")]
    pub triangle_backend: Option<PathBuf>,
    #[arg(long, requires = "triangle_backend", allow_hyphen_values = true)]
    pub triangle_backend_arg: Vec<OsString>,
    #[arg(
        long,
        requires = "triangle_backend",
        help = "Use native triangles for automatic memory context; explicit memory/fact tools remain available"
    )]
    pub triangle_context_only: bool,
}

pub struct TriangleProvider {
    options: Options,
    backend: Mutex<Option<Backend>>,
    context_tokens: usize,
    timeout: Duration,
    max_tokens: u16,
    next_turn: AtomicU64,
}
fn error(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Provider(error.to_string())
}
impl TriangleProvider {
    pub fn new(
        options: Options,
        timeout: Duration,
        max_tokens: i64,
        maximum_context_tokens: Option<usize>,
    ) -> spine_runtime::Result<Self> {
        if !(1..=2048).contains(&max_tokens) {
            return Err(error("Triangle chat requires --max-tokens in 1..2048"));
        }
        let context_tokens = maximum_context_tokens.unwrap_or(32_768).min(32_768);
        if context_tokens == 0 {
            return Err(error("Triangle context budget must be positive"));
        }
        let max_tokens = spine_runtime::bounded_completion_tokens(max_tokens, Some(context_tokens));
        Ok(Self {
            options,
            backend: Mutex::new(None),
            context_tokens,
            timeout,
            max_tokens: max_tokens as u16,
            next_turn: AtomicU64::new(0),
        })
    }
}
#[async_trait]
impl ModelProvider for TriangleProvider {
    async fn complete(&self, request: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        // Taking ownership makes future cancellation kill the in-flight child. No stale
        // unread protocol response can be reused; the next request spawns and reinstalls.
        let mut slot = self.backend.lock().await;
        let mut backend = match slot.take() {
            Some(backend) => backend,
            None => Backend::spawn(
                self.options.triangle_backend.as_ref().unwrap(),
                &self.options.triangle_backend_arg,
                self.timeout,
            )
            .map_err(error)?,
        };
        let hello = backend.call("hello", json!({})).await.map_err(error)?;
        if !hello["capabilities"]
            .as_array()
            .is_some_and(|c| c.iter().any(|v| v == "complete"))
        {
            return Err(error("Triangle backend does not support full chat"));
        }
        // The harness owns all memory selection and tool-mediated rehydration.
        // Normal chat has no private heart handle and installs no backing forest.
        // Compiled navigation remains available through `spine triangle`.
        let installed = backend
            .call(
                "install",
                json!({"forest":{
                    "schema":1,"nodes":{},"triangles":{},"roots":[]
                }}),
            )
            .await
            .map_err(error)?;
        let allowed: BTreeMap<String, Value> = if request.allow_tool_calls {
            request
                .tools
                .iter()
                .map(|t| (t.name.clone(), t.parameters.clone()))
                .collect()
        } else {
            BTreeMap::new()
        };
        let tools: Vec<Value> = request
            .tools
            .iter()
            .filter(|t| allowed.contains_key(&t.name))
            .map(|t| {
                json!({"type":"function","function":{"name":t.name,
                "description":t.description,"parameters":t.parameters}})
            })
            .collect();
        let supplied_messages = request.messages.len();
        let prepared = spine_runtime::prepare_messages(
            request.messages,
            Some(self.context_tokens),
            2048,
            i64::from(self.max_tokens),
            spine_runtime::estimated_tool_tokens(&tools),
        );
        let messages: Vec<Value> = prepared
            .iter()
            .map(|m| {
                let role = match m.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };
                let mut value = json!({"role":role,"content":m.content});
                if let Some(id) = &m.tool_call_id {
                    value["tool_call_id"] = json!(id);
                }
                if !m.tool_calls.is_empty() {
                    value["tool_calls"] = json!(
                        m.tool_calls
                            .iter()
                            .map(|t| json!({"id":t.id,
                    "type":"function","function":{"name":t.name,"arguments":t.arguments}}))
                            .collect::<Vec<_>>()
                    );
                }
                value
            })
            .collect();
        let result = backend
            .call(
                "complete",
                json!({"forest_sha256":installed["forest_sha256"],
            "messages":messages,"tools":tools,"max_tokens":self.max_tokens,
            "temperature":request.temperature.unwrap_or(0.0)}),
            )
            .await
            .map_err(error)?;
        let turn_number = self.next_turn.fetch_add(1, Ordering::Relaxed);
        let turn = decode_turn(&result, &allowed, turn_number, self.max_tokens)?;
        eprintln!(
            "{}",
            json!({"type":"triangle_chat","retrieval":"native_harness",
            "provider_added_evidence":false,"resident_records":0,
            "supplied_messages":supplied_messages,"prepared_messages":prepared.len(),
            "context_limit":self.context_tokens,
            "forest_sha256":installed["forest_sha256"],"weights_sha256":result["weights_sha256"],
            "prompt_tokens":turn.usage.prompt,"answer_tokens":turn.usage.completion,
            "tool_calls":turn.tool_calls.iter().map(|c|&c.name).collect::<Vec<_>>(),
            "navigation_unchanged":result["navigation_unchanged"],"sampling":result["sampling"],
            "gpu_reserved_bytes":result["gpu_reserved_bytes"]})
        );
        if let Some(progress) = request.progress {
            let _ = progress.send(spine_runtime::ModelProgress {
                content: turn.content.clone(),
                reasoning: None,
            });
        }
        *slot = Some(backend);
        Ok(turn)
    }
}

fn decode_turn(
    result: &Value,
    allowed: &BTreeMap<String, Value>,
    turn: u64,
    max_tokens: u16,
) -> spine_runtime::Result<ModelTurn> {
    if result["truncated"].as_bool() != Some(false)
        || result["navigation_unchanged"].as_bool() != Some(true)
    {
        return Err(error(
            "Incomplete or invalid triangle output; no assistant completion accepted",
        ));
    }
    let prompt = result["prompt_tokens"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 32768)
        .ok_or_else(|| error("Invalid prompt token count"))?;
    let completion = result["answer_tokens"]
        .as_u64()
        .filter(|n| *n <= u64::from(max_tokens))
        .ok_or_else(|| error("Invalid completion token count"))?;
    let mut rest = result["answer"]
        .as_str()
        .ok_or_else(|| error("Missing answer text"))?;
    let mut content = String::new();
    let mut calls = Vec::new();
    while let Some((before, after)) = rest.split_once("<tool_call>") {
        content.push_str(before);
        let (body, tail) = after
            .split_once("</tool_call>")
            .ok_or_else(|| error("Unclosed tool call"))?;
        let call = parse_call(body.trim(), allowed)?;
        let name = call["name"]
            .as_str()
            .filter(|name| allowed.contains_key(*name))
            .ok_or_else(|| error("Model requested an unavailable tool"))?;
        if !call["arguments"].is_object() {
            return Err(error("Tool arguments must be a JSON object"));
        }
        calls.push(ToolCall {
            id: format!("triangle-{turn}-{}", calls.len()),
            name: name.into(),
            arguments: call["arguments"].clone(),
        });
        rest = tail;
    }
    if rest.contains("</tool_call>") || rest.contains("<tool_call") {
        return Err(error("Malformed tool framing"));
    }
    content.push_str(rest);
    if content.trim().is_empty() && calls.is_empty() {
        return Err(error("Empty model turn"));
    }
    Ok(ModelTurn {
        content: content.trim().into(),
        reasoning: None,
        tool_calls: calls,
        usage: TokenUsage { prompt, completion },
    })
}

/// Match the pinned Qwen3.5 template, including string-valued numeric text and
/// multiline content. This is bounded tag framing, not a general XML interpreter.
fn parse_call(body: &str, schemas: &BTreeMap<String, Value>) -> spine_runtime::Result<Value> {
    if body.starts_with('{') {
        return serde_json::from_str(body).map_err(error);
    }
    let body = body
        .strip_prefix("<function=")
        .ok_or_else(|| error("Expected Qwen function tag"))?;
    let (name, body) = body
        .split_once('>')
        .ok_or_else(|| error("Unclosed function tag"))?;
    let schema = schemas
        .get(name)
        .ok_or_else(|| error("Model requested an unavailable tool"))?;
    let mut rest = body
        .trim_end()
        .strip_suffix("</function>")
        .ok_or_else(|| error("Unclosed function body"))?;
    let mut arguments = serde_json::Map::new();
    while !rest.trim().is_empty() {
        let body = rest
            .trim_start()
            .strip_prefix("<parameter=")
            .ok_or_else(|| error("Expected parameter tag"))?;
        let (key, body) = body
            .split_once('>')
            .ok_or_else(|| error("Unclosed parameter tag"))?;
        let (value, tail) = body
            .split_once("</parameter>")
            .ok_or_else(|| error("Unclosed parameter body"))?;
        if arguments.contains_key(key) {
            return Err(error("Duplicate tool parameter"));
        }
        let property = schema["properties"]
            .get(key)
            .ok_or_else(|| error("Unknown tool parameter"))?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let value = if property["type"] == "string" {
            json!(value)
        } else if property["type"] == "boolean" {
            // The pinned Jinja template serializes scalar booleans as True/False.
            match value.trim().to_ascii_lowercase().as_str() {
                "true" => json!(true),
                "false" => json!(false),
                _ => return Err(error("Invalid boolean tool parameter")),
            }
        } else {
            serde_json::from_str(value.trim())
                .map_err(|e| error(format!("Invalid {key} parameter: {e}")))?
        };
        arguments.insert(key.into(), value);
        rest = tail;
    }
    if let Some(required) = schema["required"].as_array()
        && required
            .iter()
            .filter_map(Value::as_str)
            .any(|k| !arguments.contains_key(k))
    {
        return Err(error("Missing required tool parameter"));
    }
    Ok(json!({"name":name,"arguments":arguments}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn native_harness_context_is_forwarded_without_private_retrieval() {
        use spine_runtime::{Message, ToolCategory, ToolRisk, ToolSpec};
        let temp = tempfile::tempdir().unwrap();
        let script = temp.path().join("backend.py");
        let capture = temp.path().join("requests.jsonl");
        std::fs::write(
            &script,
            r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    with open(sys.argv[1], 'a') as capture:
        capture.write(json.dumps(request) + '\n')
    op = request['op']
    if op == 'hello':
        result = {'capabilities':['complete']}
    elif op == 'install':
        assert request['args']['forest'] == {'schema':1,'nodes':{},'triangles':{},'roots':[]}
        result = {'forest_sha256':'0'*64,'addresses':[]}
    elif op == 'complete':
        result = {'answer':'ok','prompt_tokens':64,'answer_tokens':1,
                  'truncated':False,'navigation_unchanged':True}
    else:
        raise AssertionError('Unexpected provider-owned retrieval operation: ' + op)
    print(json.dumps({'schema':1,'id':request['id'],'ok':True,'result':result}), flush=True)
"#,
        )
        .unwrap();
        let provider = TriangleProvider::new(
            Options {
                triangle_backend: Some("python3".into()),
                triangle_backend_arg: vec![
                    script.into_os_string(),
                    capture.clone().into_os_string(),
                ],
                ..Options::default()
            },
            Duration::from_secs(10),
            64,
            Some(32_768),
        )
        .unwrap();
        let mut system = Message::new(MessageRole::System, "Native triangle context: event:only");
        system.record_id = Some("host-only-record-id".into());
        let turn = provider.complete(CompletionRequest {
            progress: None,
            messages: vec![system, Message::new(MessageRole::User, "current task"),
                Message::assistant("", None, vec![ToolCall {
                    id:"read-1".into(),name:"heart_recall".into(),arguments:json!({"query":"older source"})
                }]), Message::tool("read-1", "native tool evidence")],
            tools: vec![ToolSpec { name:"heart_recall".into(), description:"native recall".into(),
                category:ToolCategory::Internal,risk:ToolRisk::ReadOnly,
                parameters:json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}) }],
            allow_tool_calls: true, temperature: Some(0.1),
        }).await.unwrap();
        assert_eq!(turn.content, "ok");
        let requests: Vec<Value> = std::fs::read_to_string(capture)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            requests
                .iter()
                .map(|r| r["op"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["hello", "install", "complete"]
        );
        let args = &requests[2]["args"];
        assert_eq!(
            args["messages"],
            json!([
                {"role":"system","content":"Native triangle context: event:only"},
                {"role":"user","content":"current task"},
                {"role":"assistant","content":"","tool_calls":[{"id":"read-1","type":"function",
                    "function":{"name":"heart_recall","arguments":{"query":"older source"}}}]},
                {"role":"tool","content":"native tool evidence","tool_call_id":"read-1"}
            ])
        );
        assert_eq!(args["tools"][0]["function"]["name"], "heart_recall");
        assert_eq!(args["max_tokens"], 64);
    }

    fn result(answer: &str) -> Value {
        json!({"answer":answer,"prompt_tokens":20,
        "answer_tokens":15,"truncated":false,"navigation_unchanged":true})
    }
    #[test]
    fn triangle_qwen_xml_preserves_typed_parameters_and_rejects_ambiguous_calls() {
        let schemas = BTreeMap::from([(
            "file_write".into(),
            json!({"properties":{
            "path":{"type":"string"},"content":{"type":"string"},"offset":{"type":"integer"}},
            "required":["path","content"]}),
        )]);
        let body = "<function=file_write>\n<parameter=path>\n007\n</parameter>\n<parameter=content>\nhello\nworld\n\n</parameter>\n<parameter=offset>\n2\n</parameter>\n</function>";
        let call = parse_call(body, &schemas).unwrap();
        assert_eq!(
            call["arguments"],
            json!({"path":"007","content":"hello\nworld\n","offset":2})
        );
        assert!(
            parse_call(
                &body.replace("<parameter=offset>", "<parameter=path>"),
                &schemas
            )
            .is_err()
        );
        assert!(
            parse_call(
                "<function=file_write><parameter=path>x</parameter></function>",
                &schemas
            )
            .is_err()
        );
    }
    #[test]
    fn triangle_qwen_template_boolean_is_not_json_boolean_spelling() {
        let schemas = BTreeMap::from([(
            "fact_search".into(),
            json!({"properties":{
            "include_superseded":{"type":"boolean"}}}),
        )]);
        let call = parse_call(
            "<function=fact_search><parameter=include_superseded>\nFalse\n</parameter></function>",
            &schemas,
        )
        .unwrap();
        assert_eq!(call["arguments"]["include_superseded"], json!(false));
    }
    #[test]
    fn triangle_chat_validates_tool_permission_and_partial_outputs() {
        let allowed = BTreeMap::from([(
            "file_read".into(),
            json!({"properties":{"path":{"type":"string"}},"required":["path"]}),
        )]);
        let value = result(
            "<tool_call>\n{\"name\":\"file_read\",\"arguments\":{\"path\":\"fixture.txt\"}}\n</tool_call>",
        );
        let turn = decode_turn(&value, &allowed, 7, 32).unwrap();
        assert_eq!(turn.tool_calls[0].id, "triangle-7-0");
        assert!(decode_turn(&value, &BTreeMap::new(), 7, 32).is_err());
        let mut partial = value.clone();
        partial["truncated"] = json!(true);
        assert!(decode_turn(&partial, &allowed, 7, 32).is_err());
        assert!(decode_turn(&result("<tool_call>{}"), &allowed, 7, 32).is_err());
    }
}
