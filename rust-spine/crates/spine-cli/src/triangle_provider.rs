//! Full Spine provider: Rust owns memory, tools and durable transcript; Python owns tensors.
use crate::triangle_backend::Backend;
use async_trait::async_trait;
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use spine_runtime::{
    CompletionRequest, Message, MessageRole, ModelProvider, ModelTurn, RuntimeError, TokenUsage,
    ToolCall,
};
use std::{
    collections::{BTreeMap, BTreeSet},
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
    #[arg(
        long,
        requires = "triangle_backend",
        help = "Require compiled checkpoint navigation before expanding native automatic memory context"
    )]
    pub triangle_internal_context: bool,
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
        if options.triangle_backend.is_none() {
            return Err(error("Triangle chat requires --triangle-backend"));
        }
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
    async fn complete(&self, mut request: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        let packet = if self.options.triangle_internal_context {
            Some(request_packet(&request.messages)?)
        } else {
            None
        };
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
        // Reinstall this request's durable snapshot on every call, including tool,
        // repair and resumed requests. Never reuse a previous turn's authorization.
        let handoff = if let Some((system_index, packet)) = packet {
            let handoff = replay_packet(&mut backend, &packet).await?;
            let mut context = packet.context;
            context["rehydrated"] = json!(packet.sources);
            let expanded = serde_json::to_string(&context).map_err(error)?;
            request.messages[system_index].content = request.messages[system_index]
                .content
                .replace(CONTEXT_MARKER, &expanded);
            handoff
        } else {
            let installed = backend
                .call("install", json!({"forest":empty_forest()}))
                .await
                .map_err(error)?;
            Handoff {
                forest_sha256: installed["forest_sha256"].clone(),
                ..Handoff::default()
            }
        };
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
                json!({"forest_sha256":handoff.forest_sha256,
            "messages":messages,"tools":tools,"max_tokens":self.max_tokens,
            "temperature":request.temperature.unwrap_or(0.0)}),
            )
            .await
            .map_err(error)?;
        if self.options.triangle_internal_context {
            if result["forest_sha256"] != handoff.forest_sha256 {
                return Err(error(
                    "Completion forest digest differs from the verified handoff",
                ));
            }
            verify_execution(
                &result["execution"],
                0,
                handoff.final_records,
                handoff.final_triangles,
                handoff.last_execution.as_ref(),
            )?;
        }
        let turn_number = self.next_turn.fetch_add(1, Ordering::Relaxed);
        let turn = decode_turn(&result, &allowed, turn_number, self.max_tokens)?;
        eprintln!(
            "{}",
            json!({"type":"triangle_chat","retrieval":if self.options.triangle_internal_context {
                "compiled_native_handoff" } else { "native_harness" },
            "provider_added_evidence":false,"resident_records":handoff.resident_records,
            "resident_triangles":handoff.resident_triangles,"bank_count":handoff.banks,
            "branch_commands":handoff.branch_commands,"command_count":handoff.commands,
            "source_records":handoff.authorized_sources,
            "source_gate_verified":self.options.triangle_internal_context,
            "host_packet_sha256":handoff.host_packet_sha256,
            "source_gate":if self.options.triangle_internal_context { "verified" } else { "native_harness" },
            "memory_status":if handoff.resident_records == 0 { "no_memory" } else { "verified_navigation" },
            "supplied_messages":supplied_messages,"prepared_messages":prepared.len(),
            "context_limit":self.context_tokens,
            "forest_sha256":handoff.forest_sha256,"weights_sha256":result["weights_sha256"],
            "execution":result["execution"],
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

const CONTEXT_MARKER: &str = "<spine-compiled-memory-context/>";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompiledPacket {
    schema: u32,
    context: Value,
    sources: Vec<Value>,
    banks: Vec<CompiledBank>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompiledBank {
    forest: Value,
    commands: Vec<String>,
    coordinates: Vec<String>,
}

#[derive(Default)]
struct Handoff {
    forest_sha256: Value,
    host_packet_sha256: Option<String>,
    resident_records: usize,
    resident_triangles: usize,
    final_records: usize,
    final_triangles: usize,
    banks: usize,
    commands: usize,
    branch_commands: usize,
    authorized_sources: usize,
    last_execution: Option<Value>,
}

fn empty_forest() -> Value {
    json!({"schema":1,"nodes":{},"triangles":{},"roots":[]})
}

fn request_packet(messages: &[Message]) -> spine_runtime::Result<(usize, CompiledPacket)> {
    let packets: Vec<_> = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.compiled_memory.is_some())
        .collect();
    let marker_count: usize = messages
        .iter()
        .map(|message| message.content.matches(CONTEXT_MARKER).count())
        .sum();
    if packets.len() != 1 || marker_count != 1 {
        return Err(error(
            "Internal context requires exactly one system snapshot and one marker",
        ));
    }
    let (index, message) = packets[0];
    if message.role != MessageRole::System || !message.content.contains(CONTEXT_MARKER) {
        return Err(error(
            "Compiled memory snapshot and marker must belong to the same system message",
        ));
    }
    let packet: CompiledPacket =
        serde_json::from_value(message.compiled_memory.clone().unwrap()).map_err(error)?;
    if packet.schema != 1
        || !packet.context.is_object()
        || packet.sources.len() > 12
        || packet.banks.len() > 3
        || packet
            .context
            .get("rehydrated")
            .is_some_and(|value| value.as_array().is_none_or(|rows| !rows.is_empty()))
    {
        return Err(error(
            "Invalid compiled memory packet schema, context or native bounds",
        ));
    }
    let mut source_counts = BTreeMap::<&str, usize>::new();
    let mut source_ids = BTreeSet::new();
    for source in &packet.sources {
        let node = source["node_id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| error("Compiled source has no node identity"))?;
        let event = source["event_id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| error("Compiled source has no event provenance"))?;
        if source["text"].as_str().is_none_or(|text| {
            let chars = text.chars().count();
            // Preserve cognition_tools::bounded_text's native 1,000-character
            // excerpt plus its truncation marker, including Unicode boundaries.
            chars > 1000 && !(chars == 1012 && text.ends_with("\n[truncated]"))
        }) || !source_ids.insert((node, event))
        {
            return Err(error("Invalid or duplicate compiled source excerpt"));
        }
        let count = source_counts.entry(node).or_default();
        *count += 1;
        if *count > 2 {
            return Err(error(
                "Compiled source exceeds native per-node event budget",
            ));
        }
    }
    if !packet.sources.is_empty() && packet.banks.is_empty() {
        return Err(error("Compiled sources require navigation banks"));
    }
    for bank in &packet.banks {
        if bank.commands.is_empty()
            || bank.commands.len() > 128
            || bank.coordinates.is_empty()
            || bank.coordinates.len() > 16
            || bank.coordinates.iter().collect::<BTreeSet<_>>().len() != bank.coordinates.len()
            || bank.coordinates.iter().any(|a| !a.starts_with("node:"))
        {
            return Err(error("Invalid or unbounded compiled navigation walk"));
        }
        bank_records(&bank.forest)?;
    }
    Ok((index, packet))
}

#[derive(Clone)]
struct Record {
    apex: usize,
    children: Option<(usize, usize)>,
}

fn bank_records(forest: &Value) -> spine_runtime::Result<(Vec<String>, Vec<Record>, usize)> {
    let nodes = forest["nodes"]
        .as_object()
        .ok_or_else(|| error("Missing native bank nodes"))?;
    let triangles = forest["triangles"]
        .as_object()
        .ok_or_else(|| error("Missing native bank triangles"))?;
    let roots = forest["roots"]
        .as_array()
        .ok_or_else(|| error("Missing native bank roots"))?;
    if forest["schema"] != 1
        || nodes.is_empty()
        || nodes.len() + triangles.len() > 64
        || roots.len() != 1
        || nodes.keys().any(|a| !a.starts_with("node:"))
        || triangles.keys().any(|a| !a.starts_with("triangle:"))
        || nodes
            .values()
            .any(|node| node["text"] != "" || node["event_id"].as_str().is_none_or(str::is_empty))
    {
        return Err(error(
            "Invalid native bank identity, source isolation or resident bounds",
        ));
    }
    // Explicit sorting keeps the address contract independent of serde map configuration.
    let mut addresses: Vec<String> = nodes.keys().cloned().collect();
    addresses.sort();
    let mut branches: Vec<String> = triangles.keys().cloned().collect();
    branches.sort();
    addresses.extend(branches);
    let slots: BTreeMap<_, _> = addresses
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();
    let slot = |value: &Value| {
        value
            .as_str()
            .and_then(|s| slots.get(s).copied())
            .ok_or_else(|| error("Native bank has an unresolved address"))
    };
    let root = slot(&roots[0])?;
    let mut records = Vec::with_capacity(addresses.len());
    for (index, address) in addresses.iter().enumerate() {
        records.push(if nodes.contains_key(address) {
            Record {
                apex: index,
                children: None,
            }
        } else {
            let triangle = &triangles[address];
            let apex = slot(&triangle["apex"])?;
            let left = slot(&triangle["left"])?;
            let right = slot(&triangle["right"])?;
            if apex >= nodes.len()
                || left == right
                || triangle["depth"].as_u64().is_none_or(|depth| depth > 8)
            {
                return Err(error("Invalid native triangle apex, branches or depth"));
            }
            Record {
                apex,
                children: Some((left, right)),
            }
        });
    }
    fn visit(
        index: usize,
        depth: usize,
        records: &[Record],
        visiting: &mut BTreeSet<usize>,
        reached: &mut BTreeSet<usize>,
    ) -> spine_runtime::Result<()> {
        if depth > 8 || !visiting.insert(index) {
            return Err(error("Native bank contains a cycle or exceeds depth eight"));
        }
        reached.insert(index);
        reached.insert(records[index].apex);
        if let Some((left, right)) = records[index].children {
            visit(left, depth + 1, records, visiting, reached)?;
            visit(right, depth + 1, records, visiting, reached)?;
        }
        visiting.remove(&index);
        Ok(())
    }
    let mut reached = BTreeSet::new();
    visit(root, 0, &records, &mut BTreeSet::new(), &mut reached)?;
    if reached.len() != records.len() {
        return Err(error(
            "Native bank must contain exactly one complete root closure",
        ));
    }
    Ok((addresses, records, triangles.len()))
}

// This independent interpreter predicts protocol tokens only. It never selects or
// fetches source text, and actual checkpoint APEX emissions alone unlock evidence.
#[derive(Clone, Default)]
struct Navigation {
    cursor: Option<usize>,
    steps: usize,
    routes: BTreeMap<usize, usize>,
}

impl Navigation {
    fn expected(&mut self, command: &str, records: &[Record]) -> spine_runtime::Result<Vec<usize>> {
        let (token, mode) = match command {
            "peek" => (264, "peek"),
            "left" => (265, "left"),
            "right" => (266, "right"),
            _ => {
                let (mode, number) = command
                    .split_once(':')
                    .ok_or_else(|| error("Expected open:SLOT, peek, left, right or route:INDEX"))?;
                let number: usize = number.parse().map_err(error)?;
                match mode {
                    "open" if number < 64 => (number, "open"),
                    "route" if number < 8 => (256 + number, "route"),
                    _ => return Err(error("Navigation command outside compiled bounds")),
                }
            }
        };
        let mut trace = vec![token];
        match mode {
            "open" => {
                self.cursor = Some(token);
                self.steps = 0;
                self.routes.clear();
            }
            "route" => {
                let Some(cursor) = self.routes.get(&(token - 256)).copied() else {
                    trace.push(268);
                    return Ok(trace);
                };
                self.cursor = Some(cursor);
                trace.push(64 + cursor);
            }
            _ => {}
        }
        if mode == "left" || mode == "right" {
            self.steps += 1;
        }
        let Some(record) = self.cursor.and_then(|i| records.get(i)) else {
            trace.push(268);
            return Ok(trace);
        };
        if mode == "left" || mode == "right" {
            let Some((left, right)) = record.children else {
                trace.push(270);
                return Ok(trace);
            };
            if self.steps > 8 {
                trace.push(269);
                return Ok(trace);
            }
            let (next, alternate) = if mode == "left" {
                (left, right)
            } else {
                (right, left)
            };
            self.routes.insert(self.steps - 1, alternate);
            self.cursor = Some(next);
            trace.extend([128 + alternate, 64 + next, 192 + records[next].apex, 267]);
        } else {
            trace.extend([192 + record.apex, 267]);
        }
        Ok(trace)
    }
}

fn digest(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn verify_execution(
    execution: &Value,
    ticks: usize,
    records: usize,
    triangles: usize,
    previous: Option<&Value>,
) -> spine_runtime::Result<()> {
    let before = execution["ticks_before"].as_u64();
    let after = execution["ticks_after"].as_u64();
    if execution["engine"] != "merged_checkpoint_prefix"
        || execution["prefix_layers"] != 5
        || execution["resident_records"].as_u64() != Some(records as u64)
        || execution["resident_triangles"].as_u64() != Some(triangles as u64)
        || before
            .zip(after)
            .is_none_or(|(b, a)| a.checked_sub(b) != Some(ticks as u64))
        || !digest(&execution["state_before"])
        || !digest(&execution["state_after"])
        || (ticks == 0 && execution["state_before"] != execution["state_after"])
        || previous.is_some_and(|p| {
            p["ticks_after"] != execution["ticks_before"]
                || p["state_after"] != execution["state_before"]
        })
    {
        return Err(error(
            "Missing, inconsistent or non-checkpoint navigation execution attestation",
        ));
    }
    Ok(())
}

async fn replay_packet(
    backend: &mut Backend,
    packet: &CompiledPacket,
) -> spine_runtime::Result<Handoff> {
    let mut handoff = Handoff {
        host_packet_sha256: Some(hex::encode(Sha256::digest(
            serde_json::to_vec(packet).map_err(error)?,
        ))),
        ..Handoff::default()
    };
    let mut authorized = BTreeSet::new();
    if packet.banks.is_empty() {
        let installed = backend
            .call("install", json!({"forest":empty_forest()}))
            .await
            .map_err(error)?;
        if installed["addresses"] != json!([]) || !digest(&installed["forest_sha256"]) {
            return Err(error("Invalid empty native bank installation"));
        }
        handoff.forest_sha256 = installed["forest_sha256"].clone();
    }
    for (bank_index, bank) in packet.banks.iter().enumerate() {
        let (addresses, records, triangles) = bank_records(&bank.forest)?;
        let installed = backend
            .call("install", json!({"forest":bank.forest}))
            .await
            .map_err(error)?;
        if installed["addresses"] != json!(addresses) || !digest(&installed["forest_sha256"]) {
            return Err(error(
                "Backend address map or digest differs from native bank contract",
            ));
        }
        let forest_sha256 = installed["forest_sha256"].clone();
        let mut navigation = Navigation::default();
        let mut emitted = Vec::new();
        let mut traces = Vec::new();
        let mut executions = Vec::new();
        let mut last_execution = None;
        for command in &bank.commands {
            let expected = navigation.expected(command, &records)?;
            if expected.last() != Some(&267) {
                return Err(error(
                    "Native context walk does not terminate at a valid coordinate",
                ));
            }
            let result = backend
                .call(
                    "execute",
                    json!({"forest_sha256":forest_sha256,
                "command":command}),
                )
                .await
                .map_err(error)?;
            let trace: Vec<usize> =
                serde_json::from_value(result["trace"].clone()).map_err(error)?;
            if trace != expected || result["forest_sha256"] != forest_sha256 {
                return Err(error(format!(
                    "Compiled trace or forest mismatch for {command}: expected {expected:?}, received {trace:?}"
                )));
            }
            verify_execution(
                &result["execution"],
                trace.len() - 1,
                records.len(),
                triangles,
                last_execution.as_ref(),
            )?;
            for token in &trace {
                if (192..256).contains(token) {
                    let address = addresses
                        .get(token - 192)
                        .filter(|a| a.starts_with("node:"))
                        .ok_or_else(|| {
                            error("Checkpoint emitted an invalid APEX source address")
                        })?;
                    if !emitted.contains(address) {
                        emitted.push(address.clone());
                    }
                }
            }
            handoff.commands += 1;
            handoff.branch_commands += usize::from(command == "left" || command == "right");
            traces.push(trace);
            last_execution = Some(result["execution"].clone());
            executions.push(result["execution"].clone());
        }
        if emitted != bank.coordinates {
            return Err(error(
                "Actual checkpoint coordinates differ from native selected coordinate order",
            ));
        }
        authorized.extend(emitted.iter().cloned());
        eprintln!(
            "{}",
            json!({"type":"triangle_internal_bank","bank":bank_index,
            "resident_records":records.len(),"resident_triangles":triangles,
            "commands":bank.commands,"traces":traces,"emitted_coordinates":emitted,
            "forest_sha256":forest_sha256,
            "host_bank_sha256":hex::encode(Sha256::digest(serde_json::to_vec(&bank.forest).map_err(error)?)),
            "execution":executions,"source_gate":"trace_verified"})
        );
        handoff.forest_sha256 = forest_sha256;
        handoff.resident_records += records.len();
        handoff.resident_triangles += triangles;
        handoff.final_records = records.len();
        handoff.final_triangles = triangles;
        handoff.banks += 1;
        handoff.last_execution = last_execution;
    }
    if packet.sources.iter().any(|source| {
        !authorized.contains(&format!("node:{}", source["node_id"].as_str().unwrap()))
    }) {
        return Err(error(
            "Compiled source gate rejected a node not emitted by checkpoint navigation",
        ));
    }
    handoff.authorized_sources = packet.sources.len();
    Ok(handoff)
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

    fn compiled_packet() -> Value {
        let node =
            |id: &str| json!({"text":"","event_id":format!("event-{id}"),"vector":[1.0,0.0]});
        json!({"schema":1,"context":{"retained_goal":"native goal"},
            "sources":[{"node_id":"a","event_id":"event-a","text":"source A"},
                {"node_id":"b","event_id":"event-b","text":"source B"},
                {"node_id":"c","event_id":"event-c","text":"source C"}],
            "banks":[{"forest":{"schema":1,
                "nodes":{"node:a":node("a"),"node:b":node("b"),"node:c":node("c")},
                "triangles":{"triangle:root":{"apex":"node:a","left":"node:b","right":"node:c","depth":1}},
                "roots":["triangle:root"]},
                "commands":["open:3","left","route:0"],"coordinates":["node:a","node:b","node:c"]}]})
    }

    fn internal_request(packet: Value) -> CompletionRequest {
        CompletionRequest {
            progress: None,
            messages: vec![
                Message::new(
                    MessageRole::System,
                    format!("Native context: {CONTEXT_MARKER}"),
                )
                .with_compiled_memory(packet),
                Message::new(MessageRole::User, "current task"),
            ],
            tools: vec![],
            allow_tool_calls: false,
            temperature: None,
        }
    }

    #[test]
    fn internal_context_rejects_missing_ambiguous_and_preexpanded_snapshots() {
        assert!(request_packet(&[Message::new(MessageRole::System, CONTEXT_MARKER)]).is_err());
        let request = internal_request(compiled_packet());
        assert!(request_packet(&request.messages).is_ok());
        let mut duplicate = request.messages.clone();
        duplicate.push(duplicate[0].clone());
        assert!(request_packet(&duplicate).is_err());
        let mut wrong_role = request.messages.clone();
        wrong_role[0].role = MessageRole::User;
        assert!(request_packet(&wrong_role).is_err());
        let mut expanded = compiled_packet();
        expanded["context"]["rehydrated"] = expanded["sources"].clone();
        assert!(request_packet(&internal_request(expanded).messages).is_err());
        let mut source_in_forest = compiled_packet();
        source_in_forest["banks"][0]["forest"]["nodes"]["node:a"]["text"] =
            json!("bypass evidence");
        assert!(request_packet(&internal_request(source_in_forest).messages).is_err());
    }

    #[test]
    fn internal_context_preserves_native_truncated_unicode_excerpts() {
        let mut packet = compiled_packet();
        let excerpt = format!("{}\n[truncated]", "🦀".repeat(1000));
        packet["sources"][0]["text"] = json!(excerpt);
        let (_, parsed) = request_packet(&internal_request(packet.clone()).messages).unwrap();
        assert_eq!(parsed.sources[0]["text"], excerpt);
        packet["sources"][0]["text"] = json!("🦀".repeat(1001));
        assert!(request_packet(&internal_request(packet.clone()).messages).is_err());
        packet["sources"][0]["text"] = json!(format!("{}\n[truncated]", "🦀".repeat(1001)));
        assert!(request_packet(&internal_request(packet).messages).is_err());
    }

    // Deliberately fake transport fixture, not model-execution evidence. Production
    // attestation shape is exercised here; real prefix execution is a separate gate.
    #[cfg(unix)]
    fn internal_provider(temp: &std::path::Path, fault: &str) -> (TriangleProvider, PathBuf) {
        let script = temp.join("internal_backend.py");
        let capture = temp.join("requests.jsonl");
        std::fs::write(
            &script,
            r#"
import hashlib, json, sys
fault, capture = sys.argv[1:]
ticks = 0
state = '0' * 64
records = triangles = 0
def execution(before, before_state):
    return {'engine':'merged_checkpoint_prefix','prefix_layers':5,
            'ticks_before':before,'ticks_after':ticks,
            'state_before':before_state,'state_after':state,
            'resident_records':records,'resident_triangles':triangles}
for line in sys.stdin:
    request = json.loads(line)
    with open(capture, 'a') as out:
        out.write(json.dumps(request) + '\n')
    op, args = request['op'], request['args']
    if op == 'hello':
        result = {'capabilities':['complete','execute']}
    elif op == 'install':
        forest = args['forest']
        addresses = sorted(forest['nodes']) + sorted(forest['triangles'])
        records, triangles = len(addresses), len(forest['triangles'])
        digest = hashlib.sha256(json.dumps(forest, sort_keys=True).encode()).hexdigest()
        ticks += 1
        state = hashlib.sha256(str(ticks).encode()).hexdigest()
        if fault == 'addresses': addresses.reverse()
        result = {'forest_sha256':digest,'addresses':addresses}
    elif op == 'execute':
        trace = {'open:3':[3,192,267], 'left':[265,130,65,193,267],
                 'route:0':[256,66,194,267]}[args['command']]
        before, before_state = ticks, state
        ticks += len(trace)-1
        state = hashlib.sha256(str(ticks).encode()).hexdigest()
        if fault == 'trace': trace[-2] = 195
        result = {'forest_sha256':digest,'trace':trace,'execution':execution(before,before_state)}
        if fault == 'digest': result['forest_sha256'] = 'f'*64
        if fault == 'engine': result['execution']['engine'] = 'mock'
        if fault == 'ticks': result['execution']['ticks_after'] += 1
    elif op == 'complete':
        result = {'answer':'ok','prompt_tokens':64,'answer_tokens':1,
                  'truncated':False,'navigation_unchanged':True,'forest_sha256':digest,
                  'execution':execution(ticks,state)}
        if fault == 'complete_state': result['execution']['state_after'] = 'e'*64
    else: raise AssertionError(op)
    print(json.dumps({'schema':1,'id':request['id'],'ok':True,'result':result}), flush=True)
"#,
        )
        .unwrap();
        (
            TriangleProvider::new(
                Options {
                    triangle_backend: Some("python3".into()),
                    triangle_backend_arg: vec![
                        script.into_os_string(),
                        fault.into(),
                        capture.clone().into_os_string(),
                    ],
                    triangle_internal_context: true,
                    ..Options::default()
                },
                Duration::from_secs(10),
                64,
                Some(32768),
            )
            .unwrap(),
            capture,
        )
    }

    #[cfg(unix)]
    fn captured(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn internal_navigation_reinstalls_snapshot_and_only_expands_authorized_sources() {
        let temp = tempfile::tempdir().unwrap();
        let (provider, capture) = internal_provider(temp.path(), "none");
        let first = internal_request(compiled_packet());
        let snapshot = first.messages.clone();
        assert_eq!(provider.complete(first).await.unwrap().content, "ok");
        let mut next_packet = compiled_packet();
        next_packet["context"]["retained_goal"] = json!("next request snapshot");
        provider
            .complete(internal_request(next_packet))
            .await
            .unwrap();
        let requests = captured(&capture);
        assert_eq!(
            requests
                .iter()
                .map(|r| r["op"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "hello", "install", "execute", "execute", "execute", "complete", "hello",
                "install", "execute", "execute", "execute", "complete"
            ]
        );
        for (index, goal) in [(5, "native goal"), (11, "next request snapshot")] {
            let messages = &requests[index]["args"]["messages"];
            let text = messages[0]["content"].as_str().unwrap();
            let context: Value =
                serde_json::from_str(text.strip_prefix("Native context: ").unwrap()).unwrap();
            assert_eq!(context["retained_goal"], goal);
            assert_eq!(context["rehydrated"], compiled_packet()["sources"]);
            let wire = messages.to_string();
            for hidden in [
                "compiled_memory",
                "triangles",
                "coordinates",
                "vector",
                "commands",
                CONTEXT_MARKER,
            ] {
                assert!(!wire.contains(hidden), "host metadata leaked: {hidden}");
            }
        }
        assert!(snapshot[0].content.contains(CONTEXT_MARKER));
        assert_eq!(snapshot[0].compiled_memory, Some(compiled_packet()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn internal_corrupt_navigation_never_calls_complete() {
        for fault in [
            "addresses",
            "trace",
            "digest",
            "engine",
            "ticks",
            "coordinates",
            "source",
        ] {
            let temp = tempfile::tempdir().unwrap();
            let (provider, capture) = internal_provider(temp.path(), fault);
            let mut packet = compiled_packet();
            if fault == "coordinates" {
                packet["banks"][0]["coordinates"][1] = json!("node:wrong");
            }
            if fault == "source" {
                packet["sources"][1]["node_id"] = json!("never-emitted");
            }
            assert!(
                provider.complete(internal_request(packet)).await.is_err(),
                "{fault}"
            );
            assert!(
                !captured(&capture).iter().any(|r| r["op"] == "complete"),
                "{fault}"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn internal_empty_selection_and_completion_state_are_explicit() {
        let temp = tempfile::tempdir().unwrap();
        let (provider, capture) = internal_provider(temp.path(), "none");
        provider
            .complete(internal_request(
                json!({"schema":1,"context":{},"sources":[],"banks":[]}),
            ))
            .await
            .unwrap();
        assert_eq!(
            captured(&capture)
                .iter()
                .map(|r| r["op"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["hello", "install", "complete"]
        );
        let temp = tempfile::tempdir().unwrap();
        let (provider, _) = internal_provider(temp.path(), "complete_state");
        assert!(
            provider
                .complete(internal_request(compiled_packet()))
                .await
                .is_err()
        );
    }

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
