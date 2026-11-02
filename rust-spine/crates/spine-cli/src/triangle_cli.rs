//! Native heart retrieval and verified replay through a local compiled model.
use crate::triangle_backend::{Backend, MAX_LINE, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use spine_heart::{
    AgentId, CognitiveConfig, CognitiveState, Content, ContextHandle, ContextLeaf, Embedding,
    EventKind, HeartConfig, InteractionInput, KeySource, NodeId, ParticipantRole, Provenance,
    RehydrateBudget, SemanticEncoder, SignedEvent, SpineHeart, ThreadId,
};
use spine_models::{MiniLmAssets, MiniLmEncoder};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Args)]
pub struct Options {
    #[arg(long)]
    pub path: PathBuf,
    #[arg(
        long,
        help = "Heart passphrase (prefer SPINE_HEART_PASSPHRASE or prompt)"
    )]
    pub passphrase: Option<String>,
    #[arg(long, help = "Existing local MiniLM directory; never downloads")]
    pub model_dir: PathBuf,
    #[arg(long)]
    pub backend_program: PathBuf,
    #[arg(long, allow_hyphen_values = true)]
    pub backend_arg: Vec<OsString>,
    #[arg(long)]
    pub question: Option<String>,
    #[arg(long)]
    pub session_dir: Option<PathBuf>,
    #[arg(long, help = "Do not modify heart or commit completed turns")]
    pub read_only: bool,
    #[arg(long, requires = "session_dir")]
    pub save_state: bool,
    #[arg(long, requires = "session_dir")]
    pub restore_state: bool,
    #[arg(
        long,
        help = "Verified open:SLOT, peek, left, right or route:INDEX; repeatable"
    )]
    pub navigation_command: Vec<String>,
    #[arg(long, default_value_t = 160, value_parser = clap::value_parser!(u16).range(1..=512))]
    pub max_tokens: u16,
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..=3600))]
    pub timeout_seconds: u64,
    #[arg(long, default_value = "triangle")]
    pub agent: String,
    #[arg(long, default_value = "triangle")]
    pub thread: String,
}

#[derive(Clone, Debug)]
struct Record {
    apex: usize,
    children: Option<(usize, usize)>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct Navigation {
    cursor: Option<usize>,
    steps: usize,
    routes: BTreeMap<usize, usize>,
}
#[derive(Deserialize, Serialize)]
struct SavedNavigation {
    schema: u32,
    forest_sha256: String,
    native_sha256: String,
    backend_file: String,
    navigation: Navigation,
}
pub(crate) struct Installed {
    forest_sha256: String,
    native_sha256: String,
    addresses: Vec<String>,
    records: Vec<Record>,
    navigation: Navigation,
    status: String,
}

fn address(handle: ContextHandle) -> String {
    match handle {
        ContextHandle::Node(id) => format!("node:{id}"),
        ContextHandle::Triangle(id) => format!("triangle:{id}"),
    }
}
fn coordinate(handle: ContextHandle, state: &CognitiveState) -> NodeId {
    match handle {
        ContextHandle::Node(id) => id,
        ContextHandle::Triangle(id) => state.triangles.triangles[&id].apex,
    }
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn empty_forest() -> Value {
    json!({"schema":1,"nodes":{},"triangles":{},"roots":[]})
}

/// Full provenance enters the digest; an oversized installation falls back explicitly.
fn forest_export(state: &CognitiveState, events: &[SignedEvent]) -> Result<Value> {
    state.triangles.verify(&state.dcmdb)?;
    let events: BTreeMap<_, _> = events.iter().map(|e| (e.id, e)).collect();
    let mut nodes = serde_json::Map::new();
    for id in state.triangles.referenced_nodes() {
        let node = state.dcmdb.node(id).ok_or("missing native coordinate")?;
        let sources: Vec<_> = node.event_ids.iter().filter_map(|id| events.get(id)).filter_map(|e| {
            if let Content::Inline(text) = &e.body.interaction.content {
                Some(json!({"event_id":e.id.to_string(),"text":text,"sequence":e.body.device_sequence}))
            } else { None }
        }).collect();
        let last = sources
            .last()
            .ok_or("native coordinate has no inline source; cannot compile this forest")?;
        nodes.insert(format!("node:{id}"),json!({"text":last["text"],"event_id":last["event_id"],"events":sources,"vector":node.centroid}));
    }
    let mut triangles = serde_json::Map::new();
    for (id, t) in &state.triangles.triangles {
        triangles.insert(format!("triangle:{id}"),json!({"apex":format!("node:{}",t.apex),"left":address(t.left),"right":address(t.right),"depth":t.depth}));
    }
    Ok(
        json!({"schema":1,"nodes":nodes,"triangles":triangles,"roots":state.triangles.roots.iter().map(|r|address(r.handle)).collect::<Vec<_>>()}),
    )
}

/// Postcard supports the native graph's structured map keys. Every persisted map
/// in CognitiveState is ordered, so identical state and canonical events produce
/// the same bytes across process restarts. Keep the domain version explicit.
fn native_fingerprint(state: &CognitiveState, events: &[spine_heart::EventId]) -> Result<String> {
    let bytes = postcard::to_allocvec(&("spine-triangle-native-v1", state, events))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

pub(crate) async fn install(backend: &mut Backend, heart: &SpineHeart) -> Result<Installed> {
    if !heart.cognition_is_current()? {
        return Err("heart changed; cognitive projection is stale".into());
    }
    let state = heart
        .cognition()?
        .ok_or("heart has no cognitive projection")?;
    let events = heart.events_canonical()?;
    let native_sha256 =
        native_fingerprint(&state, &events.iter().map(|e| e.id).collect::<Vec<_>>())?;
    let count = state.triangles.referenced_nodes().len() + state.triangles.triangles.len();
    let depth = state
        .triangles
        .triangles
        .values()
        .map(|t| t.depth)
        .max()
        .unwrap_or(0);
    let mut status = if count > 64 {
        "native_capacity_fallback"
    } else if depth > 8 {
        "native_depth_fallback"
    } else if count == 0 {
        "native_no_routes"
    } else {
        "compiled_native_routes"
    }
    .to_string();
    let mut forest = if status == "compiled_native_routes" {
        match forest_export(&state, &events) {
            Ok(forest) => forest,
            Err(error) => {
                eprintln!("[native_source_fallback: {error}]");
                status = "native_source_fallback".into();
                empty_forest()
            }
        }
    } else {
        empty_forest()
    };
    if serde_json::to_vec(&forest)?.len() > MAX_LINE - 1024 {
        status = "native_export_budget_fallback".into();
        forest = empty_forest();
    }
    let expected: Vec<String> = forest["nodes"]
        .as_object()
        .unwrap()
        .keys()
        .chain(forest["triangles"].as_object().unwrap().keys())
        .cloned()
        .collect();
    let result = backend.call("install", json!({"forest":forest})).await?;
    let addresses: Vec<String> = serde_json::from_value(result["addresses"].clone())?;
    if addresses != expected {
        return Err("backend address map differs from native export".into());
    }
    let forest_sha256 = result["forest_sha256"]
        .as_str()
        .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or("invalid installed forest digest")?
        .to_owned();
    let slots: BTreeMap<_, _> = addresses
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();
    let records = addresses
        .iter()
        .map(|a| {
            if a.starts_with("node:") {
                Record {
                    apex: slots[a.as_str()],
                    children: None,
                }
            } else {
                let t = &forest["triangles"][a];
                Record {
                    apex: slots[t["apex"].as_str().unwrap()],
                    children: Some((
                        slots[t["left"].as_str().unwrap()],
                        slots[t["right"].as_str().unwrap()],
                    )),
                }
            }
        })
        .collect();
    eprintln!(
        "{}",
        json!({"type":"triangle_install","status":status,"native_records":count,"native_depth":depth,"forest_sha256":forest_sha256,"native_sha256":native_sha256})
    );
    Ok(Installed {
        forest_sha256,
        native_sha256,
        addresses,
        records,
        navigation: Navigation::default(),
        status,
    })
}

impl Navigation {
    fn expected(&mut self, command: &str, records: &[Record]) -> Result<Vec<usize>> {
        let (token, mode) = match command {
            "peek" => (264, "peek"),
            "left" => (265, "left"),
            "right" => (266, "right"),
            _ => {
                let (mode, number) = command
                    .split_once(':')
                    .ok_or("expected open:SLOT, peek, left, right or route:INDEX")?;
                let number: usize = number.parse()?;
                match mode {
                    "open" if number < 64 => (number, "open"),
                    "route" if number < 8 => (256 + number, "route"),
                    _ => return Err("navigation command outside compiled bounds".into()),
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
            };
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

async fn execute(
    backend: &mut Backend,
    installed: &mut Installed,
    command: &str,
) -> Result<Vec<usize>> {
    if installed.records.is_empty() {
        return Err(format!("navigation unavailable: {}", installed.status).into());
    }
    let mut candidate = installed.navigation.clone();
    let expected = candidate.expected(command, &installed.records)?;
    let result = backend
        .call(
            "execute",
            json!({"forest_sha256":installed.forest_sha256,"command":command}),
        )
        .await?;
    let trace: Vec<usize> = serde_json::from_value(result["trace"].clone())?;
    if trace != expected {
        return Err(format!(
            "compiled trace mismatch for {command}; expected {expected:?}, received {trace:?}"
        )
        .into());
    }
    installed.navigation = candidate;
    eprintln!("[verified {command} trace={trace:?}]");
    Ok(trace)
}

struct Walk {
    commands: Vec<String>,
    coordinates: Vec<String>,
    alternates: Vec<String>,
}
fn ranked_walks(
    heart: &SpineHeart,
    state: &CognitiveState,
    query: &Embedding,
    installed: &Installed,
) -> Result<(Vec<Walk>, BTreeSet<NodeId>)> {
    let mut walks = Vec::new();
    let mut visited = BTreeSet::new();
    for root in state.triangles.ranked_roots(&state.dcmdb, query, None, 3)? {
        let native = heart.rehydrate_context(
            root.handle,
            Some(query),
            RehydrateBudget {
                max_depth: 8,
                max_fanout: 8,
                max_nodes: 64,
                max_tokens: usize::MAX,
            },
        )?;
        for c in &native.coordinates {
            visited.insert(c.node_id);
        }
        if installed.status != "compiled_native_routes" {
            continue;
        }
        let root_address = address(root.handle);
        let slot = installed
            .addresses
            .iter()
            .position(|a| a == &root_address)
            .ok_or("native root not installed")?;
        let mut commands = vec![format!("open:{slot}")];
        let mut cursor = root.handle;
        for _ in 0..8 {
            let ContextHandle::Triangle(id) = cursor else {
                break;
            };
            let t = &state.triangles.triangles[&id];
            let left = dot(
                &state
                    .dcmdb
                    .node(coordinate(t.left, state))
                    .ok_or("missing left coordinate")?
                    .centroid,
                query.as_slice(),
            );
            let right = dot(
                &state
                    .dcmdb
                    .node(coordinate(t.right, state))
                    .ok_or("missing right coordinate")?
                    .centroid,
                query.as_slice(),
            );
            let (command, next) = if left > right {
                ("left", t.left)
            } else {
                ("right", t.right)
            };
            commands.push(command.into());
            cursor = next;
        }
        walks.push(Walk {
            commands,
            coordinates: native
                .coordinates
                .iter()
                .map(|c| format!("node:{}", c.node_id))
                .collect(),
            alternates: native
                .alternate_routes
                .iter()
                .copied()
                .map(address)
                .collect(),
        });
    }
    Ok((walks, visited))
}

#[derive(Debug)]
struct Segment {
    score: usize,
    start: usize,
    end: usize,
    text: String,
}
fn query_terms(question: &str) -> BTreeSet<String> {
    question
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.chars().count() > 2)
        .map(str::to_lowercase)
        .collect()
}
/// Inspect every source segment, including old events and tails; bound what enters the prompt.
fn segments(text: &str, terms: &BTreeSet<String>) -> Vec<Segment> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    (0..chars.len())
        .step_by(600)
        .map(|start| {
            let end = (start + 800).min(chars.len());
            let text: String = chars[start..end].iter().collect();
            let lower = text.to_lowercase();
            let score = terms.iter().filter(|t| lower.contains(t.as_str())).count();
            Segment {
                score,
                start,
                end,
                text,
            }
        })
        .collect()
}
fn evidence(
    state: &CognitiveState,
    events: &[SignedEvent],
    visited: &BTreeSet<NodeId>,
    query: &Embedding,
    question: &str,
) -> (Vec<Value>, Value) {
    let terms = query_terms(question);
    // The caller supplies events_canonical(); event indexes preserve its total
    // order across devices, whose individual sequence numbers are not comparable.
    let canonical_positions: BTreeMap<_, _> = events
        .iter()
        .enumerate()
        .map(|(index, event)| (event.id, index))
        .collect();
    let mut memberships = BTreeMap::new();
    for (id, node) in &state.dcmdb.nodes {
        for event in &node.event_ids {
            memberships.entry(*event).or_insert(*id);
        }
    }
    let mut candidates = Vec::new();
    let mut non_inline = 0;
    let mut source_chars = 0;
    let mut segments_total: usize = 0;
    for event in events {
        let Content::Inline(text) = &event.body.interaction.content else {
            non_inline += 1;
            continue;
        };
        source_chars += text.chars().count();
        let node = memberships.get(&event.id).copied();
        let semantic = node
            .and_then(|id| state.dcmdb.node(id))
            .map_or(0., |n| dot(&n.centroid, query.as_slice()));
        let routed = node.is_some_and(|id| visited.contains(&id));
        for segment in segments(text, &terms) {
            segments_total += 1;
            candidates.push((segment, routed, semantic, event, node));
        }
    }
    candidates.sort_by(|a, b| {
        b.0.score
            .cmp(&a.0.score)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.total_cmp(&a.2))
            .then_with(|| a.3.id.cmp(&b.3.id))
            .then_with(|| a.0.start.cmp(&b.0.start))
    });
    let mut result = Vec::new();
    let mut emitted_chars = 0;
    let mut per_event = BTreeMap::new();
    let mut seen_events = BTreeSet::new();
    for (segment, routed, _, event, node) in candidates {
        if result.len() >= 12 || emitted_chars + segment.text.chars().count() > 8192 {
            continue;
        }
        let count = per_event.entry(event.id).or_insert(0);
        if *count >= 2 {
            continue;
        }
        *count += 1;
        emitted_chars += segment.text.chars().count();
        seen_events.insert(event.id);
        result.push(json!({"node_id":node.map(|n|n.to_string()).unwrap_or_default(),"event_id":event.id.to_string(),"canonical_index":canonical_positions[&event.id],"device_id":event.body.device_id.to_string(),"device_sequence":event.body.device_sequence,"text":segment.text,"source":event.body.interaction.provenance.source_uri,"role":format!("{:?}",event.body.interaction.role),"start_char":segment.start,"end_char":segment.end,"source_chars":match &event.body.interaction.content {Content::Inline(t)=>t.chars().count(),_=>0},"retrieval":if routed {"native_route_and_query_segments"} else {"native_source_query_segments"}}));
    }
    // Relevance decides which bounded excerpts survive; chronology decides how
    // those selected excerpts are presented, so later corrections remain later.
    result.sort_by_key(|record| {
        (
            record["canonical_index"].as_u64().unwrap(),
            record["start_char"].as_u64().unwrap(),
        )
    });
    let report = json!({"budget_records":12,"budget_chars":8192,"source_events":events.len(),"selected_events":seen_events.len(),"omitted_events":events.len().saturating_sub(seen_events.len()),"non_inline_events":non_inline,"candidate_segments":segments_total,"selected_segments":result.len(),"omitted_segments":segments_total.saturating_sub(result.len()),"source_chars":source_chars,"selected_chars":emitted_chars,"presentation":"canonical_event_order_then_source_offset","scope":"all canonical inline source events, including older events and tails; citations identify provenance, not proof"});
    (result, report)
}

fn session_directory(path: &Path) -> Result<PathBuf> {
    if path.exists() && fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err("session directory must not be a symlink".into());
    }
    if !path.exists() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(path)?;
        }
        #[cfg(not(unix))]
        fs::create_dir_all(path)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(path)?.permissions().mode() & 0o077 != 0 {
            return Err("session directory must be private (chmod 700)".into());
        }
    }
    Ok(fs::canonicalize(path)?)
}
async fn save(
    backend: &mut Backend,
    installed: &Installed,
    directory: Option<&Path>,
) -> Result<()> {
    let dir = session_directory(directory.ok_or("saving requires --session-dir")?)?;
    let mut random = [0_u8; 12];
    getrandom::fill(&mut random)?;
    let backend_file = format!("navigation-{}.state", hex::encode(random));
    let path = dir.join(&backend_file);
    let response = backend
        .call(
            "save",
            json!({"forest_sha256":installed.forest_sha256,"path":path}),
        )
        .await?;
    if response["forest_sha256"].as_str() != Some(&installed.forest_sha256) {
        return Err("saved backend revision mismatch".into());
    }
    let saved = SavedNavigation {
        schema: 1,
        forest_sha256: installed.forest_sha256.clone(),
        native_sha256: installed.native_sha256.clone(),
        backend_file,
        navigation: installed.navigation.clone(),
    };
    let mut temp = tempfile::NamedTempFile::new_in(&dir)?;
    temp.write_all(&serde_json::to_vec(&saved)?)?;
    temp.as_file().sync_all()?;
    temp.persist(dir.join("navigation.json"))?;
    eprintln!(
        "[saved private navigation: {}]",
        dir.join("navigation.json").display()
    );
    Ok(())
}
async fn restore(
    backend: &mut Backend,
    installed: &mut Installed,
    directory: Option<&Path>,
) -> Result<()> {
    let dir = session_directory(directory.ok_or("restoring requires --session-dir")?)?;
    let metadata = dir.join("navigation.json");
    if fs::symlink_metadata(&metadata)?.file_type().is_symlink()
        || fs::metadata(&metadata)?.len() > 64 * 1024
    {
        return Err("invalid navigation manifest file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(&metadata)?.permissions().mode() & 0o077 != 0 {
            return Err("navigation manifest must be private".into());
        }
    }
    let saved: SavedNavigation = serde_json::from_slice(&fs::read(metadata)?)?;
    if saved.schema != 1
        || saved.forest_sha256 != installed.forest_sha256
        || saved.native_sha256 != installed.native_sha256
    {
        return Err("saved navigation is stale: heart/forest changed; no state restored".into());
    }
    if Path::new(&saved.backend_file)
        .file_name()
        .and_then(|n| n.to_str())
        != Some(saved.backend_file.as_str())
        || !saved.backend_file.starts_with("navigation-")
    {
        return Err("invalid saved backend file name".into());
    }
    if saved.navigation.cursor.is_some_and(|n| n >= 64)
        || saved.navigation.steps > 1024
        || saved
            .navigation
            .routes
            .iter()
            .any(|(k, v)| *k >= 8 || *v >= installed.records.len())
    {
        return Err("invalid saved native navigation".into());
    }
    let path = dir.join(&saved.backend_file);
    if fs::symlink_metadata(&path)?.file_type().is_symlink() {
        return Err("backend session file must not be a symlink".into());
    }
    let response = backend
        .call(
            "restore",
            json!({"forest_sha256":installed.forest_sha256,"path":path}),
        )
        .await?;
    if response["forest_sha256"].as_str() != Some(&installed.forest_sha256) {
        return Err("restored backend revision mismatch".into());
    }
    installed.navigation = saved.navigation;
    // Verify restored cursor immediately when one exists; subsequent route/descent checks use saved alternates.
    if installed.navigation.cursor.is_some() {
        execute(backend, installed, "peek").await?;
    }
    eprintln!(
        "{}",
        json!({"type":"triangle_restore","status":"restored","forest_sha256":installed.forest_sha256,"native_sha256":installed.native_sha256,"cursor":installed.navigation.cursor,"routes":installed.navigation.routes})
    );
    Ok(())
}

fn validate_generation(
    response: &Value,
    evidence: &[Value],
    max_tokens: u16,
) -> Result<(String, bool, Vec<String>)> {
    let answer = response["answer"]
        .as_str()
        .filter(|a| !a.trim().is_empty() && a.len() <= 128 * 1024)
        .ok_or("model returned empty or oversized answer")?
        .to_string();
    let prompt = response["prompt_tokens"]
        .as_u64()
        .ok_or("generation missing prompt token count")?;
    let tokens = response["answer_tokens"]
        .as_u64()
        .ok_or("generation missing answer token count")?;
    if prompt > 8192 || tokens > u64::from(max_tokens) {
        return Err("model exceeded declared token budgets".into());
    }
    let truncated = response["truncated"]
        .as_bool()
        .ok_or("generation missing completion status")?;
    let citations: Vec<String> = serde_json::from_value(response["citations"].clone())?;
    let allowed: BTreeSet<&str> = evidence
        .iter()
        .flat_map(|e| {
            [
                e["node_id"].as_str().unwrap_or(""),
                e["event_id"].as_str().unwrap_or(""),
            ]
        })
        .filter(|s| !s.is_empty())
        .collect();
    if citations
        .iter()
        .any(|c| !allowed.contains(c.as_str()) || !answer.contains(c))
    {
        return Err("model reported a citation absent from the supplied sources or answer".into());
    }
    Ok((answer, truncated, citations))
}

fn commit_turn(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    options: &Options,
    question: &str,
    answer: &str,
    installed: &Installed,
    citations: &[String],
) -> Result<()> {
    let agent = AgentId::new(options.agent.clone())?;
    let thread = ThreadId::new(options.thread.clone())?;
    // Encode and validate both messages before any write; backend partial output never reaches here.
    let mut items = Vec::new();
    for (role, text) in [
        (ParticipantRole::User, question),
        (ParticipantRole::Assistant, answer),
    ] {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "triangle_forest_sha256".into(),
            installed.forest_sha256.clone(),
        );
        metadata.insert("triangle_retrieval".into(), installed.status.clone());
        if role == ParticipantRole::Assistant {
            metadata.insert("evidence_class".into(), "unverified_model_claim".into());
            metadata.insert("citations".into(), serde_json::to_string(citations)?);
        }
        let embedding = encoder.encode(text)?;
        items.push((
            InteractionInput {
                agent_id: agent.clone(),
                thread_id: thread.clone(),
                role,
                kind: EventKind::Message,
                content: Content::Inline(text.into()),
                causal_parents: Vec::new(),
                provenance: Provenance {
                    provider: Some("local-triangle-merged".into()),
                    metadata,
                    ..Default::default()
                },
                tool: None,
                attachments: Vec::new(),
                outcome: None,
            },
            embedding,
        ));
    }
    let committed = heart.commit_embedded_batch(items)?;
    heart.compact_context(
        committed.iter().map(|(c, m)| ContextLeaf {
            node_id: m.node_id,
            chronology: c.event.body.device_sequence,
        }),
        6,
    )?;
    eprintln!(
        "{}",
        json!({"type":"triangle_commit","committed":true,"messages":2,"prior_navigation":"stale"})
    );
    Ok(())
}

pub(crate) async fn retrieve(
    backend: &mut Backend,
    installed: &mut Installed,
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    question: &str,
) -> Result<Value> {
    if question.trim().is_empty() || question.len() > 16 * 1024 {
        return Err("question must contain 1..16384 UTF-8 bytes".into());
    }
    if !heart.cognition_is_current()? {
        return Err("heart projection changed; restart triangle to refresh".into());
    }
    let query = encoder.encode(question)?;
    let state = heart.cognition()?.ok_or("missing cognitive state")?;
    let (walks, visited) = ranked_walks(heart, &state, &query, installed)?;
    for walk in walks {
        let mut coordinates = Vec::new();
        let mut alternates = Vec::new();
        for command in walk.commands {
            for token in execute(backend, installed, &command).await? {
                if (192..256).contains(&token) {
                    let a = installed
                        .addresses
                        .get(token - 192)
                        .ok_or("trace apex outside addresses")?
                        .clone();
                    if !coordinates.contains(&a) {
                        coordinates.push(a);
                    }
                }
                if (128..192).contains(&token) {
                    alternates.push(
                        installed
                            .addresses
                            .get(token - 128)
                            .ok_or("trace alternate outside addresses")?
                            .clone(),
                    );
                }
            }
        }
        if coordinates != walk.coordinates || alternates != walk.alternates {
            return Err(
                "compiled walk differs from native rehydrated coordinates/alternates".into(),
            );
        }
    }
    let events = heart.events_canonical()?;
    let (sources, report) = evidence(&state, &events, &visited, &query, question);
    eprintln!("[triangle evidence {}]", serde_json::to_string(&report)?);
    Ok(
        json!({"sources":sources,"report":report,"status":installed.status,
        "forest_sha256":installed.forest_sha256,"native_sha256":installed.native_sha256}),
    )
}

async fn ask(
    backend: &mut Backend,
    installed: &mut Installed,
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    options: &Options,
    question: &str,
) -> Result<()> {
    let retrieval = retrieve(backend, installed, heart, encoder, question).await?;
    let sources: Vec<Value> = serde_json::from_value(retrieval["sources"].clone())?;
    let result = backend
        .call(
            "generate",
            json!({"question":question,"evidence":sources,"max_tokens":options.max_tokens}),
        )
        .await?;
    let (answer, truncated, citations) =
        validate_generation(&result, &sources, options.max_tokens)?;
    println!("{answer}");
    eprintln!(
        "{}",
        json!({"type":"triangle_generation","status":installed.status,"prompt_tokens":result["prompt_tokens"],"answer_tokens":result["answer_tokens"],"truncated":truncated,"citations":citations,"read_only":options.read_only,"committed":false,"citation_scope":"supplied provenance, not verified support"})
    );
    if truncated {
        return Err("generation reached its token limit; partial answer displayed, no interaction committed".into());
    }
    if !options.read_only {
        commit_turn(
            heart, encoder, options, question, &answer, installed, &citations,
        )?;
        *installed = install(backend, heart).await?;
    }
    Ok(())
}

pub async fn run(mut options: Options) -> Result<()> {
    let passphrase = crate::resolve_heart_passphrase(options.passphrase.take(), false)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let heart = SpineHeart::open(
        HeartConfig::new(&options.path),
        KeySource::Passphrase(passphrase),
    )?;
    let encoder = MiniLmEncoder::load(MiniLmAssets::from_directory(&options.model_dir), 256)?;
    match heart.cognition()? {
        None if !options.read_only => {
            heart.rebuild_cognition(
                CognitiveConfig::new(1, encoder.manifest().clone(), 8)?,
                &encoder,
            )?;
        }
        None => {
            return Err(
                "read-only requires an existing cognitive projection; run cognition-init first"
                    .into(),
            );
        }
        Some(state) if &state.config.model != encoder.manifest() => {
            return Err("MiniLM manifest does not match this heart's cognitive projection".into());
        }
        _ => {}
    }
    if !heart.cognition_is_current()? {
        if options.read_only {
            return Err("read-only requires a current cognitive projection".into());
        }
        heart.catch_up_cognition(&encoder)?;
    }
    // Validate destination/identities before starting an expensive child or generation.
    AgentId::new(options.agent.clone())?;
    ThreadId::new(options.thread.clone())?;
    if let Some(directory) = &options.session_dir {
        session_directory(directory)?;
    }
    let mut backend = Backend::spawn(
        &options.backend_program,
        &options.backend_arg,
        Duration::from_secs(options.timeout_seconds),
    )?;
    let result = run_with_backend(&mut backend, &heart, &encoder, &options).await;
    if result.is_err() {
        backend.abort().await;
        return result;
    }
    backend.shutdown().await
}

async fn run_with_backend(
    backend: &mut Backend,
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    options: &Options,
) -> Result<()> {
    let hello = backend.call("hello", json!({})).await?;
    if hello["protocol"] != 1
        || hello["capacity"] != 64
        || hello["max_depth"] != 8
        || hello["backend"] != "merged"
    {
        return Err(
            "incompatible triangle backend: requires merged protocol1/capacity64/depth8".into(),
        );
    }
    let mut installed = install(backend, heart).await?;
    if options.restore_state {
        restore(backend, &mut installed, options.session_dir.as_deref()).await?;
    }
    for command in &options.navigation_command {
        execute(backend, &mut installed, command).await?;
    }
    if let Some(question) = &options.question {
        ask(backend, &mut installed, heart, encoder, options, question).await?;
    }
    if options.save_state {
        save(backend, &installed, options.session_dir.as_deref()).await?;
    }
    if options.question.is_some()
        || options.save_state
        || options.restore_state
        || !options.navigation_command.is_empty()
    {
        return Ok(());
    }
    eprintln!(
        "Triangle terminal: ask a question; /save, /restore, /open N, /peek, /left, /right, /route N, /quit."
    );
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    crate::terminal_input::spawn(sender);
    loop {
        print!("triangle> ");
        io::stdout().flush()?;
        let input = tokio::select! {input=receiver.recv()=>input, signal=tokio::signal::ctrl_c()=>{signal?;return Ok(());}};
        let Some(input) = input else { break };
        let input = input?;
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/quit" | "/exit" => break,
            "/save" => save(backend, &installed, options.session_dir.as_deref()).await?,
            "/restore" => restore(backend, &mut installed, options.session_dir.as_deref()).await?,
            "/peek" | "/left" | "/right" => {
                execute(backend, &mut installed, &input[1..]).await?;
            }
            s if s.starts_with("/open ") || s.starts_with("/route ") => {
                execute(backend, &mut installed, &s[1..].replacen(' ', ":", 1)).await?;
            }
            s if s.starts_with('/') => eprintln!("[unknown triangle command]"),
            question => ask(backend, &mut installed, heart, encoder, options, question).await?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn triangle_native_trace_checks_alternates_and_revisited_routes() {
        let records = vec![
            Record {
                apex: 0,
                children: None,
            },
            Record {
                apex: 1,
                children: None,
            },
            Record {
                apex: 0,
                children: Some((0, 1)),
            },
        ];
        let mut nav = Navigation::default();
        assert_eq!(nav.expected("open:2", &records).unwrap(), vec![2, 192, 267]);
        assert_eq!(
            nav.expected("right", &records).unwrap(),
            vec![266, 128, 65, 193, 267]
        );
        assert_eq!(
            nav.expected("route:0", &records).unwrap(),
            vec![256, 64, 192, 267]
        );
        assert_eq!(nav.expected("left", &records).unwrap(), vec![265, 270]);
        assert_eq!(nav.expected("route:7", &records).unwrap(), vec![263, 268]);
        assert!(nav.expected("open:64", &records).is_err());
    }
    #[test]
    fn triangle_source_segments_find_relevant_tail_and_unicode() {
        let source = format!(
            "{} launch code AURORA Ω",
            "old unrelated preamble ".repeat(200)
        );
        let terms = query_terms("What is the launch code AURORA?");
        let mut candidates = segments(&source, &terms);
        candidates.sort_by_key(|s| std::cmp::Reverse(s.score));
        assert!(candidates[0].start > 1000);
        assert!(candidates[0].text.contains("AURORA Ω"));
        assert!(candidates.iter().all(|s| s.text.chars().count() <= 800));
    }
    #[test]
    fn triangle_generation_rejects_unknown_citations_and_missing_completion() {
        let sources = vec![json!({"node_id":"node1","event_id":"event1","text":"source"})];
        let mut answer = json!({"answer":"answer [event1]","prompt_tokens":3,"answer_tokens":4,"truncated":false,"citations":["event1"]});
        assert!(validate_generation(&answer, &sources, 10).is_ok());
        answer["citations"] = json!(["invented"]);
        assert!(validate_generation(&answer, &sources, 10).is_err());
        answer["citations"] = json!([]);
        answer.as_object_mut().unwrap().remove("truncated");
        assert!(validate_generation(&answer, &sources, 10).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn triangle_session_directory_is_private_and_rejects_symlinks() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("session");
        session_directory(&path).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o077, 0);
        let link = temp.path().join("alias");
        symlink(&path, &link).unwrap();
        assert!(session_directory(&link).is_err());
    }
    #[test]
    fn triangle_native_export_and_evidence_keep_old_event_tail() {
        let dir = tempfile::tempdir().unwrap();
        let created = SpineHeart::create(
            HeartConfig::new(dir.path().join("test.spine")),
            "synthetic-only",
        )
        .unwrap();
        let manifest = spine_heart::ModelManifest {
            schema: 1,
            model_name: "triangle-test".into(),
            artifact_hash: [1; 32],
            tokenizer_hash: [2; 32],
            dimension: 3,
            normalized: true,
            quantization: None,
        };
        created
            .heart
            .initialize_cognition(CognitiveConfig::new(1, manifest, 2).unwrap())
            .unwrap();
        let query = Embedding::normalized(vec![1., 0., 0.], 3).unwrap();
        let old = format!(
            "{} launch code AURORA",
            "unrelated historical preamble ".repeat(90)
        );
        let mut leaves = Vec::new();
        for text in [&old, "new unrelated event", "latest unrelated event"] {
            let (commit, memory) = created
                .heart
                .commit_embedded(
                    InteractionInput {
                        agent_id: AgentId::new("test").unwrap(),
                        thread_id: ThreadId::new("test").unwrap(),
                        role: ParticipantRole::User,
                        kind: EventKind::Message,
                        content: Content::Inline(text.into()),
                        causal_parents: Vec::new(),
                        provenance: Provenance::default(),
                        tool: None,
                        attachments: Vec::new(),
                        outcome: None,
                    },
                    query.clone(),
                )
                .unwrap();
            leaves.push(ContextLeaf {
                node_id: memory.node_id,
                chronology: commit.event.body.device_sequence,
            });
        }
        created.heart.compact_context(leaves, 6).unwrap();
        let state = created.heart.cognition().unwrap().unwrap();
        let events = created.heart.events_canonical().unwrap();
        let exported = forest_export(&state, &events).unwrap();
        assert!(exported["nodes"].as_object().unwrap().values().any(|n| {
            n["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["text"].as_str() == Some(old.as_str()))
        }));
        let (selected, report) = evidence(
            &state,
            &events,
            &BTreeSet::new(),
            &query,
            "What is the launch code AURORA?",
        );
        let tail = selected
            .iter()
            .find(|record| record["text"].as_str().unwrap().contains("AURORA"))
            .expect("query-ranked old tail must survive chronological presentation");
        assert_eq!(tail["event_id"], events[0].id.to_string());
        assert!(tail["start_char"].as_u64().unwrap() > 1000);
        assert!(report["selected_chars"].as_u64().unwrap() <= 8192);
        assert!(selected.len() <= 12);
    }
    #[test]
    fn triangle_native_fingerprint_handles_graph_edges_and_rejects_changed_revision() {
        let manifest = spine_heart::ModelManifest {
            schema: 1,
            model_name: "triangle-graph-test".into(),
            artifact_hash: [1; 32],
            tokenizer_hash: [2; 32],
            dimension: 3,
            normalized: true,
            quantization: None,
        };
        let mut state = CognitiveState::new(CognitiveConfig::new(1, manifest, 2).unwrap()).unwrap();
        let mut ids = Vec::new();
        for (index, vector) in [vec![1., 0., 0.], vec![0., 1., 0.], vec![1., 1., 0.]]
            .into_iter()
            .enumerate()
        {
            let event_id = spine_heart::EventId::from_bytes([index as u8 + 1; 32]);
            ids.push(event_id);
            state
                .dcmdb
                .update(spine_heart::MemoryObservation {
                    event_id,
                    vector,
                    time: index as f64,
                    source: None,
                    metadata: BTreeMap::new(),
                })
                .unwrap();
        }
        assert!(
            state.dcmdb.graph.values().any(|row| !row.is_empty()),
            "fixture must exercise native graph edges/coactivation keys"
        );
        assert!(
            serde_json::to_vec(&state).is_err(),
            "regression fixture must reproduce JSON map-key rejection"
        );
        let original = native_fingerprint(&state, &ids).unwrap();
        let restored: CognitiveState =
            postcard::from_bytes(&postcard::to_allocvec(&state).unwrap()).unwrap();
        assert_eq!(original, native_fingerprint(&restored, &ids).unwrap());
        let mut changed = restored;
        *changed
            .dcmdb
            .graph
            .values_mut()
            .find_map(|row| row.values_mut().next())
            .unwrap() += 0.25;
        assert_ne!(original, native_fingerprint(&changed, &ids).unwrap());
        ids.push(spine_heart::EventId::from_bytes([99; 32]));
        assert_ne!(original, native_fingerprint(&state, &ids).unwrap());
    }
    #[test]
    fn triangle_evidence_presents_ranked_sources_in_canonical_chronology() {
        let dir = tempfile::tempdir().unwrap();
        let created = SpineHeart::create(
            HeartConfig::new(dir.path().join("chronology.spine")),
            "synthetic-only",
        )
        .unwrap();
        let manifest = spine_heart::ModelManifest {
            schema: 1,
            model_name: "triangle-chronology-test".into(),
            artifact_hash: [1; 32],
            tokenizer_hash: [2; 32],
            dimension: 3,
            normalized: true,
            quantization: None,
        };
        created
            .heart
            .initialize_cognition(CognitiveConfig::new(1, manifest, 2).unwrap())
            .unwrap();
        let query = Embedding::normalized(vec![1., 0., 0.], 3).unwrap();
        let texts = [
            "Old docking color was amber.",
            "The current docking color is now azure; replaces the earlier color.",
        ];
        for text in texts {
            created
                .heart
                .commit_embedded(
                    InteractionInput {
                        agent_id: AgentId::new("test").unwrap(),
                        thread_id: ThreadId::new("chronology").unwrap(),
                        role: ParticipantRole::User,
                        kind: EventKind::Message,
                        content: Content::Inline(text.into()),
                        causal_parents: Vec::new(),
                        provenance: Provenance::default(),
                        tool: None,
                        attachments: Vec::new(),
                        outcome: None,
                    },
                    query.clone(),
                )
                .unwrap();
        }
        let state = created.heart.cognition().unwrap().unwrap();
        let events = created.heart.events_canonical().unwrap();
        let terms = query_terms("current docking color");
        assert!(
            segments(texts[1], &terms)[0].score > segments(texts[0], &terms)[0].score,
            "fixture must rank the correction ahead of the old source"
        );
        let (selected, report) = evidence(
            &state,
            &events,
            &BTreeSet::new(),
            &query,
            "current docking color",
        );
        assert_eq!(selected.len(), 2, "old source must remain available");
        for (index, source) in selected.iter().enumerate() {
            assert_eq!(source["event_id"], events[index].id.to_string());
            assert_eq!(source["canonical_index"], index);
            assert_eq!(
                source["device_id"],
                events[index].body.device_id.to_string()
            );
            assert_eq!(
                source["device_sequence"],
                events[index].body.device_sequence
            );
            assert_eq!(source["text"], texts[index]);
        }
        assert_eq!(
            report["presentation"],
            "canonical_event_order_then_source_offset"
        );
    }
}
