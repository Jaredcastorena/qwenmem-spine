//! Bounded, native-selected topology for replay by the compiled model prefix.
//! Selection and source rendering stay in cognition_tools; this module never searches sources.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};
use spine_heart::{CognitiveState, ContextHandle, Embedding, NodeId, ResolvedCoordinate};
use spine_runtime::{Result, RuntimeError};

pub const MARKER: &str = "<spine-compiled-memory-context/>";

fn error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Provider(message.into())
}

fn address(handle: ContextHandle) -> String {
    match handle {
        ContextHandle::Node(id) => format!("node:{id}"),
        ContextHandle::Triangle(id) => format!("triangle:{id}"),
    }
}

fn coordinate(state: &CognitiveState, handle: ContextHandle) -> Result<NodeId> {
    match handle {
        ContextHandle::Node(id) => Ok(id),
        ContextHandle::Triangle(id) => state
            .triangles
            .triangles
            .get(&id)
            .map(|triangle| triangle.apex)
            .ok_or_else(|| error("Missing native triangle")),
    }
}

fn collect(
    state: &CognitiveState,
    handle: ContextHandle,
    visited: &mut BTreeSet<ContextHandle>,
    nodes: &mut BTreeSet<NodeId>,
    triangles: &mut BTreeMap<String, Value>,
) -> Result<()> {
    if !visited.insert(handle) {
        return Ok(());
    }
    match handle {
        ContextHandle::Node(id) => {
            nodes.insert(id);
        }
        ContextHandle::Triangle(id) => {
            let triangle = state
                .triangles
                .triangles
                .get(&id)
                .ok_or_else(|| error("Missing selected native triangle"))?;
            if triangle.depth > 8 {
                return Err(error(
                    "Selected triangle closure exceeds compiled depth 8; no native fallback",
                ));
            }
            nodes.insert(triangle.apex);
            triangles.insert(address(handle), json!({
                "apex":format!("node:{}", triangle.apex),
                "left":address(triangle.left),"right":address(triangle.right),"depth":triangle.depth
            }));
            collect(state, triangle.left, visited, nodes, triangles)?;
            collect(state, triangle.right, visited, nodes, triangles)?;
        }
    }
    if nodes.len() + triangles.len() > 64 {
        return Err(error(
            "Selected triangle closure exceeds 64 resident records; no native fallback",
        ));
    }
    Ok(())
}

pub fn bank(
    state: &CognitiveState,
    root: ContextHandle,
    query: &Embedding,
    selected: &[ResolvedCoordinate],
) -> Result<Option<Value>> {
    if selected.is_empty() {
        return Ok(None);
    }
    state.triangles.verify(&state.dcmdb)?;
    let mut nodes = BTreeSet::new();
    let mut triangles = BTreeMap::new();
    collect(
        state,
        root,
        &mut BTreeSet::new(),
        &mut nodes,
        &mut triangles,
    )?;
    let mut exported = BTreeMap::new();
    for id in nodes {
        let node = state
            .dcmdb
            .node(id)
            .ok_or_else(|| error("Missing native coordinate"))?;
        let event = node
            .event_ids
            .first()
            .ok_or_else(|| error("Native coordinate has no provenance"))?;
        // No source bodies are installed or exposed through the topology packet.
        exported.insert(
            format!("node:{id}"),
            json!({"text":"", "event_id":event.to_string(),"vector":node.centroid}),
        );
    }
    let addresses: Vec<_> = exported.keys().chain(triangles.keys()).cloned().collect();
    let root_slot = addresses
        .iter()
        .position(|item| item == &address(root))
        .ok_or_else(|| error("Selected root missing from compiled bank"))?;
    let expected: Vec<String> = selected
        .iter()
        .map(|item| format!("node:{}", item.node_id))
        .collect();
    let allowed: BTreeSet<_> = expected.iter().cloned().collect();
    let first = format!("node:{}", coordinate(state, root)?);
    if expected.first() != Some(&first) {
        return Err(error("Native coordinate order disagrees with root"));
    }
    let mut emitted = vec![first];
    let mut commands = vec![format!("open:{root_slot}")];
    let mut cursor = root;
    for _ in 0..8 {
        let ContextHandle::Triangle(id) = cursor else {
            break;
        };
        let triangle = &state.triangles.triangles[&id];
        let score = |handle| -> Result<f32> {
            let id = coordinate(state, handle)?;
            let node = state
                .dcmdb
                .node(id)
                .ok_or_else(|| error("Missing branch coordinate"))?;
            Ok(node
                .centroid
                .iter()
                .zip(query.as_slice())
                .map(|(a, b)| a * b)
                .sum())
        };
        let (command, next) = if score(triangle.left)? > score(triangle.right)? {
            ("left", triangle.left)
        } else {
            ("right", triangle.right)
        };
        let apex = format!("node:{}", coordinate(state, next)?);
        // Native rehydration already applied token/node/depth limits. Never release
        // an additional coordinate just because the resident closure contains it.
        if !allowed.contains(&apex) {
            break;
        }
        commands.push(command.to_owned());
        if !emitted.contains(&apex) {
            emitted.push(apex);
        }
        cursor = next;
    }
    if emitted != expected {
        return Err(error(
            "Bounded compiled route differs from native selected coordinates",
        ));
    }
    Ok(Some(json!({
        "forest":{"schema":1,"nodes":exported,"triangles":triangles,"roots":[address(root)]},
        "commands":commands,"coordinates":expected
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use spine_heart::{
        AgentId, CognitiveConfig, Content, ContextBranch, ContextLeaf, EventId, EventKind,
        HeartConfig, InteractionInput, MemoryObservation, ModelManifest, ParticipantRole,
        Provenance, RehydrateBudget, SemanticEncoder, SpineHeart, ThreadId,
    };

    fn manifest(dimension: usize) -> ModelManifest {
        ModelManifest {
            schema: 1,
            model_name: "native-triangle-handoff-test".into(),
            artifact_hash: [41; 32],
            tokenizer_hash: [42; 32],
            dimension,
            normalized: true,
            quantization: None,
        }
    }

    struct Fixture {
        state: CognitiveState,
        root: ContextHandle,
        leaves: Vec<NodeId>,
        unrelated: NodeId,
    }

    // Every triangle is created by native compaction. Pairwise compaction gives
    // capacity tests a shallow tree without altering native IDs or branch data.
    fn fixture(leaf_count: usize, balanced: bool) -> Fixture {
        let dimension = leaf_count + 2;
        let mut state =
            CognitiveState::new(CognitiveConfig::new(1, manifest(dimension), 2).unwrap()).unwrap();
        state.dcmdb.config.theta_similarity = 0.9999;
        let mut leaves = Vec::new();
        for index in 0..leaf_count {
            let mut vector = vec![0.0; dimension];
            vector[0] = 1.0;
            vector[index + 1] = 0.3;
            let node_id = state
                .dcmdb
                .update(MemoryObservation {
                    event_id: EventId::from_bytes([index as u8 + 1; 32]),
                    vector,
                    time: index as f64,
                    source: None,
                    metadata: BTreeMap::from([
                        ("token_count".into(), "40".into()),
                        ("source_body".into(), "PRIVATE-SOURCE-BODY-Ω".into()),
                    ]),
                })
                .unwrap();
            leaves.push(ContextLeaf {
                node_id,
                chronology: index as u64 + 1,
            });
        }
        assert_eq!(state.dcmdb.nodes.len(), leaf_count);
        let mut vector = vec![0.0; dimension];
        vector[dimension - 1] = 1.0;
        let unrelated = state
            .dcmdb
            .update(MemoryObservation {
                event_id: EventId::from_bytes([250; 32]),
                vector,
                time: leaf_count as f64,
                source: None,
                metadata: BTreeMap::new(),
            })
            .unwrap();
        if balanced {
            let mut level: Vec<_> = leaves.iter().copied().map(ContextBranch::leaf).collect();
            while level.len() > 1 {
                let mut next = Vec::new();
                for pair in level.chunks(2) {
                    state.triangles.roots = pair.to_vec();
                    state.triangles.compact([], &state.dcmdb, 1).unwrap();
                    assert_eq!(state.triangles.roots.len(), 1);
                    next.push(state.triangles.roots[0]);
                }
                level = next;
            }
            state.triangles.roots = level;
        } else {
            for leaf in &leaves {
                state.triangles.compact([*leaf], &state.dcmdb, 1).unwrap();
            }
        }
        assert_eq!(state.triangles.roots.len(), 1);
        let root = state.triangles.roots[0].handle;
        state.triangles.roots.push(ContextBranch::leaf(ContextLeaf {
            node_id: unrelated,
            chronology: leaf_count as u64 + 1,
        }));
        state.triangles.verify(&state.dcmdb).unwrap();
        Fixture {
            state,
            root,
            leaves: leaves.into_iter().map(|leaf| leaf.node_id).collect(),
            unrelated,
        }
    }

    fn query_for(fixture: &Fixture, node: NodeId) -> Embedding {
        Embedding::normalized(
            fixture.state.dcmdb.node(node).unwrap().centroid.clone(),
            fixture.state.config.model.dimension,
        )
        .unwrap()
    }

    fn native_budget() -> RehydrateBudget {
        RehydrateBudget {
            max_depth: 8,
            max_fanout: 3,
            max_nodes: 16,
            max_tokens: 1024,
        }
    }

    fn native_coordinates(
        fixture: &Fixture,
        query: &Embedding,
        budget: RehydrateBudget,
    ) -> Vec<ResolvedCoordinate> {
        fixture
            .state
            .triangles
            .rehydrate(fixture.root, Some(query), &fixture.state.dcmdb, budget)
            .unwrap()
            .coordinates
    }

    #[test]
    fn compiled_bank_keeps_complete_native_closure_without_source_bodies() {
        let fixture = fixture(4, true);
        let before = postcard::to_allocvec(&fixture.state).unwrap();
        let query = query_for(&fixture, fixture.leaves[3]);
        let selected = native_coordinates(&fixture, &query, native_budget());
        let packet = bank(&fixture.state, fixture.root, &query, &selected)
            .unwrap()
            .unwrap();
        let forest = &packet["forest"];
        let nodes = forest["nodes"].as_object().unwrap();
        let triangles = forest["triangles"].as_object().unwrap();
        assert_eq!(nodes.len(), 4);
        assert_eq!(triangles.len(), 3, "native nested triangles must survive");
        assert_eq!(forest["roots"], json!([address(fixture.root)]));
        assert!(!nodes.contains_key(&format!("node:{}", fixture.unrelated)));
        for id in &fixture.leaves {
            let node = &nodes[&format!("node:{id}")];
            let native = fixture.state.dcmdb.node(*id).unwrap();
            assert_eq!(node["text"], "");
            assert_eq!(node["event_id"], native.event_ids[0].to_string());
            assert_eq!(node["vector"], json!(native.centroid));
            assert_eq!(node.as_object().unwrap().len(), 3);
        }
        for triangle in fixture.state.triangles.triangles.values() {
            let exported = &triangles[&format!("triangle:{}", triangle.id)];
            assert_eq!(exported["apex"], format!("node:{}", triangle.apex));
            assert_eq!(exported["left"], address(triangle.left));
            assert_eq!(exported["right"], address(triangle.right));
            assert_eq!(exported["depth"], triangle.depth);
            for side in ["left", "right"] {
                let branch = exported[side].as_str().unwrap();
                assert!(nodes.contains_key(branch) || triangles.contains_key(branch));
            }
        }
        assert!(!packet.to_string().contains("PRIVATE-SOURCE-BODY"));
        assert_eq!(postcard::to_allocvec(&fixture.state).unwrap(), before);
    }

    #[test]
    fn compiled_bank_preserves_native_triangle_after_centroid_drift() {
        let mut fixture = fixture(2, true);
        let ContextHandle::Triangle(root_id) = fixture.root else {
            panic!("fixture must contain a genuinely compacted native triangle");
        };
        let original = fixture.state.triangles.triangles[&root_id].clone();
        let changed = fixture
            .leaves
            .iter()
            .copied()
            .find(|id| *id != original.apex)
            .unwrap();
        let apex = fixture
            .state
            .dcmdb
            .node(original.apex)
            .unwrap()
            .centroid
            .clone();
        let dot = |vector: &[f32]| apex.iter().zip(vector).map(|(a, b)| a * b).sum::<f32>();
        assert!(dot(&fixture.state.dcmdb.node(changed).unwrap().centroid) >= 0.8);

        // Model a later persisted centroid update without compacting again. Native
        // coherence controls triangle formation; verify() preserves established
        // identity and branches when their live coordinates subsequently change.
        let mut updated = vec![0.0; fixture.state.config.model.dimension];
        updated[fixture.state.config.model.dimension - 1] = 1.0;
        let node = fixture.state.dcmdb.nodes.get_mut(&changed).unwrap();
        node.centroid = updated.clone();
        node.sufficient_sum = updated
            .iter()
            .map(|value| value * node.effective_count)
            .collect();
        assert!(dot(&node.centroid) < 0.8);
        fixture
            .state
            .triangles
            .verify(&fixture.state.dcmdb)
            .unwrap();
        assert_eq!(fixture.state.triangles.triangles[&root_id], original);
        let before = postcard::to_allocvec(&fixture.state).unwrap();

        let query = query_for(&fixture, changed);
        let selected = native_coordinates(&fixture, &query, native_budget());
        assert_eq!(
            selected
                .iter()
                .map(|coordinate| coordinate.node_id)
                .collect::<Vec<_>>(),
            vec![original.apex, changed]
        );
        let packet = bank(&fixture.state, fixture.root, &query, &selected)
            .unwrap()
            .expect("structurally valid native triangle must remain exportable after drift");
        let forest = &packet["forest"];
        assert_eq!(
            forest["nodes"][format!("node:{changed}")]["vector"],
            json!(updated)
        );
        assert_eq!(
            forest["nodes"][format!("node:{}", original.apex)]["vector"],
            json!(apex)
        );
        assert_eq!(forest["roots"], json!([format!("triangle:{root_id}")]));
        assert_eq!(forest["triangles"].as_object().unwrap().len(), 1);
        assert_eq!(
            forest["triangles"][format!("triangle:{root_id}")],
            json!({
                "apex": format!("node:{}", original.apex),
                "left": address(original.left),
                "right": address(original.right),
                "depth": original.depth,
            })
        );
        assert_eq!(packet["commands"].as_array().unwrap().len(), 2);
        assert_eq!(postcard::to_allocvec(&fixture.state).unwrap(), before);
    }

    #[test]
    fn compiled_bank_matches_native_coordinate_order_and_stops_at_native_budgets() {
        let fixture = fixture(2, true);
        let apex = coordinate(&fixture.state, fixture.root).unwrap();
        let preferred = fixture
            .leaves
            .iter()
            .copied()
            .find(|id| *id != apex)
            .unwrap();
        let query = query_for(&fixture, preferred);
        let selected = native_coordinates(&fixture, &query, native_budget());
        assert_eq!(selected.len(), 2, "fixture must traverse beyond the apex");
        let packet = bank(&fixture.state, fixture.root, &query, &selected)
            .unwrap()
            .unwrap();
        assert_eq!(
            packet["coordinates"],
            json!([format!("node:{apex}"), format!("node:{preferred}")])
        );
        assert_eq!(packet["commands"].as_array().unwrap().len(), 2);
        assert!(matches!(
            packet["commands"][1].as_str(),
            Some("left" | "right")
        ));
        for budget in [
            RehydrateBudget {
                max_depth: 0,
                ..native_budget()
            },
            RehydrateBudget {
                max_nodes: 1,
                ..native_budget()
            },
            RehydrateBudget {
                max_tokens: 40,
                ..native_budget()
            },
        ] {
            let selected = native_coordinates(&fixture, &query, budget);
            assert_eq!(selected.len(), 1);
            let packet = bank(&fixture.state, fixture.root, &query, &selected)
                .unwrap()
                .unwrap();
            assert_eq!(packet["coordinates"], json!([format!("node:{apex}")]));
            assert_eq!(packet["commands"].as_array().unwrap().len(), 1);
            assert_eq!(
                packet["forest"]["nodes"].as_object().unwrap().len(),
                2,
                "the complete closure stays resident while extra evidence stays unreleased"
            );
        }
        let empty = native_coordinates(
            &fixture,
            &query,
            RehydrateBudget {
                max_tokens: 39,
                ..native_budget()
            },
        );
        assert!(empty.is_empty());
        assert!(
            bank(&fixture.state, fixture.root, &query, &empty)
                .unwrap()
                .is_none()
        );
        let mut reversed = selected;
        reversed.reverse();
        assert!(bank(&fixture.state, fixture.root, &query, &reversed).is_err());
    }

    #[test]
    fn compiled_bank_rejects_oversized_closures_even_when_selected_path_is_small() {
        let fitting = fixture(32, true);
        let query = query_for(&fitting, fitting.leaves[31]);
        let selected = native_coordinates(&fitting, &query, native_budget());
        let packet = bank(&fitting.state, fitting.root, &query, &selected)
            .unwrap()
            .unwrap();
        assert_eq!(
            packet["forest"]["nodes"].as_object().unwrap().len()
                + packet["forest"]["triangles"].as_object().unwrap().len(),
            63
        );
        let oversized = fixture(33, true);
        assert!(
            oversized
                .state
                .triangles
                .triangles
                .values()
                .all(|triangle| triangle.depth <= 8)
        );
        let query = query_for(&oversized, oversized.leaves[32]);
        let selected = native_coordinates(&oversized, &query, native_budget());
        assert!(
            selected.len() <= 8,
            "a short route must not bypass the resident bound"
        );
        let error = bank(&oversized.state, oversized.root, &query, &selected)
            .unwrap_err()
            .to_string();
        assert!(error.contains("64 resident records"), "{error}");
        assert!(error.contains("no native fallback"), "{error}");
    }

    #[test]
    fn compiled_bank_rejects_native_depth_nine_without_flattening_it() {
        for (leaf_count, succeeds) in [(9, true), (10, false)] {
            let fixture = fixture(leaf_count, false);
            let ContextHandle::Triangle(root_id) = fixture.root else {
                panic!("fixture must have a native triangle root");
            };
            assert_eq!(
                fixture.state.triangles.triangles[&root_id].depth,
                leaf_count as u32 - 1
            );
            assert!(fixture.leaves.len() + fixture.state.triangles.triangles.len() < 64);
            let query = query_for(&fixture, fixture.leaves[leaf_count - 1]);
            let selected = native_coordinates(&fixture, &query, native_budget());
            let result = bank(&fixture.state, fixture.root, &query, &selected);
            if succeeds {
                assert!(result.unwrap().is_some());
            } else {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("compiled depth 8"), "{error}");
                assert!(error.contains("no native fallback"), "{error}");
            }
        }
    }

    struct FixedEncoder {
        manifest: ModelManifest,
    }

    impl SemanticEncoder for FixedEncoder {
        fn manifest(&self) -> &ModelManifest {
            &self.manifest
        }

        fn encode(&self, _: &str) -> spine_heart::Result<Embedding> {
            Embedding::normalized(vec![1.0, 0.0, 0.0], 3)
        }
    }

    #[test]
    fn compiled_handoff_preserves_native_renderer_and_separates_evidence_from_topology() {
        let directory = tempfile::tempdir().unwrap();
        let heart = SpineHeart::create(
            HeartConfig::new(directory.path().join("renderer.spine")),
            "synthetic-only",
        )
        .unwrap()
        .heart;
        let encoder = FixedEncoder {
            manifest: manifest(3),
        };
        heart
            .initialize_cognition(CognitiveConfig::new(1, encoder.manifest.clone(), 2).unwrap())
            .unwrap();
        let thread = ThreadId::new("native-renderer").unwrap();
        let mut leaves = Vec::new();
        for index in 0..12 {
            let (commit, memory) = heart
                .commit_embedded(
                    InteractionInput {
                        agent_id: AgentId::new("main").unwrap(),
                        thread_id: thread.clone(),
                        role: ParticipantRole::User,
                        kind: EventKind::Message,
                        content: Content::Inline(format!(
                            "native evidence {index}: {}",
                            "Ω".repeat(1100)
                        )),
                        causal_parents: Vec::new(),
                        provenance: Provenance::default(),
                        tool: None,
                        attachments: Vec::new(),
                        outcome: None,
                    },
                    Embedding::normalized(
                        if index % 2 == 0 {
                            vec![1.0, 0.0, 0.0]
                        } else {
                            vec![0.82, 0.57, 0.0]
                        },
                        3,
                    )
                    .unwrap(),
                )
                .unwrap();
            if index < 2 {
                leaves.push(ContextLeaf {
                    node_id: memory.node_id,
                    chronology: commit.event.body.device_sequence,
                });
            }
        }
        heart.compact_context(leaves, 1).unwrap();
        let state = heart.cognition().unwrap().unwrap();
        assert!(
            !state.triangles.triangles.is_empty(),
            "fixture must exercise a real native triangle"
        );
        let before = postcard::to_allocvec(&state).unwrap();
        let query = encoder.encode("current request").unwrap();
        let legacy = crate::cognition_tools::rehydrate_triangle_context_for_goal(
            &heart,
            &query,
            Some(&query),
        )
        .unwrap();
        let (plain, no_banks) = crate::cognition_tools::rehydrate_triangle_context_handoff(
            &heart,
            &query,
            Some(&query),
            false,
        )
        .unwrap();
        let (compiled, banks) = crate::cognition_tools::rehydrate_triangle_context_handoff(
            &heart,
            &query,
            Some(&query),
            true,
        )
        .unwrap();
        assert_eq!(legacy, plain);
        assert_eq!(plain, compiled);
        assert!(no_banks.is_empty());
        assert!(
            banks
                .iter()
                .any(|bank| !bank["forest"]["triangles"].as_object().unwrap().is_empty())
        );
        let sources: Vec<Value> = serde_json::from_str(&compiled).unwrap();
        assert!(!sources.is_empty());
        assert!(sources.len() <= 12);
        let authorized: BTreeSet<_> = banks
            .iter()
            .flat_map(|bank| {
                bank["coordinates"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|coordinate| coordinate.as_str().unwrap())
            })
            .collect();
        for source in &sources {
            assert!(
                authorized
                    .contains(format!("node:{}", source["node_id"].as_str().unwrap()).as_str())
            );
            assert!(source["text"].as_str().unwrap().chars().count() <= 1012);
        }
        let (ordinary, absent) =
            crate::context_state::rehydrate_for_provider(&heart, &encoder, &thread, &query, false)
                .unwrap();
        let (internal, packet) =
            crate::context_state::rehydrate_for_provider(&heart, &encoder, &thread, &query, true)
                .unwrap();
        assert_eq!(ordinary, internal);
        assert_eq!(
            ordinary,
            crate::context_state::rehydrate(&heart, &encoder, &thread, &query).unwrap()
        );
        assert!(absent.is_none());
        let packet = packet.unwrap();
        assert!(packet["context"].get("rehydrated").is_none());
        let native: Value = serde_json::from_str(&ordinary).unwrap();
        assert_eq!(packet["sources"], native["rehydrated"]);
        assert!(!packet["banks"].to_string().contains("native evidence"));
        assert_eq!(
            postcard::to_allocvec(&heart.cognition().unwrap().unwrap()).unwrap(),
            before
        );
    }
}
