//! Quantify compact working handles separately from retained recursive storage.
use spine_heart::{
    ContextForest, ContextHandle, ContextLeaf, ContextTriangle, Dcmdb, DcmdbConfig, EventId,
    MemoryObservation, RehydrateBudget,
};
use std::collections::{BTreeMap, BTreeSet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 1.0;
    config.tension_promote_count = u32::MAX;
    let mut memory = Dcmdb::new(config)?;
    let mut forest = ContextForest::default();
    let mut original_nodes = BTreeSet::new();
    let mut measurements = Vec::new();
    for i in 0..96_u8 {
        let angle = f32::from(i) * 0.005;
        let node = memory.update(MemoryObservation {
            event_id: EventId::from_bytes([i + 1; 32]),
            vector: vec![angle.cos(), angle.sin(), 0.0],
            time: f64::from(i + 1),
            source: None,
            metadata: BTreeMap::new(),
        })?;
        assert!(
            original_nodes.insert(node),
            "fixture requires distinct coordinates"
        );
        forest.compact(
            [ContextLeaf {
                node_id: node,
                chronology: u64::from(i + 1),
            }],
            &memory,
            1,
        )?;
        assert_eq!(forest.roots.len(), 1);
        assert_eq!(forest.referenced_nodes(), original_nodes);
        if [1, 2, 3, 6, 12, 24, 48, 96].contains(&(i + 1)) {
            let root = forest.roots[0].handle;
            let depth = match root {
                ContextHandle::Node(_) => 0,
                ContextHandle::Triangle(id) => forest.triangles[&id].depth,
            };
            let bytes = postcard::to_allocvec(&forest)?;
            let reopened: ContextForest = postcard::from_bytes(&bytes)?;
            assert_eq!(reopened.referenced_nodes(), original_nodes);
            assert_eq!(reopened, forest);
            let bounded = forest.rehydrate(root, None, &memory, RehydrateBudget::default())?;
            measurements.push(serde_json::json!({"source_nodes":i+1,"live_roots":forest.roots.len(),
                "root_handle_bytes":postcard::to_allocvec(&root)?.len(),
                "triangle_struct_bytes":std::mem::size_of::<ContextTriangle>(),
                "stored_triangles":forest.triangles.len(),"reachable_nodes":forest.referenced_nodes().len(),
                "flat_adapter_records":forest.triangles.len()+forest.referenced_nodes().len(),
                "stored_forest_bytes":bytes.len(),"depth":depth,
                "eight_step_read_truncated":bounded.truncated,"read_coordinates":bounded.coordinates.len()}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"schema":1,
        "fixture":"distinct coherent coordinates; incremental compaction to one root; no semantic recall claim",
        "all_sources_reachable_after_roundtrip":true,"measurements":measurements}))?
    );
    Ok(())
}
