//! Controlled geometry stress test; synthetic vectors, actual native compaction.
use std::collections::BTreeMap;
use spine_heart::{ContextForest,ContextHandle,ContextLeaf,Dcmdb,DcmdbConfig,Embedding,EventId,MemoryObservation,RehydrateBudget};
use serde_json::json;
fn address(h:ContextHandle)->String {match h {ContextHandle::Node(n)=>format!("node:{n}"),ContextHandle::Triangle(t)=>format!("triangle:{t}")}}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let mut cfg=DcmdbConfig::dense(6);cfg.theta_similarity=0.999999;cfg.tension_promote_count=99;
    let mut db=Dcmdb::new(cfg)?;let mut leaves=Vec::new();let mut vectors=Vec::new();
    for i in 0..32 {
        let mut v=vec![1.0];for (bit,w) in [0.03,0.05,0.07,0.11,0.16].iter().enumerate() {v.push(if (i>>bit)&1==1 {*w} else {-*w});}
        let id=db.update(MemoryObservation{event_id:EventId::from_bytes([i as u8;32]),vector:v.clone(),time:i as f64+1.0,source:None,metadata:BTreeMap::new()})?;
        leaves.push(ContextLeaf{node_id:id,chronology:i as u64+1});vectors.push(v);
    }
    let mut forest=ContextForest::default();forest.compact(leaves,&db,1)?;forest.verify(&db)?;
    let nodes:BTreeMap<_,_>=db.nodes.iter().map(|(id,n)|(format!("node:{id}"),json!({"text":format!("Controlled geometry source {id}"),"event_id":n.event_ids[0].to_string(),"vector":n.centroid}))).collect();
    let triangles:BTreeMap<_,_>=forest.triangles.iter().map(|(id,t)|(format!("triangle:{id}"),json!({"apex":format!("node:{}",t.apex),"left":address(t.left),"right":address(t.right)}))).collect();
    let root=forest.roots[0].handle;let mut queries=Vec::new();
    for v in vectors {
        let q=Embedding::normalized(v,6)?;
        let native=forest.rehydrate(root,Some(&q),&db,RehydrateBudget{max_depth:8,max_fanout:8,max_nodes:64,max_tokens:4096})?;
        let mut cursor=root;let mut commands=Vec::new();
        for _ in 0..8 {
            let ContextHandle::Triangle(id)=cursor else {break};let t=&forest.triangles[&id];
            let score=|h| {let n=match h {ContextHandle::Node(n)=>n,ContextHandle::Triangle(id)=>forest.triangles[&id].apex};db.nodes[&n].centroid.iter().zip(q.as_slice()).map(|(a,b)|a*b).sum::<f32>()};
            if score(t.left)>score(t.right) {commands.push("left");cursor=t.left;} else {commands.push("right");cursor=t.right;}
        }
        queries.push(json!({"commands":commands,"coordinates":native.coordinates.iter().map(|c|format!("node:{}",c.node_id)).collect::<Vec<_>>(),"alternates":native.alternate_routes.iter().take(8).copied().map(address).collect::<Vec<_>>(),"truncated":native.truncated}));
    }
    println!("{}",serde_json::to_string_pretty(&json!({"schema":1,"nodes":nodes,"triangles":triangles,"roots":[address(root)],"geometry":{"max_depth":forest.triangles.values().map(|t|t.depth).max(),"queries":queries,"description":"Controlled vectors, actual Rust ContextForest compaction; not learned semantic embeddings"}}))?);
    Ok(())
}
