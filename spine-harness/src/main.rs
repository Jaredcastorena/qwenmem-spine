use std::{collections::{BTreeMap, BTreeSet}, fs, path::Path, time::Instant};
use serde_json::{json, Value};
use spine_heart::{AgentId,CognitiveConfig,Content,ContextHandle,ContextLeaf,Embedding,EventKind,
    HeartConfig,InteractionInput,KeySource,ParticipantRole,Provenance,RehydrateBudget,
    SemanticEncoder,SpineHeart,ThreadId};
use spine_models::{MiniLmAssets,MiniLmEncoder};
mod upstream;

fn handle(h: ContextHandle) -> String {
    match h { ContextHandle::Node(n)=>format!("node:{n}"), ContextHandle::Triangle(t)=>format!("triangle:{t}") }
}
fn coord(h:ContextHandle,s:&spine_heart::CognitiveState)->spine_heart::NodeId {
    match h {ContextHandle::Node(n)=>n,ContextHandle::Triangle(t)=>s.triangles.triangles[&t].apex}
}
fn dot(a:&[f32],b:&[f32])->f32 {a.iter().zip(b).map(|(x,y)|x*y).sum()}

fn export(heart:&SpineHeart,encoder:&MiniLmEncoder,corpus:&Value,out:&Path,stage:&Value)->Result<(),Box<dyn std::error::Error>> {
    let state=heart.cognition()?.unwrap();
    state.triangles.verify(&state.dcmdb)?;
    assert!(heart.cognition_is_current()?);
    let canonical=heart.events_canonical()?;
    let events:BTreeMap<_,_>=canonical.iter().map(|e|(e.id,e)).collect();
    let referenced=state.triangles.referenced_nodes();
    let mut nodes=serde_json::Map::new();
    for id in &referenced {
        let n=state.dcmdb.node(*id).unwrap();
        let bodies:Vec<Value>=n.event_ids.iter().filter_map(|id| events.get(id)).filter_map(|e| {
            if let Content::Inline(text)=&e.body.interaction.content {Some(json!({"event_id":e.id.to_string(),"text":text,"sequence":e.body.device_sequence}))} else {None}
        }).collect();
        let last=bodies.last().expect("source event");
        nodes.insert(format!("node:{id}"),json!({"text":last["text"],"event_id":last["event_id"],"events":bodies,"vector":n.centroid}));
    }
    let mut triangles=serde_json::Map::new();
    for (id,t) in &state.triangles.triangles {
        triangles.insert(format!("triangle:{id}"),json!({"apex":format!("node:{}",t.apex),"left":handle(t.left),"right":handle(t.right),"depth":t.depth}));
    }
    let mut queries=Vec::new();
    for q in corpus["queries"].as_array().unwrap() {
        let embedding=encoder.encode(q["question"].as_str().unwrap())?;
        let mut walks=Vec::new();
        for root in state.triangles.ranked_roots(&state.dcmdb,&embedding,None,3)? {
            let native=heart.rehydrate_context(root.handle,Some(&embedding),RehydrateBudget{max_depth:8,max_fanout:8,max_nodes:64,max_tokens:usize::MAX})?;
            let mut cursor=root.handle;
            let mut commands=Vec::new();
            for _ in 0..8 {
                let ContextHandle::Triangle(t)=cursor else {break};
                let t=&state.triangles.triangles[&t];
                let left=dot(&state.dcmdb.node(coord(t.left,&state)).unwrap().centroid,embedding.as_slice());
                let right=dot(&state.dcmdb.node(coord(t.right,&state)).unwrap().centroid,embedding.as_slice());
                let (command,next)=if left>right {("left",t.left)} else {("right",t.right)};
                commands.push(command); cursor=next;
            }
            walks.push(json!({"root":handle(root.handle),"commands":commands,
                "coordinates":native.coordinates.iter().map(|c|format!("node:{}",c.node_id)).collect::<Vec<_>>(),
                "alternates":native.alternate_routes.iter().take(8).copied().map(handle).collect::<Vec<_>>(),
                "truncated":native.truncated,"estimated_tokens":native.consumed_tokens}));
        }
        let cli=upstream::cognition_tools::rehydrate_triangle_context_for_goal(heart,&embedding,None)?;
        queries.push(json!({"id":q["id"],"question":q["question"],"walks":walks,"cli_evidence":serde_json::from_str::<Value>(&cli)?}));
    }
    let labels:Vec<_>=canonical.iter().map(|e|json!({"label":e.body.interaction.provenance.metadata.get("benchmark_id"),"event_id":e.id.to_string(),"content":e.body.interaction.content})).collect();
    let result=json!({"schema":1,"nodes":nodes,"triangles":triangles,
        "roots":state.triangles.roots.iter().map(|r|handle(r.handle)).collect::<Vec<_>>(),
        "native":{"projected_events":state.projected_events,"canonical_events":canonical.len(),
            "dcmdb_nodes":state.dcmdb.nodes.len(),"referenced_nodes":referenced.len(),
            "triangles":state.triangles.triangles.len(),"roots":state.triangles.roots.len(),
            "current":true,"stage":stage,"queries":queries,"event_labels":labels}});
    fs::write(out,serde_json::to_vec_pretty(&result)?)?;
    println!("export {}: {} events, {} nodes + {} triangles",out.display(),canonical.len(),referenced.len(),state.triangles.triangles.len());
    Ok(())
}

fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=4 {return Err("usage: qwenmem-spine-retention CORPUS OUTPUT_DIR MINILM_DIR".into())}
    let corpus:Value=serde_json::from_slice(&fs::read(&args[1])?)?;
    let out=Path::new(&args[2]);fs::create_dir_all(out)?;
    let encoder=MiniLmEncoder::load(MiniLmAssets::from_directory(&args[3]),256)?;
    let path=out.join("retention.spine");
    if path.exists() {return Err("refusing to overwrite existing test heart".into())}
    let created=SpineHeart::create(HeartConfig::new(&path),"synthetic-retention-test-only")?;
    created.heart.initialize_cognition(CognitiveConfig::new(1,encoder.manifest().clone(),4)?)?;
    let mut heart=created.heart; drop(created.recovery_phrase);
    let mut vectors:BTreeMap<String,Embedding>=BTreeMap::new();
    let all=corpus["events"].as_array().unwrap();
    let mut start=0;
    let clock=Instant::now();
    for stage in corpus["stages"].as_array().unwrap() {
        let end=stage["event_count"].as_u64().unwrap() as usize;
        for batch in all[start..end].chunks(24) {
            let missing:BTreeSet<String>=batch.iter().map(|e|e["text"].as_str().unwrap().to_owned()).filter(|s|!vectors.contains_key(s)).collect();
            let missing:Vec<_>=missing.into_iter().collect();
            for (text,embedding) in missing.iter().zip(encoder.encode_batch(&missing)?) {vectors.insert(text.clone(),embedding);}
            let inputs=batch.iter().map(|e| {
                let text=e["text"].as_str().unwrap();
                let mut metadata=BTreeMap::new();metadata.insert("benchmark_id".into(),e["id"].as_str().unwrap().into());
                (InteractionInput {agent_id:AgentId::new("retention").unwrap(),thread_id:ThreadId::new(format!("session-{}",e["session"])).unwrap(),
                    role:ParticipantRole::User,kind:EventKind::Message,content:Content::Inline(text.into()),causal_parents:Vec::new(),
                    provenance:Provenance{metadata,..Default::default()},tool:None,attachments:Vec::new(),outcome:None},vectors[text].clone())
            }).collect();
            let committed=heart.commit_embedded_batch(inputs)?;
            let leaves=committed.iter().map(|(c,m)|ContextLeaf{node_id:m.node_id,chronology:c.event.body.device_sequence});
            heart.compact_context(leaves,6)?;
        }
        // Close the encrypted store at every measured session boundary.
        let before=heart.events_canonical()?;
        drop(heart);
        heart=SpineHeart::open(HeartConfig::new(&path),KeySource::Passphrase("synthetic-retention-test-only".into()))?;
        assert_eq!(before,heart.events_canonical()?);
        export(&heart,&encoder,&corpus,&out.join(format!("{}.json",stage["name"].as_str().unwrap())),stage)?;
        println!("elapsed {:.1}s",clock.elapsed().as_secs_f64());
        start=end;
    }
    // Exercise the actual CLI transcript persistence adapter, including replay idempotency.
    let message=spine_runtime::Message::assistant("Persistence sentinel: transcript replay must not duplicate this result.",None,Vec::new());
    let mut message=message;
    message.record_id=Some("retention-transcript-sentinel".into());
    let first=upstream::transcript::persist(&heart,&encoder,&AgentId::new("retention")?,&ThreadId::new("adapter-check")?,&[message.clone()])?;
    let second=upstream::transcript::persist(&heart,&encoder,&AgentId::new("retention")?,&ThreadId::new("adapter-check")?,&[message])?;
    assert_eq!(first.len(),1);assert!(second.is_empty());
    heart.compact_context(first,6)?;
    heart.maintain_cognition(1)?;
    let state=heart.cognition()?.unwrap();state.triangles.verify(&state.dcmdb)?;
    fs::write(out.join("integration-checks.json"),serde_json::to_vec_pretty(&json!({"encrypted_reopen_exact":true,"cognition_current":heart.cognition_is_current()?,"transcript_idempotent":true,"maintenance_preserves_forest":true,"native_encoder":encoder.manifest(),"seconds":clock.elapsed().as_secs_f64()}))?)?;
    Ok(())
}
