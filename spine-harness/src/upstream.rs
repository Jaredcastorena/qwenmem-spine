#[allow(dead_code)]
#[path = "../../rust-spine/crates/spine-cli/src/cognition_tools.rs"]
pub mod cognition_tools;
#[allow(dead_code)]
#[path = "../../rust-spine/crates/spine-cli/src/transcript.rs"]
pub mod transcript;
#[cfg(test)]
#[path = "../../rust-spine/crates/spine-heart/tests/triangle_compaction.rs"]
mod triangle_compaction;
#[cfg(test)]
#[path = "../../rust-spine/crates/spine-heart/tests/projection_roundtrip.rs"]
mod projection_roundtrip;
#[cfg(test)]
#[path = "../../rust-spine/crates/spine-heart/tests/dcmdb_maintenance.rs"]
mod dcmdb_maintenance;
