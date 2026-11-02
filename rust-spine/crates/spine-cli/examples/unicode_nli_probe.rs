//! Diagnostic only: replay saved evidence through the real native grounding gate.
//! No assistant generation, heart writes, altered extractor, or altered NLI weights.
use serde_json::{Value, json};
use spine_heart::{AtomicClaim, ClaimExtractor, ClaimRelation, NliReport};
use std::{fs, path::Path};

#[allow(dead_code)]
mod original_gate {
    // Reuse the exact application gate, including its private repair predicate.
    include!("../src/grounding.rs");

    pub fn forced(
        gate: &GroundingGate,
        claims: &[spine_heart::AtomicClaim],
        evidence: &[String],
        threshold: f32,
    ) -> spine_heart::Result<GroundingDecision> {
        let report = gate.verifier.verify(claims, evidence)?;
        Ok(GroundingDecision {
            claim_count: claims.len(),
            needs_repair: needs_repair(claims.len(), &report, threshold),
            report,
        })
    }
}

fn cell(
    gate: &original_gate::GroundingGate,
    id: &str,
    answer: &str,
    evidence: &[String],
    forced: Option<&AtomicClaim>,
    threshold: f32,
) -> spine_heart::Result<Value> {
    let claims = forced.map_or_else(
        || ClaimExtractor::new().map(|extractor| extractor.extract(answer)),
        |claim| Ok(vec![claim.clone()]),
    )?;
    let decision = if forced.is_some() {
        original_gate::forced(gate, &claims, evidence, threshold)?
    } else {
        gate.verify(answer, evidence, threshold)?
    };
    assert_eq!(claims.len(), decision.claim_count);
    let NliReport {
        evaluated_pairs,
        coverage,
        contradiction,
        ..
    } = decision.report;
    Ok(json!({
        "id":id, "claim_mode":if forced.is_some(){"forced_explicit"}else{"original_extractor"},
        "claims":claims, "claim_count":decision.claim_count,
        "hypotheses":claims.iter().map(AtomicClaim::hypothesis).collect::<Vec<_>>(),
        "evidence":evidence, "evidence_chars":evidence.iter().map(|s|s.chars().count()).collect::<Vec<_>>(),
        "report":decision.report, "nli_evaluated":evaluated_pairs > 0,
        // Native empty reports contain zeroes; null distinguishes no evaluation
        // from an actual model probability of zero.
        "nli_entailment":(evaluated_pairs > 0).then_some(coverage),
        "nli_contradiction":(evaluated_pairs > 0).then_some(contradiction),
        "coverage_threshold":threshold, "repair_triggered":decision.needs_repair,
        "repair_generation_run":false,
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: unicode_nli_probe MODEL_DIR INPUT_JSON OUTPUT_JSON".into());
    }
    let input: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let answer = input["answer"].as_str().ok_or("missing answer")?;
    let threshold = input["coverage_threshold"]
        .as_f64()
        .ok_or("missing threshold")? as f32;
    let excerpt = vec![
        input["source_excerpt"]
            .as_str()
            .ok_or("missing excerpt")?
            .to_owned(),
    ];
    let complete = vec![
        input["source_full"]
            .as_str()
            .ok_or("missing source")?
            .to_owned(),
    ];
    let selected: Vec<String> = serde_json::from_value(input["actual_grounding_evidence"].clone())?;
    let forced = AtomicClaim {
        subject: "The exact Zircon recovery label".into(),
        relation: ClaimRelation::Equals,
        value: input["asserted_literal"]
            .as_str()
            .ok_or("missing asserted literal")?
            .into(),
        time: None,
        excerpt: answer.into(),
    };
    let gate = original_gate::GroundingGate::load(Path::new(&args[1]))?;
    let rows = vec![
        cell(&gate, "A", answer, &excerpt, None, threshold)?,
        cell(&gate, "B", answer, &complete, None, threshold)?,
        cell(&gate, "C", answer, &excerpt, Some(&forced), threshold)?,
        cell(&gate, "D", answer, &complete, Some(&forced), threshold)?,
    ];
    let supplemental = vec![
        cell(&gate, "actual_original", answer, &selected, None, threshold)?,
        cell(
            &gate,
            "actual_forced",
            answer,
            &selected,
            Some(&forced),
            threshold,
        )?,
        cell(
            &gate,
            "correct_literal_control",
            answer,
            &complete,
            Some(&AtomicClaim {
                value: input["expected_literal"]
                    .as_str()
                    .ok_or("missing expected literal")?
                    .into(),
                ..forced.clone()
            }),
            threshold,
        )?,
    ];
    let result = json!({"schema":1, "input":input,
        "pair_order":["hypothesis", "evidence"], "native_evidence_top_m":3,
        "native_tokenizer_pair_limit":512, "rows":rows,"supplemental":supplemental});
    fs::write(&args[3], serde_json::to_string_pretty(&result)? + "\n")?;
    for row in result["rows"]
        .as_array()
        .unwrap()
        .iter()
        .chain(result["supplemental"].as_array().unwrap())
    {
        println!(
            "{}",
            json!({"id":row["id"],"claims":row["claim_count"],
            "pairs":row["report"]["evaluated_pairs"],"entailment":row["nli_entailment"],
            "contradiction":row["nli_contradiction"],"repair":row["repair_triggered"]})
        );
    }
    Ok(())
}
