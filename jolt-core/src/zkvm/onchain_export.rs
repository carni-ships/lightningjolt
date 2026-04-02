use crate::field::JoltField;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::subprotocols::sumcheck::ClearSumcheckProof;
use crate::transcripts::Transcript;

#[cfg(not(feature = "zk"))]
use crate::poly::commitment::dory::{ArkFr, ArkG1, ArkG2, ArkGT, ArkworksVerifierSetup, BN254};
#[cfg(not(feature = "zk"))]
use dory::DoryWitnessData;
/// Per-stage instance configuration for on-chain verification.
#[cfg(not(feature = "zk"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct StageInstanceConfig {
    /// Number of sumcheck rounds per instance.
    pub num_rounds: Vec<usize>,
    /// Maximum polynomial degree across instances.
    pub max_degree: usize,
}

/// Read-write checking configuration.
#[cfg(not(feature = "zk"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct RwConfigExport {
    pub ram_rw_phase1_num_rounds: u8,
    pub ram_rw_phase2_num_rounds: u8,
    pub registers_rw_phase1_num_rounds: u8,
    pub registers_rw_phase2_num_rounds: u8,
}

/// One-hot params configuration.
#[cfg(not(feature = "zk"))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct OneHotConfigExport {
    pub log_k_chunk: usize,
    pub lookups_ra_virtual_log_k_chunk: usize,
    pub k_chunk: usize,
    pub ram_k: usize,
    pub bytecode_k: usize,
    pub instruction_d: usize,
    pub bytecode_d: usize,
    pub ram_d: usize,
}

/// Data captured during verification for on-chain export.
/// Returned by `JoltVerifier::verify_for_export()`.
#[cfg(not(feature = "zk"))]
#[derive(Debug, Clone)]
pub struct OnChainExportData {
    /// Per-stage sumcheck challenge vectors (hex-encoded).
    pub stage_challenges: Vec<Vec<String>>,
    /// UniSkip polynomial coefficients for stages 1 and 2 (hex-encoded).
    pub uniskip_polys: Vec<Vec<String>>,
    /// UniSkip challenges for stages 1 and 2 (hex-encoded).
    pub uniskip_challenges: Vec<String>,
    /// Per-stage compressed polynomial coefficients (hex-encoded).
    /// Outer: stages 1-7, Inner: rounds, Innermost: coefficients.
    pub stage_compressed_polys: Vec<Vec<Vec<String>>>,
    /// Per-stage sumcheck input claims (hex-encoded).
    /// Outer: per batched sumcheck call, Inner: per instance.
    pub sumcheck_input_claims: Vec<Vec<String>>,
    /// Per-stage flushed claims (hex-encoded). One Vec<String> per flush call.
    pub flush_history: Vec<Vec<String>>,
    /// All opening claims: (OpeningId debug string, claim value hex).
    pub opening_claims: Vec<(String, String)>,
    /// Serialized polynomial commitment bytes (hex-encoded).
    pub commitment_bytes: Vec<String>,
    /// Transcript state (hex) at each stage boundary.
    /// Index 0 = after preamble+commitments, 1 = after stage 1, ..., 7 = after stage 7.
    pub transcript_states: Vec<String>,
    /// Transcript nRounds at each stage boundary (same indices as transcript_states).
    pub transcript_n_rounds: Vec<u32>,
    pub trace_length: usize,
    pub ram_k: usize,
    pub entry_address: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub heap_size: u64,
    /// Program inputs as hex bytes.
    pub inputs_hex: String,
    /// Program outputs as hex bytes.
    pub outputs_hex: String,
    /// Panic flag (0 or 1).
    pub panic: u64,
    /// Spartan key num_rows_bits (tau vector length for stage 1).
    pub num_rows_bits: usize,
    /// Per-stage instance configurations (num_rounds, max_degree).
    pub stage_instance_configs: Vec<StageInstanceConfig>,
    /// Per-stage intermediate values needed for algebraic verification.
    /// Key-value pairs of named hex values (e.g., "unmapEval" → "0x...").
    pub stage_intermediate_values: Vec<Vec<(String, String)>>,
    /// ReadWriteConfig phase rounds.
    pub rw_config: RwConfigExport,
    /// OneHotParams key values.
    pub one_hot_config: OneHotConfigExport,
    /// All verifier accumulator openings: (OpeningId debug string, point hex[], value hex).
    pub accumulator_openings: Vec<(String, Vec<String>, String)>,
    /// Stage 8: committed polynomial evaluation claims (before RLC).
    pub stage8_committed_claims: Vec<String>,
    /// Stage 8: scaling factors (Lagrange factors for dense/advice polys, 1 for RA polys).
    pub stage8_scaling_factors: Vec<String>,
    /// Stage 8: unified opening point (r_address || r_cycle in big-endian).
    pub stage8_opening_point: Vec<String>,
    /// Stage 8: joint_claim = Σ γ^i * claim_i (for transcript binding).
    pub stage8_joint_claim: String,
    /// Per-stage combined initial claim (after batching + pow2 scaling).
    pub combined_claims: Vec<String>,
    /// Full Dory witness JSON for gnark Groth16 circuit (gnark-compatible format).
    pub dory_witness_json: Option<serde_json::Value>,
    /// Karabina-transformed commitment for MiMC InputHash (12 Fp values in A0-A11 order).
    /// A0-A5 = a0 - 9*a1 (Karabina compressed), A6-A11 = raw a1 components.
    /// Order matches gnark E12 layout: C0.B0, C1.B0, C0.B1, C1.B1, C0.B2, C1.B2.
    pub commitment_karabina: Vec<String>,
}

/// Convert a field element to a hex string (big-endian, 0x-prefixed, 32 bytes).
pub fn fr_to_hex<F: JoltField>(f: &F) -> String {
    let mut buf = vec![];
    f.serialize_uncompressed(&mut buf).unwrap();
    buf.reverse(); // LE to BE
    format!("0x{}", to_hex(&buf))
}

/// Convert bytes to a hex string (0x-prefixed).
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    format!("0x{}", to_hex(bytes))
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Extract compressed polynomial coefficients from a ClearSumcheckProof.
pub fn extract_compressed_polys<F: JoltField, ProofTranscript: Transcript>(
    proof: &ClearSumcheckProof<F, ProofTranscript>,
) -> Vec<Vec<String>> {
    proof
        .compressed_polys
        .iter()
        .map(|poly| {
            poly.coeffs_except_linear_term
                .iter()
                .map(|c| fr_to_hex(c))
                .collect()
        })
        .collect()
}

/// Capture pending opening claims from the accumulator as hex strings.
/// Must be called before `flush_to_transcript`.
pub fn capture_pending_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
) -> Vec<String> {
    accumulator
        .peek_pending_claims()
        .iter()
        .map(|c| fr_to_hex(c))
        .collect()
}

/// Serialize Dory witness data to gnark-compatible JSON.
#[cfg(not(feature = "zk"))]
pub fn dory_witness_to_json(
    witness: &DoryWitnessData<ArkFr, ArkG1, ArkG2, ArkGT>,
    setup: &ArkworksVerifierSetup,
) -> serde_json::Value {
    use ark_bn254::{Fq, Fq12, Fr, G1Affine, G2Affine};
    use ark_ec::CurveGroup;
    use ark_ff::PrimeField;
    use serde_json::json;

    fn fq_hex(f: &Fq) -> String {
        let bigint = f.into_bigint();
        let limbs = bigint.as_ref();
        let mut hex = String::new();
        for &limb in limbs.iter().rev() {
            hex.push_str(&format!("{:016x}", limb));
        }
        let trimmed = hex.trim_start_matches('0');
        format!("0x{}", if trimmed.is_empty() { "0" } else { trimmed })
    }

    fn scalar_hex(f: &Fr) -> String {
        let bigint = f.into_bigint();
        let limbs = bigint.as_ref();
        let mut hex = String::new();
        for &limb in limbs.iter().rev() {
            hex.push_str(&format!("{:016x}", limb));
        }
        let trimmed = hex.trim_start_matches('0');
        format!("0x{}", if trimmed.is_empty() { "0" } else { trimmed })
    }

    fn ark_fr_hex(f: &ArkFr) -> String {
        scalar_hex(&f.0)
    }

    fn g1_json(p: &ArkG1) -> serde_json::Value {
        let affine: G1Affine = p.0.into_affine();
        json!({"x": fq_hex(&affine.x), "y": fq_hex(&affine.y)})
    }

    fn g2_json(p: &ArkG2) -> serde_json::Value {
        let affine: G2Affine = p.0.into_affine();
        json!({
            "x": {"a0": fq_hex(&affine.x.c0), "a1": fq_hex(&affine.x.c1)},
            "y": {"a0": fq_hex(&affine.y.c0), "a1": fq_hex(&affine.y.c1)},
        })
    }

    fn gt_json(gt: &ArkGT) -> serde_json::Value {
        let f: &Fq12 = &gt.0;
        json!({
            "c0": {
                "b0": {"a0": fq_hex(&f.c0.c0.c0), "a1": fq_hex(&f.c0.c0.c1)},
                "b1": {"a0": fq_hex(&f.c0.c1.c0), "a1": fq_hex(&f.c0.c1.c1)},
                "b2": {"a0": fq_hex(&f.c0.c2.c0), "a1": fq_hex(&f.c0.c2.c1)},
            },
            "c1": {
                "b0": {"a0": fq_hex(&f.c1.c0.c0), "a1": fq_hex(&f.c1.c0.c1)},
                "b1": {"a0": fq_hex(&f.c1.c1.c0), "a1": fq_hex(&f.c1.c1.c1)},
                "b2": {"a0": fq_hex(&f.c1.c2.c0), "a1": fq_hex(&f.c1.c2.c1)},
            },
        })
    }

    // Serialize Dory protocol messages in transcript order.
    // These are the exact bytes that JoltToDoryTranscript::append_serde feeds
    // into the Keccak transcript, enabling on-chain challenge derivation.
    fn serde_hex<S: dory::primitives::serialization::DorySerialize>(s: &S) -> String {
        let mut buf = Vec::new();
        s.serialize_compressed(&mut buf).unwrap();
        format!(
            "0x{}",
            buf.iter().map(|b| format!("{b:02x}")).collect::<String>()
        )
    }

    let mut transcript_messages: Vec<serde_json::Value> = Vec::new();
    // VMV messages (3)
    transcript_messages.push(json!(serde_hex(&witness.vmv_message.c)));
    transcript_messages.push(json!(serde_hex(&witness.vmv_message.d2)));
    transcript_messages.push(json!(serde_hex(&witness.vmv_message.e1)));
    // Per-round messages: 6 first + 6 second = 12 per round
    for round_idx in 0..witness.num_rounds {
        let fm = &witness.first_messages[round_idx];
        transcript_messages.push(json!(serde_hex(&fm.d1_left)));
        transcript_messages.push(json!(serde_hex(&fm.d1_right)));
        transcript_messages.push(json!(serde_hex(&fm.d2_left)));
        transcript_messages.push(json!(serde_hex(&fm.d2_right)));
        transcript_messages.push(json!(serde_hex(&fm.e1_beta)));
        transcript_messages.push(json!(serde_hex(&fm.e2_beta)));
        // beta challenge derived here (not serialized)
        let sm = &witness.second_messages[round_idx];
        transcript_messages.push(json!(serde_hex(&sm.c_plus)));
        transcript_messages.push(json!(serde_hex(&sm.c_minus)));
        transcript_messages.push(json!(serde_hex(&sm.e1_plus)));
        transcript_messages.push(json!(serde_hex(&sm.e1_minus)));
        transcript_messages.push(json!(serde_hex(&sm.e2_plus)));
        transcript_messages.push(json!(serde_hex(&sm.e2_minus)));
        // alpha challenge derived here (not serialized)
    }
    // gamma challenge derived here (not serialized)
    // Final messages (2)
    transcript_messages.push(json!(serde_hex(&witness.final_message.e1)));
    transcript_messages.push(json!(serde_hex(&witness.final_message.e2)));
    // d challenge derived here (not serialized)

    // Build debug GT accumulation values (matching gnark circuit loop)
    let num_rounds = witness.num_rounds;
    let mut debug_c_values = Vec::new();
    let mut debug_d1_values = Vec::new();
    let mut debug_d2_values = Vec::new();
    {
        use dory::primitives::arithmetic::{Field as DoryField, Group};
        let mut gt_c = witness.vmv_message.c;
        let mut gt_d1 = witness.commitment;
        let mut gt_d2 = witness.vmv_message.d2;

        let setup_inner = setup.clone().into_inner();
        for round_idx in 0..num_rounds {
            let alpha = &witness.alphas[round_idx];
            let beta = &witness.betas[round_idx];
            let alpha_inv = alpha.inv().unwrap();
            let beta_inv = beta.inv().unwrap();
            let alpha_beta = *alpha * *beta;
            let alpha_inv_beta_inv = alpha_inv * beta_inv;
            let round = num_rounds - round_idx;

            gt_c = gt_c
                + setup_inner.chi[round]
                + gt_d2.scale(beta)
                + gt_d1.scale(&beta_inv)
                + witness.second_messages[round_idx].c_plus.scale(alpha)
                + witness.second_messages[round_idx].c_minus.scale(&alpha_inv);

            gt_d1 = witness.first_messages[round_idx].d1_left.scale(alpha)
                + witness.first_messages[round_idx].d1_right
                + setup_inner.delta_1l[round].scale(&alpha_beta)
                + setup_inner.delta_1r[round].scale(beta);

            gt_d2 = witness.first_messages[round_idx].d2_left.scale(&alpha_inv)
                + witness.first_messages[round_idx].d2_right
                + setup_inner.delta_2l[round].scale(&alpha_inv_beta_inv)
                + setup_inner.delta_2r[round].scale(&beta_inv);

            debug_c_values.push(gt_json(&gt_c));
            debug_d1_values.push(gt_json(&gt_d1));
            debug_d2_values.push(gt_json(&gt_d2));
        }
    }

    // Karabina encoding of commitment for on-chain MiMC hash
    let commitment_karabina = {
        let c = &witness.commitment.0;
        let nine = Fq::from(9u64);
        let karabina = |a0: &Fq, a1: &Fq| -> String {
            let val = *a0 - nine * a1;
            fq_hex(&val)
        };
        let pairs: [(&Fq, &Fq); 6] = [
            (&c.c0.c0.c0, &c.c0.c0.c1),
            (&c.c1.c0.c0, &c.c1.c0.c1),
            (&c.c0.c1.c0, &c.c0.c1.c1),
            (&c.c1.c1.c0, &c.c1.c1.c1),
            (&c.c0.c2.c0, &c.c0.c2.c1),
            (&c.c1.c2.c0, &c.c1.c2.c1),
        ];
        let mut vals: Vec<String> = Vec::new();
        for (a0, a1) in &pairs {
            vals.push(karabina(a0, a1));
        }
        for (_, a1) in &pairs {
            vals.push(fq_hex(a1));
        }
        json!(vals)
    };

    // Test pairing for Go-side sanity check
    let setup_inner = setup.clone().into_inner();
    let test_pairing = <BN254 as dory::primitives::arithmetic::PairingCurve>::pair(
        &setup_inner.g1_0,
        &setup_inner.g2_0,
    );

    json!({
        "num_rounds": num_rounds,
        "alpha": witness.alphas.iter().map(ark_fr_hex).collect::<Vec<_>>(),
        "beta": witness.betas.iter().map(ark_fr_hex).collect::<Vec<_>>(),
        "gamma": ark_fr_hex(&witness.gamma),
        "d": ark_fr_hex(&witness.d),
        "s1_coords": witness.s1_coords.iter().map(ark_fr_hex).collect::<Vec<_>>(),
        "s2_coords": witness.s2_coords.iter().map(ark_fr_hex).collect::<Vec<_>>(),
        "final_p1_g2": g2_json(&witness.final_p1_g2),
        "final_p2_g2": g2_json(&witness.final_p2_g2),
        "commitment": gt_json(&witness.commitment),
        "evaluation": ark_fr_hex(&witness.evaluation),
        "vmv_c": gt_json(&witness.vmv_message.c),
        "vmv_d2": gt_json(&witness.vmv_message.d2),
        "vmv_e1": g1_json(&witness.vmv_message.e1),
        "first_messages": witness.first_messages.iter().map(|m| json!({
            "d1_left": gt_json(&m.d1_left),
            "d1_right": gt_json(&m.d1_right),
            "d2_left": gt_json(&m.d2_left),
            "d2_right": gt_json(&m.d2_right),
            "e1_beta": g1_json(&m.e1_beta),
            "e2_beta": g2_json(&m.e2_beta),
        })).collect::<Vec<_>>(),
        "second_messages": witness.second_messages.iter().map(|m| json!({
            "c_plus": gt_json(&m.c_plus),
            "c_minus": gt_json(&m.c_minus),
            "e1_plus": g1_json(&m.e1_plus),
            "e1_minus": g1_json(&m.e1_minus),
            "e2_plus": g2_json(&m.e2_plus),
            "e2_minus": g2_json(&m.e2_minus),
        })).collect::<Vec<_>>(),
        "final_e1": g1_json(&witness.final_message.e1),
        "final_e2": g2_json(&witness.final_message.e2),
        "transcript_messages": transcript_messages,
        "commitment_karabina": commitment_karabina,
        "test_pairing": gt_json(&test_pairing),
        "debug_c_after_rounds": debug_c_values,
        "debug_d1_after_rounds": debug_d1_values,
        "debug_d2_after_rounds": debug_d2_values,
        "setup": {
            "chi": setup_inner.chi.iter().map(gt_json).collect::<Vec<_>>(),
            "delta_1l": setup_inner.delta_1l.iter().map(gt_json).collect::<Vec<_>>(),
            "delta_1r": setup_inner.delta_1r.iter().map(gt_json).collect::<Vec<_>>(),
            "delta_2l": setup_inner.delta_2l.iter().map(gt_json).collect::<Vec<_>>(),
            "delta_2r": setup_inner.delta_2r.iter().map(gt_json).collect::<Vec<_>>(),
            "g1_0": g1_json(&setup_inner.g1_0),
            "g2_0": g2_json(&setup_inner.g2_0),
            "h1": g1_json(&setup_inner.h1),
            "h2": g2_json(&setup_inner.h2),
            "ht": gt_json(&setup_inner.ht),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_fr_to_hex() {
        let f = Fr::from(42u64);
        let hex = fr_to_hex(&f);
        assert_eq!(
            hex,
            "0x000000000000000000000000000000000000000000000000000000000000002a"
        );
    }

    #[test]
    fn test_fr_to_hex_large() {
        let f = Fr::from(0xdeadbeef_u64);
        let hex = fr_to_hex(&f);
        assert!(hex.starts_with("0x"));
        assert!(hex.ends_with("deadbeef"));
    }
}
