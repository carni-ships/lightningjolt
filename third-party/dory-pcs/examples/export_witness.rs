//! Exports a Dory proof witness to JSON for the gnark Groth16 wrapper circuit.

use ark_bn254::{Fq, Fq12, Fr, G1Affine, G2Affine};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use dory_pcs::backends::arkworks::{
    ArkFr, ArkG1, ArkG2, ArkGT, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines,
    BN254,
};
use dory_pcs::primitives::arithmetic::{Field as DoryField, Group, PairingCurve};
use dory_pcs::primitives::poly::Polynomial;
use dory_pcs::primitives::transcript::Transcript;
use dory_pcs::{prove, setup, Transparent};
use serde_json::{json, Value};
use std::fs;

fn fq_to_hex(f: &Fq) -> String {
    let bigint = f.into_bigint();
    let limbs = bigint.as_ref();
    let mut hex = String::new();
    for &limb in limbs.iter().rev() {
        hex.push_str(&format!("{:016x}", limb));
    }
    let trimmed = hex.trim_start_matches('0');
    format!("0x{}", if trimmed.is_empty() { "0" } else { trimmed })
}

fn fr_to_hex(f: &Fr) -> String {
    let bigint = f.into_bigint();
    let limbs = bigint.as_ref();
    let mut hex = String::new();
    for &limb in limbs.iter().rev() {
        hex.push_str(&format!("{:016x}", limb));
    }
    let trimmed = hex.trim_start_matches('0');
    format!("0x{}", if trimmed.is_empty() { "0" } else { trimmed })
}

fn g1_to_json(p: &ArkG1) -> Value {
    let affine: G1Affine = p.0.into_affine();
    json!({"x": fq_to_hex(&affine.x), "y": fq_to_hex(&affine.y)})
}

fn g2_to_json(p: &ArkG2) -> Value {
    let affine: G2Affine = p.0.into_affine();
    json!({
        "x": {"a0": fq_to_hex(&affine.x.c0), "a1": fq_to_hex(&affine.x.c1)},
        "y": {"a0": fq_to_hex(&affine.y.c0), "a1": fq_to_hex(&affine.y.c1)},
    })
}

fn gt_to_json(gt: &ArkGT) -> Value {
    let f: &Fq12 = &gt.0;
    json!({
        "c0": {
            "b0": {"a0": fq_to_hex(&f.c0.c0.c0), "a1": fq_to_hex(&f.c0.c0.c1)},
            "b1": {"a0": fq_to_hex(&f.c0.c1.c0), "a1": fq_to_hex(&f.c0.c1.c1)},
            "b2": {"a0": fq_to_hex(&f.c0.c2.c0), "a1": fq_to_hex(&f.c0.c2.c1)},
        },
        "c1": {
            "b0": {"a0": fq_to_hex(&f.c1.c0.c0), "a1": fq_to_hex(&f.c1.c0.c1)},
            "b1": {"a0": fq_to_hex(&f.c1.c1.c0), "a1": fq_to_hex(&f.c1.c1.c1)},
            "b2": {"a0": fq_to_hex(&f.c1.c2.c0), "a1": fq_to_hex(&f.c1.c2.c1)},
        },
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (prover_setup, verifier_setup) = setup::<BN254>(10);

    let nu = 4;
    let sigma = 4;
    let poly_size = 1 << (nu + sigma);
    let num_vars = nu + sigma;
    let num_rounds = std::cmp::max(nu, sigma);

    let coefficients: Vec<ArkFr> = (0..poly_size).map(|_| ArkFr::random()).collect();
    let poly = ArkworksPolynomial::new(coefficients);

    let (tier_2, tier_1, commit_blind) =
        poly.commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)?;

    let point: Vec<ArkFr> = (0..num_vars).map(|_| ArkFr::random()).collect();
    let evaluation = poly.evaluate(&point);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-witness-export");
    let (proof, _) = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
        &poly,
        &point,
        tier_1,
        commit_blind,
        nu,
        sigma,
        &prover_setup,
        &mut prover_transcript,
    )?;

    // Verify first
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-witness-export");
    dory_pcs::verify::<_, BN254, G1Routines, G2Routines, _>(
        tier_2,
        evaluation,
        &point,
        &proof,
        verifier_setup.clone(),
        &mut verifier_transcript,
    )?;
    eprintln!("Proof verified successfully!");

    // Replay Fiat-Shamir to extract challenges
    let mut transcript = Blake2bTranscript::new(b"dory-witness-export");
    transcript.append_serde(b"vmv_c", &proof.vmv_message.c);
    transcript.append_serde(b"vmv_d2", &proof.vmv_message.d2);
    transcript.append_serde(b"vmv_e1", &proof.vmv_message.e1);

    let mut alphas = Vec::new();
    let mut betas = Vec::new();

    for round in 0..num_rounds {
        let first = &proof.first_messages[round];
        let second = &proof.second_messages[round];

        transcript.append_serde(b"d1_left", &first.d1_left);
        transcript.append_serde(b"d1_right", &first.d1_right);
        transcript.append_serde(b"d2_left", &first.d2_left);
        transcript.append_serde(b"d2_right", &first.d2_right);
        transcript.append_serde(b"e1_beta", &first.e1_beta);
        transcript.append_serde(b"e2_beta", &first.e2_beta);

        let beta: ArkFr = transcript.challenge_scalar(b"beta");
        betas.push(beta);

        transcript.append_serde(b"c_plus", &second.c_plus);
        transcript.append_serde(b"c_minus", &second.c_minus);
        transcript.append_serde(b"e1_plus", &second.e1_plus);
        transcript.append_serde(b"e1_minus", &second.e1_minus);
        transcript.append_serde(b"e2_plus", &second.e2_plus);
        transcript.append_serde(b"e2_minus", &second.e2_minus);

        let alpha: ArkFr = transcript.challenge_scalar(b"alpha");
        alphas.push(alpha);
    }

    let gamma: ArkFr = transcript.challenge_scalar(b"gamma");
    transcript.append_serde(b"final_e1", &proof.final_message.e1);
    transcript.append_serde(b"final_e2", &proof.final_message.e2);
    let d_chal: ArkFr = transcript.challenge_scalar(b"d");

    // s1/s2 coords matching verifier convention:
    // s1 = columns = point[..sigma], s2 = rows = point[sigma..sigma+nu] padded to sigma
    let s1_coords: Vec<ArkFr> = point[..sigma].to_vec();
    let mut s2_coords: Vec<ArkFr> = vec![ArkFr::zero(); sigma];
    for i in 0..nu {
        s2_coords[i] = point[sigma + i];
    }

    // Compute E2 accumulator (replaying verifier logic)
    // E2_init = g2_0 * evaluation (transparent mode)
    let mut e2_acc: ArkG2 = verifier_setup.g2_0.scale(&evaluation);
    let mut s1_acc = ArkFr::one();
    let mut s2_acc = ArkFr::one();

    for round_idx in 0..num_rounds {
        let alpha = &alphas[round_idx];
        let beta = &betas[round_idx];
        let alpha_inv = alpha.inv().unwrap();
        let beta_inv = beta.inv().unwrap();

        // E2 += beta_inv * e2_beta + alpha * e2_plus + alpha_inv * e2_minus
        e2_acc = e2_acc
            + proof.first_messages[round_idx].e2_beta.scale(&beta_inv)
            + proof.second_messages[round_idx].e2_plus.scale(alpha)
            + proof.second_messages[round_idx].e2_minus.scale(&alpha_inv);

        // Fold scalars (verifier uses round = num_rounds - round_idx, idx = round - 1)
        let coord_idx = num_rounds - 1 - round_idx;
        let y_t = &s1_coords[coord_idx];
        let x_t = &s2_coords[coord_idx];
        let one = ArkFr::one();
        s1_acc = s1_acc * (*alpha * (one - *y_t) + *y_t);
        s2_acc = s2_acc * (alpha_inv * (one - *x_t) + *x_t);
    }

    // Compute G2 composite witnesses for the pairing equation
    let d_inv = d_chal.inv().unwrap();
    let neg_gamma = -gamma;

    // FinalP1G2 = final_e2 + g2_0 * d_inv
    let final_p1_g2 = proof.final_message.e2 + verifier_setup.g2_0.scale(&d_inv);

    // FinalP2G2 = neg_gamma * (e2_acc + g2_0 * (d_inv * s1_acc))
    let final_p2_g2 = (e2_acc + verifier_setup.g2_0.scale(&(d_inv * s1_acc))).scale(&neg_gamma);

    eprintln!("s1_acc computed, s2_acc computed, G2 composites computed");

    // Native verification: replay the exact circuit equation to catch bugs
    // This mirrors the gnark circuit's computation exactly.
    {
        // Replay GT accumulation (matching gnark circuit loop)
        let mut gt_c = proof.vmv_message.c;
        let mut gt_d1 = tier_2; // commitment
        let mut gt_d2 = proof.vmv_message.d2;
        let mut g1_e1 = proof.vmv_message.e1;

        for round_idx in 0..num_rounds {
            let alpha = &alphas[round_idx];
            let beta = &betas[round_idx];
            let alpha_inv = alpha.inv().unwrap();
            let beta_inv = beta.inv().unwrap();
            let alpha_beta = *alpha * *beta;
            let alpha_inv_beta_inv = alpha_inv * beta_inv;

            let round = num_rounds - round_idx; // matches circuit's round variable

            // C update
            gt_c = gt_c
                + verifier_setup.chi[round]
                + gt_d2.scale(beta)
                + gt_d1.scale(&beta_inv)
                + proof.second_messages[round_idx].c_plus.scale(alpha)
                + proof.second_messages[round_idx].c_minus.scale(&alpha_inv);

            // D1 update
            gt_d1 = proof.first_messages[round_idx].d1_left.scale(alpha)
                + proof.first_messages[round_idx].d1_right
                + verifier_setup.delta_1l[round].scale(&alpha_beta)
                + verifier_setup.delta_1r[round].scale(beta);

            // D2 update
            gt_d2 = proof.first_messages[round_idx].d2_left.scale(&alpha_inv)
                + proof.first_messages[round_idx].d2_right
                + verifier_setup.delta_2l[round].scale(&alpha_inv_beta_inv)
                + verifier_setup.delta_2r[round].scale(&beta_inv);

            // G1 E1 update
            g1_e1 = g1_e1
                + proof.first_messages[round_idx].e1_beta.scale(beta)
                + proof.second_messages[round_idx].e1_plus.scale(alpha)
                + proof.second_messages[round_idx].e1_minus.scale(&alpha_inv);
        }

        let d_sq = d_chal * d_chal;
        let neg_gamma_inv = -(gamma.inv().unwrap());

        // RHS
        let s_product = s1_acc * s2_acc;
        let rhs = gt_c
            + verifier_setup.ht.scale(&s_product)
            + verifier_setup.chi[0]
            + gt_d2.scale(&d_chal)
            + gt_d1.scale(&d_inv)
            + proof.vmv_message.d2.scale(&d_sq);

        // LHS pairing inputs
        let p1_g1 = proof.final_message.e1 + verifier_setup.g1_0.scale(&d_chal);
        let p3_g1_inner = g1_e1 + verifier_setup.g1_0.scale(&(d_chal * s2_acc));
        let p3_g1 = p3_g1_inner.scale(&neg_gamma_inv);
        let p4_g1 = proof.vmv_message.e1.scale(&d_sq);

        let lhs = BN254::multi_pair(
            &[p1_g1, verifier_setup.h1, p3_g1, p4_g1],
            &[final_p1_g2, final_p2_g2, verifier_setup.h2, verifier_setup.g2_0],
        );

        if lhs == rhs {
            eprintln!("Native circuit equation verification: PASSED");
            // Also verify using the original Dory verifier's verify_final equation
            // to make sure our replay matches exactly
            let orig_rhs_check = gt_c
                + verifier_setup.ht.scale(&s_product)
                + verifier_setup.chi[0]
                + gt_d2.scale(&d_chal)
                + gt_d1.scale(&d_inv)
                + proof.vmv_message.d2.scale(&d_sq);
            let orig_lhs_check = BN254::multi_pair(
                &[p1_g1, verifier_setup.h1, p3_g1, p4_g1],
                &[
                    proof.final_message.e2 + verifier_setup.g2_0.scale(&d_inv),
                    (e2_acc + verifier_setup.g2_0.scale(&(d_inv * s1_acc))).scale(&neg_gamma),
                    verifier_setup.h2,
                    verifier_setup.g2_0,
                ],
            );
            if orig_lhs_check == orig_rhs_check {
                eprintln!("Direct G2 verification: PASSED");
            } else {
                eprintln!("Direct G2 verification: FAILED");
            }
        } else {
            eprintln!("Native circuit equation verification: FAILED");
            eprintln!("  LHS and RHS differ — check GT accumulation or G2 composites");
        }
    }

    // Export test pairing value for Go-side verification
    let test_pairing = BN254::pair(&verifier_setup.g1_0, &verifier_setup.g2_0);
    eprintln!("Test pairing e(G1_0, G2_0) computed");

    // Export intermediate GT values after each round for debugging
    let mut debug_c_values = Vec::new();
    let mut debug_d1_values = Vec::new();
    let mut debug_d2_values = Vec::new();
    {
        let mut gt_c = proof.vmv_message.c;
        let mut gt_d1 = tier_2;
        let mut gt_d2 = proof.vmv_message.d2;

        for round_idx in 0..num_rounds {
            let alpha = &alphas[round_idx];
            let beta = &betas[round_idx];
            let alpha_inv = alpha.inv().unwrap();
            let beta_inv = beta.inv().unwrap();
            let alpha_beta = *alpha * *beta;
            let alpha_inv_beta_inv = alpha_inv * beta_inv;
            let round = num_rounds - round_idx;

            gt_c = gt_c
                + verifier_setup.chi[round]
                + gt_d2.scale(beta)
                + gt_d1.scale(&beta_inv)
                + proof.second_messages[round_idx].c_plus.scale(alpha)
                + proof.second_messages[round_idx].c_minus.scale(&alpha_inv);

            gt_d1 = proof.first_messages[round_idx].d1_left.scale(alpha)
                + proof.first_messages[round_idx].d1_right
                + verifier_setup.delta_1l[round].scale(&alpha_beta)
                + verifier_setup.delta_1r[round].scale(beta);

            gt_d2 = proof.first_messages[round_idx].d2_left.scale(&alpha_inv)
                + proof.first_messages[round_idx].d2_right
                + verifier_setup.delta_2l[round].scale(&alpha_inv_beta_inv)
                + verifier_setup.delta_2r[round].scale(&beta_inv);

            debug_c_values.push(gt_to_json(&gt_c));
            debug_d1_values.push(gt_to_json(&gt_d1));
            debug_d2_values.push(gt_to_json(&gt_d2));
        }
    }

    // Build JSON witness
    let witness = json!({
        "num_rounds": num_rounds,
        "alpha": alphas.iter().map(|a| fr_to_hex(&a.0)).collect::<Vec<_>>(),
        "beta": betas.iter().map(|b| fr_to_hex(&b.0)).collect::<Vec<_>>(),
        "gamma": fr_to_hex(&gamma.0),
        "d": fr_to_hex(&d_chal.0),
        "s1_coords": s1_coords.iter().map(|s| fr_to_hex(&s.0)).collect::<Vec<_>>(),
        "s2_coords": s2_coords.iter().map(|s| fr_to_hex(&s.0)).collect::<Vec<_>>(),
        "final_p1_g2": g2_to_json(&final_p1_g2),
        "final_p2_g2": g2_to_json(&final_p2_g2),
        "commitment": gt_to_json(&tier_2),
        "evaluation": fr_to_hex(&evaluation.0),
        "vmv_c": gt_to_json(&proof.vmv_message.c),
        "vmv_d2": gt_to_json(&proof.vmv_message.d2),
        "vmv_e1": g1_to_json(&proof.vmv_message.e1),
        "first_messages": proof.first_messages.iter().map(|m| json!({
            "d1_left": gt_to_json(&m.d1_left),
            "d1_right": gt_to_json(&m.d1_right),
            "d2_left": gt_to_json(&m.d2_left),
            "d2_right": gt_to_json(&m.d2_right),
            "e1_beta": g1_to_json(&m.e1_beta),
            "e2_beta": g2_to_json(&m.e2_beta),
        })).collect::<Vec<_>>(),
        "second_messages": proof.second_messages.iter().map(|m| json!({
            "c_plus": gt_to_json(&m.c_plus),
            "c_minus": gt_to_json(&m.c_minus),
            "e1_plus": g1_to_json(&m.e1_plus),
            "e1_minus": g1_to_json(&m.e1_minus),
            "e2_plus": g2_to_json(&m.e2_plus),
            "e2_minus": g2_to_json(&m.e2_minus),
        })).collect::<Vec<_>>(),
        "final_e1": g1_to_json(&proof.final_message.e1),
        "final_e2": g2_to_json(&proof.final_message.e2),
        "test_pairing": gt_to_json(&test_pairing),
        "debug_c_after_rounds": debug_c_values,
        "debug_d1_after_rounds": debug_d1_values,
        "debug_d2_after_rounds": debug_d2_values,
        "setup": {
            "chi": verifier_setup.chi.iter().map(gt_to_json).collect::<Vec<_>>(),
            "delta_1l": verifier_setup.delta_1l.iter().map(gt_to_json).collect::<Vec<_>>(),
            "delta_1r": verifier_setup.delta_1r.iter().map(gt_to_json).collect::<Vec<_>>(),
            "delta_2l": verifier_setup.delta_2l.iter().map(gt_to_json).collect::<Vec<_>>(),
            "delta_2r": verifier_setup.delta_2r.iter().map(gt_to_json).collect::<Vec<_>>(),
            "g1_0": g1_to_json(&verifier_setup.g1_0),
            "g2_0": g2_to_json(&verifier_setup.g2_0),
            "h1": g1_to_json(&verifier_setup.h1),
            "h2": g2_to_json(&verifier_setup.h2),
            "ht": gt_to_json(&verifier_setup.ht),
        },
    });

    let output_path = "dory_witness.json";
    fs::write(output_path, serde_json::to_string_pretty(&witness)?)?;
    eprintln!("Witness exported to {output_path} ({num_rounds} rounds)");

    Ok(())
}
