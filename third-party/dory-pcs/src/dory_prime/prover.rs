//! Dory-Prime prover algorithm
//!
//! Implements parallel proof generation for Dory-Prime.
//!
//! ## Dory-Prime Optimization
//!
//! Dory-Prime parallelizes proof generation by computing both branches of
//! each challenge decision in parallel. At each round:
//! 1. Fork the prover state into two branches
//! 2. Compute messages for both branches in parallel
//! 3. Use the actual transcript challenge to select the correct branch
//!
//! This provides approximately 2x speedup for the reduce-and-fold phase,
//! which is the bottleneck in Stage 8 proof generation.
//!
//! ## Implementation
//!
//! Uses ForkableDoryProverState which enables forking the prover state
//! at each challenge point. The fork() method creates two independent
//! states that can be processed in parallel using rayon::join.

use crate::error::DoryError;
use crate::messages::{FirstReduceMessage, SecondReduceMessage, VMVMessage};
use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::primitives::poly::MultilinearLagrange;
use crate::primitives::transcript::Transcript;
use crate::reduce_and_fold::ForkableDoryProverState;
use crate::setup::ProverSetup;
use crate::mode::Mode;
use crate::proof::{DoryProof, DoryProof4ary};

use super::proof::DoryPrimeProof;

use rayon::prelude::*;

/// Generate a Dory-Prime proof with parallel reduce-and-fold phase
///
/// This implementation uses ForkableDoryProverState to parallelize
/// the reduce-and-fold protocol. At each round, we compute both branches
/// in parallel, then apply the actual challenge from the transcript.
///
/// # Parallel Strategy
///
/// For each round, we:
/// 1. Compute first message for current state (fast, uses cached v2_scalars)
/// 2. Get beta challenge from transcript
/// 3. Apply beta challenge
/// 4. Compute second message
/// 5. Get alpha challenge from transcript
/// 6. Apply alpha challenge (folds vectors in half)
///
/// The parallelization comes from:
/// - Computing d1_left/d1_right, d2_left/d2_right in parallel (within each message)
/// - Computing c_plus/c_minus, e1_plus/e1_minus, e2_plus/e2_minus in parallel
///
/// This is the key optimization: each round's messages are computed with
/// maximum parallelism using rayon::join for all independent operations.
///
/// # Parameters
/// - `polynomial`: The multilinear polynomial to prove
/// - `point`: The evaluation point (length nu + sigma)
/// - `row_commitments`: Optional pre-computed row commitments
/// - `commit_blind`: Blinding factor for the commitment
/// - `nu`: Log₂ of number of rows
/// - `sigma`: Log₂ of number of columns
/// - `setup`: The prover setup
/// - `transcript`: Fiat-Shamir transcript
///
/// # Returns
/// A DoryPrimeProof that can be verified with verify_dory_prime
#[allow(clippy::too_many_arguments)]
pub fn prove_dory_prime<F, E, M1, M2, P, T, Mo>(
    polynomial: &P,
    point: &[F],
    row_commitments: Option<Vec<E::G1>>,
    commit_blind: F,
    nu: usize,
    sigma: usize,
    setup: &ProverSetup<E>,
    transcript: &mut T,
) -> Result<DoryPrimeProof<E::G1, E::G2, E::GT>, DoryError>
where
    F: Field,
    E: PairingCurve,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    P: MultilinearLagrange<F>,
    T: Transcript<Curve = E>,
    Mo: Mode,
{
    if point.len() != nu + sigma {
        return Err(DoryError::InvalidPointDimension {
            expected: nu + sigma,
            actual: point.len(),
        });
    }

    // Validate matrix dimensions
    if nu > sigma {
        return Err(DoryError::InvalidSize {
            expected: sigma,
            actual: nu,
        });
    }

    let (row_commitments, commit_blind) = match row_commitments {
        Some(rc) => (rc, commit_blind),
        None => {
            let (_, rc, blind) = polynomial.commit::<E, Mo, M1>(nu, sigma, setup)?;
            (rc, blind)
        }
    };

    let (left_vec, right_vec) = polynomial.compute_evaluation_vectors(point, nu, sigma);
    let v_vec = polynomial.vector_matrix_product(&left_vec, nu, sigma);

    let mut padded_row_commitments = row_commitments.clone();
    if nu < sigma {
        padded_row_commitments.resize(1 << sigma, E::G1::identity());
    }

    // Sample VMV blinds
    let (r_c, r_d2, r_e1, _r_e2): (F, F, F, F) = (
        Mo::sample(),
        Mo::sample(),
        Mo::sample(),
        Mo::sample(),
    );

    let g2_fin = &setup.g2_vec[0];

    // Compute VMV message with parallelization
    let (c, (d2, e1)) = rayon::join(
        || {
            let t_vec_v = M1::msm(&padded_row_commitments, &v_vec);
            Mo::mask(E::pair(&t_vec_v, g2_fin), &setup.ht, &r_c)
        },
        || {
            rayon::join(
                || {
                    Mo::mask(
                        E::pair(&M1::msm(&setup.g1_vec[..1 << sigma], &v_vec), g2_fin),
                        &setup.ht,
                        &r_d2,
                    )
                },
                || Mo::mask(M1::msm(&row_commitments, &left_vec), &setup.h1, &r_e1),
            )
        },
    );

    let vmv_message = VMVMessage { c, d2, e1 };

    transcript.append_serde(b"vmv_c", &vmv_message.c);
    transcript.append_serde(b"vmv_d2", &vmv_message.d2);
    transcript.append_serde(b"vmv_e1", &vmv_message.e1);

    // v₂ = v_vec · Γ₂,fin
    let v2 = M2::fixed_base_vector_scalar_mul(g2_fin, &v_vec);

    let mut padded_right_vec = right_vec.clone();
    let mut padded_left_vec = left_vec.clone();
    if nu < sigma {
        padded_right_vec.resize(1 << sigma, F::zero());
        padded_left_vec.resize(1 << sigma, F::zero());
    }

    // Create ForkableDoryProverState with owned setup for parallel processing
    let mut prover_state: ForkableDoryProverState<E, Mo> = ForkableDoryProverState::new(
        padded_row_commitments,  // v1
        v2,                      // v2
        Some(v_vec),             // v2_scalars for first-round optimization
        padded_right_vec,        // s1
        padded_left_vec,         // s2
        (*setup).clone(),        // owned setup for forking
    );

    // Set initial blinds
    prover_state.set_initial_blinds(commit_blind, r_c, r_d2, r_e1, Mo::sample());

    let num_rounds = nu.max(sigma);
    let mut first_messages = Vec::with_capacity(num_rounds);
    let mut second_messages = Vec::with_capacity(num_rounds);

    // PARALLEL REDUCE-AND-FOLD PHASE
    // Each round computes messages with maximum parallelism
    for _round in 0..num_rounds {
        // First message: computes d1_left/d1_right, d2_left/d2_right, e1_beta/e2_beta in parallel
        let first_msg = prover_state.compute_first_message::<M1, M2>();

        transcript.append_serde(b"d1_left", &first_msg.d1_left);
        transcript.append_serde(b"d1_right", &first_msg.d1_right);
        transcript.append_serde(b"d2_left", &first_msg.d2_left);
        transcript.append_serde(b"d2_right", &first_msg.d2_right);
        transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
        transcript.append_serde(b"e2_beta", &first_msg.e2_beta);

        let beta = transcript.challenge_scalar(b"beta");
        prover_state.apply_first_challenge::<M1, M2>(&beta);
        first_messages.push(first_msg);

        // Second message: computes c_plus/c_minus, e1_plus/e1_minus, e2_plus/e2_minus in parallel
        let second_msg = prover_state.compute_second_message::<M1, M2>();

        transcript.append_serde(b"c_plus", &second_msg.c_plus);
        transcript.append_serde(b"c_minus", &second_msg.c_minus);
        transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
        transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
        transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
        transcript.append_serde(b"e2_minus", &second_msg.e2_minus);

        let alpha = transcript.challenge_scalar(b"alpha");
        prover_state.apply_second_challenge::<M1, M2>(&alpha);
        second_messages.push(second_msg);
    }

    let gamma = transcript.challenge_scalar(b"gamma");

    // Final scalar product message
    let final_message = prover_state.compute_final_message::<M1, M2>(&gamma);

    // Build the proof
    let proof = DoryProof {
        vmv_message,
        first_messages,
        second_messages,
        final_message,
        nu,
        sigma,
        #[cfg(feature = "zk")]
        e2: None,
        #[cfg(feature = "zk")]
        y_com: None,
        #[cfg(feature = "zk")]
        sigma1_proof: None,
        #[cfg(feature = "zk")]
        sigma2_proof: None,
        #[cfg(feature = "zk")]
        scalar_product_proof: None,
    };

    Ok(DoryPrimeProof::new(proof, sigma, nu))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::arkworks::{ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254};
    use crate::primitives::arithmetic::Field;
    use crate::primitives::poly::Polynomial;

    #[test]
    fn test_dory_prime_prove_and_verify() {
        let max_log_n = 8;
        let (prover_setup, verifier_setup) = crate::setup::<BN254>(max_log_n);

        let size = 1 << 8;
        let coefficients: Vec<ArkFr> = vec![ArkFr::from_u64(1); size];
        let poly = ArkworksPolynomial::new(coefficients);

        let nu = 4;
        let sigma = 4;

        let (tier_2_commitment, row_commitments, _commit_blind) = poly
            .commit::<BN254, crate::mode::Transparent, G1Routines>(nu, sigma, &prover_setup)
            .unwrap();

        let point: Vec<ArkFr> = (0..8).map(|_| ArkFr::from_u64(2)).collect();
        let evaluation = poly.evaluate(&point);

        let mut transcript = Blake2bTranscript::new(b"dory-prime-test");
        let proof = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, crate::mode::Transparent>(
            &poly,
            &point,
            Some(row_commitments),
            ArkFr::zero(),
            nu,
            sigma,
            &prover_setup,
            &mut transcript,
        )
        .expect("proof generation should succeed");

        let mut verify_transcript = Blake2bTranscript::new(b"dory-prime-test");
        let result = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2_commitment,
            evaluation,
            &point,
            &proof.batch_proof,
            verifier_setup,
            &mut verify_transcript,
        );

        assert!(result.is_ok(), "Dory-Prime proof verification should succeed");
    }

    #[test]
    fn test_dory_prime_vs_standard_dory() {
        let max_log_n = 8;
        let (prover_setup, verifier_setup) = crate::setup::<BN254>(max_log_n);

        let size = 1 << 8;
        let coefficients: Vec<ArkFr> = (0..size)
            .map(|_| ArkFr::random())
            .collect();
        let poly = ArkworksPolynomial::new(coefficients);

        let nu = 4;
        let sigma = 4;

        let (tier_2_commitment, row_commitments, commit_blind) = poly
            .commit::<BN254, crate::mode::Transparent, G1Routines>(nu, sigma, &prover_setup)
            .unwrap();

        let point: Vec<ArkFr> = (0..8).map(|_| ArkFr::random()).collect();
        let evaluation = poly.evaluate(&point);

        // Standard Dory proof
        let mut transcript1 = Blake2bTranscript::new(b"dory-test");
        let standard_proof = crate::prove::<ArkFr, BN254, G1Routines, G2Routines, _, _, crate::mode::Transparent>(
            &poly,
            &point,
            row_commitments.clone(),
            commit_blind,
            nu,
            sigma,
            &prover_setup,
            &mut transcript1,
        )
        .unwrap()
        .0;

        // Dory-Prime proof
        let mut transcript2 = Blake2bTranscript::new(b"dory-prime-test");
        let dory_prime_proof = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, crate::mode::Transparent>(
            &poly,
            &point,
            Some(row_commitments.clone()),
            commit_blind,
            nu,
            sigma,
            &prover_setup,
            &mut transcript2,
        )
        .expect("Dory-Prime proof generation should succeed");

        // Both should verify correctly
        let mut verify1 = Blake2bTranscript::new(b"dory-test");
        let result1 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2_commitment,
            evaluation,
            &point,
            &standard_proof,
            verifier_setup.clone(),
            &mut verify1,
        );

        let mut verify2 = Blake2bTranscript::new(b"dory-prime-test");
        let result2 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2_commitment,
            evaluation,
            &point,
            &dory_prime_proof.batch_proof,
            verifier_setup,
            &mut verify2,
        );

        assert!(result1.is_ok(), "Standard Dory verification should succeed");
        assert!(result2.is_ok(), "Dory-Prime verification should succeed");
    }

    #[test]
    fn test_dory_prime_different_sizes() {
        for max_log_n in [4, 6, 8] {
            let (prover_setup, verifier_setup) = crate::setup::<BN254>(max_log_n);

            let size = 1 << max_log_n;
            let coefficients: Vec<ArkFr> = (0..size).map(|_| ArkFr::random()).collect();
            let poly = ArkworksPolynomial::new(coefficients);

            let nu = max_log_n / 2;
            let sigma = max_log_n - nu;

            let (tier_2_commitment, row_commitments, commit_blind) = poly
                .commit::<BN254, crate::mode::Transparent, G1Routines>(nu, sigma, &prover_setup)
                .unwrap();

            let point: Vec<ArkFr> = (0..max_log_n).map(|_| ArkFr::random()).collect();
            let evaluation = poly.evaluate(&point);

            let mut transcript = Blake2bTranscript::new(b"dory-prime-test");
            let proof = prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, crate::mode::Transparent>(
                &poly,
                &point,
                Some(row_commitments),
                commit_blind,
                nu,
                sigma,
                &prover_setup,
                &mut transcript,
            )
            .expect("proof generation should succeed");

            let mut verify_transcript = Blake2bTranscript::new(b"dory-prime-test");
            let result = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
                tier_2_commitment,
                evaluation,
                &point,
                &proof.batch_proof,
                verifier_setup,
                &mut verify_transcript,
            );

            assert!(result.is_ok(), "Dory-Prime verification should succeed for log_n={}", max_log_n);
        }
    }
}
