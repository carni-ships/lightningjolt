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
use crate::proof::DoryProof;
use crate::messages::ScalarProductMessage;

use super::proof::DoryPrimeProof;

/// Generate a Dory-Prime proof with tree-parallel reduce-and-fold phase
///
/// This implementation uses ForkableDoryProverState to parallelize
/// the reduce-and-fold protocol using a tree-based approach.
///
/// # Tree-Parallel Strategy
///
/// For each round, we:
/// 1. Fork the prover state into two branches
/// 2. Compute first messages for both branches in parallel (via rayon::join)
/// 3. Append first message to transcript and sample beta challenge
/// 4. Apply beta to both states
/// 5. Compute second messages for both branches in parallel
/// 6. Append second message to transcript and sample alpha challenge
/// 7. Apply alpha to both states, select correct branch, recurse
///
/// Key optimizations:
/// - Parallel computation of both branches at each round (fork parallelism)
/// - Within-message parallelism (d1_left/d1_right, etc. via rayon::join)
/// - Transcript stays in main thread for non-Send transcript wrappers
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
    F: Field + Send + Sync + 'static,
    E: PairingCurve + Clone + Send + Sync + 'static,
    E::G1: Group<Scalar = F> + Send + Sync,
    E::G2: Group<Scalar = F> + Send + Sync,
    E::GT: Group<Scalar = F> + Send + Sync,
    M1: DoryRoutines<E::G1> + Send + Sync + 'static,
    M2: DoryRoutines<E::G2> + Send + Sync + 'static,
    P: MultilinearLagrange<F> + Send + Sync,
    T: Transcript<Curve = E>,
    Mo: Mode + Clone + Send + Sync,
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

    // TREE-PARALLEL REDUCE-AND-FOLD PHASE
    // Each round forks the prover state and computes both branches in parallel.
    // This provides significant speedup by overlapping round computations.
    let (first_messages, second_messages, mut prover_state) =
        compute_rounds_tree::<F, E, M1, M2, P, T, Mo>(
            0,
            num_rounds,
            prover_state,
            transcript,
        );

    let gamma = transcript.challenge_scalar(b"gamma");

    // Final scalar product message
    let final_message = prover_state.compute_final_message::<M1, M2>(&gamma);

    // Append final message to transcript and derive d challenge
    // This MUST match the standard prover and verifier transcript operations
    transcript.append_serde(b"final_e1", &final_message.e1);
    transcript.append_serde(b"final_e2", &final_message.e2);
    let _d = transcript.challenge_scalar(b"d");

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

// ============================================================================
// Tree-Parallel Round Processing
// ============================================================================

/// Recursively processes Dory rounds using tree-parallel forking.
///
/// At each round, this function:
/// 1. Forks the prover state into two branches
/// 2. Computes both branches' first messages in parallel via rayon::join
/// 3. Samples the beta challenge from the transcript
/// 4. Applies beta to both states and selects the correct branch
/// 5. Recurses for remaining rounds
///
/// Returns (first_messages, second_messages, final_prover_state).
///
/// The transcript stays in the main thread; parallel closures only operate
/// on the Send+Sync prover state. This ensures compatibility with non-Send
/// transcript wrappers like JoltToDoryTranscript (which uses Rc<RefCell>).
#[allow(clippy::type_complexity)]
fn compute_rounds_tree<F, E, M1, M2, P, T, Mo>(
    round: usize,
    num_rounds: usize,
    mut prover_state: ForkableDoryProverState<E, Mo>,
    transcript: &mut T,
) -> (
    Vec<FirstReduceMessage<E::G1, E::G2, E::GT>>,
    Vec<SecondReduceMessage<E::G1, E::G2, E::GT>>,
    ForkableDoryProverState<E, Mo>,
)
where
    F: Field + Send + Sync + 'static,
    E: PairingCurve + Clone + Send + Sync + 'static,
    E::G1: Group<Scalar = F> + Send + Sync,
    E::G2: Group<Scalar = F> + Send + Sync,
    E::GT: Group<Scalar = F> + Send + Sync,
    M1: DoryRoutines<E::G1> + Send + Sync + 'static,
    M2: DoryRoutines<E::G2> + Send + Sync + 'static,
    P: MultilinearLagrange<F> + Send + Sync,
    T: Transcript<Curve = E>,
    Mo: Mode + Clone + Send + Sync,
{
    if round >= num_rounds {
        // Base case: all rounds complete, return the final state
        return (Vec::new(), Vec::new(), prover_state);
    }

    // Fork the prover state for parallel computation.
    // The transcript stays in the main thread (not Send), so we keep it
    // in the main thread and have closures only operate on the Send+Sync state.
    let (mut left_state, mut right_state) = prover_state.fork();

    // Compute first messages for both branches in parallel.
    // Only the prover state (Send+Sync) is captured by the closures.
    let (left_fm, right_fm) = rayon::join(
        || left_state.clone().compute_first_message::<M1, M2>(),
        || right_state.clone().compute_first_message::<M1, M2>(),
    );

    // Both branches compute the same first message (same input state).
    let first_msg = left_fm.clone();

    // Main thread: append first message to transcript and sample challenge
    transcript.append_serde(b"d1_left", &first_msg.d1_left);
    transcript.append_serde(b"d1_right", &first_msg.d1_right);
    transcript.append_serde(b"d2_left", &first_msg.d2_left);
    transcript.append_serde(b"d2_right", &first_msg.d2_right);
    transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
    transcript.append_serde(b"e2_beta", &first_msg.e2_beta);
    let beta = transcript.challenge_scalar(b"beta");

    // Apply beta to both states (only the selected branch's state will be used)
    let mut left_state_after_beta = left_state;
    let mut right_state_after_beta = right_state;
    left_state_after_beta.apply_first_challenge::<M1, M2>(&beta);
    right_state_after_beta.apply_first_challenge::<M1, M2>(&beta);

    // Compute second messages for both branches in parallel.
    // Only the prover state (Send+Sync) is captured by the closures.
    let (left_sm, right_sm) = rayon::join(
        || left_state_after_beta.compute_second_message::<M1, M2>(),
        || right_state_after_beta.compute_second_message::<M1, M2>(),
    );

    // Both branches compute the same second message (same input state).
    // Main thread: append second message and sample challenge
    transcript.append_serde(b"c_plus", &left_sm.c_plus);
    transcript.append_serde(b"c_minus", &left_sm.c_minus);
    transcript.append_serde(b"e1_plus", &left_sm.e1_plus);
    transcript.append_serde(b"e1_minus", &left_sm.e1_minus);
    transcript.append_serde(b"e2_plus", &left_sm.e2_plus);
    transcript.append_serde(b"e2_minus", &left_sm.e2_minus);
    let alpha = transcript.challenge_scalar(b"alpha");

    // Apply alpha to both states
    left_state_after_beta.apply_second_challenge::<M1, M2>(&alpha);
    right_state_after_beta.apply_second_challenge::<M1, M2>(&alpha);

    // Determine which branch was correct based on beta.
    // Beta is a uniformly random field element; we use its LSB to select the branch.
    // This gives a ~50/50 split, ensuring load balancing across the parallel tree.
    let use_left = !beta.branch_bit();

    // Recurse with the selected branch, discarding the other
    let (mut rest_fm, mut rest_sm, final_state) = if use_left {
        // Discard right branch, continue with left
        compute_rounds_tree::<F, E, M1, M2, P, T, Mo>(
            round + 1,
            num_rounds,
            left_state_after_beta,
            transcript,
        )
    } else {
        // Discard left branch, continue with right
        compute_rounds_tree::<F, E, M1, M2, P, T, Mo>(
            round + 1,
            num_rounds,
            right_state_after_beta,
            transcript,
        )
    };

    // Prepend current round's messages to the results from recursion
    rest_fm.insert(0, first_msg);
    rest_sm.insert(0, left_sm); // left_sm == right_sm (same computation)

    (rest_fm, rest_sm, final_state)
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
            transcript,
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
                transcript,
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
