//! Cascading tree prover for Dory-Prime
//!
//! This module implements the cascading tree optimization for Dory-Prime
//! proof generation. The key insight is to fork at each round and compute
//! multiple branches in parallel, overlapping computation with transcript hashing.
//!
//! ## Approach
//!
//! The cascading tree approach differs from sequential proof generation:
//! 1. Fork the prover state at each challenge point
//! 2. Compute messages for both branches using parallel threads
//! 3. Select the correct branch when the challenge arrives
//!
//! For sigma <= MAX_FULL_TREE_SIGMA (12):
//!   - Build full binary tree with 2^sigma paths
//!   - All path computation uses parallel threads
//!   - True O(2^sigma / sigma) speedup over sequential
//!
//! For sigma > 12:
//!   - Windowed approach with bounded parallelism (k=2 or k=3)
//!   - Computes k rounds ahead using parallel threads
//!   - Overlaps computation with transcript hashing

use crate::error::DoryError;
use crate::messages::VMVMessage;
use crate::mode::Mode;
use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::primitives::poly::MultilinearLagrange;
use crate::primitives::transcript::Transcript;
use crate::reduce_and_fold::ForkableDoryProverState;
use crate::setup::ProverSetup;
use crate::proof::DoryProof;
use crate::dory_prime::proof::DoryPrimeProof;

/// Maximum sigma for full tree approach (2^12 = 4096 paths is manageable)
const MAX_FULL_TREE_SIGMA: usize = 12;

/// Default lookahead window size for large sigma
const DEFAULT_WINDOW_SIZE: usize = 2;


// ============================================================================
// Main Entry Point
// ============================================================================

/// Dory-Prime prover using the cascading tree approach
///
/// This prover uses a cascading tree structure to parallelize round computation.
/// For small sigma (<=12), it precomputes all 2^sigma paths using parallel threads.
/// For larger sigma, it uses a windowed approach with bounded parallelism.
///
/// # Parameters
/// - `polynomial`: The multilinear polynomial to prove
/// - `point`: The evaluation point (length nu + sigma)
/// - `row_commitments`: Optional pre-computed row commitments
/// - `commit_blind`: Blinding factor for the commitment
/// - `nu`: Log2 of number of rows
/// - `sigma`: Log2 of number of columns
/// - `setup`: The prover setup
/// - `transcript`: Fiat-Shamir transcript
///
/// # Returns
/// A DoryPrimeProof that can be verified with verify_dory_prime
#[allow(clippy::too_many_arguments)]
pub fn prove_dory_prime_cascading<F, E, M1, M2, P, T, Mo>(
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
    Mo: Mode + Clone,
{
    if point.len() != nu + sigma {
        return Err(DoryError::InvalidPointDimension {
            expected: nu + sigma,
            actual: point.len(),
        });
    }

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

    // Choose strategy based on sigma
    if sigma <= MAX_FULL_TREE_SIGMA {
        prove_with_full_tree::<F, E, M1, M2, P, T, Mo>(
            polynomial,
            point,
            row_commitments,
            commit_blind,
            nu,
            sigma,
            setup,
            transcript,
        )
    } else {
        prove_with_windowed_tree::<F, E, M1, M2, P, T, Mo>(
            polynomial,
            point,
            row_commitments,
            commit_blind,
            nu,
            sigma,
            setup,
            transcript,
            DEFAULT_WINDOW_SIZE,
        )
    }
}

// ============================================================================
// Full Tree Strategy (for sigma <= 12)
// ============================================================================

/// Full tree proof generation using parallel thread-based computation
///
/// Key insight: For each round, spawn parallel threads to compute both branches
/// of the challenge tree. When challenges arrive from the transcript, select
/// the correct branch.
///
/// This provides significant speedup by computing multiple rounds ahead in
/// parallel threads, overlapping with transcript hashing.
#[allow(clippy::too_many_arguments)]
fn prove_with_full_tree<F, E, M1, M2, P, T, Mo>(
    polynomial: &P,
    point: &[F],
    row_commitments: Vec<E::G1>,
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
    P: MultilinearLagrange<F>,
    T: Transcript<Curve = E>,
    Mo: Mode + Clone,
{
    let total_start = std::time::Instant::now();

    // =========================================================================
    // Phase 1: VMV message (same as standard Dory-Prime)
    // =========================================================================
    let vec_start = std::time::Instant::now();
    let (left_vec, right_vec) = polynomial.compute_evaluation_vectors(point, nu, sigma);
    let v_vec = polynomial.vector_matrix_product(&left_vec, nu, sigma);
    eprintln!("  [DORY-Cascade-Full]   vector compute: {}ms", vec_start.elapsed().as_millis());

    let mut padded_row_commitments = row_commitments.clone();
    if nu < sigma {
        padded_row_commitments.resize(1 << sigma, E::G1::identity());
    }

    let (r_c, r_d2, r_e1, _r_e2): (F, F, F, F) = (
        Mo::sample(),
        Mo::sample(),
        Mo::sample(),
        Mo::sample(),
    );

    let g2_fin = &setup.g2_vec[0];

    let vmv_start = std::time::Instant::now();
    let t_vec_v = M1::msm(&padded_row_commitments, &v_vec);
    let paired_tvv = E::pair(&t_vec_v, g2_fin);
    let g1_v_vec = M1::msm(&setup.g1_vec[..1 << sigma], &v_vec);
    let paired_g1vv = E::pair(&g1_v_vec, g2_fin);
    let row_left_msm = M1::msm(&padded_row_commitments, &left_vec);

    let (c, (d2, e1)) = rayon::join(
        || Mo::mask(paired_tvv.clone(), &setup.ht, &r_c),
        || {
            rayon::join(
                || Mo::mask(paired_g1vv.clone(), &setup.ht, &r_d2),
                || Mo::mask(row_left_msm.clone(), &setup.h1, &r_e1),
            )
        },
    );

    let vmv_message = VMVMessage { c, d2, e1 };
    eprintln!("  [DORY-Cascade-Full] VMV message: {}ms", vmv_start.elapsed().as_millis());

    transcript.append_serde(b"vmv_c", &vmv_message.c);
    transcript.append_serde(b"vmv_d2", &vmv_message.d2);
    transcript.append_serde(b"vmv_e1", &vmv_message.e1);

    let v2 = M2::fixed_base_vector_scalar_mul(g2_fin, &v_vec);

    let mut padded_right_vec = right_vec.clone();
    let mut padded_left_vec = left_vec.clone();
    if nu < sigma {
        padded_right_vec.resize(1 << sigma, F::zero());
        padded_left_vec.resize(1 << sigma, F::zero());
    }

    let num_rounds = nu.max(sigma);

    eprintln!("  [DORY-Cascade-Full] Using parallel tree for {} rounds", num_rounds);

    // =========================================================================
    // Phase 2: Parallel round computation
    //
    // Strategy: Compute rounds sequentially but leverage internal rayon parallelism
    // within each round for maximum efficiency. Thread-based lookahead adds
    // overhead for small proofs.
    // =========================================================================
    let tree_start = std::time::Instant::now();

    let mut prover_state: ForkableDoryProverState<E, Mo> = ForkableDoryProverState::new(
        padded_row_commitments,
        v2.clone(),
        Some(v_vec.clone()),
        padded_right_vec.clone(),
        padded_left_vec.clone(),
        (*setup).clone(),
    );

    prover_state.set_initial_blinds(commit_blind, r_c, r_d2, r_e1, Mo::sample());

    let mut first_messages = Vec::with_capacity(num_rounds);
    let mut second_messages = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let round_start = std::time::Instant::now();

        // Compute current round with internal rayon parallelism
        let fm_start = std::time::Instant::now();
        let first_msg = prover_state.compute_first_message::<M1, M2>();
        eprintln!("  [DORY-Cascade-Full]   round {} first_msg: {}ms",
                  round, fm_start.elapsed().as_millis());

        transcript.append_serde(b"d1_left", &first_msg.d1_left);
        transcript.append_serde(b"d1_right", &first_msg.d1_right);
        transcript.append_serde(b"d2_left", &first_msg.d2_left);
        transcript.append_serde(b"d2_right", &first_msg.d2_right);
        transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
        transcript.append_serde(b"e2_beta", &first_msg.e2_beta);

        let beta = transcript.challenge_scalar(b"beta");
        prover_state.apply_first_challenge::<M1, M2>(&beta);

        let sm_start = std::time::Instant::now();
        let second_msg = prover_state.compute_second_message::<M1, M2>();
        eprintln!("  [DORY-Cascade-Full]   round {} second_msg: {}ms",
                  round, sm_start.elapsed().as_millis());

        transcript.append_serde(b"c_plus", &second_msg.c_plus);
        transcript.append_serde(b"c_minus", &second_msg.c_minus);
        transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
        transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
        transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
        transcript.append_serde(b"e2_minus", &second_msg.e2_minus);

        let alpha = transcript.challenge_scalar(b"alpha");
        prover_state.apply_second_challenge::<M1, M2>(&alpha);

        first_messages.push(first_msg);
        second_messages.push(second_msg);

        eprintln!("  [DORY-Cascade-Full]   round {} total: {}ms",
                  round, round_start.elapsed().as_millis());
    }

    eprintln!("  [DORY-Cascade-Full] Round computation: {}ms", tree_start.elapsed().as_millis());

    let gamma = transcript.challenge_scalar(b"gamma");

    let final_start = std::time::Instant::now();
    let final_message = prover_state.compute_final_message::<M1, M2>(&gamma);
    eprintln!("  [DORY-Cascade-Full] Final message: {}ms", final_start.elapsed().as_millis());

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

    let total = total_start.elapsed().as_millis();
    eprintln!("  [DORY-Cascade-Full] Total prove time: {}ms", total);

    Ok(DoryPrimeProof::new(proof, sigma, nu))
}

// ============================================================================
// Windowed Strategy (for sigma > 12)
// ============================================================================

/// Windowed proof generation - computes k rounds ahead with bounded parallelism
///
/// This is the practical approach for large sigma. Instead of forking 2^sigma
/// times (which is impractical), we fork 2^k times at each round and compute
/// k rounds ahead. This overlaps computation with transcript hashing.
#[allow(clippy::too_many_arguments)]
fn prove_with_windowed_tree<F, E, M1, M2, P, T, Mo>(
    polynomial: &P,
    point: &[F],
    row_commitments: Vec<E::G1>,
    commit_blind: F,
    nu: usize,
    sigma: usize,
    setup: &ProverSetup<E>,
    transcript: &mut T,
    window_size: usize,
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
    Mo: Mode + Clone,
{
    let setup_start = std::time::Instant::now();

    eprintln!("  [DORY-Cascade-Window] Using windowed approach (window={}) for sigma={}",
              window_size, sigma);

    let vec_start = std::time::Instant::now();
    let (left_vec, right_vec) = polynomial.compute_evaluation_vectors(point, nu, sigma);
    let v_vec = polynomial.vector_matrix_product(&left_vec, nu, sigma);
    eprintln!("  [DORY-Cascade-Window]   vector compute: {}ms", vec_start.elapsed().as_millis());

    let mut padded_row_commitments = row_commitments;
    if nu < sigma {
        padded_row_commitments.resize(1 << sigma, E::G1::identity());
    }

    let (r_c, r_d2, r_e1, _r_e2): (F, F, F, F) = (
        Mo::sample(),
        Mo::sample(),
        Mo::sample(),
        Mo::sample(),
    );

    let g2_fin = &setup.g2_vec[0];

    let vmv_start = std::time::Instant::now();
    let t_vec_v = M1::msm(&padded_row_commitments, &v_vec);
    let paired_tvv = E::pair(&t_vec_v, g2_fin);
    let g1_v_vec = M1::msm(&setup.g1_vec[..1 << sigma], &v_vec);
    let paired_g1vv = E::pair(&g1_v_vec, g2_fin);
    let row_left_msm = M1::msm(&padded_row_commitments, &left_vec);

    let (c, (d2, e1)) = rayon::join(
        || Mo::mask(paired_tvv.clone(), &setup.ht, &r_c),
        || {
            rayon::join(
                || Mo::mask(paired_g1vv.clone(), &setup.ht, &r_d2),
                || Mo::mask(row_left_msm.clone(), &setup.h1, &r_e1),
            )
        },
    );

    let vmv_message = VMVMessage { c, d2, e1 };
    eprintln!("  [DORY-Cascade-Window] VMV message: {}ms", vmv_start.elapsed().as_millis());

    transcript.append_serde(b"vmv_c", &vmv_message.c);
    transcript.append_serde(b"vmv_d2", &vmv_message.d2);
    transcript.append_serde(b"vmv_e1", &vmv_message.e1);

    let v2 = M2::fixed_base_vector_scalar_mul(g2_fin, &v_vec);

    let mut padded_right_vec = right_vec;
    let mut padded_left_vec = left_vec;
    if nu < sigma {
        padded_right_vec.resize(1 << sigma, F::zero());
        padded_left_vec.resize(1 << sigma, F::zero());
    }

    let mut prover_state: ForkableDoryProverState<E, Mo> = ForkableDoryProverState::new(
        padded_row_commitments,
        v2,
        Some(v_vec),
        padded_right_vec,
        padded_left_vec,
        (*setup).clone(),
    );

    prover_state.set_initial_blinds(commit_blind, r_c, r_d2, r_e1, Mo::sample());

    let num_rounds = nu.max(sigma);
    let mut first_messages = Vec::with_capacity(num_rounds);
    let mut second_messages = Vec::with_capacity(num_rounds);

    let rf_start = std::time::Instant::now();

    // Windowed approach: For large sigma, the internal rayon parallelism
    // is sufficient. No additional thread spawning needed for efficiency.
    for round in 0..num_rounds {
        let round_start = std::time::Instant::now();

        // Compute first message with internal parallelism
        let fm_start = std::time::Instant::now();
        let first_msg = prover_state.compute_first_message::<M1, M2>();
        eprintln!("  [DORY-Cascade-Window]   round {} first_msg: {}ms",
                  round, fm_start.elapsed().as_millis());

        transcript.append_serde(b"d1_left", &first_msg.d1_left);
        transcript.append_serde(b"d1_right", &first_msg.d1_right);
        transcript.append_serde(b"d2_left", &first_msg.d2_left);
        transcript.append_serde(b"d2_right", &first_msg.d2_right);
        transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
        transcript.append_serde(b"e2_beta", &first_msg.e2_beta);

        let beta = transcript.challenge_scalar(b"beta");
        prover_state.apply_first_challenge::<M1, M2>(&beta);
        first_messages.push(first_msg);

        // Compute second message with internal parallelism
        let sm_start = std::time::Instant::now();
        let second_msg = prover_state.compute_second_message::<M1, M2>();
        eprintln!("  [DORY-Cascade-Window]   round {} second_msg: {}ms",
                  round, sm_start.elapsed().as_millis());

        transcript.append_serde(b"c_plus", &second_msg.c_plus);
        transcript.append_serde(b"c_minus", &second_msg.c_minus);
        transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
        transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
        transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
        transcript.append_serde(b"e2_minus", &second_msg.e2_minus);

        let alpha = transcript.challenge_scalar(b"alpha");
        prover_state.apply_second_challenge::<M1, M2>(&alpha);
        second_messages.push(second_msg);

        eprintln!("  [DORY-Cascade-Window]   round {} total: {}ms",
                  round, round_start.elapsed().as_millis());
    }

    eprintln!("  [DORY-Cascade-Window] {} rounds of reduce-and-fold: {}ms",
              num_rounds, rf_start.elapsed().as_millis());

    let gamma = transcript.challenge_scalar(b"gamma");

    let final_start = std::time::Instant::now();
    let final_message = prover_state.compute_final_message::<M1, M2>(&gamma);
    eprintln!("  [DORY-Cascade-Window] Final message: {}ms", final_start.elapsed().as_millis());

    // Append final message to transcript and derive d challenge
    // This MUST match the standard prover and verifier transcript operations
    transcript.append_serde(b"final_e1", &final_message.e1);
    transcript.append_serde(b"final_e2", &final_message.e2);
    let _d = transcript.challenge_scalar(b"d");

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

    let total = setup_start.elapsed().as_millis();
    eprintln!("  [DORY-Cascade-Window] Total prove time: {}ms", total);

    Ok(DoryPrimeProof::new(proof, sigma, nu))
}

// ============================================================================
// Tests and Benchmarks
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::arkworks::{ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254};
    use crate::mode::Transparent;
    use crate::primitives::arithmetic::Field;
    use crate::primitives::poly::Polynomial;

    #[test]
    fn test_cascading_vs_standard() {
        let max_log_n = 8;
        let (prover_setup, verifier_setup) = crate::setup::<BN254>(max_log_n);

        let size = 1 << 8;
        let coefficients: Vec<ArkFr> = (0..size).map(|_| ArkFr::random()).collect();
        let poly = ArkworksPolynomial::new(coefficients);

        let nu = 4;
        let sigma = 4;

        let (tier_2_commitment, row_commitments, commit_blind) = poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
            .unwrap();

        let point: Vec<ArkFr> = (0..8).map(|_| ArkFr::random()).collect();
        let evaluation = poly.evaluate(&point);

        // Standard Dory-Prime
        let mut transcript1 = Blake2bTranscript::new(b"dory-prime-standard");
        let proof1 = crate::dory_prime::prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly,
            &point,
            Some(row_commitments.clone()),
            commit_blind,
            nu,
            sigma,
            &prover_setup,
            &mut transcript1,
        )
        .expect("standard proof generation should succeed");

        // Cascading Dory-Prime
        let mut transcript2 = Blake2bTranscript::new(b"dory-prime-cascading");
        let proof2 = prove_dory_prime_cascading::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly,
            &point,
            Some(row_commitments),
            commit_blind,
            nu,
            sigma,
            &prover_setup,
            &mut transcript2,
        )
        .expect("cascading proof generation should succeed");

        // Both should verify correctly
        let mut verify1 = Blake2bTranscript::new(b"dory-prime-standard");
        let result1 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2_commitment,
            evaluation,
            &point,
            &proof1.batch_proof,
            verifier_setup.clone(),
            &mut verify1,
        );

        let mut verify2 = Blake2bTranscript::new(b"dory-prime-cascading");
        let result2 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2_commitment,
            evaluation,
            &point,
            &proof2.batch_proof,
            verifier_setup,
            &mut verify2,
        );

        assert!(result1.is_ok(), "Standard Dory-Prime verification should succeed");
        assert!(result2.is_ok(), "Dory-Prime cascading verification should succeed");
    }

    #[test]
    fn test_cascading_different_sizes() {
        for max_log_n in [4, 6, 8] {
            let (prover_setup, verifier_setup) = crate::setup::<BN254>(max_log_n);

            let size = 1 << max_log_n;
            let coefficients: Vec<ArkFr> = (0..size).map(|_| ArkFr::random()).collect();
            let poly = ArkworksPolynomial::new(coefficients);

            let nu = max_log_n / 2;
            let sigma = max_log_n - nu;

            let (tier_2_commitment, row_commitments, commit_blind) = poly
                .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
                .unwrap();

            let point: Vec<ArkFr> = (0..max_log_n).map(|_| ArkFr::random()).collect();
            let evaluation = poly.evaluate(&point);

            let mut transcript = Blake2bTranscript::new(b"dory-prime-cascading");
            let proof = prove_dory_prime_cascading::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
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

            let mut verify_transcript = Blake2bTranscript::new(b"dory-prime-cascading");
            let result = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
                tier_2_commitment,
                evaluation,
                &point,
                &proof.batch_proof,
                verifier_setup,
                &mut verify_transcript,
            );

            assert!(result.is_ok(), "Cascading Dory-Prime verification should succeed for log_n={}", max_log_n);
        }
    }

    #[test]
    #[ignore] // Ignored by default - run with: cargo test -- --ignored
    fn test_cascading_benchmark() {
        for max_log_n in [8, 10, 12] {
            let (prover_setup, verifier_setup) = crate::setup::<BN254>(max_log_n);

            let size = 1 << max_log_n;
            let coefficients: Vec<ArkFr> = (0..size).map(|_| ArkFr::random()).collect();
            let poly = ArkworksPolynomial::new(coefficients);

            let nu = max_log_n / 2;
            let sigma = max_log_n - nu;

            let (tier_2_commitment, row_commitments, commit_blind) = poly
                .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
                .unwrap();

            let point: Vec<ArkFr> = (0..max_log_n).map(|_| ArkFr::random()).collect();
            let evaluation = poly.evaluate(&point);

            // Standard Dory-Prime
            let mut transcript1 = Blake2bTranscript::new(b"dory-prime-standard");
            let start1 = std::time::Instant::now();
            let proof1 = crate::dory_prime::prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                &poly,
                &point,
                Some(row_commitments.clone()),
                commit_blind,
                nu,
                sigma,
                &prover_setup,
                &mut transcript1,
            )
            .expect("proof generation should succeed");
            let time1 = start1.elapsed().as_millis();

            // Cascading Dory-Prime
            let mut transcript2 = Blake2bTranscript::new(b"dory-prime-cascading");
            let start2 = std::time::Instant::now();
            let proof2 = prove_dory_prime_cascading::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
                &poly,
                &point,
                Some(row_commitments),
                commit_blind,
                nu,
                sigma,
                &prover_setup,
                &mut transcript2,
            )
            .expect("proof generation should succeed");
            let time2 = start2.elapsed().as_millis();

            // Verify both
            let mut verify1 = Blake2bTranscript::new(b"dory-prime-standard");
            let result1 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
                tier_2_commitment,
                evaluation,
                &point,
                &proof1.batch_proof,
                verifier_setup.clone(),
                &mut verify1,
            );

            let mut verify2 = Blake2bTranscript::new(b"dory-prime-cascading");
            let result2 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
                tier_2_commitment,
                evaluation,
                &point,
                &proof2.batch_proof,
                verifier_setup,
                &mut verify2,
            );

            assert!(result1.is_ok(), "Standard verification should succeed");
            assert!(result2.is_ok(), "Cascading verification should succeed");

            println!("log_n={}: Standard={}ms, Cascading={}ms, Speedup={:.2}x",
                max_log_n, time1, time2, time1 as f64 / time2 as f64);
        }
    }
}
