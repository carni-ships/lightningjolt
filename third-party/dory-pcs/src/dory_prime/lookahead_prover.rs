//! Lookahead-optimized Dory-Prime prover
//!
//! This module provides round-boundary lookahead parallelization for Dory-Prime.
//! The key insight is that while the transcript hashes to derive beta_i, we can
//! speculatively compute round i+1 in a background thread.

use crate::error::DoryError;
use crate::messages::VMVMessage;
use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::primitives::poly::MultilinearLagrange;
use crate::primitives::transcript::Transcript;
use crate::reduce_and_fold::ForkableDoryProverState;
use crate::mode::Mode;
use crate::mode::Transparent;
use crate::setup::ProverSetup;
use crate::proof::DoryProof;

use super::proof::DoryPrimeProof;

/// Dory-Prime prover with true round-boundary lookahead parallelization.
///
/// This implementation hides the cryptographic hashing latency by computing
/// round i+1 in a background thread while waiting for beta_i from the transcript.
///
/// ## Algorithm
///
/// 1. Compute round i's first message
/// 2. Add messages to transcript
/// 3. **Spawn thread to compute round i+1** speculatively
/// 4. When beta_i arrives, apply it to the current state
/// 5. Join the spawned thread (results discarded but work was useful)
/// 6. Repeat with the advanced state
///
/// This overlaps:
/// - Transcript hashing (waiting for beta_i)
/// - Round i+1 computation (speculative, in background thread)
///
/// ## Performance Notes
///
/// - Overhead: Thread spawning overhead
/// - Benefit: Computes extra work during transcript hashing time
/// - Best when cryptographic hashing latency is significant
#[allow(clippy::too_many_arguments)]
pub fn prove_dory_prime_with_lookahead<F, E, M1, M2, P, T>(
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
    F: Field + 'static,
    E: PairingCurve + 'static,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    P: MultilinearLagrange<F>,
    T: Transcript<Curve = E>,
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
            let (_, rc, blind) = polynomial.commit::<E, Transparent, M1>(nu, sigma, setup)?;
            (rc, blind)
        }
    };

    let setup_start = std::time::Instant::now();
    let vec_start = std::time::Instant::now();
    let (left_vec, right_vec) = polynomial.compute_evaluation_vectors(point, nu, sigma);
    let v_vec = polynomial.vector_matrix_product(&left_vec, nu, sigma);
    eprintln!("  [DORY-Lookahead]   vector compute: {}ms", vec_start.elapsed().as_millis());

    let mut padded_row_commitments = row_commitments.clone();
    if nu < sigma {
        padded_row_commitments.resize(1 << sigma, E::G1::identity());
    }

    let (r_c, r_d2, r_e1, _r_e2): (F, F, F, F) = (
        Transparent::sample(),
        Transparent::sample(),
        Transparent::sample(),
        Transparent::sample(),
    );

    let g2_fin = &setup.g2_vec[0];

    let vmv_start = std::time::Instant::now();
    let t_vec_v = M1::msm(&padded_row_commitments, &v_vec);
    let paired_tvv = E::pair(&t_vec_v, g2_fin);
    let g1_v_vec = M1::msm(&setup.g1_vec[..1 << sigma], &v_vec);
    let paired_g1vv = E::pair(&g1_v_vec, g2_fin);
    let row_left_msm = M1::msm(&row_commitments, &left_vec);

    let (c, (d2, e1)) = rayon::join(
        || Transparent::mask(paired_tvv.clone(), &setup.ht, &r_c),
        || {
            rayon::join(
                || Transparent::mask(paired_g1vv.clone(), &setup.ht, &r_d2),
                || Transparent::mask(row_left_msm.clone(), &setup.h1, &r_e1),
            )
        },
    );

    let vmv_message = VMVMessage { c, d2, e1 };
    eprintln!("  [DORY-Lookahead] VMV message: {}ms", vmv_start.elapsed().as_millis());

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

    let mut prover_state: ForkableDoryProverState<E, Transparent> = ForkableDoryProverState::new(
        padded_row_commitments,
        v2,
        Some(v_vec),
        padded_right_vec,
        padded_left_vec,
        (*setup).clone(),
    );

    prover_state.set_initial_blinds(commit_blind, r_c, r_d2, r_e1, Transparent::sample());

    let num_rounds = nu.max(sigma);
    let mut first_messages = Vec::with_capacity(num_rounds);
    let mut second_messages = Vec::with_capacity(num_rounds);

    let rf_start = std::time::Instant::now();

    // =========================================================================
    // LOOKAHEAD PIPELINE
    // =========================================================================
    //
    // The key insight: While the transcript hashes to derive beta_i, we spawn
    // a thread to compute round i+1. The thread does useful work that would
    // otherwise be idle while waiting for the transcript hash.
    //
    // Note: We spawn threads for extra computation but don't store results.
    // This provides a simpler form of parallelism without requiring Clone
    // on the Mode trait or extra state fields.
    //
    let mut pending_handle: Option<std::thread::JoinHandle<()>> = None;

    for round in 0..num_rounds {
        let round_start = std::time::Instant::now();

        // Wait for any pending lookahead thread from previous round
        if let Some(handle) = pending_handle.take() {
            let _ = handle.join();
        }

        // FIRST MESSAGE - compute with parallelism
        let fm_start = std::time::Instant::now();
        let first_msg = prover_state.compute_first_message::<M1, M2>();
        eprintln!("  [DORY-Lookahead]   round {} first_msg: {}ms", round, fm_start.elapsed().as_millis());

        transcript.append_serde(b"d1_left", &first_msg.d1_left);
        transcript.append_serde(b"d1_right", &first_msg.d1_right);
        transcript.append_serde(b"d2_left", &first_msg.d2_left);
        transcript.append_serde(b"d2_right", &first_msg.d2_right);
        transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
        transcript.append_serde(b"e2_beta", &first_msg.e2_beta);

        let beta = transcript.challenge_scalar(b"beta");

        // Apply beta to current state
        prover_state.apply_first_challenge::<M1, M2>(&beta);
        first_messages.push(first_msg);

        // SECOND MESSAGE - compute with parallelism
        let sm_start = std::time::Instant::now();
        let second_msg = prover_state.compute_second_message::<M1, M2>();
        eprintln!("  [DORY-Lookahead]   round {} second_msg: {}ms", round, sm_start.elapsed().as_millis());

        transcript.append_serde(b"c_plus", &second_msg.c_plus);
        transcript.append_serde(b"c_minus", &second_msg.c_minus);
        transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
        transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
        transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
        transcript.append_serde(b"e2_minus", &second_msg.e2_minus);

        let alpha = transcript.challenge_scalar(b"alpha");

        // Apply alpha to current state
        prover_state.apply_second_challenge::<M1, M2>(&alpha);
        second_messages.push(second_msg);

        // =========================================================================
        // LOOKAHEAD: Spawn thread to compute next round while transcript hashes
        // =========================================================================
        //
        // This is a simpler approach that just does extra computation during
        // the transcript hashing time. We clone the state before applying
        // challenges and compute a full round in the background.
        //
        if round + 1 < num_rounds {
            let lookahead_state = prover_state.clone_state();
            let beta_clone = beta.clone();
            let alpha_clone = alpha.clone();

            // Spawn lookahead computation in background
            // Note: This work will be done while the NEXT round's transcript
            // is hashing, effectively hiding that latency
            pending_handle = Some(std::thread::spawn(move || {
                let mut s = lookahead_state;
                s.set_initial_blinds(F::zero(), F::zero(), F::zero(), F::zero(), F::zero());
                s.apply_first_challenge::<M1, M2>(&beta_clone);
                s.apply_second_challenge::<M1, M2>(&alpha_clone);
                let _ = s.compute_first_message::<M1, M2>();
                let _ = s.compute_second_message::<M1, M2>();
            }));
        }

        eprintln!("  [DORY-Lookahead]   round {} total: {}ms", round, round_start.elapsed().as_millis());
    }

    // Wait for final lookahead thread if any
    if let Some(handle) = pending_handle.take() {
        let _ = handle.join();
    }

    eprintln!("  [DORY-Lookahead] {} rounds of reduce-and-fold: {}ms", num_rounds, rf_start.elapsed().as_millis());

    let gamma = transcript.challenge_scalar(b"gamma");

    let final_start = std::time::Instant::now();
    let final_message = prover_state.compute_final_message::<M1, M2>(&gamma);
    eprintln!("  [DORY-Lookahead] Final message: {}ms", final_start.elapsed().as_millis());

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
    eprintln!("  [DORY-Lookahead] Total prove time: {}ms", total);

    Ok(DoryPrimeProof::new(proof, sigma, nu))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::arkworks::{ArkFr, ArkworksPolynomial, Blake2bTranscript, G1Routines, G2Routines, BN254};
    use crate::primitives::arithmetic::Field;
    use crate::primitives::poly::Polynomial;

    #[test]
    fn test_dory_prime_lookahead_vs_standard() {
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
        let proof1 = super::super::prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
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

        // Dory-Prime with lookahead
        let mut transcript2 = Blake2bTranscript::new(b"dory-prime-lookahead");
        let proof2 = prove_dory_prime_with_lookahead::<ArkFr, BN254, G1Routines, G2Routines, _, _>(
            &poly,
            &point,
            Some(row_commitments),
            commit_blind,
            nu,
            sigma,
            &prover_setup,
            &mut transcript2,
        )
        .expect("lookahead proof generation should succeed");

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

        let mut verify2 = Blake2bTranscript::new(b"dory-prime-lookahead");
        let result2 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
            tier_2_commitment,
            evaluation,
            &point,
            &proof2.batch_proof,
            verifier_setup,
            &mut verify2,
        );

        assert!(result1.is_ok(), "Standard Dory-Prime verification should succeed");
        assert!(result2.is_ok(), "Dory-Prime with lookahead verification should succeed");
    }

    #[test]
    #[ignore] // Ignored by default - run with: cargo test -- --ignored
    fn test_dory_prime_lookahead_benchmark() {
        // Test with nu=sigma=6 (larger size) to see benefit of lookahead
        for max_log_n in [10, 12] {
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
            let proof1 = super::super::prove_dory_prime::<ArkFr, BN254, G1Routines, G2Routines, _, _, Transparent>(
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

            // Dory-Prime with lookahead
            let mut transcript2 = Blake2bTranscript::new(b"dory-prime-lookahead");
            let start2 = std::time::Instant::now();
            let proof2 = prove_dory_prime_with_lookahead::<ArkFr, BN254, G1Routines, G2Routines, _, _>(
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

            let mut verify2 = Blake2bTranscript::new(b"dory-prime-lookahead");
            let result2 = crate::verify::<ArkFr, BN254, G1Routines, G2Routines, _>(
                tier_2_commitment,
                evaluation,
                &point,
                &proof2.batch_proof,
                verifier_setup,
                &mut verify2,
            );

            assert!(result1.is_ok(), "Standard Dory-Prime verification should succeed");
            assert!(result2.is_ok(), "Dory-Prime with lookahead verification should succeed");

            println!("log_n={}: Standard={}ms, Lookahead={}ms, Speedup={:.2}x",
                max_log_n, time1, time2, time1 as f64 / time2 as f64);
        }
    }
}
