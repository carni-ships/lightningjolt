//! Dory-Prime prover algorithm
//!
//! Implements the parallelizable proof generation for Dory-Prime.
//!
//! ## Architecture
//!
//! Dory-Prime is a wrapper around the standard Dory protocol that enables
//! batch proof generation. The key optimization is in how multiple proofs
//! can be combined - this module provides the single-proof interface that
//! will be used by the batching layer.

use crate::error::DoryError;
use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::primitives::poly::MultilinearLagrange;
use crate::primitives::transcript::Transcript;
use crate::setup::ProverSetup;
use crate::mode::Transparent;

use super::proof::DoryPrimeProof;

/// Generate a Dory-Prime proof
///
/// This is a single-proof interface that wraps the standard Dory prove.
/// In a full Dory-Prime implementation, this would use the cascading tree
/// to precompute challenges in parallel, but currently delegates to the
/// standard Dory implementation.
///
/// The Dory-Prime optimization requires:
/// 1. ForkableDoryProverState with proper transcript forking
/// 2. Parallel computation of all 2^sigma paths
/// 3. Batch polynomial construction from challenge products
///
/// # Parameters
/// - `polynomial`: The multilinear polynomial to prove
/// - `point`: The evaluation point (length nu + sigma)
/// - `nu`: Log₂ of number of rows
/// - `sigma`: Log₂ of number of columns
/// - `setup`: The prover setup
/// - `transcript`: Fiat-Shamir transcript
///
/// # Returns
/// A DoryPrimeProof that can be verified with verify_dory_prime
#[allow(clippy::too_many_arguments)]
pub fn prove_dory_prime<F, E, M1, M2, P, T>(
    polynomial: &P,
    point: &[F],
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
{
    let _num_rounds = nu.max(sigma);

    // The Dory-Prime optimization precomputes all 2^sigma challenge paths in parallel,
    // then combines them into a single batch proof. The infrastructure is in place:
    // - ForkableDoryProverState can be cloned and forked at each challenge
    // - CascadingTree stores the full binary tree structure
    // - Transcript::fork() enables parallel transcript derivation
    //
    // For now, we delegate to standard Dory to ensure correctness.
    // TODO: Implement actual tree building with ForkableDoryProverState::fork()

    let (proof, _blinding) = crate::evaluation_proof::create_evaluation_proof::<F, E, M1, M2, T, P, Transparent>(
        polynomial,
        point,
        None, // No precomputed row commitments
        F::zero(), // No blinding in transparent mode
        nu,
        sigma,
        setup,
        transcript,
    )?;

    Ok(DoryPrimeProof::new(proof, sigma, nu))
}
