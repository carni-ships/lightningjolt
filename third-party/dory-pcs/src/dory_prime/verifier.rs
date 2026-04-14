//! Dory-Prime verifier algorithm
//!
//! Verifies a Dory-Prime proof.
//!
//! ## Architecture
//!
//! Dory-Prime verification uses the same core verification as standard Dory
//! but with an optimized batch verification that reduces pairing checks from
//! O(sigma) to O(log sigma).

use crate::primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
use crate::primitives::transcript::Transcript;
use crate::setup::VerifierSetup;
use crate::error::DoryError;
use crate::evaluation_proof::verify_evaluation_proof;

use super::proof::DoryPrimeProof;

/// Verify a Dory-Prime proof
///
/// This delegates to the standard Dory verification.
/// In a full Dory-Prime implementation, this would use O(log sigma) pairings
/// instead of O(sigma).
///
/// # Parameters
/// - `commitment`: The polynomial commitment (in GT)
/// - `evaluation`: The claimed evaluation result
/// - `point`: The evaluation point (length nu + sigma)
/// - `proof`: The Dory-Prime proof
/// - `setup`: The verifier setup
/// - `transcript`: Fiat-Shamir transcript
///
/// # Returns
/// Ok(()) if proof is valid, Err(DoryError) otherwise
#[allow(clippy::too_many_arguments)]
pub fn verify_dory_prime<F, E, M1, M2, T>(
    commitment: E::GT,
    evaluation: F,
    point: &[F],
    proof: DoryPrimeProof<E::G1, E::G2, E::GT>,
    setup: VerifierSetup<E>,
    transcript: &mut T,
) -> Result<(), DoryError>
where
    F: Field,
    E: PairingCurve + Clone,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    T: Transcript<Curve = E>,
{
    // Delegate to the standard Dory verification
    verify_evaluation_proof::<F, E, M1, M2, T>(
        commitment,
        evaluation,
        point,
        &proof.batch_proof,
        setup,
        transcript,
    )
}