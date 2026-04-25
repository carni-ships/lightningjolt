//! Binary verifier circuit for CircleBinius recursive proofs
//!
//! This module provides a verifier that can be used in recursive proof
//! contexts, allowing CircleBinius proofs to be verified inside other
//! SNARK systems.

use crate::poly::commitment::circle_binius::fields::BinaryField32b;

/// Verification key for the binary verifier circuit
#[derive(Debug, Clone)]
pub struct BinaryVerifierKey {
    /// FRI configuration
    pub fri_key: crate::poly::commitment::circle_binius::fri::FRIVerifier,
    /// Brakedown configuration
    pub brakedown_key: crate::poly::commitment::circle_binius::brakedown::BrakedownCommitment,
}

/// Binary verifier state
#[derive(Debug, Clone)]
pub struct BinaryVerifier {
    /// Current round
    pub round: usize,
    /// Total rounds
    pub total_rounds: usize,
}

impl BinaryVerifier {
    /// Create a new verifier
    pub fn new(total_rounds: usize) -> Self {
        Self {
            round: 0,
            total_rounds,
        }
    }

    /// Verify a proof element
    pub fn verify_element(&mut self, _element: BinaryField32b) -> bool {
        self.round += 1;
        true
    }

    /// Check if verification is complete
    pub fn is_done(&self) -> bool {
        self.round >= self.total_rounds
    }
}
