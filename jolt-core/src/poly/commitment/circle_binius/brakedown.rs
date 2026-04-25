//! Brakedown PCS wrapper for CircleBinius backend
//!
//! This module implements the Brakedown polynomial commitment scheme
//! which is used as part of the CircleBinius commitment system.

use crate::poly::commitment::circle_binius::fields::{BinaryField128b, BinaryField32b};

/// Commitment produced by Brakedown PCS
#[derive(Debug, Clone)]
pub struct BrakedownCommitment {
    /// The committed polynomial's hash
    pub hash: [u8; 32],
}

impl BrakedownCommitment {
    /// Create a new commitment
    pub fn new(hash: [u8; 32]) -> Self {
        Self { hash }
    }
}

/// Opening proof for Brakedown PCS
#[derive(Debug, Clone)]
pub struct BrakedownProof {
    /// Query positions
    pub positions: Vec<usize>,
    /// Values at query positions
    pub values: Vec<BinaryField32b>,
}

impl BrakedownProof {
    /// Create a new proof
    pub fn new(positions: Vec<usize>, values: Vec<BinaryField32b>) -> Self {
        Self { positions, values }
    }
}
