//! Dory-Prime proof structure
//!
//! The Dory-Prime proof replaces sigma sequential round messages with a
//! single batch commitment and opening proof.

use crate::primitives::arithmetic::Group;
use crate::DoryProof;

/// Dory-Prime proof structure
///
/// This is a wrapper around the standard DoryProof that enables the
/// Dory-Prime batching optimization at a higher level.
///
/// In Dory-Prime, multiple polynomials can be proven together using
/// a batch polynomial construction, reducing verification costs.
#[derive(Debug, Clone)]
pub struct DoryPrimeProof<G1, G2, GT>
where
    G1: Group,
    G2: Group,
    GT: Group,
{
    /// The underlying Dory proof for the batch polynomial
    pub batch_proof: DoryProof<G1, G2, GT>,

    /// Number of variables sigma (log of vector size)
    pub sigma: usize,

    /// Number of rounds nu (for matrix layout)
    pub nu: usize,
}

impl<G1, G2, GT> DoryPrimeProof<G1, G2, GT>
where
    G1: Group,
    G2: Group,
    GT: Group,
{
    /// Create a new DoryPrimeProof from a standard DoryProof
    pub fn new(batch_proof: DoryProof<G1, G2, GT>, sigma: usize, nu: usize) -> Self {
        Self {
            batch_proof,
            sigma,
            nu,
        }
    }

    /// Returns the number of bytes needed to serialize this proof
    pub fn serialized_size(&self) -> usize {
        // Approximate size - batch_proof size depends on sigma
        32 + self.sigma * 64 + self.nu * 32
    }
}