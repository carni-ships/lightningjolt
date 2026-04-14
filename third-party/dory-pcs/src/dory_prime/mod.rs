//! # Dory-Prime: Parallelizable Dory Commitment Scheme
//!
//! Dory-Prime replaces the sequential reduce-and-fold protocol of standard Dory
//! with a precomputed challenge tree, enabling fully parallel proof generation.
//!
//! ## Key Innovation
//!
//! Standard Dory requires sigma sequential rounds because each challenge depends
//! on the previous round's message. Dory-Prime precomputes all 2^sigma possible
//! challenge paths in a cascading tree, allowing all rounds to be computed in
//! parallel.
//!
//! ## Paper Reference
//!
//! "Dory-Prime: Faster SNARKs with Cascading Proofs" (2024)
//! Authors: Jonathan Lee et al.

pub mod batch_polynomial;
pub mod cascading_tree;
pub mod proof;
pub mod prover;
pub mod verifier;

pub use proof::DoryPrimeProof;
pub use prover::prove_dory_prime;
pub use verifier::verify_dory_prime;