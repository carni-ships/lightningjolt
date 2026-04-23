//! Batch prover for aggregating multiple transaction proofs into a single Dory opening proof.
//!
//! ## Overview
//!
//! This module provides batch aggregation utilities for Dory commitments and hints.
//! The key insight is that Dory commitments support homomorphic combination.
//!
//! ## Architecture Notes
//!
//! **IMPORTANT**: True batch proving (single Dory proof for N transactions) requires
//! prover-side architectural changes. The prover must be aware of multiple transactions
//! from the start so it can build a combined RLC polynomial from all traces together.
//!
//! Current architecture:
//! - Each `prover()` call runs Stages 1-8 for ONE transaction
//! - Opening hints are generated during `generate_and_commit_witness_polynomials()`
//! - Stage 8 uses `build_streaming_rlc()` which reads from a single trace
//!
//! For batch proving, we'd need to:
//! 1. Extract opening hints and commitments from each transaction
//! 2. Build a combined RLC polynomial that reads from ALL traces
//! 3. Call `PCS::prove()` once with the combined hint
//!
//! This requires modifications to `prove_stage8()` and the guest interface.

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme, ZkEvalCommitment};
use crate::utils::math::Math;

/// Maximum number of transactions to batch
pub const MAX_BATCH_SIZE: usize = 1024;

/// Data for a single transaction in batch Stage 8
#[derive(Debug)]
pub struct BatchTxData<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Opening proof hints for this transaction's polynomials
    pub opening_hints: Vec<PCS::OpeningProofHint>,
    /// Claims at opening point for each polynomial (same order as hints)
    pub polynomial_claims: Vec<F>,
    /// Commitments for each polynomial
    pub commitments: Vec<PCS::Commitment>,
    /// Joint claim for this transaction
    pub joint_claim: F,
}

/// Combined result from batch Stage 8 proving
#[derive(Debug)]
pub struct BatchStage8Proof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// The combined Dory opening proof
    pub proof: PCS::Proof,
    /// Combined hint for the aggregated commitments
    pub combined_hint: PCS::OpeningProofHint,
    /// Combined commitment: sum_i(gamma^i * commitment_i)
    pub combined_commitment: PCS::Commitment,
    /// Combined claim: sum_i(gamma^i * joint_claim_i)
    pub combined_claim: F,
    /// Number of transactions in the batch
    pub num_transactions: usize,
}

/// Batch prover utilities for combining hints and commitments
pub struct BatchProver<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
{
    _phantom: std::marker::PhantomData<(F, C, PCS)>,
}

impl<F, C, PCS> BatchProver<F, C, PCS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
{
    /// Combine multiple transaction hints into a single hint
    ///
    /// Uses `PCS::combine_hints()` to aggregate row commitments via RLC.
    /// The result can be used with `PCS::prove()` to generate a batch proof.
    pub fn combine_hints(
        tx_data: &[BatchTxData<F, PCS>],
        gamma_powers: &[F],
        num_rows: usize,
    ) -> PCS::OpeningProofHint {
        // Collect all hints from all transactions
        let all_hints: Vec<_> = tx_data
            .iter()
            .flat_map(|tx| tx.opening_hints.iter().cloned())
            .collect();

        PCS::combine_hints(all_hints, gamma_powers, num_rows)
    }

    /// Combine multiple transaction commitments into a single commitment
    ///
    /// Uses `PCS::combine_commitments()` to aggregate GT commitments via RLC.
    pub fn combine_commitments(
        tx_data: &[BatchTxData<F, PCS>],
        gamma_powers: &[F],
    ) -> PCS::Commitment {
        let all_commitments: Vec<_> = tx_data
            .iter()
            .flat_map(|tx| tx.commitments.iter().cloned())
            .collect();

        PCS::combine_commitments(&all_commitments, gamma_powers)
    }

    /// Compute the combined joint claim from all transactions
    pub fn combine_claims(tx_data: &[BatchTxData<F, PCS>], batch_gamma: F) -> F {
        let mut combined_claim = F::zero();

        // Compute gamma powers iteratively to avoid needing exp_u64
        let mut gamma_power = F::one();
        for tx in tx_data.iter() {
            combined_claim = combined_claim + gamma_power * tx.joint_claim;
            gamma_power = gamma_power * batch_gamma;
        }

        combined_claim
    }

    /// Generate gamma powers for batch aggregation
    ///
    /// For each transaction i and polynomial j, computes:
    /// gamma_powers[i * num_polys_per_tx + j] = batch_gamma^i * inner_gamma^j
    pub fn compute_gamma_powers(
        num_transactions: usize,
        num_polys_per_tx: usize,
        batch_gamma: F,
        transcript: &mut impl crate::transcripts::Transcript,
    ) -> Vec<F> {
        let inner_gamma_powers = transcript.challenge_scalar_powers(num_polys_per_tx);
        let mut gamma_powers = Vec::with_capacity(num_transactions * num_polys_per_tx);

        let mut tx_gamma = F::one();
        for _tx_idx in 0..num_transactions {
            for poly_idx in 0..num_polys_per_tx {
                let inner_gamma = inner_gamma_powers
                    .get(poly_idx)
                    .copied()
                    .unwrap_or(F::zero());
                gamma_powers.push(tx_gamma * inner_gamma);
            }
            tx_gamma = tx_gamma * batch_gamma;
        }

        gamma_powers
    }
}
