# Batch Opening Proof Aggregation Implementation Plan

## Executive Summary

**Goal**: Aggregate all transaction proofs into a single block proof, reducing Phase 2 verification from O(n) individual Dory verifications to O(1).

**Current State**:
- 162 proofs -> 162 individual Dory verifications
- Phase 2 (verification) takes ~1.3s

**Target State**:
- Single aggregated proof
- Phase 2 verification time ~= 8ms

## Key Insight: Prover-Side Aggregation Required

Post-hoc aggregation doesn't work because:
1. Each `joint_opening_proof` is computed with different Fiat-Shamir challenges
2. Dory verification requires matching the exact transcript state
3. Cannot combine already-computed proofs; must compute combined proof from combined polynomials

## Files to Modify

| File | Changes |
|------|---------|
| `jolt-core/src/zkvm/prover.rs` | Add `prove_stage8_batch()` |
| `jolt-core/src/zkvm/verifier.rs` | Add `verify_stage8_batch()` |
| `jolt-core/src/poly/opening_proof.rs` | Extend `DoryOpeningState` for batch |
| `jolt-core/src/poly/rlc_polynomial.rs` | Add batch RLC support |
| `jolt-core/src/zkvm/proof_serialization.rs` | Add `BatchBlockProof` type |
| `ethrex-trace/examples/ethrex_block_prove.rs` | Integrate batch flow |

## Implementation Steps

### Step 1: Add Batch Proof Types

```rust
pub struct BatchBlockProof<F, C, PCS, FS> {
    pub combined_commitment: PCS::Commitment,
    pub combined_opening_proof: PCS::Proof,
    pub sumcheck_proofs: Vec<SumcheckProofBundle<F, C, FS>>,
    pub commitment_sets: Vec<Vec<PCS::Commitment>>,
    pub claims: Vec<F>,
    pub batch_gamma: F,
}
```

### Step 2: Extend DoryOpeningState for Batch

```rust
impl<F: JoltField> DoryOpeningState<F> {
    pub fn build_batch_rlc<PCS: CommitmentScheme<Field = F>>(
        &self,
        transaction_polys: Vec<Vec<(CommittedPolynomial, F, PCS::OpeningProofHint)>>,
        batch_gamma: F,
        num_rows: usize,
    ) -> (MultilinearPolynomial<F>, PCS::OpeningProofHint) {
        // 1. Compute combined polynomial coefficients
        // 2. Combine hints using the same coefficients
        // 3. Build combined RLC polynomial
    }
}
```

### Step 3: Modify Prover Stage 8

```rust
pub fn prove_stage8_batch(
    &mut self,
    transaction_data: Vec<TransactionBatchData<F, PCS>>,
    batch_gamma: F,
) -> BatchStage8Proof<F, PCS> {
    // 1. Collect polynomial data and claims from all transactions
    // 2. Compute combined commitment using homomorphic properties
    // 3. Build combined RLC polynomial
    // 4. Generate single Dory proof for combined polynomial
}
```

### Step 4: Modify Verifier Stage 8

```rust
pub fn verify_stage8_batch(
    &mut self,
    batch_proof: &BatchStage8Proof<F, PCS>,
    expected_claims: &[F],
) -> Result<bool, ProofVerifyError> {
    // 1. Compute combined claim from all transaction claims
    // 2. Single Dory verification
    // 3. Bind opening inputs for Fiat-Shamir
}
```

## Expected Performance Impact

| Phase | Current | Expected |
|-------|---------|----------|
| Phase 1 | ~15.4s | ~16.2s (+5%) |
| Phase 2 | ~1.2s | ~0.01s (-99%) |
| Total | ~16.6s | ~16.2s (-2%) |

**Note**: Phase 1 time may increase slightly due to combined RLC computation, but Phase 2 becomes negligible.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory pressure | High | Process in batches of 50-100 |
| Transcript mismatch | Critical | Use same gamma challenge binding |
| Different trace lengths | Medium | Zero-pad shorter traces |
| ZK mode compatibility | Medium | Maintain y_com commitments |

## Implementation Order

1. **Phase 1**: Add `BatchBlockProof` type, extend `DoryOpeningState`
2. **Phase 2**: Implement `prove_stage8_batch()` and `verify_stage8_batch()`
3. **Phase 3**: Update `ethrex_block_prove.rs` with batch flow
4. **Phase 4**: Optimize with chunked processing

---

*Created: 2026-04-22*
