# Jolt Verifier Data Flow Analysis: Exact Proof Consumption by Stage

This document details the exact proof data consumed by each verification stage, transcript operations, and opening claim accumulation patterns for on-chain implementation.

## Preamble (Pre-Stage 1)

**Fiat-Shamir Preamble** appended to transcript in this order:
1. `max_input_size` (u64)
2. `max_output_size` (u64)
3. `heap_size` (u64)
4. `inputs` (byte array - entire input)
5. `outputs` (byte array - entire output)
6. `panic` (u64 - 0 or 1)
7. `ram_K` (u64)
8. `trace_length` (u64)
9. `entry_address` (u64)

**Then append proof metadata:**
- All commitments from `proof.commitments` array
- `untrusted_advice_commitment` (if present)
- `trusted_advice_commitment` (if present)

---

## Stage 1: Spartan Outer (UniSkip + Remaining Rounds)

### UniSkip First Round (Stage 1a)
**Proof Data Consumed:**
- `proof.stage1_uni_skip_first_round_proof` - contains:
  - Univariate polynomial coefficients OR Pedersen commitments (ZK mode)
  - Univariate-skip domain: `OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE` (typically 256)
  - Extended domain: `OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE`
  - Num coefficients: `OUTER_FIRST_ROUND_POLY_NUM_COEFFS`

**Transcript Operations:**
1. Sample tau (τ_high and τ_rest): `transcript.challenge_vector_optimized(num_rows_bits)` → 1 challenge sampled
2. Append/verify uni-skip first round polynomial proof
3. Sample r0 challenge: `transcript.challenge_scalar_optimized()`

**Opening Accumulator Updates:**
- **Opening ID:** `OpeningId::Polynomial(VirtualPolynomial::UnivariateSkip, SumcheckId::SpartanOuter)`
- **Opening Point:** [r0] (single point, big-endian)
- **Claim:** evaluation of uni-skip polynomial at r0
- Status: marked as pending claim

### Regular Rounds (Stage 1b)
**Sumcheck Instance:**
- `OuterRemainingSumcheckVerifier` - 1 instance
- Num rounds: typically `log2(trace_length) - 1` (univariate skip handles 1 round)
- Degree bound: 3 (cubic sumcheck)

**Proof Data Consumed:**
- `proof.stage1_sumcheck_proof.compressed_polys` - array of:
  - For each of `num_rounds` rounds:
    - Compressed polynomial with degree 3
    - Format: `[c_2, c_3]` (linear term is computed, constant/degree-1 omitted)

**Transcript Operations:**
1. Append input claim: `transcript.append_scalar(b"sumcheck_claim", &input_claim)`
2. Sample batching coefficient: `transcript.challenge_scalar()` → 1 coefficient (only 1 sumcheck instance)
3. For each round i in 0..num_rounds:
   - Append compressed poly: `transcript.append_scalars(b"sumcheck_poly", &poly.coeffs_except_linear_term)`
   - Sample challenge: `transcript.challenge_scalar_optimized()` → r_i
4. After all rounds, flush opening claims to transcript: `opening_accumulator.flush_to_transcript()`

**Opening Accumulator Updates:**
- **Opening Points:** [r0 || r_1 || r_2 || ... || r_n] in big-endian
  - r0 from uni-skip, r_i from regular sumcheck rounds
- **Multiple opening IDs** from polynomial evaluation at the full point:
  - Virtual polynomials from the R1CS: PC, NextPC, Rd, Imm, Rs1Value, Rs2Value, RdWriteValue, etc.
  - Committed polynomials: InstructionRa (multiple indices), TrustedAdvice, UntrustedAdvice
  - Each opening uses SumcheckId::SpartanOuter

**Output from Stage 1:**
- `r_stage1`: Vec of all sumcheck challenges [r_1, r_2, ..., r_n] (NOT including r0)
- Opening claims are flushed to transcript and encoded as opening point + claim value

---

## Stage 2: RAM/Register Mixed Sumchecks (UniSkip + Multiple Instances)

### UniSkip First Round (Stage 2a)
**Proof Data Consumed:**
- `proof.stage2_uni_skip_first_round_proof`
- Univariate-skip domain: `PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE` (typically 64)
- Num coefficients: `PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS`

**Transcript Operations:**
1. Append/verify uni-skip polynomial proof
2. Sample r0: `transcript.challenge_scalar_optimized()`

**Opening Accumulator Updates:**
- **Opening ID:** `OpeningId::Polynomial(VirtualPolynomial::UnivariateSkip, SumcheckId::SpartanProductVirtualization)`

### Regular Rounds (Stage 2b)
**Sumcheck Instances (5 batched together):**
1. `RamReadWriteCheckingVerifier`
   - Num rounds: typically `rw_config.ram_rw_phase1_num_rounds + rw_config.ram_rw_phase2_num_rounds`
   - Degree: 2-3
2. `ProductVirtualRemainderVerifier`
   - Num rounds: varies based on uni-skip params
3. `InstructionLookupsClaimReductionSumcheckVerifier`
   - Num rounds: typically 1
4. `RamRafEvaluationSumcheckVerifier`
   - Num rounds: varies
5. `OutputSumcheckVerifier`
   - Num rounds: 1

**Proof Data Consumed:**
- `proof.stage2_sumcheck_proof.compressed_polys` - batched array of compressed polynomials
- Max num rounds across all instances determines proof length
- Each instance with fewer rounds sends constant polynomials for skipped rounds

**Transcript Operations:**
1. For each instance, append input claim: `transcript.append_scalar(b"sumcheck_claim", &claim)`
2. Sample 5 batching coefficients: `transcript.challenge_vector(5)`
3. For each round i in 0..max_rounds:
   - Append batched compressed poly: `transcript.append_scalars(b"sumcheck_poly", &batched_poly.coeffs_except_linear_term)`
   - Sample challenge: `transcript.challenge_scalar_optimized()` → r_i
4. Flush opening claims to transcript

**Opening Accumulator Updates:**
- **Opening Points:** [r0 || r_1 || ... || r_n] (r0 from uni-skip)
- Multiple opening IDs from all 5 instances:
  - RamReadWriteCheckingVerifier: RamVal, RamReadValue, RamWriteValue openings
  - ProductVirtualRemainderVerifier: Product virtual polynomial openings
  - InstructionLookupsClaimReductionVerifier: Instruction-related openings
  - RamRafEvaluationSumcheckVerifier: Bytecode-related openings
  - OutputSumcheckVerifier: Output constraints openings

**Challenges Needed:**
- `ram_val_check_gamma`: sampled with `transcript.append_bytes(b"ram_val_check_gamma", &[])` then `challenge_scalar()`
  - Used for RAM value checking constraint

**Output from Stage 2:**
- `r_stage2`: Vec of all sumcheck challenges
- All opening claims from 5 instances accumulated

---

## Stage 3: Shift + Instruction + Registers Claim Reduction

**Sumcheck Instances (3 batched):**
1. `ShiftSumcheckVerifier`
   - Num rounds: `log2(trace_length)` bits of trace
   - Degree: 2
2. `InstructionInputSumcheckVerifier`
   - Num rounds: 1 (virtual polynomial evaluation)
3. `RegistersClaimReductionSumcheckVerifier`
   - Num rounds: typically 1-2
   - Handles register read-write checking phase 1

**Proof Data Consumed:**
- `proof.stage3_sumcheck_proof.compressed_polys`
- Max num rounds typically ~log2(T)

**Transcript Operations:**
1. Append 3 input claims
2. Sample 3 batching coefficients
3. For each round: append compressed poly, sample challenge
4. Flush opening claims

**Opening Accumulator Updates:**
- **Opening Point:** [r_1, r_2, ..., r_k] (k = max_rounds)
- Multiple opening IDs from 3 instances using SumcheckId::SpartanShift, SumcheckId::InstructionInputVirtualization, SumcheckId::RegistersClaimReduction

**Output from Stage 3:**
- `r_stage3`: array of challenges

---

## Stage 4: Registers RW Checking + RAM Value Check (2-phase sumcheck)

**Sumcheck Instances (2 batched):**
1. `RegistersReadWriteCheckingVerifier`
   - Num rounds: `rw_config.registers_rw_phase1_num_rounds + rw_config.registers_rw_phase2_num_rounds`
2. `RamValCheckSumcheckVerifier`
   - Num rounds: RAM phase 1 + phase 2
   - Uses `ram_val_check_gamma` from stage 2

**Additional Operations Before Sumcheck:**
```
verifier_accumulate_advice(
    ram_K,
    program_io,
    untrusted_advice_commitment.is_some(),
    trusted_advice_commitment.is_some(),
    opening_accumulator
);
```
- Adds advice polynomial openings to accumulator if present

**Proof Data Consumed:**
- `proof.stage4_sumcheck_proof.compressed_polys`

**Transcript Operations:**
1. Append 2 input claims
2. Sample 2 batching coefficients
3. For each round: append compressed poly, sample challenge
4. Flush opening claims

**Opening Accumulator Updates:**
- Advice polynomial openings at r_address points (from TrustedAdvice, UntrustedAdvice)
- RAM and register value polynomial openings

**Output from Stage 4:**
- `r_stage4`: array of challenges

---

## Stage 5: Lookups Read-RAF + RAM RA Claim Reduction + Registers Val Evaluation

**Sumcheck Instances (3 batched):**
1. `InstructionReadRafSumcheckVerifier`
   - Num rounds: `log2(trace_length)` + bytecode-related bits
2. `RamRaClaimReductionSumcheckVerifier`
   - Num rounds: `log2(ram_K)` related
3. `RegistersValEvaluationSumcheckVerifier`
   - Num rounds: 1

**Proof Data Consumed:**
- `proof.stage5_sumcheck_proof.compressed_polys`

**Transcript Operations:**
1. Append 3 input claims
2. Sample 3 batching coefficients
3. For each round: append compressed poly, sample challenge
4. Flush opening claims

**Opening Accumulator Updates:**
- Lookups RA polynomial openings
- RAM RA polynomial openings

**Output from Stage 5:**
- `r_stage5`: array of challenges

---

## Stage 6: Bytecode Read-RAF + Booleanity + Hamming + Virtual RA + Inc Reduction + Advice Phase 1

**Sumcheck Instances (6-8 batched, variable):**
1. `BytecodeReadRafSumcheckVerifier`
2. `BooleanitySumcheckVerifier`
3. `HammingBooleanitySumcheckVerifier`
4. `RamRaVirtualSumcheckVerifier`
5. `LookupsRaSumcheckVerifier`
6. `IncClaimReductionSumcheckVerifier`
7. `AdviceClaimReductionVerifier` (TrustedAdvice, if present) - Phase 1: cycle variables
8. `AdviceClaimReductionVerifier` (UntrustedAdvice, if present) - Phase 1: cycle variables

**Advice Instances Initialization:**
```rust
if self.trusted_advice_commitment.is_some() {
    advice_reduction_verifier_trusted = Some(AdviceClaimReductionVerifier::new(
        AdviceKind::Trusted,
        program_io.memory_layout,
        trace_length,
        opening_accumulator
    ));
}
if self.proof.untrusted_advice_commitment.is_some() {
    advice_reduction_verifier_untrusted = Some(AdviceClaimReductionVerifier::new(
        AdviceKind::Untrusted,
        program_io.memory_layout,
        trace_length,
        opening_accumulator
    ));
}
```

**Proof Data Consumed:**
- `proof.stage6_sumcheck_proof.compressed_polys`
- Max num rounds: typically up to log2(T) + log_K related bits

**Transcript Operations:**
1. Append input claims for each active instance
2. Sample batching coefficients
3. For each round: append compressed poly, sample challenge
4. Flush opening claims

**Opening Accumulator Updates:**
- Multiple opening IDs from all instances
- For advice instances: opening points at r_cycle (phase 1)

**Output from Stage 6:**
- `r_stage6`: array of challenges

---

## Stage 7: Hamming Weight + Advice Phase 2 (Address Variables)

**Sumcheck Instances (1-3 batched, variable):**
1. `HammingWeightClaimReductionVerifier`
   - Num rounds: `log_k_chunk` bits for address variables
   - Uses opening point from booleanity: `[r_cycle || r_address]`
2. `AdviceClaimReductionVerifier` (TrustedAdvice, if present) - Phase 2: address variables
3. `AdviceClaimReductionVerifier` (UntrustedAdvice, if present) - Phase 2: address variables

**Phase Transition:**
```rust
params.phase = ReductionPhase::AddressVariables;
```

**Proof Data Consumed:**
- `proof.stage7_sumcheck_proof.compressed_polys`

**Transcript Operations:**
1. Append input claims
2. Sample batching coefficients
3. For each round: append compressed poly, sample challenge
4. Flush opening claims

**Opening Accumulator Updates:**
- **Critical:** Unified opening point [r_address || r_cycle] in big-endian
  - `r_address` comes from stage 7 (hamming weight phase)
  - `r_cycle` comes from stage 6 (booleanity)
  - Length: `log_k_chunk + log2(trace_length)`
- Opening IDs include:
  - All InstructionRa(i) for i in 0..instruction_d, SumcheckId::HammingWeightClaimReduction
  - All BytecodeRa(i) for i in 0..bytecode_d
  - All RamRa(i) for i in 0..ram_d
  - All committed RA polynomials from all stages

**Output from Stage 7:**
- `r_stage7`: array of challenges

---

## Stage 8: Dory Batch Opening (Final Opening Proof)

**Input Data:**
- Opening point from stage 7: `[r_address_stage7 || r_cycle_stage6]` (big-endian)
- Unified for all polynomials being opened

**Polynomials Opened (in order):**

1. **Dense Polynomials (with Lagrange factors):**
   - RamInc with claim from IncClaimReduction, scaled by `eq(r_address, 0)`
   - RdInc with claim from IncClaimReduction, scaled by `eq(r_address, 0)`

2. **Sparse RA Polynomials (1:1 mapping):**
   - For i in 0..instruction_d: InstructionRa(i)
   - For i in 0..bytecode_d: BytecodeRa(i)
   - For i in 0..ram_d: RamRa(i)

3. **Advice Polynomials (with Lagrange factors if present):**
   - TrustedAdvice (if present) - scaled by Lagrange factor
   - UntrustedAdvice (if present) - scaled by Lagrange factor

**Proof Data Consumed:**
- `proof.joint_opening_proof` - Dory opening proof
  - RLC accumulation over all polynomials
  - Uses gamma challenge sampled as: `transcript.challenge_scalar_powers(num_polys)`

**Transcript Operations:**
1. For non-ZK mode: append all claims: `transcript.append_scalars(b"rlc_claims", &claims)`
2. Sample gamma powers: `transcript.challenge_scalar_powers(num_polynomials)`
3. Compute constraint coefficients: `gamma_i * scaling_factor_i`
4. Append Dory opening proof to transcript

**Opening Accumulator:**
- All 7 stages' opening claims should have been accumulated to exactly match these polynomials
- Final consistency check: opening accumulator contains exactly the right set of opening IDs

---

## Opening Accumulator Pattern: Full Trace

**After each stage's sumcheck:**
1. Instances call `sumcheck.cache_openings(opening_accumulator, r_slice)`
2. This appends virtual polynomial opening claims to pending lists
3. `opening_accumulator.flush_to_transcript(transcript)` or stored for ZK mode
4. In ZK mode: pending claims are committed and commitment appended to transcript
5. After all 7 stages, accumulator contains exactly the polynomials needed for stage 8

**Opening Point Construction:**
- **Stage 1:** r_0 (uni-skip) || r_1, r_2, ..., r_n (regular rounds)
  - Big-endian format with r_0 as highest bit
- **Stage 2-7:** Similar pattern, r_0 from uni-skip prepended
- **Stage 7 Output:** Complete [r_address || r_cycle] point passed to stage 8

---

## Summary: Proof Data Breakdown

| Stage | Proof Components | Num Rounds | Num Instances | Degree |
|-------|-----------------|-----------|---------------|--------|
| 1     | Uni-skip + sumcheck | 1 + log₂T-1 | 1             | 1, 3   |
| 2     | Uni-skip + 5 sumchecks | 1 + max(various) | 5   | 1-3    |
| 3     | 3 sumchecks     | max(log₂T, 2) | 3             | 2-3    |
| 4     | 2 sumchecks     | max(log₂T, log₂K) | 2        | 2-3    |
| 5     | 3 sumchecks     | max(various) | 3             | 2-3    |
| 6     | 6-8 sumchecks   | max(various) | 6-8           | 2-3    |
| 7     | 1-3 sumchecks   | log_k_chunk | 1-3           | 2      |
| 8     | Dory opening    | N/A        | 1 (batch)     | N/A    |

---

## Config Parameters Affecting Proof Size

From `ReadWriteConfig`:
- `ram_rw_phase1_num_rounds`: typically log₂(T)
- `ram_rw_phase2_num_rounds`: typically log₂(ram_K)
- `registers_rw_phase1_num_rounds`: typically log₂(T)
- `registers_rw_phase2_num_rounds`: typically log₂(32) = 5

From `OneHotConfig`:
- `log_k_chunk`: 4 or 8 (impacts stage 7 and stage 8)
- `lookups_ra_virtual_log_k_chunk`: varies (impacts instruction lookup stages)

---

## Fiat-Shamir Challenge Sampling Order

1. Preamble data (program I/O, params)
2. All commitments
3. **Stage 1:**
   - Uni-skip polynomial + tau
   - r0 challenge
   - Sumcheck round polynomials
   - r_1, r_2, ... challenges
   - Opening claims (flushed)
4. **Stage 2-7:** Similar pattern per stage
5. **Stage 8:**
   - Dory opening proof
   - Dory verification challenges

---

## Critical Data Structures for On-Chain Verification

**Must be reconstructed from proof:**
1. `OneHotParams::from_config(proof.one_hot_config, bytecode_k, ram_k)`
2. Opening points for each stage (reconstructed from challenges)
3. Polynomial claims (accumulated in opening_accumulator)
4. Constraint coefficients for BlindFold (ZK mode)

**Must be validated:**
1. Proof `ram_K` is power-of-2 and >= min required
2. `rw_config` is valid (phases don't exceed trace length / register count)
3. `one_hot_config` is valid (chunk sizes are 4 or 8, etc.)

