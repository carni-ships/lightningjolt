# Jolt Verifier Code References

This document maps the analysis to exact code locations for implementation.

## Key File Paths

### Main Verifier
- **Verifier entry:** `jolt-core/src/zkvm/verifier.rs:340` - `JoltVerifier::verify()` method
- **Stage functions:** Lines 483-1360 - `verify_stage1()` through `verify_stage7()`
- **Fiat-Shamir preamble:** `jolt-core/src/zkvm/mod.rs:169-185` - `fiat_shamir_preamble()`

### Proof Structures
- **JoltProof struct:** `jolt-core/src/zkvm/proof_serialization.rs:35-63`
- **SumcheckInstanceProof enum:** `jolt-core/src/subprotocols/sumcheck.rs:700+`
- **UniSkipFirstRoundProofVariant:** `jolt-core/src/subprotocols/univariate_skip.rs`

### Opening Accumulator
- **VerifierOpeningAccumulator:** `jolt-core/src/poly/opening_proof.rs:229-250`
- **OpeningId enum:** `jolt-core/src/poly/opening_proof.rs:170-177`
- **SumcheckId enum:** `jolt-core/src/poly/opening_proof.rs:138-162`

### Configuration
- **ReadWriteConfig:** `jolt-core/src/zkvm/config.rs:28-103`
- **OneHotConfig:** `jolt-core/src/zkvm/config.rs:110-198`
- **OneHotParams:** `jolt-core/src/zkvm/config.rs:204-310`

---

## Stage-by-Stage Implementation Guide

### Stage 1: Spartan Outer

**UniSkip Verification:**
```
Location: jolt-core/src/zkvm/spartan/mod.rs:20-41
Function: verify_stage1_uni_skip()
Returns: (OuterUniSkipParams, F::Challenge)

InverseCode:
- Creates OuterUniSkipVerifier with key and transcript
- Calls UniSkipFirstRoundProof::verify() OR zk_proof.verify_transcript()
- Returns params (contains tau) and challenge (r0)
```

**Regular Sumcheck:**
```
Location: jolt-core/src/zkvm/verifier.rs:495-510
Code: BatchedSumcheck::verify() for OuterRemainingSumcheckVerifier
- Inputs: 1 instance with degree 3
- Returns: (batching_coeffs, r_stage1 challenges)
```

**OuterRemainingSumcheckVerifier Construction:**
```
Location: jolt-core/src/zkvm/spartan/outer.rs:350+
- Created from: spartan_key, trace_length, uni_skip_params, opening_accumulator
- Num rounds: derived from spartan_key
- Cached openings: all virtual/committed polys at [r0 || r_1 || ... || r_n]
```

### Stage 2: RAM Mixed Sumchecks

**UniSkip Verification:**
```
Location: jolt-core/src/zkvm/spartan/mod.rs:43-63
Function: verify_stage2_uni_skip()
Returns: (ProductVirtualUniSkipParams, F::Challenge)
```

**5 Sumcheck Instances:**
```
1. RamReadWriteCheckingVerifier - jolt-core/src/zkvm/ram/read_write_checking.rs
   - Num rounds: ram_rw_phase1 + ram_rw_phase2 (from proof.rw_config)
   
2. ProductVirtualRemainderVerifier - jolt-core/src/zkvm/spartan/product.rs
   - Num rounds: derived from uni_skip_params
   
3. InstructionLookupsClaimReductionSumcheckVerifier - jolt-core/src/zkvm/instruction_lookups/
   - Num rounds: typically 1
   
4. RamRafEvaluationSumcheckVerifier - jolt-core/src/zkvm/ram/raf_evaluation.rs
   - Num rounds: varies
   
5. OutputSumcheckVerifier - jolt-core/src/zkvm/ram/output_check.rs
   - Num rounds: 1
```

**Critical Challenge Sampling:**
```
Location: jolt-core/src/zkvm/verifier.rs:772-773
Code: 
  transcript.append_bytes(b"ram_val_check_gamma", &[]);
  let ram_val_check_gamma: F = transcript.challenge_scalar();
```

### Stage 3: Shift + Instruction + Registers

**3 Instances:**
```
1. ShiftSumcheckVerifier - jolt-core/src/zkvm/spartan/shift.rs
   - Num rounds: log2(trace_length)
   
2. InstructionInputSumcheckVerifier - jolt-core/src/zkvm/spartan/instruction_input.rs
   - Num rounds: 1
   
3. RegistersClaimReductionSumcheckVerifier - jolt-core/src/zkvm/registers/claim_reduction.rs
   - Num rounds: 1-2
```

### Stage 4: Registers + RAM Value Check

**Pre-Sumcheck Advice Accumulation:**
```
Location: jolt-core/src/zkvm/verifier.rs:764-770
Function: verifier_accumulate_advice()
- Adds TrustedAdvice and UntrustedAdvice openings to accumulator
- Called from jolt-core/src/zkvm/ram/mod.rs
```

**2 Instances:**
```
1. RegistersReadWriteCheckingVerifier
2. RamValCheckSumcheckVerifier - uses ram_val_check_gamma from stage 2
```

### Stage 5: Lookups + RAM RA + Registers Val

**3 Instances:**
```
1. InstructionReadRafSumcheckVerifier - jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs
2. RamRaClaimReductionSumcheckVerifier - jolt-core/src/zkvm/ram/ra_claim_reduction.rs
3. RegistersValEvaluationSumcheckVerifier - jolt-core/src/zkvm/registers/val_evaluation.rs
```

### Stage 6: Complex Mixed Sumchecks

**Base 6 Instances:**
```
1. BytecodeReadRafSumcheckVerifier - jolt-core/src/zkvm/bytecode/read_raf_checking.rs
2. BooleanitySumcheckVerifier - jolt-core/src/subprotocols/booleanity.rs
3. HammingBooleanitySumcheckVerifier - jolt-core/src/zkvm/ram/hamming_booleanity.rs
4. RamRaVirtualSumcheckVerifier - jolt-core/src/zkvm/ram/ra_virtual.rs
5. LookupsRaSumcheckVerifier - jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs
6. IncClaimReductionSumcheckVerifier - jolt-core/src/zkvm/ram/inc_claim_reduction.rs
```

**Advice Phase 1 (Cycle Variables):**
```
Location: jolt-core/src/zkvm/verifier.rs:949-964
Code:
- Initialize AdviceClaimReductionVerifier for TrustedAdvice (if present)
- Initialize AdviceClaimReductionVerifier for UntrustedAdvice (if present)
- Both use phase 1 (cycle variables)
```

### Stage 7: Hamming Weight + Advice Phase 2

**Hamming Weight Verifier:**
```
Location: jolt-core/src/zkvm/ram/hamming_weight_claim_reduction.rs
- Num rounds: log_k_chunk
- Input: opening point from booleanity
- Output: opening point [r_address || r_cycle]
```

**Advice Phase 2 (Address Variables):**
```
Location: jolt-core/src/zkvm/verifier.rs:1295-1314
Code:
- Update params.phase = ReductionPhase::AddressVariables
- Only include if phase 1 had rounds to process
```

### Stage 8: Dory Opening Verification

**Opening Point Extraction:**
```
Location: jolt-core/src/zkvm/verifier.rs:1373-1380
Code:
let (opening_point, _) = opening_accumulator.get_committed_polynomial_opening(
    CommittedPolynomial::InstructionRa(0),
    SumcheckId::HammingWeightClaimReduction,
);
let log_k_chunk = one_hot_params.log_k_chunk;
let r_address_stage7 = &opening_point.r[..log_k_chunk];
```

**Polynomial Collection:**
```
Location: jolt-core/src/zkvm/verifier.rs:1382-1462
Code:
1. Dense polynomials (RamInc, RdInc) with Lagrange factor eq(r_address, 0)
2. Sparse RA polynomials (InstructionRa, BytecodeRa, RamRa)
3. Advice polynomials (if present) with Lagrange factors
```

**RLC and Gamma Sampling:**
```
Location: jolt-core/src/zkvm/verifier.rs:1464-1488
Code:
- Append claims (non-ZK mode only)
- Sample gamma_powers: transcript.challenge_scalar_powers(num_polys)
- Compute constraint_coeffs: gamma_i * scaling_factor_i
- Verify opening proof: proof.joint_opening_proof
```

---

## Opening Accumulator Data Flow

**Initialization:**
```
Location: jolt-core/src/zkvm/verifier.rs:268-272
Code: VerifierOpeningAccumulator::new(log_T, zk_mode)
```

**After Each Stage's Sumcheck:**
```
Location: jolt-core/src/subprotocols/sumcheck.rs:459-490
Code:
1. sumcheck.cache_openings(opening_accumulator, r_slice)
   - Calls OpeningAccumulator trait methods
   - Appends opening claims to pending lists
   
2. opening_accumulator.flush_to_transcript(transcript)
   - (In non-ZK mode) Encodes pending claims to transcript
   - (In ZK mode) Stored for BlindFold constraints
```

**Opening ID Assignment:**
```
Different sumchecks use different SumcheckId values:
- SpartanOuter (Stage 1)
- SpartanProductVirtualization (Stage 2)
- SpartanShift (Stage 3)
- SpartanProductVirtualization (Stage 2)
- InstructionClaimReduction (Stage 2)
- InstructionInputVirtualization (Stage 3)
- RamReadWriteChecking (Stage 2)
- RamValCheck (Stage 4)
- RamRaClaimReduction (Stage 5)
- IncClaimReduction (Stage 6)
- HammingWeightClaimReduction (Stage 7)
- AdviceClaimReduction (Stage 6-7)
- etc. (see poly/opening_proof.rs:138-162)
```

---

## Transcript Label Constants

| Operation | Label | Location |
|-----------|-------|----------|
| Sumcheck claim | `b"sumcheck_claim"` | sumcheck.rs:49 |
| Compressed polynomial | `b"sumcheck_poly"` | sumcheck.rs:126 |
| Commitment (ZK) | `b"sumcheck_commitment"` | sumcheck.rs:294 |
| Output claims (ZK) | `b"output_claims_coms"` | sumcheck.rs:348 |
| RAM value gamma | `b"ram_val_check_gamma"` | verifier.rs:772 |
| RLC claims | `b"rlc_claims"` | verifier.rs:1469 |
| Input constraint labels | Various per instance | |
| Output constraint labels | Various per instance | |

---

## Configuration Validation

**ReadWriteConfig Validation:**
```
Location: jolt-core/src/zkvm/config.rs:66-93
Checks:
- ram_rw_phase1_num_rounds <= log_T
- ram_rw_phase2_num_rounds <= log_ram_K
- registers_rw_phase1_num_rounds <= log_T
- registers_rw_phase2_num_rounds <= log_register_count (5 for RV64)
```

**OneHotConfig Validation:**
```
Location: jolt-core/src/zkvm/config.rs:156-197
Checks:
- log_k_chunk is 4 or 8
- lookups_ra_virtual_log_k_chunk >= log_k_chunk
- lookups_ra_virtual_log_k_chunk <= LOG_K (128)
- lookups_ra_virtual_log_k_chunk is multiple of log_k_chunk
- LOG_K is divisible by lookups_ra_virtual_log_k_chunk
```

**Proof Validation:**
```
Location: jolt-core/src/zkvm/verifier.rs:300-317
Checks:
- one_hot_config.validate()
- ram_K is power-of-two
- ram_K >= min_ram_K
- rw_config.validate(log_T, log_ram_K)
```

---

## Non-ZK vs ZK Mode Differences

**Non-ZK Mode:**
- Proof data: `SumcheckInstanceProof::Clear(ClearSumcheckProof)`
  - Contains: compressed_polys with coefficients visible
- Opening accumulator: populated during verification
- No BlindFold constraints

**ZK Mode:**
- Proof data: `SumcheckInstanceProof::Zk(ZkSumcheckProof)`
  - Contains: round_commitments, poly_degrees, output_claims_commitments
  - Actual coefficients hidden
- Opening accumulator: pending claims committed, commitments sent to transcript
- BlindFold constraints: verifies sumcheck with hidden coefficients
- Location: jolt-core/src/zkvm/verifier.rs:1028-1281 - `verify_blindfold()`

