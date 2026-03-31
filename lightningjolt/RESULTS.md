# LightningJolt — Jolt Prover Optimization Results

**Hardware:** Apple M3 Pro (6P + 6E cores, 18GB unified memory)
**Date:** 2026-03-28
**Jolt commit:** eb370ec2 (a16z/jolt)

## Benchmark Results

### Proving Performance (RAYON_NUM_THREADS=10)

| Tier | Mutations | Prove (s) | Preprocess (s) | Verify (s) | Total (s) |
|------|-----------|-----------|----------------|------------|-----------|
| minimal | 4 | **1.55** | 4.19 | 0.06 | 6.09 |
| small | 16 | **3.49** | 4.19 | 0.06 | 8.05 |
| medium | 32 | **5.95** | 4.28 | 0.07 | 10.59 |
| large | 64 | **9.04** | 4.24 | 0.07 | 13.66 |

*After dory-pcs pairing optimization (batch affine conversion + G1 base caching).*

### Comparison vs SP1 (CPU baselines, same hardware)

| Tier | Jolt (s) | SP1 (s) | Speedup |
|------|----------|---------|---------|
| minimal | 1.55 | 8.0 | **5.2x** |
| small | 3.49 | 15.0 | **4.3x** |
| medium | 5.95 | 25.0 | **4.2x** |
| large | 9.04 | 60.0 | **6.6x** |

**Jolt wins every tier by 4.2x-6.6x** (with caveats — see notes).

### Optimization Impact (before → after dory-pcs changes)

| Tier | Before | After | Improvement |
|------|--------|-------|-------------|
| minimal | 2.59s | 1.55s | **40%** |
| small | 5.54s | 3.49s | **37%** |
| medium | 7.34s | 5.95s | **19%** |
| large | 10.51s | 9.04s | **14%** |

### Thread Tuning (minimal tier)

| Threads | Prove (s) | vs Best |
|---------|-----------|---------|
| 1 | 10.25 | 5.98x |
| 2 | 4.88 | 2.85x |
| 4 | 2.98 | 1.74x |
| 6 (P-cores only) | 2.28 | 1.33x |
| 8 | 1.94 | 1.13x |
| **10** | **1.71** | **1.00x** |
| 12 (all cores) | 2.10 | 1.22x |

**Optimal: 10 threads.** Using all 12 cores is 22% slower due to E-core straggler effects with rayon's work-stealing. The sweet spot is P-cores (6) + some E-cores (4), where memory bandwidth isn't saturated but parallelism is high.

### Field Arithmetic Microbenchmarks

| Operation | Time | Throughput |
|-----------|------|------------|
| BN254 Fr multiply | 10.6 ns | 94 Mops/s |
| BN254 Fr addition | 1.8 ns | 539 Mops/s |
| BN254 Fr inverse | 1.29 us | 775 Kops/s |
| Poly eval (Horner, 1024 coeffs) | 13.7 us | 74.7 Mcoeffs/s |

## Optimization Findings

### 1. AArch64 Assembly Montgomery Multiply — NOT Beneficial

**Result: LLVM already generates near-optimal code.**

| Implementation | Time per multiply |
|---------------|-------------------|
| arkworks generic Rust | **10.5 ns** |
| Hand-written aarch64 asm | 12.9 ns (23% slower) |

LLVM compiles `u64 as u128 * u64 as u128` to `MUL`+`UMULH` instruction pairs and optimizes the carry chain better than hand-written sequential assembly. Apple M3's wide out-of-order engine benefits from LLVM's instruction reordering.

**Conclusion:** Unlike x86_64 (where ark-ff's `asm` feature provides ~2x speedup via BMI2/ADX), aarch64 field arithmetic is already well-optimized by LLVM. Do not pursue custom assembly.

### 2. Thread Count Tuning — 15-22% Improvement

Setting `RAYON_NUM_THREADS=10` instead of the default 12 provides consistent improvement across all tiers. This is due to Apple M3 Pro's heterogeneous core architecture where E-cores (efficiency) are ~40% slower than P-cores for field arithmetic. Rayon's work-stealing creates load imbalance when E-cores can't keep up.

### 3. Metal GPU Field Arithmetic — Working, Not Yet Faster

**Status: Unblocked.** We wrote and validated Metal compute shaders for BN254 Montgomery
multiplication. Correctness: 100% match with arkworks across all test sizes.

**Shader:** `shaders/bn254_field.metal` — schoolbook 256×256→512 bit multiply + word-by-word
Montgomery reduction using 32-bit operations.

**Performance scaling (batch field multiply):**

| Elements | CPU 1-thread | CPU 12-thread | GPU Metal | GPU vs CPU-1 | GPU vs CPU-12 |
|----------|-------------|--------------|-----------|-------------|--------------|
| 1K | 75 Mops/s | 2.2 Mops/s | 2.1 Mops/s | 0.03x | 0.99x |
| 4K | 78 Mops/s | 8.3 Mops/s | 7.8 Mops/s | 0.10x | 0.93x |
| 16K | 61 Mops/s | 25 Mops/s | 20 Mops/s | 0.33x | 0.80x |
| 64K | 61 Mops/s | 79 Mops/s | 41 Mops/s | 0.67x | 0.52x |
| 256K | 76 Mops/s | 187 Mops/s | 62 Mops/s | 0.81x | 0.33x |
| 1M | 76 Mops/s | 228 Mops/s | 66 Mops/s | 0.87x | 0.29x |
| **4M** | 76 Mops/s | 297 Mops/s | **92 Mops/s** | **1.22x** | 0.31x |

**Key findings:**
- GPU only beats single-threaded CPU at 4M+ elements (1.22x)
- GPU never beats multi-threaded CPU — rayon parallelism on 12 cores with native 64-bit
  multiply always wins over Metal's 32-bit emulation
- GPU dispatch overhead (~0.4ms) dominates at small sizes
- GPU throughput peaks at ~92 Mops/s vs CPU's 76 Mops/s single-thread / 430 Mops/s parallel

### 4. Metal GPU MSM (Pippenger) — Correct, But Slower Than CPU

**Status: Working.** Full Pippenger's bucket method on Metal GPU with correct results at all sizes.

**Shader:** `shaders/bn254_msm.metal` — Fq base field arithmetic, G1 affine/projective types,
mixed EC addition, scatter (scalar→bucket assignment) and accumulate (bucket EC sums) kernels.

**Bindings:** `src/metal_msm.rs::msm` module — `MetalMsm` orchestrates Pippenger:
GPU scatter → CPU sort by bucket → GPU accumulate → CPU bucket reduction → CPU window combination.

**Performance:**

| Points | CPU MSM (arkworks) | GPU MSM (Metal) | GPU/CPU |
|--------|--------------------|-----------------|---------|
| 16 | 1.4ms | 199ms | 0.01x |
| 256 | 2.0ms | 20ms | 0.10x |
| 1K | 3.5ms | 35ms | 0.10x |
| 4K | 9.0ms | 170ms | 0.05x |
| 16K | 15ms | 434ms | 0.03x |
| 64K | 49ms | 560ms | 0.09x |

**GPU MSM is ~10-30x slower than CPU MSM.** The predicted 20-30% gain from hybrid
MSM did NOT materialize. Root causes:

1. **32-bit arithmetic penalty:** Each EC mixed addition requires ~7 Fq multiplies.
   Each Fq multiply is 8×8 = 64 32-bit multiplies + 8 reduction iterations.
   On CPU, the same operation uses 4×4 = 16 native 64-bit multiplies.
   **Metal pays a 4x arithmetic penalty** per field operation.

2. **Sequential bucket accumulation:** Each bucket's points must be added sequentially
   (EC addition is not commutative-friendly for parallel reduction). Threads with
   large buckets become stragglers.

3. **CPU roundtrip per window:** Scatter results are read back to CPU for sorting,
   then re-uploaded. With 16-22 windows, this adds ~7ms of overhead.

4. **arkworks is already parallel:** CPU MSM uses rayon internally with optimized
   Pippenger, benefiting from native 64-bit multiply and wide out-of-order execution.

**Conclusion:** Metal GPU is fundamentally disadvantaged for modular arithmetic
because Apple Silicon GPUs lack native 64-bit integer multiply. The unified memory
advantage doesn't overcome the 4x per-operation penalty. **Do not pursue GPU MSM
further on Apple Silicon for BN254.**

### 5. Apple Neural Engine — Not Applicable

ANE is fixed-function for INT8/FP16 matrix ops. No modular arithmetic capability. Cannot program directly (only through CoreML black box). Not viable for ZK.

## Prover Internal Profiling (Chrome trace analysis)

### Large Tier (64 mutations, 9.6s prove, RAYON_NUM_THREADS=10)

| Stage | Time | % | Dominant Operation |
|-------|------|---|-------------------|
| Witness gen + Dory commit | 3.5s | 36% | `multi_pair_g2_setup` (42 calls, 16s aggregate) |
| Stage 8: Dory opening proof | 2.6s | 27% | `create_evaluation_proof` (13 Dory rounds × pairings) |
| Stage 6: Booleanity + RA | 1.5s | 16% | `InstructionRaSumcheck` + `BooleanitySumcheck` |
| Stage 5: Instruction ReadRAF | 0.8s | 9% | `InstructionReadRafSumcheck` |
| Stage 1: Outer sumcheck | 0.3s | 3% | `OuterLinearStage` |
| Stage 4: Registers + RAM val | 0.3s | 3% | Register read-write checking |
| Stage 3: Shift + InstructionInput | 0.2s | 2% | Shift/input sumchecks |
| Stage 2: Product virtual + RAM | 0.2s | 2% | Product virtual sumcheck |
| Stage 7: Hamming weight | 0.04s | 0.5% | Hamming weight reduction |

### Minimal Tier (4 mutations, 1.55s prove)

| Stage | Time | % |
|-------|------|---|
| Stage 8: Dory opening proof | 0.70s | 45% |
| Witness gen + Dory commit | 0.53s | 34% |
| Stage 6 | 0.15s | 9% |
| Stage 5 | 0.09s | 5% |
| Stages 1-4, 7 | 0.09s | 6% |

### Key Finding: Dory pairings dominate prove time

The Dory polynomial commitment scheme dominates the prover:
- **Witness commitment** calls `multi_pair_g2_setup` 42× for tier-2 aggregation
- **Opening proof** (Stage 8) runs 13 Dory rounds, each with pairing operations
- The `multi_pair_g2_setup` function alone consumes 25.4s aggregate across 68 calls (10 threads)

The sumcheck stages (1-7) are already well-parallelized with rayon.

### 6. Dory Pairing Optimization — 14-40% Improvement (IMPLEMENTED)

**Changes to local fork (`third-party/dory-pcs/src/backends/arkworks/ark_pairing.rs`):**

1. **Montgomery's batch affine conversion**: Added `batch_g1_to_affine()` and `batch_g2_to_affine()` using the batch inversion trick — converts N projective points to affine with 1 field inversion + 3(N-1) multiplications instead of N individual inversions. Applied to all parallel multi-pairing functions before chunked Miller loops.

2. **Increased chunk sizes**: MIN_CHUNK 32→128, MAX_CHUNK 128→512. Reduces per-chunk overhead (affine prep + Miller loop startup cost) with fewer, larger chunks.

**Changes to jolt-core (`jolt-core/src/poly/commitment/dory/commitment_scheme.rs`):**

3. **G1 affine base caching**: Added `RwLock<Arc<Vec<G1Affine>>>` cache for the G1 SRS bases used in tier-1 commitments. Previously, `process_chunk` and `process_chunk_onehot` converted the same `g1_vec` (projective→affine) on every call — 10,240 times per prove. Now computed once via `normalize_batch` and shared across all rayon threads via `Arc`.

### 7. Pipeline Witness Commitment — NOT FEASIBLE

**Investigated and determined infeasible.** The Fiat-Shamir transcript requires all polynomial commitments to be appended before stage 1 samples its first challenge. This is a protocol-level constraint:
- `generate_and_commit_witness_polynomials()` adds tier-2 commitments to transcript
- `prove_stage1()` derives challenges from transcript state including all commitments
- Cannot defer any commitment past the transcript append without breaking soundness

### Remaining Optimization Opportunities

1. **Reduce Dory opening proof rounds** — Stage 8 runs 13 rounds. Batching more polynomial openings per round (if possible) would reduce pairing count. Requires changes to the Dory PCS protocol.

2. **Switch to HyperKZG for smaller proofs** — Jolt supports both Dory and HyperKZG commitment schemes. HyperKZG uses KZG-based openings which are faster than Dory's pairing-heavy approach, but requires a trusted setup.

3. **Further Miller loop optimization** — The Miller loop itself is the remaining bottleneck in `multi_pair_g2_setup`. Potential approaches: shared final exponentiation batching, lazy reduction in Fq12 arithmetic.

## Architecture Notes

**Jolt's sumcheck prover** (`jolt-core/src/subprotocols/sumcheck.rs`):
- Individual `compute_message()` implementations use `par_iter()` internally
- The outer sumcheck round loop is inherently sequential (each round depends on the previous challenge)
- Polynomial binding is already well-parallelized with rayon
- Dory commitment batch operations are parallelized

**Key insight:** There is no single "10x speedup" available. The codebase is already well-optimized for CPU parallelism. The gains achieved are:
1. Thread count tuning (done: 15-22%)
2. Metal GPU (investigated: not viable for BN254 on Apple Silicon — see §4)
3. Dory pairing optimization (done: 14-40% via batch affine + G1 caching)
4. Witness/proving pipeline overlap (investigated: infeasible due to Fiat-Shamir constraint)

## SP1 Comparison Caveats

The SP1 baselines include in-circuit Ed25519 signature verification (via precompile), while our Jolt guest only checks a signature count. A fairer comparison would:
- Add Ed25519 verification to the Jolt guest (via jolt-inlines or RISC-V)
- OR remove Ed25519 from SP1 for benchmark parity

SP1 also supports recursive IVC composition which Jolt doesn't have yet.

## Recommendations

1. **Use RAYON_NUM_THREADS=10** for all Jolt proving on M3 Pro
2. **Do not invest in custom aarch64 assembly** — LLVM is already optimal
3. **Metal GPU is not viable for BN254 arithmetic on Apple Silicon** — both field ops and MSM are slower than parallel CPU due to lack of native 64-bit multiply
4. **Dory pairing optimization delivers 14-40% speedup** — batch affine conversion in dory-pcs + G1 base caching in jolt-core. The local fork at `third-party/dory-pcs/` should be maintained.
5. **Pipeline overlap is not feasible** — Fiat-Shamir transcript dependency blocks concurrent witness commitment and proving.
5. **Pipeline witness commitment with proving** — Start sumcheck stages before all tier-2 commitments finish. This is the most feasible optimization we can implement ourselves.
6. **For production:** Use Jolt for Merkle-only proving (current setup, 3-6x faster than SP1), keep SP1 for business logic proving (Ed25519 precompile, recursive IVC needed)
7. **For maximum prover speed on M3 Pro:** The current ~10s large-tier prove time is dominated by pairing math. A 2x speedup would require either (a) faster BN254 pairing implementations, (b) HyperKZG commitment scheme, or (c) a different proof system entirely (FRI-based with Goldilocks field).
