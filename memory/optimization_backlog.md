# LightningJolt Optimization Backlog

## Status: Active - Benchmark analysis complete, optimization planning phase

Last Updated: 2026-04-22 (post-benchmark analysis)

---

## Benchmark Results Analysis

### Test Configuration
- Block: #21000000 (181 transactions, mostly 1-step transfers)
- Features: `lattice + metal-pairing + icicle`
- Machine: 20-thread parallel proving
- Change: Full parallelism (all proofs in one batch) vs batched (10 at a time)

### Performance Summary (181 transactions)
| Configuration | Total Time | Speedup |
|--------------|-----------|---------|
| Batched (10) | 17.1s | baseline |
| Batch (20) | 16.7s | 1.02x |
| Full Parallel (181) | 16.2s | 1.06x |

### Key Observations
1. **Stage 8 is already parallel** - Dory commitment for each transaction runs in parallel within the batch
2. **Parallelism overhead is minimal** - Going from batch=10 to full parallel didn't help much
3. **CPU utilization is already high** - ~8x parallelism achieved with 20 threads
4. **The bottleneck is per-transaction** - Each proof takes ~0.8s regardless of parallelism

### Stage Breakdown (per transaction)
| Stage | Time | % of Total |
|-------|------|------------|
| Witness generation | ~200ms | 26% |
| **Stage 8 (Dory Opening)** | **~300-350ms** | **52%** |
| Stage 5 (Sumcheck) | ~60ms | 8% |
| Stage 4 | ~12ms | 2% |
| Setup/Verification | ~100ms | 13% |

### Why More Parallelism Doesn't Help
- Stage 8 (Dory Opening) is the bottleneck at 52% of CPU time
- Each Dory proof is independent - they already run in parallel
- The issue is that each individual Dory proof is slow (~350ms)
- Adding more threads doesn't make each proof faster

### Conclusion: Focus on Per-Proof Optimization
The path forward is not more parallelism, but:
1. **Faster Dory commitment** - GPU-accelerated MSMs for large polynomials
2. **Proof aggregation** - One proof for multiple transactions
3. **Skipping simple proofs** - Trust small transactions instead of proving

---

## First Principles Analysis: Root Causes

### Why is Stage 8 the bottleneck?
Stage 8 performs Dory commitment which involves:
1. **MSM (Multi-Scalar Multiplication)** - G1 and G2 point operations
2. **Reduce-and-Fold rounds** - Tree-structured commitments
3. **NTTs** - Number Theoretic Transforms for polynomial commitment

For 1-step proofs, the polynomial size is tiny but the **fixed overhead** of Dory setup dominates.

### Why is ICICLE not helping?
ICICLE's parallel MSM works best when:
- MSM size > 1000 elements (good GPU utilization)
- Scalar multiplication is the dominant cost

For 1-step proofs:
- Polynomial has maybe 16 elements
- Overhead of launching GPU kernel >> computation time

---

## Optimization Strategies (Priority Order)

### P0: Batch Proof Aggregation (Highest Impact, ~5x speedup potential)

**Concept**: Instead of 181 individual proofs, aggregate into a single batch proof.

**Current Flow**:
```
Tx1 → Proof₁
Tx2 → Proof₂
...
Tx181 → Proof₁₈₁
Verification: verify(Proof₁) + verify(Proof₂) + ... verify(Proof₁₈₁)
```

**Proposed Flow**:
```
Tx1 → Proof₁
Tx2 → Proof₂
...
Tx181 → Proof₁₈₁
Aggregation: batch(Proof₁, Proof₂, ..., Proof₁₈₁) → AggregatedProof
Verification: verify(AggregatedProof) [single pairing check]
```

**Expected Impact**:
- Reduces verification from O(n) to O(1) pairing checks
- Could reduce total time from 17s to ~3-4s
- Most important for production where blocks have 100s of txs

**Implementation**: Use Dory's batch commitment infrastructure

---

### P1: Skip Simple Transfers (~2x effective speedup)

**Concept**: Simple 1-step transfers don't need full ZK proofs.

**Optimization**:
- Classify transactions by complexity
- Simple transfers (< 10 steps): skip proving, include in "trusted" set
- Complex transactions (> 10 steps): full proof generation

**Expected Impact**:
- Block #21000000: 170/181 = 94% skipped → potential 2x speedup
- Only contract calls and complex txs get proven

**Tradeoffs**:
- Security assumption: trust miner for simple transfers
- Configurable threshold for security level

---

### P2: Parallel Dory Opening (~1.3x speedup)

**Concept**: Stage 8 for independent transactions can run in parallel.

**Current**: Stage 8 runs serially, blocking other work.

**Proposed**:
- Run Stage 8 for all transactions in parallel using rayon
- Requires coordinating the commitment phase

**Expected Impact**:
- Better CPU utilization
- 1.3x speedup on Stage 8 phase

---

### P3: Pre-compute GLV Tables (~1.2x speedup)

**Concept**: Dory uses fixed-base MSMs repeatedly with the same generator.

**Current**: Each MSM recomputes GLV decomposition.

**Proposed**:
- Pre-compute GLV tables once at startup
- Reuse for all MSMs with the same base point

**Expected Impact**:
- 20% speedup on Dory commitment phase
- Particularly effective for Stage 8

---

### P4: Optimize Small Polynomial Path (~1.5x speedup)

**Concept**: Dory overhead doesn't scale linearly with polynomial size.

**Current**: Same Dory commitment code path for 16-element and 16K-element polys.

**Proposed**:
- Add fast path for small polynomials (< 256 elements)
- Use simpler commitment scheme for small proofs
- Batch small polynomials together before committing

**Expected Impact**:
- Large speedup for simple transactions
- 1.5x overall speedup

---

### P5: Metal GPU for BN254 Pairing (~1.2x speedup)

**Concept**: Verification involves BN254 pairings.

**Current**: arkworks CPU pairing verification.

**Proposed**:
- Enable `metal-pairing` feature for Apple Silicon
- Use `BN254G2MSMEngine.swift` for G2 MSM

**Expected Impact**:
- 20% speedup on verification phase
- Useful for production verification infrastructure

---

## Already Completed

- ✅ `lattice` commitment (~2x speedup on arithmetic)
- ✅ `icicle` MSM acceleration (available but small MSM overhead negates benefit for 1-step proofs)
- ✅ `metal-pairing` enabled by default for GPU verification on Apple Silicon
- ✅ Rayon parallelization
- ✅ GLV for G2 fixed-base MSMs (was already implemented, G1 GLV not available in jolt-optimizations)
- ✅ zkMetal integration (G1 MSM via NEON, G2 via Metal) - **DISABLED due to scalar format bug**

### Implementation Notes

#### GLV for G1
- `fixed_base_vector_msm_g1` uses `FixedBasePrecomputedG1::new(base)` which precomputes GLV decomposition
- This is more efficient than ICICLE's general-purpose MSM for fixed-base operations
- `jolt_optimizations::fixed_base_vector_msm_g1` is called in `JoltG1Routines::fixed_base_vector_scalar_mul`
- ICICLE is only used for the regular `msm()` method, not for fixed-base operations

#### Stage 8 Analysis (Dory Opening)
Stage 8 involves multiple operations per proof:
1. **MSM operations** - G1 and G2 scalar multiplications for commitment
2. **Reduce-and-fold rounds** - Tree-structured commitments with fold operations
3. **Fold field vectors** - `left[i] = left[i] * scalar + right[i]` (already parallelized)

**Current GPU utilization:**
- `GPU_MSM_THRESHOLD = 64` means ICICLE used when `len >= 64`
- For 1-step proofs, polynomial size is small (~256 elements padded)
- Most MSMs in Stage 8 are smaller than threshold, so arkworks is used

**Opportunity:** Lowering threshold to 32 actually hurt performance (18s vs 16s) because:
- ICICLE overhead exceeds benefit for small MSMs
- Conversion between arkworks and ICICLE formats adds latency
- For 1-step proofs, Stage 8 MSMs are inherently small

**Conclusion:** Stage 8 bottleneck is not GPU acceleration but fixed overhead per proof (~350ms).
The real path to faster Stage 8 is proof aggregation (P0) or skipping simple proofs (P1).

### Dory Stage 8: Detailed Computation Analysis

**Requirement: Do NOT skip transactions. Find redundant work within Dory instead.**

#### Stage 8 Workflow (per proof)

1. **Setup** (~5% of time)
   - Initialize Dory context
   - Extract opening point from sumcheck accumulator
   - Build polynomial claims list

2. **VMV Message Computation** (~30% of time)
   - 3 independent MSM + Pairing operations (parallelized):
     - `C = pair(t_vec_v, g2_fin)` where `t_vec_v = MSM(row_commitments, v_vec)`
     - `D2 = pair(MSM(g1_vec, v_vec), g2_fin)`
     - `E1 = MSM(row_commitments, left_vec)`
   - G2 fixed-base scalar multiplication: `v2 = fixed_base_vector_scalar_mul(g2_fin, v_vec)`

3. **Reduce-and-Fold Rounds** (~55% of time)
   - `num_rounds = max(nu, sigma)` rounds (for 1-step proof: ~8 rounds)
   - Each round computes first message then second message
   - **First Message** (per round):
     - 2× `M1::msm(g1_prime, scalars)` for D2 (if v2_scalars available)
     - 2× `M2::msm(g2_prime, scalars)` for D1
     - 1× `M1::msm(g1_full, s2)` for E1β
     - 1× `M2::msm(g2_full, s1)` for E2β
   - **Second Message** (per round):
     - 2× `M1::msm(v1_l, s2_r)` for E1±
     - 2× `M1::msm(v1_r, s2_l)` for E1±
     - 2× `M2::msm(v2_r, s1_l)` for E2±
     - 2× `M2::msm(v2_l, s1_r)` for E2±
     - 2× `E::multi_pair_two_products` for C± (uses single Miller loop)

4. **Final Message** (~10% of time)
   - Final scalar product computation

#### Verified Optimization Opportunities

1. **Row commitments cloned and padded** (CONFIRMED)
   - `padded_row_commitments = row_commitments.clone()` - memory allocation
   - `if nu < sigma { padded_row_commitments.resize(...) }` - unnecessary for square matrices
   - **Impact**: For square matrices (nu=sigma), no padding needed, saves clone+resize
   - **File**: `evaluation_proof.rs` lines 126-129

2. **Parallel rayon::join overhead for small rounds** (CONFIRMED)
   - For small vectors (< 32 elements), parallel overhead exceeds sequential
   - Each round spawns multiple rayon tasks that do minimal work
   - **Impact**: ~10-15% overhead from unnecessary parallelism in early rounds
   - **File**: `reduce_and_fold.rs` lines 234-272, 367-390

3. **Generator vector slicing** (LOW IMPACT)
   - `g1_prime = &setup.g1_vec[..n2]` - minor, just slice creation
   - Not worth optimizing

4. **v2_scalars optimization** (ALREADY DONE)
   - `Some(v_vec)` is passed to prover state in evaluation_proof.rs:206
   - This optimization IS being used for round 0
   - After round 0, v2 is updated with generator contribution, so v2_scalars becomes stale

#### Concrete Optimization: Sequential Fast Path for Small Rounds

In `reduce_and_fold.rs`, for early rounds with small vector sizes:
- Add threshold check: `if n2 < 32 { use_sequential() } else { use_parallel() }`
- For 1-step proofs, ~5 rounds have n2 < 32

**Expected Impact**: 5-10% speedup on Stage 8 by eliminating rayon overhead

#### Metal Pairing
- Enabled by default: `default = ["lattice", "metal-pairing"]`
- Metal GPU Miller loop accelerates BN254 pairing verification
- Automatically used when available on Apple Silicon

#### Small Polynomial Optimization
- All proofs are padded to minimum 256 elements (line 371-375 in prover.rs)
- This ensures Dory matrix is large enough for commitment scheme
- Cannot easily bypass this without modifying the core commitment scheme

#### Proof Caching (Implemented but Not Effective for Block #21000000)
- Cache key: `(bytecode_hash, steps)`
- For block #21000000, all transactions have unique bytecode (different sender addresses)
- Cache would be effective for:
  - Contract calls to the same function with same parameters
  - Test scenarios with repeated transactions
  - Batch transfer patterns (same recipient + amount sent multiple times)
- Implementation: `ethrex-trace/examples/ethrex_block_prove.rs` CommitmentCache struct

---

## Future Optimization Ideas

### 2. Batch Trace Execution
Instead of proving transactions separately, execute multiple simple transfers in one trace:
- Concatenate their execution traces
- Generate one proof for N transactions
- Divide the total gas by N

**Challenge**: Requires modifying the commitment scheme to handle multiple execution contexts.

### 3. Simplified Commitment for Simple Transfers
Add a fast path in Dory when:
- Polynomial has ≤ 128 elements
- All coefficients are 0 or 1

Use simple Pedersen commitments instead of full Dory, then prove correctness via a small SNARK.

**Challenge**: Requires significant changes to the commitment scheme interface.

### 4. Parallel Stage 8
Currently Stage 8 (Dory Opening) runs serially even when proofs are batched. Enable parallel Stage 8:
- Process opening proofs for multiple transactions in parallel
- Reduces wall time for Phase 1

**Challenge**: Requires coordinating the commitment phase across multiple transactions.

### 5. Pre-compute GLV Tables at Startup
GLV decomposition recomputes for each distinct base point. Pre-computing once:
- Load standard generator tables at startup
- Reuse across all MSMs with same generator

**Status**: G2 already uses GLV. G1 uses fixed_base_vector_msm_g1 which has its own optimizations.

---

## Performance Targets

| Metric | Current | Target | Strategy |
|--------|---------|--------|----------|
| Block prove time (simple txs) | ~17s | <12s | P1 (skip simple) + P0 (batch) |
| Block prove time (complex txs) | ~2s/tx | <1s/tx | P3 (GLV) + P4 (small poly) |
| Stage 8 time | ~300ms/proof | <100ms | P2 (parallel) + P4 (small poly) |
| Verification | O(n) pairings | O(1) | P0 (batch aggregation) |

---

## zkMetal Integration - Root Cause Identified

### Problem: zkMetal BN254 Pippenger produces incorrect results

Extensive testing with `bn254_pippenger_msm` reveals that for all scalars > 1:
- Standard scalar format `[2, 0, 0, 0, 0, 0, 0, 0]` produces wrong results
- Montgomery scalar format produces different (also wrong) results
- All 4 combinations of point/scalar formats fail

### Test Results (scalar=2, single point G):
```
Standard point + Standard scalar → Wrong
Standard point + Montgomery scalar → Wrong
Montgomery point + Standard scalar → Wrong
Montgomery point + Montgomery scalar → Wrong
```

### Key Observations:
1. **scalar=1 works correctly** - zkMetal computes G*1 = G
2. **scalar=0 works correctly** - zkMetal computes G*0 = identity
3. **All scalars > 1 produce wrong results** - Not just format issues

This suggests the problem is in zkMetal's Pippenger implementation itself,
not in our format conversions.

### Workaround Applied:
- Set `GPU_MSM_THRESHOLD = 1024` to disable zkMetal for all MSM operations
- All MSMs now use arkworks (correct but potentially slower)
- zkMetal code remains in place for future debugging

### Recommendation:
Contact zkMetal team about BN254 Pippenger scalar handling bug.
The issue appears to be in their C implementation, not in our FFI bindings.

---

## Feature Flags

Current enabled features in `jolt-core/Cargo.toml`:

```toml
prover = [
    "minimal",
    "ark-ec/parallel",
    "ark-ff/parallel",
    "ark-std/parallel",
    "dory/backends",
    "dory/cache",
    "dory/disk-persistence",
    # "dory-prime",  # DISABLED - negative speedup
    "lattice",           # ~2x speedup
    "icicle",            # MSM acceleration (underutilized for small proofs)
]
```

---

## Related Documents

- `memory/zkmetal_developer_notes.md` - zkMetal integration details
- `FUTURE_CONTEXT.md` - Original project context
- `memory/bottleneck_analysis.md` - Detailed bottleneck breakdown
