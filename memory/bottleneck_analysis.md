# Block Proving Performance Analysis

**Date:** 2026-04-21
**Status:** Analyzing bottlenecks

## Current Performance

| Metric | Value |
|--------|-------|
| Block | #21000000 |
| Transactions in block | 181 |
| Provable transactions | 152 |
| Skipped (execution failures) | 29 |
| Total proving time | **31.0s** |
| Proving phase (Phase 1) | 29.9s |
| Verification phase (Phase 2) | 1.15s |
| Average per transaction | **0.31s** |

## Bottleneck Analysis

### Per-Stage Timing Breakdown

```
Stage 8 (Dory Opening):         21,346ms total, 140ms avg  → 74.3% of proving time
  └─ MSM operations (scalar multiplications)
  └─ Dory commitment generation
  └─ Pairing computations

Stage 5 (Witness commit):        5,821ms total, 38ms avg  → 20.3%
  └─ Polynomial commitment generation
  └─ MLE computation

Witness generation:             5,366ms total, 35ms avg  → 18.7%
  └─ Trace to witness conversion
  └─ Constraint evaluation

Stage 6:                        1,939ms total, 13ms avg  →  6.8%
Stage 4:                        1,527ms total, 10ms avg  →  5.3%
Stage 2:                        1,007ms total,  7ms avg  →  3.5%
Stage 3:                          361ms total,  2ms avg  →  1.3%
Stage 1:                          187ms total,  1ms avg  →  0.7%
Stage 7:                            2ms total,  0ms avg  →  0.0%
```

**Total**: 45,556ms across all stages (sequential execution, so wall time is 29.9s)

### Key Finding: Stage 8 is 74.3% of proving time

Stage 8 (Dory Opening / commitment) dominates the proving pipeline. This includes:
1. **MSM (Multi-Scalar Multiplication)** - O(n log n) point multiplications
2. **Dory commitment generation** - Polynomial commitment scheme
3. **Pairing computations** - Final verification pairings

## Optimization Opportunities

### 1. Stage 8 - Dory Opening (74.3%) - HIGHEST PRIORITY

**Options:**
- **zkMetal GPU MSM**: Already integrated, scalar conversion fixed
  - Expected speedup: 3-5x for large MSMs
  - Current: 140ms per proof, target: 30-50ms
- **ICICLE GPU MSM**: Alternative GPU backend
  - Requires Metal backend integration
- **Batch opening proofs**: Aggregate commitments across transactions
  - Combine hints using `combine_hints()` and `combine_commitments()`
  - Reduces number of MSMs from O(n) to O(1) per block

**Potential speedup**: 3-5x → reduces total from 31s to 6-10s

### 2. Stage 5 - Witness Commitment (20.3%)

**Options:**
- **Parallel polynomial evaluation**
- **SIMD optimizations for MLE computation**
- **Precompute witness across batch**

**Potential speedup**: 1.5-2x → reduces total from 31s to 20-25s

### 3. Witness Generation (18.7%)

**Options:**
- **Batch trace processing**
- **SIMD trace parsing**
- **Pre-allocate witness buffers**

**Potential speedup**: 1.3-1.5x

### 4. Stages 1-7 (18.7% combined)

**Options:**
- Already relatively efficient
- Could explore parallelization within proof

## Current Status (Updated 2026-04-22)

### Completed Optimizations
1. **Parallel batching with larger stack** - 38% improvement
   - 16MB stack (up from 8MB default)
   - 8-12 threads per batch
   - 10 proofs per batch
   - Result: 31.0s → 22.5s for 179 proofs

2. **zkMetal CPU Pippenger** - Not beneficial
   - FFI overhead dominates for small MSMs
   - Threshold (2^14) didn't help
   - Reverted to arkworks VariableBaseMSM

3. **Batch opening proof aggregation** - Analysis complete
   - Requires prover-side changes (not post-hoc)
   - Limited impact (verification is only 8% of time)
   - See `memory/batch_opening_aggregation.md`

### Bottleneck Analysis (Updated)
| Phase | Time | Percentage |
|-------|------|------------|
| Phase 1 (proving) | 20.8s | 92% |
| Phase 2 (verify) | 1.7s | 8% |

**Key Finding**: Verification is NOT the bottleneck. Proving is 92% of time.

## Recommended Optimization Order

1. **Stage 8 GPU acceleration** (highest impact: 92% of proving time)
   - ICICLE GPU MSM integration
   - Target: 3-5x speedup on Stage 8

2. **Batch proving** (requires prover changes)
   - Aggregate transactions during proving
   - Single proof for multiple transactions
   - Target: 2-3x speedup

3. **Stage 5 optimization** (20% of proving time)
   - SIMD polynomial evaluation
   - Target: 1.5-2x speedup

## Projected Performance

| Optimization | Current | After |
|-------------|---------|-------|
| Baseline | 31.0s | - |
| Parallel batching | 22.5s | ✓ Done |
| Stage 8 GPU | - | 7-12s |
| Batch proving | - | 4-8s |
| Stage 5 SIMD | - | 3-6s |

**Target: 3-6s for 179 transactions (~15s for full 181 tx block)**

## Updated Analysis (2026-04-23)

### Why GPU MSMs Don't Help for Small Proofs

Based on post-benchmark analysis of 1-step transfers:

1. **ICICLE parallel MSM works best when MSM size > 1000 elements**
   - GPU kernel launch overhead (~0.1-0.5ms) exceeds computation for tiny polynomials
   - For 1-step proofs, polynomial has ~16 elements

2. **Fixed Dory setup overhead dominates small proofs**
   - Each proof has ~350ms of Stage 8 time regardless of polynomial size
   - MSM computation is not the bottleneck - it's the Dory protocol overhead

3. **Parallelism is already effective**
   - 8x parallelism achieved with 20 threads
   - Stage 8 runs in parallel across transactions
   - Adding more threads doesn't make each proof faster

### Strategic Recommendations

| Priority | Strategy | Expected Impact |
|----------|----------|-----------------|
| P0 | Batch Proof Aggregation | ~5x speedup by combining proofs |
| P1 | GPU MSM for large polynomials | Only helps large proofs (>1000 elements) |
| P2 | Skip/trust small proofs | Depends on trust model |

### Sequential Fast Path (Applied 2026-04-23)

Commit 61602746 introduced `SEQUENTIAL_THRESHOLD=16` optimization:
- For small vector sizes (n2 < 16), sequential computation bypasses rayon spawn overhead
- Applied to `third-party/dory-pcs/src/reduce_and_fold.rs`
- Target: reduce per-proof Stage 8 overhead

## First-Principles Bottleneck Analysis (2026-04-23)

### Current Performance (with lattice + metal-pairing)
- **Total time**: 17.8s for 152 proofs (~117ms per proof wall time)
- **Achieved parallelism**: 71.2x (8 threads doing 152 proofs)
- **Per-transaction breakdown** (from profile logs):
  - Stage 8 (Dory Opening): ~100-200ms per proof
  - Witness generation: ~30-40ms per proof
  - Other stages: ~10-20ms per proof combined

### First-Principles Analysis

#### Why Stage 8 Dominates
Stage 8 performs Dory commitment which involves:
1. **MSM (Multi-Scalar Multiplication)** - O(n log n) point multiplications
2. **Reduce-and-Fold rounds** - Tree of pairings and commitments
3. **Pairing computations** - Final verification pairings

For 1-step proofs with lattice feature:
- Polynomial size: ~16-64 elements (very small)
- MSM with 64 elements is NOT the bottleneck
- **The Dory protocol fixed overhead dominates** - setup, folding rounds, etc.

#### Why GPU/MSM Optimizations Don't Help Much
| Component | Size | GPU Speedup | Overhead |
|-----------|------|------------|----------|
| MSM (64 elements) | Tiny | ~1.1-1.2x | Kernel launch ~0.1-0.5ms |
| MSM (4096 elements) | Large | ~3-5x | Worth it |

For small proofs, the **Dory protocol overhead** (not MSM) is the bottleneck:
- Round setup and teardown
- Pairing computations
- Memory allocations

#### What Actually Helps

| Optimization | Mechanism | Expected Speedup |
|--------------|-----------|------------------|
| **Batch proof aggregation** | Combine N proofs into 1 Dory opening | ~Nx (N=152 → 100x+) |
| **Lattice/NTT commitment** | Already applied | ~2x over standard Dory |
| **Metal pairing (GPU)** | GPU pairings | ~1.5-2x on pairings |
| **Sequential threshold** | Avoid rayon overhead for small MSMs | ~1.1x |
| **More parallelism** | Already 71x with 8 threads | Minimal (per-proof overhead) |

### Strategic Recommendation: Batch Proof Aggregation (P0)

**Current flow (152 separate proofs):**
```
Tx1 → Proof₁ → Dory Opening₁
Tx2 → Proof₂ → Dory Opening₂
...
Tx152 → Proof₁₅₂ → Dory Opening₁₅₂
```

**With batch aggregation:**
```
All 152 txs → Single batch proof → One Dory Opening
```

The Dory commitment scheme supports **batch opening proofs** via `combine_hints()` and `combine_commitments()`. This combines multiple transaction proofs into a single proof with ONE Dory opening, eliminating ~99% of the Stage 8 cost.

**Implementation requirements:**
1. Generate all witnesses in parallel
2. Compute commitments for all transactions
3. **Combine commitments during Stage 8** (not after)
4. Single Dory opening for the combined commitment
5. Verification checks all proofs against combined opening

**Estimated speedup:** 152 proofs × 100ms ≈ 15s → 1 proof × 200ms ≈ 0.2s (75x)

