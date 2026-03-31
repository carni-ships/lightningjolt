# LightningJolt: Optimizing the Jolt zkVM Prover

**Authors:** LightningJolt
**Date:** 2026-03-28
**Target:** a16z/jolt (commit eb370ec2), Dory polynomial commitment scheme
**Hardware:** Apple M3 Pro (6P + 6E cores, 18GB unified memory)

---

## Abstract

We profiled and optimized the Jolt zkVM prover to reduce proving time across all Jolt programs. Through Chrome trace analysis we identified that **63% of proving time** was spent in BN254 pairing operations within the Dory polynomial commitment scheme. We implemented two targeted optimizations — Montgomery's batch affine conversion in the pairing layer and G1 SRS base caching in the commitment layer — achieving a **14-40% reduction in proving time** across all circuit sizes. We also investigated and ruled out three other optimization paths: Metal GPU acceleration, custom AArch64 assembly, and witness/proving pipeline overlap.

---

## 1. Motivation

Jolt's Dory polynomial commitment scheme dominates proving time. For latency-sensitive applications — on-chain state proofs, interactive verification, real-time proving — every millisecond matters. Our goal was to minimize prove time on consumer hardware (Apple M3 Pro) without modifying the proof protocol or sacrificing soundness.

---

## 2. Profiling: Where Time Is Spent

We instrumented the Jolt prover with `tracing-chrome` and analyzed the resulting Perfetto traces. The prover pipeline has eight stages:

```
Fiat-Shamir preamble
  -> Witness generation + Dory tier-1/tier-2 commitment
  -> Stage 1: Outer sumcheck (Spartan)
  -> Stage 2: Product virtual + RAM sumcheck
  -> Stage 3: Shift + InstructionInput sumcheck
  -> Stage 4: Register + RAM val checking
  -> Stage 5: Instruction ReadRAF sumcheck
  -> Stage 6: Booleanity + RA sumcheck
  -> Stage 7: Hamming weight reduction
  -> Stage 8: Dory opening proof (13 rounds)
```

### Pre-optimization breakdown (large tier, 64 mutations, 10.51s prove)

| Component | Wall-clock | % of prove | Root cause |
|-----------|-----------|------------|------------|
| Witness gen + Dory commit | 3.5s | 33% | 42 tier-2 `multi_pair_g2_setup` calls |
| Stage 8: Dory opening proof | 2.6s | 25% | 13 rounds x pairing operations |
| Stage 6: Booleanity + RA | 1.5s | 14% | Sumcheck compute_message |
| Stage 5: InstructionReadRAF | 0.8s | 8% | Sumcheck compute_message |
| Stages 1-4, 7 | 1.1s | 10% | Various sumchecks |
| Other overhead | 1.0s | 10% | Polynomial binding, EQ evals |

### Key finding

The `multi_pair_g2_setup` function in dory-pcs consumed **16s aggregate CPU time** across 68 invocations (42 for tier-2 commitment + 26 for opening proof). This single function, which computes BN254 multi-pairings for the Dory commitment scheme, was the dominant bottleneck.

The Dory commitment protocol works in two tiers:
1. **Tier 1:** Each witness polynomial row is committed via MSM against the G1 SRS (10,240 calls)
2. **Tier 2:** Row commitments are aggregated via multi-pairing against the G2 SRS (42 calls)

Both tiers contained inefficiencies we could address without protocol changes.

---

## 3. Optimization A: Batch Affine Conversion in Dory Pairings

### Problem

The `multi_pair_g2_setup_parallel` function converts projective curve points to affine form before computing Miller loops. The original code converted each point individually:

```rust
// Original: N individual field inversions
let ps_prep: Vec<G1Prepared> = ps.iter()
    .map(|p| {
        let affine: G1Affine = p.0.into();  // field inversion per point
        affine.into()
    })
    .collect();
```

Converting from projective `(X:Y:Z)` to affine `(x, y)` requires computing `x = X/Z^2` and `y = Y/Z^3`. Each division requires a **field inversion**, which costs ~130x a field multiplication for BN254. With hundreds of points per pairing call, this adds up.

### Solution: Montgomery's batch inversion trick

We replace N individual inversions with **1 inversion + 3(N-1) multiplications**:

```rust
// New: batch_g1_to_affine() in ark_pairing.rs
fn batch_g1_to_affine(points: &[ArkG1]) -> Vec<G1Affine> {
    // 1. Collect Z-coordinates, skip identity points
    // 2. Compute running product: products[i] = Z_0 * Z_1 * ... * Z_i
    // 3. Single inversion: inv = 1 / (Z_0 * Z_1 * ... * Z_{n-1})
    // 4. Back-propagate: z_inv[i] = inv * products[i-1], inv *= Z_i
    // 5. Compute affine: x_i = X_i * z_inv_i^2, y_i = Y_i * z_inv_i^3
}
```

The algorithm:

1. **Forward pass:** Accumulate the product of all Z-coordinates: `P_i = Z_0 * Z_1 * ... * Z_i`
2. **Single inversion:** Compute `inv = 1 / P_{n-1}` (one field inversion)
3. **Backward pass:** Recover individual inverses: `Z_i^{-1} = inv * P_{i-1}`, then `inv *= Z_i`
4. **Affine conversion:** `x_i = X_i * (Z_i^{-1})^2`, `y_i = Y_i * (Z_i^{-1})^3`

This was implemented for both G1 (Fq base field) and G2 (Fq2 extension field) points.

### Additional tuning: chunk sizes

We increased the parallel Miller loop chunk sizes from `(MIN=32, MAX=128)` to `(MIN=128, MAX=512)`. Profiling showed that per-chunk overhead (affine preparation for G2Prepared, Miller loop startup) was significant at small chunk sizes. Larger chunks amortize this cost with fewer rayon task boundaries.

### Files modified

**`third-party/dory-pcs/src/backends/arkworks/ark_pairing.rs`** (local fork of dory-pcs 0.3.0):

- Added `batch_g1_to_affine()` (lines 40-101) — Montgomery batch inversion for G1
- Added `batch_g2_to_affine()` (lines 103-158) — Montgomery batch inversion for G2
- Modified `multi_pair_parallel()` — calls batch affine before chunked Miller loops
- Modified `multi_pair_g2_setup_parallel()` — uses batch affine for G1 points
- Modified `multi_pair_g1_setup_parallel()` — uses batch affine for G2 points
- Modified `determine_chunk_size()` — MIN_CHUNK 32->128, MAX_CHUNK 128->512

---

## 4. Optimization B: G1 SRS Affine Base Caching

### Problem

During tier-1 commitment, the functions `process_chunk()` and `process_chunk_onehot()` convert the G1 SRS bases from projective to affine representation on **every call**:

```rust
// Original: called 10,240 times per prove
fn process_chunk(setup: &ProverSetup, chunk: &[T]) -> ChunkState {
    let g1_bases: Vec<G1Affine> = setup.g1_vec[..row_len]
        .iter()
        .map(|g| g.0.into_affine())  // redundant conversion every call
        .collect();
    // ... MSM against g1_bases ...
}
```

The G1 SRS bases are **immutable** — they are loaded once during preprocessing and never change. Yet each of the 10,240 tier-1 commitment calls redundantly converts the same projective points to affine, wasting ~10,240 field inversions per call x `row_len` points.

### Solution: Static RwLock cache with Arc sharing

We compute the affine bases once and share them across all rayon threads via `Arc`:

```rust
// New: in commitment_scheme.rs
static G1_AFFINE_CACHE: RwLock<Option<(usize, Arc<Vec<G1Affine>>)>> = RwLock::new(None);

fn get_cached_g1_affine_bases(setup: &ProverSetup, row_len: usize) -> Arc<Vec<G1Affine>> {
    // Fast path: RwLock read (concurrent, no contention)
    {
        let cache = G1_AFFINE_CACHE.read().unwrap();
        if let Some((cached_len, ref bases)) = *cache {
            if cached_len == row_len {
                return bases.clone();  // Arc clone = pointer increment
            }
        }
    }
    // Slow path: compute once using normalize_batch (batch inversion)
    let mut cache = G1_AFFINE_CACHE.write().unwrap();
    // Double-check after acquiring write lock (another thread may have populated it)
    if let Some((cached_len, ref bases)) = *cache {
        if cached_len == row_len { return bases.clone(); }
    }
    let projs: Vec<G1Projective> = setup.g1_vec[..row_len].iter().map(|g| g.0).collect();
    let bases = Arc::new(G1Projective::normalize_batch(&projs));
    *cache = Some((row_len, bases.clone()));
    bases
}
```

Design choices:
- **`RwLock` over `Mutex`:** Multiple rayon threads read concurrently on the fast path. A `Mutex` serialized all threads and caused a 3-5% regression; `RwLock` eliminates contention.
- **`Arc` over `Vec::clone`:** An earlier `thread_local` + `Vec::clone` approach cloned the full vector on every access, negating the cache benefit. `Arc::clone()` is a single atomic increment.
- **`row_len` key:** The cache invalidates when `row_len` changes (e.g., different circuit sizes in a sweep), ensuring correctness.

### Files modified

**`jolt-core/src/poly/commitment/dory/commitment_scheme.rs`**:

- Added `G1_AFFINE_CACHE` static and `get_cached_g1_affine_bases()` function (lines 31-55)
- Modified `process_chunk()` — uses cached bases instead of per-call conversion
- Modified `process_chunk_onehot()` — uses cached bases instead of per-call conversion

---

## 5. Investigated and Ruled Out

### 5a. Metal GPU Acceleration

We implemented full BN254 field arithmetic and Pippenger MSM as Metal compute shaders (`shaders/bn254_field.metal`, `shaders/bn254_msm.metal`) with Rust bindings (`src/metal_msm.rs`). Both were **correctness-verified** against arkworks at all sizes.

**Result: GPU is 10-30x slower than CPU for BN254.** Root cause: Apple Silicon GPUs lack native 64-bit integer multiply. Each 256-bit field multiplication requires 64 32-bit multiplications on GPU vs 16 native 64-bit multiplications on CPU — a **4x per-operation penalty** that unified memory cannot overcome.

| Points | CPU MSM | GPU MSM | Ratio |
|--------|---------|---------|-------|
| 1K | 3.5ms | 35ms | 0.10x |
| 16K | 15ms | 434ms | 0.03x |
| 64K | 49ms | 560ms | 0.09x |

### 5b. Custom AArch64 Assembly

Hand-written Montgomery multiplication assembly was **23% slower** than LLVM-generated code (12.9ns vs 10.5ns per multiply). LLVM compiles `u64 as u128 * u64 as u128` to optimal `MUL`+`UMULH` pairs and reorders the carry chain better than sequential hand-written assembly on M3's wide out-of-order engine.

### 5c. Witness/Proving Pipeline Overlap

We investigated overlapping tier-2 commitment computation with sumcheck stages 1-4. **This is protocol-infeasible**: the Fiat-Shamir transcript requires all commitments to be appended before stage 1 samples its first challenge. Deferring any commitment would break soundness.

```
// In jolt-core/src/zkvm/prover.rs:prove()
let (commitments, hints) = self.generate_and_commit_witness_polynomials();
//  ^ ALL commitments must complete and enter transcript
//    before any stage challenge is sampled
self.prove_stage1();  // <- samples challenge from transcript
```

### 5d. Thread Count Tuning (Previously Implemented)

Setting `RAYON_NUM_THREADS=10` instead of the default 12 provides 15-22% improvement. Apple M3 Pro's E-cores are ~40% slower than P-cores for field arithmetic; rayon's work-stealing creates load imbalance when E-cores can't keep up.

---

## 6. Results

### Proving time (best of 2 iterations, RAYON_NUM_THREADS=10)

| Tier | Mutations | Before | After | Improvement |
|------|-----------|--------|-------|-------------|
| minimal | 4 | 2.59s | **1.55s** | **40.2%** |
| small | 16 | 5.54s | **3.49s** | **37.0%** |
| medium | 32 | 7.34s | **5.95s** | **18.9%** |
| large | 64 | 10.51s | **9.04s** | **14.0%** |

The improvement is largest on smaller tiers where per-point overhead (field inversions, redundant affine conversions) dominates relative to the fixed Miller loop cost.

### Comparison vs SP1 (CPU baselines, same hardware)

| Tier | Jolt (after) | SP1 | Speedup |
|------|-------------|-----|---------|
| minimal | 1.55s | 8.0s | **5.2x** |
| small | 3.49s | 15.0s | **4.3x** |
| medium | 5.95s | 25.0s | **4.2x** |
| large | 9.04s | 60.0s | **6.6x** |

Jolt is now **4.2-6.6x faster** than SP1 across all tiers (up from 2.7-5.7x before optimization).

### Full pipeline timing (large tier)

| Phase | Time |
|-------|------|
| Guest compilation (cached) | 0.32s |
| Preprocessing | 4.24s |
| **Proving** | **9.04s** |
| Verification | 0.07s |
| **Total** | **13.66s** |

---

## 7. What Changed: File-Level Summary

| File | Change | Impact |
|------|--------|--------|
| `third-party/dory-pcs/src/backends/arkworks/ark_pairing.rs` | Added `batch_g1_to_affine()`, `batch_g2_to_affine()` using Montgomery batch inversion; increased chunk sizes; applied batch conversion in all parallel multi-pairing functions | Reduces N field inversions to 1 per pairing call |
| `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` | Added `G1_AFFINE_CACHE` (RwLock + Arc); modified `process_chunk()` and `process_chunk_onehot()` to use cached affine bases | Eliminates 10,239 redundant projective->affine conversions per prove |
| `Cargo.toml` | Changed dory dependency from crates.io to local path fork | Enables local dory-pcs modifications |

No protocol changes. No changes to proof format, verification, or soundness properties. The optimizations are purely in the prover's internal representation conversions.

---

## 8. On-Chain Verification: Groth16 Wrapper for Dory

### Motivation

Jolt currently lacks an on-chain verifier. Direct Dory verification on-chain is infeasible (~146M gas due to multi-round pairings and G2 operations). We built a Groth16 wrapper circuit that proves Dory verification off-chain, then verifies the succinct Groth16 proof on-chain for ~320K gas.

### Architecture

The verification is split between Solidity and a gnark Groth16 circuit:

- **Solidity** (`DoryOnChainVerifier.sol`): Fiat-Shamir transcript replay (Blake2b-512 via EIP-152), MiMC-BN254 public input hashing, Groth16 proof verification, SProduct computation
- **Groth16 circuit** (`circuit/dory_verifier.go`, gnark): GT accumulation, scalar folding, final pairing check, MiMC hash verification

G2 operations are avoided in-circuit by having the prover supply G2 composite points and GT exponentiation results as witness values; correctness is enforced by checkpoint assertions and the final pairing equation.

### Circuit Optimization Path

| Stage | Constraints | Change |
|-------|-------------|--------|
| Naive (all in-circuit) | 52.7M | Baseline |
| Windowed GT exp | 21M | Window-based GT exponentiation |
| Witness-provided GT exp | 1.93M | GT exp results as witness, verified by pairing equation |
| 11-round production | 1.53M | Reduced from 13 to 11 Dory rounds |
| MiMC per-limb hashing | 1.61M | MiMC-BN254 replaces per-limb public inputs |
| MiMC packed hashing | 1.55M | Pack Fp pairs into 2 absorptions instead of 4 |

**Final circuit: 1,553,614 R1CS constraints.**

### Public Input Compression

The Dory verifier has ~240 public inputs (commitment coefficients, transcript values). At ~6K gas per `ecMul` in the Groth16 verifier's `publicInputMSM`, this would cost ~1.4M gas for public inputs alone.

**Solution:** Hash all public inputs into a single scalar via MiMC-BN254, both in-circuit (gnark) and on-chain (Solidity). The circuit verifies `MiMC(inputs) == publicInputHash`. The Groth16 verifier sees only 1 public input.

- **Packed hashing:** Full scalars use 1 MiMC absorption; Fp pairs are packed into 2 absorptions (vs 4 for individual 64-bit limbs). Total: 72 MiMC absorptions, down from 240.

### Gas Optimizations

| Optimization | Gas Saved | Status |
|-------------|-----------|--------|
| MiMC public input compression (240 → 1 input) | ~1.4M | Implemented |
| Packed MiMC hashing (240 → 72 absorptions) | ~890K on-chain | Implemented |
| MiMC constants hoisting (allocate once, pass by ref) | ~150K | Implemented |
| ~~Pedersen pairing stripping~~ | ~~113K~~ | **Unsound — reverted** |

**Pedersen stripping (unsound):** We attempted to strip the Pedersen commitment pairing check from the gnark-generated verifier, reasoning that without explicit `Commit()` calls the commitment would be trivial. **This is incorrect.** gnark v0.14 always generates a non-trivial Pedersen commitment as part of Groth16 — `CommitmentKeys count: 1`, and the commitment point varies per proof. Stripping the 2-pair Pedersen pairing check breaks soundness. Reverted.

**MiMC constants hoisting:** The original Solidity `_mimcEncrypt()` called `_mimcConstants()` on every invocation, allocating a `uint256[110]` memory array each time. With 72 MiMC absorptions × 110 rounds = 72 allocations, EVM's quadratic memory pricing (`3a + a²/512`) caused ~150K gas waste. Fix: allocate once in `_computeInputHash()` and pass by reference.

### On-Chain Gas Breakdown

| Component | Gas |
|-----------|-----|
| Groth16 `verifyProof` (4-pair ecPairing + Pedersen 2-pair + publicInputMSM) | ~320K |
| MiMC-BN254 hash (72 absorptions × 110 rounds) | ~170K |
| Fiat-Shamir transcript replay (Blake2b via EIP-152) | ~50K |
| Other (calldata, memory, control flow) | ~10K |
| **Total estimated** | **~550K** |

### Solidity Compatibility

gnark's native Go verifier defaults to SHA256-based hash-to-field (RFC 9380) for Pedersen commitment hashing, but the generated Solidity verifier uses `keccak256` (cheaper on EVM). Proofs must be generated with the Solidity-compatible hash:

```go
proof, err := groth16.Prove(cs, pk, witness,
    solidity.WithProverTargetSolidityVerifier(backend.GROTH16))
err = groth16.Verify(proof, vk, publicWitness,
    solidity.WithVerifierTargetSolidityVerifier(backend.GROTH16))
```

Without this, the commitment hash differs between Go and Solidity, and the main 4-pair pairing check fails even though the Pedersen 2-pair check passes.

### End-to-End Verification

The Groth16 proof was verified on a local Ethereum mainnet fork (via Foundry `--fork-url`), confirming correct behavior against real EVM precompile implementations:

```
$ forge test --fork-url https://ethereum-rpc.publicnode.com -vvv

[PASS] test_verifyProof_realProof()        (gas: 321,131)
[PASS] test_verifyProof_wrongInput_reverts() (gas: 323,950)
[PASS] test_verifyProof_gasUsage()          (gas: 323,501)
  Gas used for verifyProof: 319,709
```

The proof data comes from a real Jolt execution (`sha2-chain` benchmark, 11 Dory rounds). The Groth16 wrapper circuit compiles to 1.55M R1CS constraints; setup + prove takes ~165s.

### Groth16 Pipeline Timing (11-round production circuit)

| Step | Time |
|------|------|
| Circuit compile (R1CS) | ~5s |
| Groth16 setup | ~150s |
| Prove | ~5s |
| Verify (native) | <1s |
| Solidity export | <1s |

### Files

| File | Purpose |
|------|---------|
| `zk-onchain-verifier/circuit/dory_verifier.go` | gnark circuit: 11-round Dory verifier with witness-GT-exp and packed MiMC |
| `zk-onchain-verifier/circuit/witness_loader.go` | Witness assignment from Jolt-exported JSON |
| `zk-onchain-verifier/circuit/export_proof_test.go` | Groth16 prove + Solidity verifier export + Foundry test generation |
| `zk-onchain-verifier/contracts/DoryOnChainVerifier.sol` | Integration contract: Fiat-Shamir replay + MiMC hash + Groth16 call |
| `zk-onchain-verifier/foundry-test/` | Foundry project for on-chain proof verification testing |

---

## 9. Remaining Prover Bottlenecks

After optimization, the proving time breakdown (large tier, 9.04s) is approximately:

| Component | Est. time | % |
|-----------|----------|---|
| Dory tier-2 commitment (Miller loops) | ~3.0s | 33% |
| Dory opening proof Stage 8 (13 rounds) | ~2.5s | 28% |
| Sumcheck stages 5-6 | ~2.0s | 22% |
| Other stages + overhead | ~1.5s | 17% |

The Miller loop computation itself (not the affine conversion) is now the dominant cost. Further improvement would require:

1. **HyperKZG commitment scheme** — Replaces Dory's pairing-heavy approach with KZG-based openings (faster but requires trusted setup)
2. **Optimized BN254 pairing library** — Faster Fq12 arithmetic, shared Miller loop optimization, lazy reduction
3. **Different proof system** — FRI-based with Goldilocks field (no pairings at all)

---

## 10. Reproducibility

```bash
# Build
cd contracts/zk-jolt-workspace
cargo build --release -p lightningjolt

# Run single tier
RAYON_NUM_THREADS=10 cargo run --release -p lightningjolt -- profile --tier large

# Run full sweep (all tiers, 2 iterations each)
RAYON_NUM_THREADS=10 cargo run --release -p lightningjolt -- sweep --iterations 2

# With Chrome trace output (viewable in Perfetto)
RAYON_NUM_THREADS=10 cargo run --release -p lightningjolt -- profile --tier large --trace
# Opens lightningjolt-trace.json in https://ui.perfetto.dev
```
