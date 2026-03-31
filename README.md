# Lightningjolt

Optimizing the [Jolt](https://github.com/a16z/jolt) zkVM prover and building the first on-chain Dory verifier.

## What This Is

Jolt is a zkVM for RISC-V (RV64IMAC) that uses sumcheck-based protocols and the Dory polynomial commitment scheme. This fork contains two lines of work, applicable to any Jolt program:

1. **Prover optimization (LightningJolt)** — 14-40% faster proving via batch affine conversion in BN254 pairings and G1 SRS caching
2. **On-chain verification** — A Groth16 wrapper circuit that makes Dory proofs verifiable on Ethereum for ~320K gas

## Prover Optimization

The Dory commitment scheme dominates Jolt's proving time (~63% spent in BN254 pairings). We implemented:

- **Montgomery batch affine conversion** in the pairing layer — replaces N field inversions with 1 inversion + 3(N-1) multiplications
- **G1 SRS base caching** — eliminates 10,240 redundant projective-to-affine conversions per prove via a static `RwLock<Arc>` cache

| Circuit size | Before | After | Improvement |
|--------------|--------|-------|-------------|
| minimal | 2.59s | 1.55s | **40%** |
| small | 5.54s | 3.49s | **37%** |
| medium | 7.34s | 5.95s | **19%** |
| large | 10.51s | 9.04s | **14%** |

Hardware: Apple M3 Pro, `RAYON_NUM_THREADS=10`.

We also investigated and ruled out Metal GPU acceleration (4x penalty from lack of native 64-bit integer multiply), custom AArch64 assembly (LLVM already optimal), and witness/proving pipeline overlap (blocked by Fiat-Shamir transcript ordering).

See [`LightningJolt/RESEARCH-SUMMARY.md`](LightningJolt/RESEARCH-SUMMARY.md) for the full write-up.

## On-Chain Dory Verification

Direct Dory verification costs ~146M gas. We built a Groth16 wrapper circuit in gnark that proves Dory verification off-chain, producing a succinct proof verifiable on-chain.

**Circuit:** 1.55M R1CS constraints (down from 52.7M through witness-provided GT exponentiation, round reduction, and packed MiMC hashing).

**Gas cost:** ~320K for `verifyProof` (4-pair ecPairing + Pedersen commitment check + public input MSM).

**Verified on Ethereum mainnet fork:**

```
$ cd contracts/zk-onchain-verifier/foundry-test
$ forge test --fork-url https://ethereum-rpc.publicnode.com -vvv

[PASS] test_verifyProof_realProof()          (gas: 321,131)
[PASS] test_verifyProof_wrongInput_reverts() (gas: 323,950)
[PASS] test_verifyProof_gasUsage()           (gas: 323,501)
  Gas used for verifyProof: 319,709
```

The on-chain verifier code lives in a sibling directory (`contracts/zk-onchain-verifier/`).

## Repository Structure

```
jolt-core/              # Core proving system (with Dory pairing optimizations)
third-party/dory-pcs/   # Local fork of dory-pcs with batch affine conversion
LightningJolt/                # Profiler, benchmarks, Metal GPU experiments, research reports
  RESEARCH-SUMMARY.md   # Full optimization research write-up
  RESULTS.md            # Benchmark results and findings
  shaders/              # Metal compute shaders (BN254 field ops, MSM)
  src/                  # Profiler, thread tuner, Metal MSM bindings
```

## Quick Start

```bash
# Build
cargo build --release -p jolt-core

# Run benchmarks
RAYON_NUM_THREADS=10 cargo run --release -p lightningjolt -- profile --tier large

# Run full sweep
RAYON_NUM_THREADS=10 cargo run --release -p lightningjolt -- sweep --iterations 2

# Generate Chrome trace (viewable in Perfetto)
RAYON_NUM_THREADS=10 cargo run --release -p lightningjolt -- profile --tier large --trace
```

## Upstream

Based on [a16z/jolt](https://github.com/a16z/jolt) at commit `eb370ec2`. See the upstream repo for Jolt documentation, the Jolt Book, and general usage.

## Papers

- [Jolt: SNARKs for Virtual Machines via Lookups](https://eprint.iacr.org/2023/1217) — Arasu Arun, Srinath Setty, Justin Thaler
- [Twist and Shout: Faster memory checking arguments](https://eprint.iacr.org/2025/105) — Srinath Setty, Justin Thaler
- [Unlocking the lookup singularity with Lasso](https://eprint.iacr.org/2023/1216) — Srinath Setty, Justin Thaler, Riad Wahby
