//! Persistia Jolt Prover — Benchmark & Evaluation Harness
//!
//! Evaluates Jolt zkVM performance for Persistia state transition proofs.
//! Generates synthetic workloads at various sizes to measure proving time,
//! verification time, and correctness.
//!
//! Usage:
//!   cargo run --release -p persistia-jolt-host -- bench
//!   cargo run --release -p persistia-jolt-host -- bench --tier small
//!   cargo run --release -p persistia-jolt-host -- compare

use clap::{Parser, Subcommand};
use std::time::Instant;
use tracing::info;

const HEX: [u8; 16] = *b"0123456789abcdef";

#[derive(Parser)]
#[command(
    name = "persistia-jolt-bench",
    about = "Jolt zkVM benchmark for Persistia"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run benchmark suite with synthetic workloads
    Bench {
        /// Benchmark tier: minimal, small, medium, large, all
        #[arg(long, default_value = "all")]
        tier: String,
        /// Number of iterations per tier (best-of-N)
        #[arg(long, default_value = "1")]
        iterations: u32,
    },
    /// Print comparison table with SP1 baseline times
    Compare,
}

// ─── Benchmark Tiers ────────────────────────────────────────────────────

struct BenchTier {
    name: &'static str,
    mutations: usize,
    signatures: u32,
    active_nodes: u32,
}

const TIERS: &[BenchTier] = &[
    BenchTier {
        name: "minimal",
        mutations: 4,
        signatures: 1,
        active_nodes: 1,
    },
    BenchTier {
        name: "small",
        mutations: 16,
        signatures: 3,
        active_nodes: 4,
    },
    BenchTier {
        name: "medium",
        mutations: 32,
        signatures: 5,
        active_nodes: 7,
    },
    BenchTier {
        name: "large",
        mutations: 64,
        signatures: 10,
        active_nodes: 15,
    },
];

// SP1 baseline times from METAL_GPU_RESEARCH.md (M3 Pro, CPU mode)
const SP1_BASELINE_SECS: &[(&str, f64)] = &[
    ("minimal", 8.0),
    ("small", 15.0),
    ("medium", 25.0),
    ("large", 60.0),
];

// ─── SHA-256 helpers (host-side, matching guest logic) ──────────────────

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    jolt_inlines_sha2::Sha256::digest(data)
}

fn sha256_leaf_host(key: &[u8], value: &[u8]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(5 + key.len() * 2 + 1 + value.len() * 2);
    buf.extend_from_slice(b"leaf:");
    for &b in key {
        buf.push(HEX[(b >> 4) as usize]);
        buf.push(HEX[(b & 0xf) as usize]);
    }
    buf.push(b':');
    for &b in value {
        buf.push(HEX[(b >> 4) as usize]);
        buf.push(HEX[(b & 0xf) as usize]);
    }
    sha256_bytes(&buf)
}

fn sha256_node_host(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 128];
    let mut pos = 0;
    for &b in left.iter() {
        buf[pos] = HEX[(b >> 4) as usize];
        buf[pos + 1] = HEX[(b & 0xf) as usize];
        pos += 2;
    }
    for &b in right.iter() {
        buf[pos] = HEX[(b >> 4) as usize];
        buf[pos + 1] = HEX[(b & 0xf) as usize];
        pos += 2;
    }
    sha256_bytes(&buf)
}

// ─── Synthetic Data Generation ──────────────────────────────────────────

fn generate_synthetic_block(
    tier: &BenchTier,
) -> (
    u32,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    [u8; 32],
    u32,
    u32,
    u64,
) {
    let mut keys_flat = Vec::new();
    let mut key_lens = Vec::new();
    let mut values_flat = Vec::new();
    let mut value_lens = Vec::new();
    let mut leaves = Vec::new();

    for i in 0..tier.mutations {
        let key = format!("state:key:{i:04}");
        let value = format!("value:{i:04}:{:016x}", rand::random::<u64>());
        let key_bytes = key.as_bytes();
        let val_bytes = value.as_bytes();

        keys_flat.extend_from_slice(key_bytes);
        key_lens.push(key_bytes.len() as u8);
        values_flat.extend_from_slice(val_bytes);
        value_lens.push(val_bytes.len() as u8);

        leaves.push(sha256_leaf_host(key_bytes, val_bytes));
    }

    // Compute expected Merkle root
    let mut padded = leaves.clone();
    while padded.len() < 1 || padded.len().count_ones() != 1 {
        padded.push([0u8; 32]);
    }
    while padded.len() > 1 {
        let mut next = Vec::new();
        for chunk in padded.chunks(2) {
            next.push(sha256_node_host(&chunk[0], &chunk[1]));
        }
        padded = next;
    }

    (
        tier.mutations as u32,
        keys_flat,
        key_lens,
        values_flat,
        value_lens,
        padded[0],
        tier.signatures,
        tier.active_nodes,
        42u64,
    )
}

// ─── Benchmark Runner ───────────────────────────────────────────────────

struct BenchResult {
    tier_name: String,
    mutations: usize,
    signatures: u32,
    preprocess_secs: f64,
    prove_time_secs: f64,
    verify_time_secs: f64,
    proof_valid: bool,
}

fn run_benchmark(tier: &BenchTier) -> BenchResult {
    info!(
        "Generating synthetic block: {} mutations, {} sigs...",
        tier.mutations, tier.signatures
    );
    let (mc, kf, kl, vf, vl, root, sigs, an, bn) = generate_synthetic_block(tier);

    info!("Compiling guest program...");
    let target_dir = "/tmp/jolt-persistia-targets";
    let mut program = guest::compile_prove_block(target_dir);

    info!("Preprocessing...");
    let pp_start = Instant::now();
    let shared = guest::preprocess_shared_prove_block(&mut program);
    let prover_pp = guest::preprocess_prover_prove_block(shared.clone());
    let verifier_pp =
        guest::preprocess_verifier_prove_block(shared, prover_pp.generators.to_verifier_setup());
    let pp_elapsed = pp_start.elapsed();
    info!("Preprocessing: {:.3}s", pp_elapsed.as_secs_f64());

    let prove_fn = guest::build_prover_prove_block(program, prover_pp);
    let verify_fn = guest::build_verifier_prove_block(verifier_pp);

    info!("PROVING ({} mutations)...", mc);
    let prove_start = Instant::now();
    let (output, proof, program_io) = prove_fn(mc, &kf, &kl, &vf, &vl, root, sigs, an, bn);
    let prove_elapsed = prove_start.elapsed();
    info!("Prove time: {:.3}s", prove_elapsed.as_secs_f64());
    info!("Output root: {}", hex::encode(output));

    info!("VERIFYING...");
    let verify_start = Instant::now();
    let valid = verify_fn(
        mc,
        &kf,
        &kl,
        &vf,
        &vl,
        root,
        sigs,
        an,
        bn,
        output,
        program_io.panic,
        proof,
    );
    let verify_elapsed = verify_start.elapsed();
    info!(
        "Verify: {:.3}s, valid: {}",
        verify_elapsed.as_secs_f64(),
        valid
    );

    BenchResult {
        tier_name: tier.name.to_string(),
        mutations: tier.mutations,
        signatures: tier.signatures,
        preprocess_secs: pp_elapsed.as_secs_f64(),
        prove_time_secs: prove_elapsed.as_secs_f64(),
        verify_time_secs: verify_elapsed.as_secs_f64(),
        proof_valid: valid,
    }
}

fn print_results(results: &[BenchResult]) {
    println!();
    println!("========================================================================");
    println!(" JOLT BENCHMARK RESULTS — Persistia State Transition Proof");
    println!("========================================================================");
    println!(
        "{:<10} {:>6} {:>6} {:>10} {:>10} {:>10} {:>6}",
        "Tier", "Mut", "Sigs", "Preproc", "Prove(s)", "Verify", "Valid"
    );
    println!("{:-<65}", "");
    for r in results {
        println!(
            "{:<10} {:>6} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>6}",
            r.tier_name,
            r.mutations,
            r.signatures,
            r.preprocess_secs,
            r.prove_time_secs,
            r.verify_time_secs,
            if r.proof_valid { "OK" } else { "FAIL" }
        );
    }
    println!();
}

fn print_comparison(results: &[BenchResult]) {
    println!();
    println!("========================================================================");
    println!(" JOLT vs SP1 COMPARISON (Persistia State Transition)");
    println!("========================================================================");
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>8}",
        "Tier", "Jolt(s)", "SP1(s)", "Ratio", "Winner"
    );
    println!("{:-<52}", "");
    for r in results {
        let sp1_s = SP1_BASELINE_SECS
            .iter()
            .find(|(name, _)| *name == r.tier_name)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let ratio = if sp1_s > 0.0 {
            r.prove_time_secs / sp1_s
        } else {
            0.0
        };
        let winner = if ratio < 0.95 {
            "JOLT"
        } else if ratio > 1.05 {
            "SP1"
        } else {
            "TIE"
        };
        println!(
            "{:<10} {:>10.3} {:>10.1} {:>9.2}x {:>8}",
            r.tier_name, r.prove_time_secs, sp1_s, ratio, winner
        );
    }
    println!();
    println!("NOTES:");
    println!("  - SP1 baselines: M3 Pro CPU, from METAL_GPU_RESEARCH.md");
    println!("  - SP1 has Ed25519 precompile; Jolt verifies sig count only");
    println!("  - SP1 supports recursive IVC; Jolt does not yet");
    println!("  - Jolt uses Lasso+Sumcheck; SP1 uses STARKs+Plonky3");
    println!();
}

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench { tier, iterations } => {
            let tiers: Vec<&BenchTier> = if tier == "all" {
                TIERS.iter().collect()
            } else {
                TIERS.iter().filter(|t| t.name == tier).collect()
            };
            if tiers.is_empty() {
                eprintln!("Unknown tier. Available: minimal, small, medium, large, all");
                std::process::exit(1);
            }

            println!("Persistia Jolt Benchmark Suite");
            println!("Tiers: {}, Iterations: {}\n", tiers.len(), iterations);

            let mut all_results = Vec::new();
            for (idx, t) in tiers.iter().enumerate() {
                println!(
                    "[{}/{}] Tier: {} ({} mutations, {} sigs)",
                    idx + 1,
                    tiers.len(),
                    t.name,
                    t.mutations,
                    t.signatures
                );
                let mut best: Option<BenchResult> = None;
                for i in 0..iterations {
                    println!("  Run {}/{}...", i + 1, iterations);
                    let result = run_benchmark(t);
                    if best.is_none()
                        || result.prove_time_secs < best.as_ref().unwrap().prove_time_secs
                    {
                        best = Some(result);
                    }
                }
                all_results.push(best.unwrap());
                println!();
            }

            print_results(&all_results);
            print_comparison(&all_results);
        }
        Commands::Compare => {
            println!("Running all tiers for comparison...\n");
            let mut results = Vec::new();
            for t in TIERS {
                println!("[{}]", t.name);
                results.push(run_benchmark(t));
                println!();
            }
            print_comparison(&results);
        }
    }
}
