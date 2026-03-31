//! Rayon thread pool tuner for Apple Silicon.
//!
//! M3 Pro has 6 performance + 6 efficiency cores. The optimal thread count
//! for ZK proving may differ from the hardware thread count because:
//! - E-cores are ~40% slower than P-cores for field arithmetic
//! - Rayon's work-stealing can leave P-cores waiting on E-core stragglers
//! - Memory bandwidth saturates before all cores are busy
//!
//! This module sweeps thread counts to find the empirical optimum.

use std::time::Instant;

const HEX: [u8; 16] = *b"0123456789abcdef";

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

fn prove_with_threads(tier: &BenchTier, num_threads: usize) -> f64 {
    // Build a custom rayon pool with the specified thread count
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let (mc, kf, kl, vf, vl, root, sigs, an, bn) = generate_synthetic_block(tier);

    pool.install(|| {
        let target_dir = "/tmp/jolt-lightningjolt-targets";
        let mut program = guest::compile_prove_block(target_dir);
        let shared = guest::preprocess_shared_prove_block(&mut program);
        let prover_pp = guest::preprocess_prover_prove_block(shared.clone());
        let verifier_pp = guest::preprocess_verifier_prove_block(
            shared,
            prover_pp.generators.to_verifier_setup(),
        );
        let prove_fn = guest::build_prover_prove_block(program, prover_pp);
        let verify_fn = guest::build_verifier_prove_block(verifier_pp);

        let start = Instant::now();
        let (output, proof, program_io) = prove_fn(mc, &kf, &kl, &vf, &vl, root, sigs, an, bn);
        let prove_secs = start.elapsed().as_secs_f64();

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
        assert!(
            valid,
            "Proof verification failed with {num_threads} threads"
        );

        prove_secs
    })
}

pub fn run_tune(tier_name: &str) {
    tracing_subscriber::fmt()
        .with_env_filter("jolt_core=warn")
        .init();

    let tier = TIERS
        .iter()
        .find(|t| t.name == tier_name)
        .unwrap_or_else(|| {
            eprintln!("Unknown tier: {tier_name}. Available: minimal, small, medium, large");
            std::process::exit(1);
        });

    let hw_threads = num_cpus();
    println!("LightningJolt Thread Tuner — M3 Pro ({hw_threads} hardware threads)");
    println!("Tier: {} ({} mutations)", tier.name, tier.mutations);
    println!("=========================================================\n");

    // Sweep from P-core-only (6) to all cores (12) plus over-subscription
    let thread_counts: Vec<usize> = vec![1, 2, 4, 6, 8, 10, 12];
    let thread_counts: Vec<usize> = thread_counts
        .into_iter()
        .filter(|&t| t <= hw_threads + 2)
        .collect();

    println!("{:>8} {:>10} {:>10}", "Threads", "Prove(s)", "vs Best");
    println!("{:-<32}", "");

    let mut results: Vec<(usize, f64)> = Vec::new();

    for &threads in &thread_counts {
        print!("{threads:>8} ");
        let prove_secs = prove_with_threads(tier, threads);
        results.push((threads, prove_secs));

        let best_so_far = results.iter().map(|r| r.1).fold(f64::MAX, f64::min);
        let ratio = prove_secs / best_so_far;
        let marker = if (ratio - 1.0).abs() < 0.02 {
            " ← best"
        } else {
            ""
        };
        println!("{prove_secs:>9.3}s {ratio:>9.2}x{marker}");
    }

    let (best_threads, best_time) = results
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("\nOptimal: {best_threads} threads → {best_time:.3}s");
    println!("\nTo apply: set RAYON_NUM_THREADS={best_threads} before running prover");

    // P-core vs all-core analysis
    if let (Some(p_only), Some(all)) = (
        results.iter().find(|r| r.0 == 6),
        results.iter().find(|r| r.0 == 12),
    ) {
        let e_core_overhead = (all.1 / p_only.1 - 1.0) * 100.0;
        if e_core_overhead > 5.0 {
            println!("\nE-core impact: using all 12 cores is {e_core_overhead:.0}% slower than P-cores only (6)");
            println!("Recommendation: RAYON_NUM_THREADS=6 (P-cores only)");
        } else if e_core_overhead < -5.0 {
            println!(
                "\nE-cores help: all 12 cores is {:.0}% faster than P-cores only",
                -e_core_overhead
            );
        } else {
            println!("\nE-cores have minimal impact ({e_core_overhead:+.0}%)");
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
