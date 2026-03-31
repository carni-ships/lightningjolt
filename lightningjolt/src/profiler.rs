//! Per-stage profiler for the Jolt proving pipeline.
//!
//! Wraps the Jolt guest prover with fine-grained timing instrumentation
//! and optional Chrome trace output for Perfetto visualization.

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

// SP1 baselines (M3 Pro CPU)
const SP1_BASELINE_SECS: &[(&str, f64)] = &[
    ("minimal", 8.0),
    ("small", 15.0),
    ("medium", 25.0),
    ("large", 60.0),
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

struct StageTimings {
    tier_name: String,
    mutations: usize,
    compile_secs: f64,
    preprocess_secs: f64,
    prove_secs: f64,
    verify_secs: f64,
    total_secs: f64,
    proof_valid: bool,
    rayon_threads: usize,
}

fn run_single_profile(tier: &BenchTier) -> StageTimings {
    let rayon_threads = rayon::current_num_threads();
    println!("  Rayon threads: {rayon_threads}");
    println!(
        "  Generating synthetic block: {} mutations...",
        tier.mutations
    );

    let (mc, kf, kl, vf, vl, root, sigs, an, bn) = generate_synthetic_block(tier);

    let total_start = Instant::now();

    // Phase 1: Compile guest to RISC-V ELF
    let compile_start = Instant::now();
    let target_dir = "/tmp/jolt-lightningjolt-targets";
    let mut program = guest::compile_prove_block(target_dir);
    let compile_secs = compile_start.elapsed().as_secs_f64();
    println!("  Compile:      {compile_secs:>8.3}s");

    // Phase 2: Preprocessing (one-time per circuit)
    let pp_start = Instant::now();
    let shared = guest::preprocess_shared_prove_block(&mut program);
    let prover_pp = guest::preprocess_prover_prove_block(shared.clone());
    let verifier_pp =
        guest::preprocess_verifier_prove_block(shared, prover_pp.generators.to_verifier_setup());
    let preprocess_secs = pp_start.elapsed().as_secs_f64();
    println!("  Preprocess:   {preprocess_secs:>8.3}s");

    let prove_fn = guest::build_prover_prove_block(program, prover_pp);
    let verify_fn = guest::build_verifier_prove_block(verifier_pp);

    // Phase 3: Proving
    let prove_start = Instant::now();
    let (output, proof, program_io) = prove_fn(mc, &kf, &kl, &vf, &vl, root, sigs, an, bn);
    let prove_secs = prove_start.elapsed().as_secs_f64();
    println!("  Prove:        {prove_secs:>8.3}s");

    // Phase 4: Verification
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
    let verify_secs = verify_start.elapsed().as_secs_f64();
    println!(
        "  Verify:       {verify_secs:>8.3}s  ({})",
        if valid { "OK" } else { "FAIL" }
    );

    let total_secs = total_start.elapsed().as_secs_f64();
    println!("  Total:        {total_secs:>8.3}s");

    StageTimings {
        tier_name: tier.name.to_string(),
        mutations: tier.mutations,
        compile_secs,
        preprocess_secs,
        prove_secs,
        verify_secs,
        total_secs,
        proof_valid: valid,
        rayon_threads,
    }
}

pub fn run_profile(tier_name: &str, trace: bool, iterations: u32) {
    let _guard = if trace {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file("lightningjolt-trace.json")
            .build();
        tracing_subscriber::fmt()
            .with_env_filter("jolt_core=info")
            .finish();
        use tracing_subscriber::layer::SubscriberExt;
        let subscriber = tracing_subscriber::registry().with(chrome_layer);
        tracing::subscriber::set_global_default(subscriber).ok();
        println!("Chrome trace enabled → lightningjolt-trace.json");
        Some(guard)
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("jolt_core=warn")
            .init();
        None
    };

    let tiers: Vec<&BenchTier> = if tier_name == "all" {
        TIERS.iter().collect()
    } else {
        TIERS.iter().filter(|t| t.name == tier_name).collect()
    };
    if tiers.is_empty() {
        eprintln!("Unknown tier: {tier_name}. Available: minimal, small, medium, large, all");
        std::process::exit(1);
    }

    println!("LightningJolt Profiler — Apple M3 Pro (12 cores, 18GB)");
    println!("================================================\n");

    let mut results = Vec::new();
    for t in &tiers {
        println!(
            "[{}] {} mutations, {} sigs",
            t.name, t.mutations, t.signatures
        );
        let mut best: Option<StageTimings> = None;
        for i in 0..iterations {
            if iterations > 1 {
                println!("  --- Run {}/{iterations} ---", i + 1);
            }
            let timing = run_single_profile(t);
            if best.is_none() || timing.prove_secs < best.as_ref().unwrap().prove_secs {
                best = Some(timing);
            }
        }
        results.push(best.unwrap());
        println!();
    }

    print_profile_table(&results);
}

pub fn run_sweep(iterations: u32) {
    tracing_subscriber::fmt()
        .with_env_filter("jolt_core=warn")
        .init();

    println!("LightningJolt Sweep — All Tiers × Optimized Settings");
    println!("===============================================\n");

    let mut results = Vec::new();
    for tier in TIERS {
        println!("[{}] {} mutations", tier.name, tier.mutations);
        let mut best: Option<StageTimings> = None;
        for i in 0..iterations {
            if iterations > 1 {
                println!("  --- Run {}/{iterations} ---", i + 1);
            }
            let timing = run_single_profile(tier);
            if best.is_none() || timing.prove_secs < best.as_ref().unwrap().prove_secs {
                best = Some(timing);
            }
        }
        results.push(best.unwrap());
        println!();
    }

    print_profile_table(&results);
    print_comparison_table(&results);
}

fn print_profile_table(results: &[StageTimings]) {
    println!("==========================================================================");
    println!(" LIGHTNINGJOLT PROFILE RESULTS");
    println!("==========================================================================");
    println!(
        "{:<10} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>5}",
        "Tier", "Mut", "Compile", "Preproc", "Prove", "Verify", "Total", "Thds"
    );
    println!("{:-<70}", "");
    for r in results {
        println!(
            "{:<10} {:>5} {:>7.3}s {:>7.3}s {:>7.3}s {:>7.3}s {:>7.3}s {:>5}",
            r.tier_name,
            r.mutations,
            r.compile_secs,
            r.preprocess_secs,
            r.prove_secs,
            r.verify_secs,
            r.total_secs,
            r.rayon_threads
        );
    }
    println!();
}

fn print_comparison_table(results: &[StageTimings]) {
    println!("==========================================================================");
    println!(" LIGHTNINGJOLT vs SP1 COMPARISON");
    println!("==========================================================================");
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>8}",
        "Tier", "Jolt(s)", "SP1(s)", "Ratio", "Winner"
    );
    println!("{:-<52}", "");
    for r in results {
        let sp1_s = SP1_BASELINE_SECS
            .iter()
            .find(|(n, _)| *n == r.tier_name)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let ratio = if sp1_s > 0.0 {
            r.prove_secs / sp1_s
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
            r.tier_name, r.prove_secs, sp1_s, ratio, winner
        );
    }
    println!();
}
