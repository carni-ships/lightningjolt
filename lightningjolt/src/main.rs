//! LightningJolt — Jolt Prover Optimizer for Apple Silicon
//!
//! Profiles and optimizes the Jolt proving pipeline
//! state transition proofs on M3 Pro hardware.
//!
//! Usage:
//!   cargo run --release -p lightningjolt -- profile --tier minimal
//!   cargo run --release -p lightningjolt -- profile --tier all --trace
//!   cargo run --release -p lightningjolt -- tune-threads
//!   cargo run --release -p lightningjolt -- sweep
//!   cargo run --release -p lightningjolt -- gpu --size 65536

pub mod aarch64_mont;
pub mod metal_msm;
mod profiler;
mod thread_tuner;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "lightningjolt", about = "Jolt prover optimizer for Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Profile proving pipeline with per-stage timing breakdown
    Profile {
        #[arg(long, default_value = "minimal")]
        tier: String,
        /// Emit Chrome trace JSON for Perfetto visualization
        #[arg(long)]
        trace: bool,
        /// Number of iterations (reports best-of-N)
        #[arg(long, default_value = "1")]
        iterations: u32,
    },
    /// Find optimal rayon thread count for this hardware
    TuneThreads {
        #[arg(long, default_value = "minimal")]
        tier: String,
    },
    /// Sweep all tiers with optimized settings, compare against baseline
    Sweep {
        #[arg(long, default_value = "1")]
        iterations: u32,
    },
    /// Benchmark Metal GPU field arithmetic
    Gpu {
        /// Number of field elements to process
        #[arg(long, default_value = "65536")]
        size: usize,
    },
    /// Benchmark Metal GPU MSM (multi-scalar multiplication)
    Msm {
        /// Number of points for MSM
        #[arg(long, default_value = "4096")]
        size: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Profile {
            tier,
            trace,
            iterations,
        } => profiler::run_profile(&tier, trace, iterations),
        Commands::TuneThreads { tier } => thread_tuner::run_tune(&tier),
        Commands::Sweep { iterations } => profiler::run_sweep(iterations),
        Commands::Gpu { size } => {
            if let Err(e) = metal_msm::gpu::benchmark_gpu(size) {
                eprintln!("GPU benchmark failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Msm { size } => {
            if let Err(e) = metal_msm::msm::benchmark_msm(size) {
                eprintln!("GPU MSM benchmark failed: {e}");
                std::process::exit(1);
            }
        }
    }
}
