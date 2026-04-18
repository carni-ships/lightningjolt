//! # dory
//!
//! A high performance and modular implementation of the Dory polynomial commitment scheme.
//!
//! Dory is a transparent polynomial commitment scheme with excellent asymptotic
//! performance, based on the work of Jonathan Lee
//! ([eprint 2020/1274](https://eprint.iacr.org/2020/1274)).
//!
//! ## Key Features
//!
//! - **Transparent setup** with automatic disk persistence
//! - **Logarithmic proof size**: O(log n) group elements
//! - **Logarithmic verification**: O(log n) GT exps and 1 multi-pairing
//! - **Performance optimizations**: Optional prepared point caching (~20-30% speedup) and parallelization
//! - **Flexible matrix layouts**: Supports both square and non-square matrices (nu ≤ sigma)
//! - **Homomorphic properties**: Com(r₁·P₁ + r₂·P₂ + ... + rₙ·Pₙ) = r₁·Com(P₁) + r₂·Com(P₂) + ... + rₙ·Com(Pₙ)
//! - **Zero-knowledge mode**: Optional hiding proofs via the `zk` feature flag
//!
//! ## Structure
//!
//! ### Core Modules
//! - [`primitives`] - Core traits and abstractions
//!   - [`primitives::arithmetic`] - Field, group, and pairing curve traits
//!   - [`primitives::poly`] - Multilinear polynomial traits and operations
//!   - [`primitives::transcript`] - Fiat-Shamir transcript trait
//!   - [`primitives::serialization`] - Serialization abstractions
//! - [`mod@setup`] - Transparent setup generation for prover and verifier
//! - [`evaluation_proof`] - Evaluation proof creation and verification
//! - [`reduce_and_fold`] - Inner product protocol state machines (prover/verifier)
//! - [`messages`] - Protocol message structures (VMV, reduce rounds, scalar product)
//! - [`proof`] - Complete proof data structure
//! - [`error`] - Error types
//!
//! ### Backend Implementations
//! - `backends` - Concrete backend implementations (available with feature flags)
//!   - `backends::arkworks` - Arkworks backend with BN254 curve (requires `arkworks` feature)
//!
//! ## Usage
//!
//! ### Basic Example
//!
//! ```ignore
//! use dory_pcs::{setup, prove, verify, Transparent};
//! use dory_pcs::backends::arkworks::{BN254, G1Routines, G2Routines, Blake2bTranscript};
//!
//! // 1. Generate setup (automatically loads from/saves to disk)
//! let (prover_setup, verifier_setup) = setup::<BN254>(max_log_n);
//!
//! // 2. Commit to polynomial
//! let (tier_2_commitment, tier_1_commitments, commit_blind) = polynomial
//!     .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)?;
//!
//! // 3. Generate evaluation proof
//! let mut prover_transcript = Blake2bTranscript::new(b"domain-separation");
//! let proof = prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
//!     &polynomial, &point, tier_1_commitments, commit_blind, nu, sigma,
//!     &prover_setup, &mut prover_transcript
//! )?;
//!
//! // 4. Verify
//! let mut verifier_transcript = Blake2bTranscript::new(b"domain-separation");
//! verify::<_, BN254, G1Routines, G2Routines, _>(
//!     tier_2_commitment, evaluation, &point, &proof,
//!     verifier_setup, &mut verifier_transcript
//! )?;
//! ```
//!
//! ### Performance Optimization
//!
//! Enable prepared point caching for ~20-30% pairing speedup (requires `cache` feature):
//! ```ignore
//! use dory_pcs::backends::arkworks::init_cache;
//!
//! let (prover_setup, verifier_setup) = setup::<BN254>(max_log_n);
//! init_cache(&prover_setup.g1_vec, &prover_setup.g2_vec);
//! // Subsequent operations will automatically use cached prepared points
//! ```
//!
//! ### Examples
//!
//! See the `examples/` directory for complete demonstrations:
//! - `basic_e2e.rs` - Standard square matrix workflow
//! - `homomorphic.rs` - Homomorphic combination of multiple polynomials
//! - `non_square.rs` - Non-square matrix layout (nu < sigma)
//! - `zk_e2e.rs` - Zero-knowledge proof workflow (requires `zk` feature)
//! - `zk_statistical.rs` - Chi-squared uniformity and witness-independence tests (requires `zk` feature)
//!
//! ## Feature Flags
//!
//! - `backends` - Enable concrete backends (currently Arkworks BN254, includes `disk-persistence`)
//! - `cache` - Enable prepared point caching (~20-30% speedup, requires `parallel`)
//! - `zk` - Enable zero-knowledge mode with hiding commitments and proofs
//! - `parallel` - Enable Rayon parallelization for MSMs and pairings
//! - `disk-persistence` - Enable automatic setup caching to disk

/// Error types for Dory PCS operations
pub mod error;
pub mod evaluation_proof;
pub mod messages;
pub mod mode;
pub mod primitives;
pub mod proof;
pub mod reduce_and_fold;
#[cfg(feature = "lattice")]
pub mod reduce_and_fold_lattice;
pub mod setup;

/// Batch proof generation utilities for combining multiple proofs efficiently
pub mod batch;

/// Dory-Prime: parallelizable proof generation via challenge tree
///
/// This module provides an alternative proof generation algorithm that computes
/// all 2^sigma possible challenge paths in parallel, enabling faster prover
/// time through parallel computation.
#[cfg(any(feature = "parallel", feature = "dory-prime"))]
pub mod dory_prime;

#[cfg(any(feature = "arkworks", feature = "lattice"))]
pub mod backends;

pub use error::DoryError;
pub use evaluation_proof::create_evaluation_proof;
pub use messages::{
    FirstReduceMessage, FirstReduceMessage4, ScalarProductMessage, ScalarProductProof,
    SecondReduceMessage, SecondReduceMessage4, VMVMessage,
};
#[cfg(feature = "zk")]
pub use messages::{Sigma1Proof, Sigma2Proof};
#[cfg(feature = "zk")]
pub use mode::ZK;
pub use mode::{Mode, Transparent};
use primitives::arithmetic::{DoryRoutines, Field, Group, PairingCurve};
pub use primitives::poly::{MultilinearLagrange, Polynomial};
use primitives::serialization::{DoryDeserialize, DorySerialize};
pub use proof::{DoryProof, DoryProof4ary};
pub use reduce_and_fold::{DoryProverState, DoryVerifierState, ForkableDoryProverState};
pub use setup::{ProverSetup, VerifierSetup};

/// Dory Fiat-Shamir challenges and coordinates extracted during verification.
/// Used to build gnark Groth16 circuit witness for on-chain verification.
#[allow(missing_docs)]
pub struct DoryWitnessData<F, G1, G2, GT> {
    pub alphas: Vec<F>,
    pub betas: Vec<F>,
    pub gamma: F,
    pub d: F,
    pub s1_coords: Vec<F>,
    pub s2_coords: Vec<F>,
    pub num_rounds: usize,
    pub commitment: GT,
    pub evaluation: F,
    pub vmv_message: VMVMessage<G1, GT>,
    pub first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    pub second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    pub final_message: ScalarProductMessage<G1, G2>,
    pub final_p1_g2: G2,
    pub final_p2_g2: G2,
    pub s_product: F,
}

/// Generate or load prover and verifier setups from disk
///
/// Creates or loads the transparent setup parameters for Dory PCS.
/// First attempts to load from disk; if not found, generates new setup and saves to disk.
///
/// Supports polynomials up to 2^max_log_n coefficients arranged as 2^ν × 2^σ matrices
/// where ν + σ = max_log_n and ν ≤ σ (flexible matrix layouts including square and non-square).
///
/// Setup file location (OS-dependent):
/// - Linux: `~/.cache/dory/dory_{max_log_n}.urs`
/// - macOS: `~/Library/Caches/dory/dory_{max_log_n}.urs`
/// - Windows: `{FOLDERID_LocalAppData}\dory\dory_{max_log_n}.urs`
///
/// # Parameters
/// - `max_log_n`: Maximum log₂ of polynomial size
///
/// # Returns
/// `(ProverSetup, VerifierSetup)` - Setup parameters for proving and verification
///
/// # Performance
/// To enable prepared point caching (~20-30% speedup), use the `cache` feature and call
/// `init_cache(&prover_setup.g1_vec, &prover_setup.g2_vec)` after setup.
///
/// # Panics
/// Load only the prover setup from disk, skipping verifier deserialization.
/// Falls back to generating both and saving if not on disk.
///
/// # Panics
/// Panics if the setup file exists but is corrupted.
pub fn setup_prover_only<E: PairingCurve>(max_log_n: usize) -> ProverSetup<E>
where
    ProverSetup<E>: DorySerialize + DoryDeserialize,
    VerifierSetup<E>: DorySerialize + DoryDeserialize,
{
    #[cfg(all(feature = "disk-persistence", not(target_arch = "wasm32")))]
    {
        // Fast path: try uncompressed cache first (no EC decompression needed)
        match setup::load_fast_prover::<E>(max_log_n) {
            Ok(prover) => {
                tracing::debug!("Loaded prover from fast cache");
                return prover;
            }
            Err(_) => {
                tracing::debug!("Fast cache not available, trying compressed .urs");
            }
        }

        // Slow path: load from compressed .urs file
        match setup::load_prover_setup::<E>(max_log_n) {
            Ok(prover) => {
                // Save fast cache for next time
                setup::save_fast_prover(&prover, max_log_n);
                return prover;
            }
            Err(DoryError::InvalidURS(msg)) if msg.contains("not found") => {
                tracing::debug!("Setup file not found, will generate new one");
            }
            Err(e) => {
                panic!("Failed to load setup from disk: {e}");
            }
        }

        let prover_setup = ProverSetup::new(max_log_n);
        let verifier_setup = prover_setup.to_verifier_setup();
        setup::save_setup(&prover_setup, &verifier_setup, max_log_n);
        setup::save_fast_prover(&prover_setup, max_log_n);
        prover_setup
    }

    #[cfg(any(not(feature = "disk-persistence"), target_arch = "wasm32"))]
    {
        ProverSetup::new(max_log_n)
    }
}

/// Load prover and verifier setups from disk, or generate and cache if not found.
///
/// # Panics
/// Panics if the setup file exists on disk but is corrupted or cannot be deserialized.
pub fn setup<E: PairingCurve>(max_log_n: usize) -> (ProverSetup<E>, VerifierSetup<E>)
where
    ProverSetup<E>: DorySerialize + DoryDeserialize,
    VerifierSetup<E>: DorySerialize + DoryDeserialize,
{
    #[cfg(all(feature = "disk-persistence", not(target_arch = "wasm32")))]
    {
        match setup::load_setup::<E>(max_log_n) {
            Ok(saved) => return saved,
            Err(DoryError::InvalidURS(msg)) if msg.contains("not found") => {
                // File doesn't exist, we'll generate new setup
                tracing::debug!("Setup file not found, will generate new one");
            }
            Err(e) => {
                // File exists but is corrupted - unrecoverable
                panic!("Failed to load setup from disk: {e}");
            }
        }

        // Setup not found on disk - generate new setup
        tracing::info!(
            "Setup not found on disk, generating new setup for max_log_n={}",
            max_log_n
        );
        let prover_setup = ProverSetup::new(max_log_n);
        let verifier_setup = prover_setup.to_verifier_setup();

        // Save to disk
        setup::save_setup(&prover_setup, &verifier_setup, max_log_n);

        (prover_setup, verifier_setup)
    }

    #[cfg(any(not(feature = "disk-persistence"), target_arch = "wasm32"))]
    {
        tracing::info!("Generating new setup for max_log_n={}", max_log_n);

        let prover_setup = ProverSetup::new(max_log_n);
        let verifier_setup = prover_setup.to_verifier_setup();

        (prover_setup, verifier_setup)
    }
}

/// Force generate new prover and verifier setups and save to disk
///
/// Always generates fresh setup parameters, ignoring any saved values on disk.
/// Saves the newly generated setup to disk, overwriting any existing setup file
/// for the given max_log_n.
///
/// Use this when you want to explicitly regenerate the setup (e.g., for testing
/// or when you suspect the saved setup file is corrupted).
///
/// # Parameters
/// - `max_log_n`: Maximum log₂ of polynomial size
///
/// # Returns
/// `(ProverSetup, VerifierSetup)` - Newly generated setup parameters
///
/// # Availability
/// This function is only available when the `disk-persistence` feature is enabled.
#[cfg(all(feature = "disk-persistence", not(target_arch = "wasm32")))]
pub fn generate_urs<E: PairingCurve>(max_log_n: usize) -> (ProverSetup<E>, VerifierSetup<E>)
where
    ProverSetup<E>: DorySerialize + DoryDeserialize,
    VerifierSetup<E>: DorySerialize + DoryDeserialize,
{
    tracing::info!("Force-generating new setup for max_log_n={}", max_log_n);

    let prover_setup = ProverSetup::new(max_log_n);
    let verifier_setup = prover_setup.to_verifier_setup();

    // Overwrites existing
    setup::save_setup(&prover_setup, &verifier_setup, max_log_n);

    (prover_setup, verifier_setup)
}

/// Evaluate a polynomial at a point and create proof
///
/// Creates an evaluation proof for a polynomial at a given point using precomputed
/// tier-1 commitments (row commitments).
///
/// # Workflow
/// 1. Call `polynomial.commit(nu, sigma, setup)` to get `(tier_2, row_commitments, commit_blind)`
/// 2. Call this function with the `row_commitments` and `commit_blind` to create the proof
/// 3. Use `tier_2` for verification via the `verify()` function
///
/// # Parameters
/// - `polynomial`: Polynomial implementing MultilinearLagrange trait
/// - `point`: Evaluation point (length must equal nu + sigma)
/// - `row_commitments`: Tier-1 commitments (row commitments in G1) from `polynomial.commit()`
/// - `commit_blind`: GT-level blinding scalar from `polynomial.commit()` (zero in Transparent mode)
/// - `nu`: Log₂ of number of rows (constraint: nu ≤ sigma for non-square matrices)
/// - `sigma`: Log₂ of number of columns
/// - `setup`: Prover setup
/// - `transcript`: Fiat-Shamir transcript
///
/// # Returns
/// The evaluation proof containing VMV message, reduce messages, and final message
///
/// # Homomorphic Properties
/// Proofs can be created for homomorphically combined polynomials. If you have
/// commitments Com(P₁), Com(P₂), ..., Com(Pₙ) and want to prove evaluation of
/// r₁·P₁ + r₂·P₂ + ... + rₙ·Pₙ, you can:
/// 1. Combine tier-2 commitments: r₁·Com(P₁) + r₂·Com(P₂) + ... + rₙ·Com(Pₙ)
/// 2. Combine tier-1 commitments element-wise
/// 3. Generate proof using this function with the combined polynomial
///
/// See `examples/homomorphic.rs` for a complete demonstration.
///
/// # Errors
/// Returns `DoryError` if:
/// - Point dimension doesn't match nu + sigma
/// - Polynomial size doesn't match 2^(nu + sigma)
/// - Number of row commitments doesn't match 2^nu
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "prove")]
pub fn prove<F, E, M1, M2, P, T, Mo>(
    polynomial: &P,
    point: &[F],
    row_commitments: Vec<E::G1>,
    commit_blind: F,
    nu: usize,
    sigma: usize,
    setup: &ProverSetup<E>,
    transcript: &mut T,
) -> Result<(DoryProof<E::G1, E::G2, E::GT>, Option<F>), DoryError>
where
    F: Field,
    E: PairingCurve,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    P: MultilinearLagrange<F>,
    T: primitives::transcript::Transcript<Curve = E>,
    Mo: Mode,
{
    evaluation_proof::create_evaluation_proof::<F, E, M1, M2, T, P, Mo>(
        polynomial,
        point,
        Some(row_commitments),
        commit_blind,
        nu,
        sigma,
        setup,
        transcript,
    )
}

/// Verify an evaluation proof
///
/// Verifies that a committed polynomial evaluates to the claimed value at the given point.
/// The matrix dimensions (nu, sigma) are extracted from the proof.
///
/// Works with both square and non-square matrix layouts (nu ≤ sigma), and can verify
/// proofs for homomorphically combined polynomials.
///
/// # Parameters
/// - `commitment`: Polynomial commitment (in GT) - can be a combined commitment for homomorphic proofs
/// - `evaluation`: Claimed evaluation result
/// - `point`: Evaluation point (length must equal proof.nu + proof.sigma)
/// - `proof`: Evaluation proof to verify (contains nu and sigma)
/// - `setup`: Verifier setup
/// - `transcript`: Fiat-Shamir transcript
///
/// # Returns
/// `Ok(())` if proof is valid, `Err(DoryError)` otherwise
///
/// # Errors
/// Returns `DoryError::InvalidProof` if the proof is invalid, or other variants
/// if the input parameters are incorrect (e.g., point dimension mismatch).
#[tracing::instrument(skip_all, name = "verify")]
pub fn verify<F, E, M1, M2, T>(
    commitment: E::GT,
    evaluation: F,
    point: &[F],
    proof: &DoryProof<E::G1, E::G2, E::GT>,
    setup: VerifierSetup<E>,
    transcript: &mut T,
) -> Result<(), DoryError>
where
    F: Field,
    E: PairingCurve + Clone,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    T: primitives::transcript::Transcript<Curve = E>,
{
    evaluation_proof::verify_evaluation_proof::<F, E, M1, M2, T>(
        commitment, evaluation, point, proof, setup, transcript,
    )
}

/// Extract Dory witness data by replaying the Fiat-Shamir transcript.
///
/// This replays the proof messages on the given transcript to derive the
/// alpha/beta/gamma/d challenges and compute the E2 accumulator and G2
/// composite witnesses needed by the gnark Groth16 circuit.
///
/// The transcript must be in the same state as when `verify()` would be called
/// (i.e., all prior protocol messages already absorbed).
///
/// # Errors
/// Returns `DoryError::InvalidPointDimension` if point length doesn't match proof dimensions.
///
/// # Panics
/// Panics if challenge scalar inversion fails (zero challenge — cryptographically negligible).
#[allow(clippy::type_complexity)]
pub fn extract_witness_data<F, E, M1, M2, T>(
    commitment: E::GT,
    evaluation: F,
    point: &[F],
    proof: &DoryProof<E::G1, E::G2, E::GT>,
    setup: &VerifierSetup<E>,
    transcript: &mut T,
) -> Result<DoryWitnessData<F, E::G1, E::G2, E::GT>, DoryError>
where
    F: Field,
    E: PairingCurve + Clone,
    E::G1: Group<Scalar = F>,
    E::G2: Group<Scalar = F>,
    E::GT: Group<Scalar = F>,
    M1: DoryRoutines<E::G1>,
    M2: DoryRoutines<E::G2>,
    T: primitives::transcript::Transcript<Curve = E>,
{
    let nu = proof.nu;
    let sigma = proof.sigma;
    let num_rounds = sigma;

    if point.len() != nu + sigma {
        return Err(DoryError::InvalidPointDimension {
            expected: nu + sigma,
            actual: point.len(),
        });
    }

    // Replay VMV message
    transcript.append_serde(b"vmv_c", &proof.vmv_message.c);
    transcript.append_serde(b"vmv_d2", &proof.vmv_message.d2);
    transcript.append_serde(b"vmv_e1", &proof.vmv_message.e1);

    // s1/s2 coords matching verifier convention
    let s1_coords: Vec<F> = point[..sigma].to_vec();
    let mut s2_coords: Vec<F> = vec![F::zero(); sigma];
    s2_coords[..nu].copy_from_slice(&point[sigma..sigma + nu]);

    let mut alphas = Vec::with_capacity(num_rounds);
    let mut betas = Vec::with_capacity(num_rounds);

    // E2 accumulator (transparent mode)
    let mut e2_acc: E::G2 = setup.g2_0.scale(&evaluation);
    let mut s1_acc = F::one();
    let mut s2_acc = F::one();

    for round_idx in 0..num_rounds {
        let first_msg = &proof.first_messages[round_idx];
        let second_msg = &proof.second_messages[round_idx];

        transcript.append_serde(b"d1_left", &first_msg.d1_left);
        transcript.append_serde(b"d1_right", &first_msg.d1_right);
        transcript.append_serde(b"d2_left", &first_msg.d2_left);
        transcript.append_serde(b"d2_right", &first_msg.d2_right);
        transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
        transcript.append_serde(b"e2_beta", &first_msg.e2_beta);
        let beta: F = transcript.challenge_scalar(b"beta");
        betas.push(beta);

        transcript.append_serde(b"c_plus", &second_msg.c_plus);
        transcript.append_serde(b"c_minus", &second_msg.c_minus);
        transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
        transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
        transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
        transcript.append_serde(b"e2_minus", &second_msg.e2_minus);
        let alpha: F = transcript.challenge_scalar(b"alpha");
        alphas.push(alpha);

        // Update E2 accumulator and scalar accumulators
        let alpha_inv = alpha.inv().unwrap();
        let beta_inv = beta.inv().unwrap();

        e2_acc = e2_acc
            + second_msg.e2_plus.scale(&alpha)
            + second_msg.e2_minus.scale(&alpha_inv)
            + first_msg.e2_beta.scale(&beta_inv);

        let coord_idx = num_rounds - 1 - round_idx;
        let y_t = &s1_coords[coord_idx];
        let x_t = &s2_coords[coord_idx];
        let one = F::one();
        s1_acc = s1_acc * (alpha * (one - *y_t) + *y_t);
        s2_acc = s2_acc * (alpha_inv * (one - *x_t) + *x_t);
    }

    let gamma: F = transcript.challenge_scalar(b"gamma");
    transcript.append_serde(b"final_e1", &proof.final_message.e1);
    transcript.append_serde(b"final_e2", &proof.final_message.e2);
    let d: F = transcript.challenge_scalar(b"d");

    // Compute G2 composite witnesses for the gnark pairing equation
    let d_inv = d.inv().unwrap();
    let neg_gamma = F::zero() - gamma;

    let final_p1_g2 = proof.final_message.e2 + setup.g2_0.scale(&d_inv);
    let final_p2_g2 = (e2_acc + setup.g2_0.scale(&(d_inv * s1_acc))).scale(&neg_gamma);

    let s_product = s1_acc * s2_acc;

    Ok(DoryWitnessData {
        alphas,
        betas,
        gamma,
        d,
        s1_coords,
        s2_coords,
        num_rounds,
        commitment,
        evaluation,
        vmv_message: proof.vmv_message.clone(),
        first_messages: proof.first_messages.clone(),
        second_messages: proof.second_messages.clone(),
        final_message: proof.final_message.clone(),
        final_p1_g2,
        final_p2_g2,
        s_product,
    })
}
