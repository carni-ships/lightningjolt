//! Backend implementations for Dory primitives
//!
//! This module provides concrete implementations of the abstract traits
//! defined in the primitives module. Currently supports:
//! - arkworks: BN254 pairing curve implementation using Arkworks
//! - hybrid: GPU-accelerated MSM using ICICLE (requires `icicle` feature)

#[cfg(feature = "arkworks")]
pub mod arkworks;

#[cfg(feature = "arkworks")]
pub use arkworks::*;

/// Hybrid GPU-accelerated backend using ICICLE.
///
/// This module provides `HybridG1Routines` and `HybridG2Routines` that
/// automatically dispatch to ICICLE GPU for MSMs with size >= GPU_MSM_THRESHOLD (512).
///
/// Enable with the `icicle` feature:
/// ```toml
/// dory = { features = ["backends", "icicle"] }
/// ```
///
/// For Metal GPU:
/// ```toml
/// dory = { features = ["backends", "icicle-metal"] }
/// ```
///
/// For CUDA GPU:
/// ```toml
/// dory = { features = ["backends", "icicle-cuda"] }
/// ```
#[cfg(feature = "icicle")]
pub mod hybrid;

#[cfg(feature = "icicle")]
pub use hybrid::{GPU_MSM_THRESHOLD, HybridG1Routines, HybridG2Routines};

/// NTT backend for lattice-based commitments (requires `lattice` feature)
#[cfg(feature = "lattice")]
pub mod ntt;

#[cfg(feature = "lattice")]
pub use ntt::{Fp128Element, Fp128Field, KyberCurve};
