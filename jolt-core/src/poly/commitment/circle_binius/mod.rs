//! CircleBinius backend for Jolt
//!
//! This module provides a post-quantum, no-trusted-setup commitment scheme
//! based on Circle STARKs and Binius Brakedown PCS.
//!
//! # Architecture
//!
//! ```
//! RISC-V bytecode
//!       ↓
//! Jolt Arithmetization (unchanged)
//!       ↓
//! Multi-Tower Trace Layout (binary tower fields)
//!       ↓
//! Circle FFT Commitment
//!       ↓
//! Binius Brakedown PCS (hash-based)
//!       ↓
//! Binary FRI + GKR Batching
//!       ↓
//! Convergent Binary Verifier Circuit
//! ```
//!
//! # Components
//!
//! - `fields`: Binary tower field implementations (F_{2^8}, F_{2^{32}}, etc.)
//! - `tower_trace`: Adapter converting Jolt trace columns to tower format
//! - `circle_fft`: Circle group FFT implementation
//! - `brakedown`: Brakedown PCS wrapper (uses binius crate)
//! - `fri`: Circle FRI implementation
//! - `verifier`: Binary verifier circuit for recursive proofs

pub mod fields;
pub mod tower_trace;
pub mod commitment_scheme;

#[cfg(feature = "circle-binius")]
pub mod circle_fft;

#[cfg(feature = "circle-binius")]
pub mod brakedown;

#[cfg(feature = "circle-binius")]
pub mod fri;

#[cfg(feature = "circle-binius")]
pub mod verifier;

// Re-export for convenience (types are crate-public)
pub(crate) use fields::{BinaryField8b, BinaryField32b, BinaryField64b, BinaryField128b};
pub use tower_trace::TowerTrace;