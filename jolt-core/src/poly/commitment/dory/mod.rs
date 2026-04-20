//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.

mod commitment_scheme;
mod dory_globals;
#[cfg(feature = "icicle")]
pub mod icicle_msm;
mod jolt_dory_routines;
#[cfg(feature = "metal-pairing")]
pub(crate) mod metal_pairing;
#[cfg(feature = "zkmetal")]
pub mod zkmetal_cpu;
mod wrappers;

#[cfg(test)]
mod tests;

#[cfg(feature = "zk")]
pub use commitment_scheme::bind_opening_inputs_zk;
pub use commitment_scheme::{bind_opening_inputs, extract_dory_witness, DoryCommitmentScheme};
pub use dory_globals::{DoryContext, DoryGlobals, DoryLayout};
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, BN254,
};
