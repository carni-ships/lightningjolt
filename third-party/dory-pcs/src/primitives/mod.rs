//! # Primitives
//! This submodule defines the basic EC and FS related tools that Dory is built upon
pub mod arithmetic;
pub mod poly;
pub mod serialization;
pub mod transcript;
pub use serialization::*;

// Lattice-based primitives
#[cfg(feature = "lattice")]
pub mod lattice_trait;
#[cfg(feature = "lattice")]
pub mod lattice_curve;

// Re-export lattice types
#[cfg(feature = "lattice")]
pub use lattice_trait::{LatticeCurve, LatticeElement, LatticeField};
#[cfg(feature = "lattice")]
pub use lattice_curve::KyberCurve;
