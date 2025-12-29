//! Ra System mathematical constants - Rust bindings
//!
//! This crate provides type-safe access to the Ra System constants from
//! "The Rods of Amon Ra" by Wesley H. Bateman.
//!
//! # Overview
//!
//! The Ra System defines:
//! - Fundamental constants (Ankh, Pi variants, Phi variants)
//! - 27 Repitans (semantic sectors)
//! - 6 RAC levels (access sensitivity)
//! - 5 Omega formats (coherence depth)
//!
//! # Example
//!
//! ```
//! use ra_core::{ANKH, Repitan, RacLevel, OmegaFormat};
//! use ra_core::gates::access_level;
//!
//! // Check access at 80% coherence for RAC1
//! let result = access_level(0.8, RacLevel::RAC1);
//! assert!(result.is_full_access());
//!
//! // Create a validated Repitan
//! let r = Repitan::new(9).unwrap();
//! assert!((r.value() - 9.0/27.0).abs() < 1e-10);
//! ```
//!
//! # Invariants
//!
//! All 17 invariants from ra_integration_spec.md are enforced:
//! - Constant invariants (I1-I6)
//! - Ordering invariants (O1-O4)
//! - Conversion invariants (C1-C3)
//! - Range invariants (R1-R4)

pub mod constants;
pub mod repitans;
pub mod rac;
pub mod omega;
pub mod spherical;
pub mod gates;

// Re-export commonly used items
pub use constants::*;
pub use repitans::Repitan;
pub use rac::{RacLevel, RacValue};
pub use omega::OmegaFormat;
pub use spherical::RaCoordinate;
pub use gates::AccessResult;
