# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0.0] - 2025-12-29

### Added

- Initial release of Ra System Haskell bindings
- `Ra.Constants`: Typed constants with newtype wrappers (Ankh, Hunab, OmegaRatio, etc.)
- `Ra.Repitans`: Repitan type with smart constructor validating range [1, 27]
- `Ra.Rac`: RacLevel ADT with Enum/Bounded instances and value functions
- `Ra.Omega`: OmegaFormat enum with conversion functions
- `Ra.Spherical`: RaCoordinate type with θ/φ/h/r dimensional mapping
- `Ra.Gates`: Access gating logic per spec Section 4
- QuickCheck properties for all 17 invariants from ra_integration_spec.md
