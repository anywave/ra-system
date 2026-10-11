# Ra System

Mathematical framework derived from "The Rods of Amon Ra" by Wesley H. Bateman (1992-1997).

Portable, language-agnostic specification with generated bindings for Python, TypeScript, Rust, and Haskell.

## Status and scope

This is an exploratory modelling framework. The constants are taken from Bateman's numerical system and are used here as
design parameters; their harmonic names describe a method, not a measured physical effect. Modules under `haskell/`
and `haskell/clash/` that use words such as biofield, healing, orgone, scalar, radionics or propulsion are speculative
simulations. Nothing in this repository has been validated experimentally, and none of it is a medical device or a
medical, therapeutic or health claim.

## Architecture

```
ra-system/
├── schema/                 # Layer 0: Canonical Schema
│   └── ra_system_schema.json
├── spec/                   # Layer 0: Mathematical Specification
│   └── ra_integration_spec.md
├── data/                   # Source constants
│   └── ra_constants_v2.json
├── python/                 # Layer 1: Python bindings
├── typescript/             # Layer 1: TypeScript bindings
├── rust/                   # Layer 1: Rust bindings
└── haskell/                # Layer 1: Haskell bindings
```

## Layer 0: Canonical Schema

The source of truth for all language bindings.

- **`schema/ra_system_schema.json`** - JSON Schema validating constants structure
- **`spec/ra_integration_spec.md`** - Pure mathematical specification

## Core Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Ankh | 5.08938 | π_red × 1.62 = 3 × 84823/50000 (the Ankh wheel: circumference 5.08938, diameter 1.62) |
| π_red | 3.141592592 | Red Pi = 84823/27000 (π rounded at a base-27 scale; not π) |
| π_green | 3.142696805 | Green Pi = (20/9)√2 |
| φ (Omega Major) | 1.62 | Bateman's 1.62; his erratum places it in the Omega Major window |
| φ_green | 1.629174025 | Green Phi per Bateman's supplement erratum |
| Ω | 1.005662978 | Omega ratio = 32√2/45 (format conversion) |
| r | 1.000351482 | Colour ratio = 60000√2/84823 (Red → Green → Blue) |
| Hunab | 1.0602875 | Red Hunab = 84823/80000 |
| H-Bar | 1.0546875 | 135/128 = Green Hunab / Ω |

Values are Bateman's, as printed in *The Rods of Amon Ra* and verified page by page by the lattice's TR node
(2026-08/09); see [CHANGELOG.md](CHANGELOG.md). Names such as H-Bar, Rydberg and fine structure are his; the values
are not the physical constants.

## Dimensional Mapping

| Dimension | Ra Mapping | Range | Use |
|-----------|------------|-------|-----|
| θ (theta) | 27 Repitans | [1, 27] | Semantic sector |
| φ (phi) | 6 RACs | [1, 6] | Access sensitivity |
| h (harmonic) | 5 Omega formats | [0, 4] | Coherence depth |
| r (radius) | Ankh-normalized | [0, 1] | Emergence intensity |

## RAC Levels (Resonant Access Constants)

| Level | Value | Derivation |
|-------|-------|------------|
| RAC₁ | 0.6361725 | Ankh / 8 |
| RAC₂ | 0.628318519 | π_red / 5 |
| RAC₃ | 0.57255525 | π_red × 3⁶/4000 = 1.62 × Hunab / 3 |
| RAC₄ | 0.523598765 | π_red / 6 |
| RAC₅ | 0.4580442 | Ankh × 9/100 = π_red × 3⁶/5000 |
| RAC₆ | 0.3998594565 | π_red × 0.09√2 |

In Bateman's corpus every Rac is an arc of Red Pi, a turn of a measuring wheel; the access-level reading is this
framework's modelling layer.

## Omega Format Hierarchy

In magnitude (the enum index follows the printed column layout, not this order):

```
Omega Major --(Ω/r)-- Red --(r)-- Green --(r)-- Blue --(Ω/r)-- Omega Minor
     = Green ÷ Ω                                         = Green × Ω
```

Each step applies as a power of the number of lengths in play: once for a length, squared for an area, cubed for a
volume.

## References

Third-party papers used as background are linked rather than redistributed:

- Chen, X. and Sun, L. "Bayesian Temporal Factorization for Multidimensional Time Series Prediction." arXiv:1910.06366. https://arxiv.org/abs/1910.06366
- Amirzadeh, R., Thiruvady, D., Nazari, A. and Ee, M. S. "Dynamic Bayesian Networks for Predicting Cryptocurrency Price Directions: Uncovering Causal Relationships." arXiv:2306.08157. https://arxiv.org/abs/2306.08157
- Wyant, J. C. "Zernike Polynomials." University of Arizona, College of Optical Sciences (2003).
- Anonymous. *The Book of Aquarius* (2011), bookofaquarius.forgottenbooks.org.
- Bateman, W. H. *The Rods of Amon Ra* (1992-1997).

## Citation

See [CITATION.cff](CITATION.cff).

## Security

See [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
