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
| Ankh | 5.08938 | Master harmonic (π_red × φ_green) |
| π_red | 3.141592592 | Red Trac Pi |
| φ_green | 1.62 | Green Phi (threshold harmonic) |
| Ω | 1.005662978 | Omega ratio (format conversion) |
| Hunab | 1.0602875 | Harmonic scalar unit |

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
| RAC₂ | 0.628318519 | 2π/10 |
| RAC₃ | 0.57255525 | φ × Hunab × ⅓ |
| RAC₄ | 0.523598765 | π/6 |
| RAC₅ | 0.4580442 | Ankh × 9/100 |
| RAC₆ | 0.3998594565 | RAC lattice |

## Omega Format Hierarchy

```
Red → Omega Major → Green → Omega Minor → Blue
         ↑                      ↑
      ÷ Ω                    × Ω
```

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
