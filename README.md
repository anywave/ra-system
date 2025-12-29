# Ra System

Mathematical framework derived from "The Rods of Amon Ra" by Wesley H. Bateman (1992-1997).

Portable, language-agnostic specification with generated bindings for Python, TypeScript, Rust, and Haskell.

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

## License

Apache-2.0
