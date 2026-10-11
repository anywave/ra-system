# Changelog

All notable changes to the Python, Rust and TypeScript bindings and the shared data file are recorded here. The
Haskell bindings keep their own log in `haskell/CHANGELOG.md`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Values in this package are Wesley H. Bateman's, from *The Rods of Amon Ra* (1992-1997), as printed and as verified
page by page by the lattice's TR node in Aug-Sep 2026. They are the author's own units and ratios, not physics
claims. Each correction below names the lattice post (TR-###) and the archive page or printed folio that settles it.

## [0.1.2] - 2026-10-10

Version 0.1.1 was reserved for the 2026-10 public cleanup (PR #1), which merged without a version bump. Neither is
on PyPI yet; PyPI has 0.1.0.

### Fixed

- `RED_PI` is Bateman's Red Pi, `84823/27000 = 3.141592592...`, not π (`3.14159265`). Red Pi is π rounded at a
  base-27 scale and differs from π at the 8th decimal (TR-291, TR-349).
- `GREEN_PI` is `(20/9)√2 = 3.142696805` (was `3.14754099`, no source). Named GREEN PI at volume_1 p.0129,
  procedure 7-3 (TR-300, TR-301).
- `BLUE_PI` is `Green Pi × r = 3.143801406` (was `3.15349386`, no source); volume_1 p.0122, printed 114 (TR-308).
- `HUNAB` is Bateman's Red Hunab, `84823/80000 = 1.0602875` (was `1.05946`, the 12th root of 2, which is not his
  value). Volume_1 p.0150 (TR-291); labelled HUNAB 247 times in the corpus (FP-1399).
- `H_BAR` is `135/128 = 1.0546875` (was `1.05346545`), the Omega Major window of the Hunab row on vol2a
  g00194_src01_p0010, "OMEGA LEVELS: KEY NUMBERS" (TR-304; confirmed on supplement p.0002, TR-312).
- Invariant I3 is now `H-Bar = Green Hunab / Ω` (exact). The old `Hunab / Ω` only held at 1e-4 because both
  constants were wrong; with Bateman's Hunab it is 3.6e-4 off (TR-304).
- `RED_PHI`, `GREEN_PHI`, `BLUE_PHI` follow Bateman's own erratum on supplement g00857_src03_p0002: Omega Major 1.62,
  Red 1.6286016, Green 1.629174025, Blue 1.629746650. The erratum withdraws the Vol 2/3 Red Phi 1.619430799
  (FP-1399, TR-312, TR-316). The old 1.614 and 1.626 had no source. **Behaviour note:** `GREEN_PHI` moves from 1.62
  to 1.629174025; see "Open" below.
- Invariant I1 is now `Ankh = Red Pi × 1.62`, where 1.62 is the Ankh wheel diameter (volume_1 p.0168, TR-298) and
  the Omega Major Phi (`OMEGA_MAJOR_PHI`). `ANKH = 3 × 84823/50000` (TR-329).
- The gate `COHERENCE_FLOOR` uses `OMEGA_MAJOR_PHI / ANKH`, so its value (0.3183098..., = 1/Red Pi) is unchanged.
- `OMEGA` is the exact `32√2/45` (TR-304); printed 1.005662978 (TR-302).
- `FINE_STRUCTURE` is exactly `1/729` and documented as Bateman's f.s.Δ unit, not the physical fine-structure
  constant (0.0072973526) (TR-323, TR-331).
- RAC values are computed as Red Pi × the fractions printed on vol2b g00585_src09_p0038, printed 418 (TR-298); they
  agree with the old printed decimals to their last digit. Derivation comments fixed (`2π/10`, `π/6` and
  "RAC lattice terminus" replaced by Red Pi / 5, Red Pi / 6, Red Pi × 0.09√2).
- Pyramid-division comments: base = (729/10) × Red Pi, so base / RAC 4 = 72.9 × 6 = 437.4, not "27 × φ" (TR-331).
- Python, TypeScript: the Omega-format conversion matrix is built from the exact ladder
  Omega Major --(Ω/r)-- Red --(r)-- Green --(r)-- Blue --(Ω/r)-- Omega Minor (TR-302, TR-308, TR-312). It reproduces
  every printed factor to print width; roundtrip tolerance tightened from 1e-6 to 1e-9.
- Docs: the format "hierarchy" was described as `Red > Omega Major > Green > Omega Minor > Blue`. That is the printed
  column layout; in magnitude the order is Omega Major < Red < Green < Blue < Omega Minor (TR-302).
- `data/ra_constants_v2.json`:
  - Fine Structure (Ra) `0.0001371742` -> `1/729`; the printed value is kept as `printed_value` (lost decimal,
    BAT-002; TR-281, TR-323).
  - Rydberg `ra_value` `1.097369` -> `800/729 = 1.09739369` (extraction dropped two digits; TR-320, TR-322).
  - `lyman_m2_connection` `1.062875` -> `1.0602875` (dropped '0', BAT-028; TR-288).
  - `green_to_omega_minor` `1.00562978` -> `1.005662978` (dropped '6').
  - "Phi (full precision)" `1.619430799` -> `1.6286016` per the author's erratum; withdrawn value kept as
    `printed_value` (FP-1399, TR-312).
  - H-Bar derivation corrected to Green Hunab / Ω; RAC derivations corrected; erratum and conflict notes added.
  - Metadata no longer claims every value is "preserved exactly as stated"; corrections are marked.

### Added

- `COLOUR_RATIO` (r = 60000√2/84823 = 1.000351482; TR-291, TR-300; printed at vol2a g00215_src01_p0031, TR-349),
  `OMEGA_OVER_R` (84823/84375; TR-308, TR-312), `GREEN_HUNAB` ((3/4)√2; TR-320, FP-1430), `OMEGA_MAJOR_PHI` (1.62),
  `RA_RYDBERG` (800/729; TR-320, TR-322) and `BALMER_CONSTANT` (729/2; TR-288).
- Python: `convert_omega(..., degree=1)`. Ladder steps apply as a power of the number of lengths in play: r² for areas
  (volume_1 p.0117, printed 109: "Red from Blue by the ratio 1.000703088"; TR-294, TR-301), r³ for volumes (TR-309).
  TypeScript `convertOmega(..., degree = 1)` likewise.
- Python: `RAC_UNITS_OF_RED_PI`, all ten Rac units of TR-298 (RAC 1-7, MIR Rac, Block Rac 1-2).
- `data/ra_constants_v2.json`: Lyman M6 = `6561/7 = 937.2857142857` Å. Volume_1 p.0100 prints 93.72857142 (×10);
  the Supplement's 932.7857142 is a transposition and the page proves itself: 1093.5 / 93.72857142 = 35/3
  (TR-280 §7, TR-286).
- Tests pinning each TR-verified value (`python/tests/test_tr_extractions.py`, `python/tests/test_data_json.py`,
  `rust/tests/tr_extractions.rs`, `typescript/tests/tr_extractions.test.ts`), including a guard that the unsourced
  "51.851651° face angle" never enters the package. That number appears nowhere in the 863 pages; Bateman's
  `51.851 (51 23/27)` on vol2a g00343_src02_p0035 is a count of Red Rac 3 units, not an angle (TR-278, TR-290).

### Not changed, on purpose

- Spectral values are not normalised toward reference wavelengths. His tables sit a constant ~545 ppm from measured
  hydrogen lines because he used his own Rydberg (800/729, close to R∞) rather than R_H; the offset is authorial
  (TR-280 §7.3, TR-283, TR-349). Do not "fix" it.
- Haskell bindings: not edited in this release (no Haskell toolchain to test with). They already carry the correct
  Hunab and H-Bar; their Phi values predate the erratum.

### Open (needs a ruling)

- Green Phi: volume_1 p.0027 and the vol2a p.0010 master table call 1.62 the Green Phi; the supplement erratum puts
  1.62 in the Omega Major window and prints Green Phi 1.629174025. This release follows the erratum (TR-312: "the
  erratum that supersedes"). Revert `GREEN_PHI` to 1.62 if RADIX rules the other way; nothing else depends on it.
- Rydberg window: volume_1 p.0126 calls 800/729 the Green Rydberg (TR-320); the vol2a KEY NUMBERS sheet carries
  729/800 in its Omega Major column (TR-304). Recorded, not resolved.
