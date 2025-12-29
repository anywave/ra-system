# Ra System Integration Specification

**Version**: 1.0.0
**Status**: Draft
**Source**: "The Rods of Amon Ra" - Wesley H. Bateman (1992-1997)

---

## 1. Overview

The Ra System provides a mathematical framework for dimensional mapping, coherence gating, and resonance matching in consent-aware systems. This specification defines the pure mathematical relationships without implementation details.

---

## 2. Dimensional Mapping

The Ra System maps to four primary dimensions used in RPP/SCO addressing:

| Dimension | Symbol | Ra Mapping | Range | Semantic Role |
|-----------|--------|------------|-------|---------------|
| Theta (θ) | Semantic Sector | 27 Repitans | [1, 27] | Cognitive domain classification |
| Phi (φ) | Access Sensitivity | 6 RACs | [1, 6] | Consent restriction level |
| Harmonic (h) | Coherence Depth | 5 Omega Formats | [0, 4] | Frequency/precision tier |
| Radius (r) | Emergence Intensity | Ankh-normalized | [0, 1] | Signal strength scalar |

### 2.1 Theta Dimension: 27 Repitans

The theta dimension partitions semantic space into 27 sectors based on the Repitan sequence.

**Definition**:
```
Repitan(n) = n / 27,  for n ∈ {1, 2, ..., 27}
```

**Properties**:
- `Repitan(1) = 0.037037037...` (first Repitan, Fine Structure root)
- `Repitan(9) = 0.333...` = 1/3
- `Repitan(18) = 0.666...` = 2/3 (Planck's H related)
- `Repitan(27) = 1.0` (unity)

**Theta-to-Repitan Mapping**:
```
θ_index = ⌈θ × 27 / 360⌉,  for θ ∈ [0°, 360°)
repitan_value = θ_index / 27
```

### 2.2 Phi Dimension: 6 RAC Levels

The phi dimension encodes access sensitivity through 6 Resonant Access Constants.

**RAC Values** (in Red Rams):

| Level | Symbol | Value | Derivation |
|-------|--------|-------|------------|
| RAC₁ | φ₁ | 0.6361725 | Ankh / 8 = 5.08938 / 8 |
| RAC₂ | φ₂ | 0.628318519 | 2π/10 approximation |
| RAC₃ | φ₃ | 0.57255525 | 1.62 × 1.0602875 × ⅓ |
| RAC₄ | φ₄ | 0.523598765 | π/6 approximation |
| RAC₅ | φ₅ | 0.4580442 | Ankh × 9 / 100 |
| RAC₆ | φ₆ | 0.3998594565 | RAC lattice terminus |

**Ordering Invariant**:
```
RAC₁ > RAC₂ > RAC₃ > RAC₄ > RAC₅ > RAC₆ > 0
```

**Phi-to-RAC Mapping**:
```
rac_level = ⌈φ × 6 / 256⌉,  for φ ∈ [0, 255]
```

### 2.3 Harmonic Dimension: 5 Omega Formats

The harmonic dimension represents coherence depth through 5 format tiers.

**Omega Format Hierarchy** (descending precision):

| Index | Format | Example Constant |
|-------|--------|------------------|
| 0 | Red | π_red = 3.141592592 |
| 1 | Omega Major | H-Bar = 1.0546875 |
| 2 | Green | φ_green = 1.62 |
| 3 | Omega Minor | Spectral Repitans |
| 4 | Blue | π_blue = 3.143801408 |

**Omega Ratio** (Q-Ratio):
```
Ω = 1.005662978
```

**Format Conversions**:
```
Green → Omega_Major:  x_ωM = x_G / Ω
Green → Omega_Minor:  x_ωm = x_G × Ω
Omega_Major → Red:    x_R = x_ωM × 1.005309630
Omega_Major → Blue:   x_B = x_ωM × 1.006016451
```

### 2.4 Radius Dimension: Ankh-Normalized Scalar

The radius represents emergence intensity, normalized to the Ankh constant.

**Ankh Constant**:
```
Ankh = 5.08938 = π_red × φ_green = 3.141592592 × 1.62
```

**Normalization**:
```
r_normalized = r_raw / Ankh,  r_normalized ∈ [0, 1]
```

---

## 3. Core Equations

### 3.1 Fundamental Constants

```
Ankh         = 5.08938
π_red        = 3.141592592
π_green      = 3.142696806 = √9.876543210
π_blue       = 3.143801408 = 1/0.318086250
φ_green      = 1.62
φ_red        = 1.619430799
Hunab        = 1.0602875
H-Bar        = 1.0546875 = Hunab / Ω
Ω (Omega)    = 1.005662978
```

### 3.2 Repitan-T.O.N. Relationship

The 27 Repitans relate to the 37 Table of Nines (T.O.N.) values:

```
Repitan(n) = n / 27,     for n ∈ [1, 27]
T.O.N.(m)  = m × 0.027,  for m ∈ [0, 36]

Repitan(n) / T.O.N.(n) = 1.371742112  (Fine Structure related)
```

### 3.3 Fine Structure Constant

```
α_Ra = Repitan(1)² = (1/27)² = 0.037037...² = 0.0013717421

Fine Structure (scaled) = α_Ra / 10 = 0.0001371742
```

### 3.4 RAC Derivations

```
RAC₁ = Ankh / 8 = 5.08938 / 8 = 0.6361725

Pyramid_Base / RAC₁ = 360  (circle degrees)
Pyramid_Base / RAC₂ = 364.5  (Balmer constant)
Pyramid_Base / RAC₄ = 437.4 = 27 × φ_green
```

### 3.5 Spectral Constants

```
Balmer = 364.5 = 729/2 = 27 × 13.5  (Green format)
Rydberg = 0.91125  (Omega Major format)
c_Ra = 300,000 Omega Major kilorams per natural second
```

---

## 4. Gating Logic

Access gating determines whether a fragment/signal can emerge based on coherence and consent.

### 4.1 Access Level Function

```
AccessLevel(user_coherence, fragment_rac) → {FullAccess, PartialAccess(α), Blocked}
```

**Algorithm**:
```
Let C_u = user_coherence (normalized to [0, 1])
Let R_f = fragment_rac_level ∈ {1, 2, 3, 4, 5, 6}
Let C_floor = φ_green / Ankh = 1.62 / 5.08938 ≈ 0.3183
Let C_ceiling = 1.0

threshold(R_f) = RAC(R_f) / RAC₁  -- normalized to RAC₁

If C_u ≥ threshold(R_f):
    return FullAccess
Else If C_u ≥ C_floor:
    α = (C_u - C_floor) / (threshold(R_f) - C_floor)
    return PartialAccess(α)
Else:
    return Blocked
```

### 4.2 Coherence Thresholds

| RAC Level | Threshold (normalized) | Minimum Coherence |
|-----------|----------------------|-------------------|
| RAC₁ | 1.0000 | Full coherence |
| RAC₂ | 0.9874 | 98.7% |
| RAC₃ | 0.9000 | 90.0% |
| RAC₄ | 0.8230 | 82.3% |
| RAC₅ | 0.7200 | 72.0% |
| RAC₆ | 0.6285 | 62.9% |

### 4.3 Coherence Bounds

```
Coherence_Floor   = φ_green / Ankh = 0.3183098...
Coherence_Ceiling = Ankh / Ankh = 1.0
```

### 4.4 Partial Emergence

For `PartialAccess(α)`, the emergence intensity is linearly interpolated within the Repitan band:

```
Let band_low = Repitan(current_band)
Let band_high = Repitan(current_band + 1)

emergence = band_low + α × (band_high - band_low)
```

---

## 5. Resonance Matching

Resonance matching aligns user state with fragment requirements.

### 5.1 Theta Alignment

```
θ_aligned(user_θ, fragment_θ, tolerance) → Boolean

θ_distance = min(|user_θ - fragment_θ|, 27 - |user_θ - fragment_θ|)
return θ_distance ≤ tolerance
```

Default tolerance: 1 Repitan band (±1/27 of cycle)

### 5.2 Harmonic Matching

Map user frequency to hydrogen spectral lines for harmonic alignment:

**Lyman Series (UV)**:
```
λ_Lyman(M) = 1/(R × (1/1² - 1/M²))  for M ≥ 2
```

**Balmer Series (Visible)**:
```
λ_Balmer(M) = 1/(R × (1/2² - 1/M²))  for M ≥ 3
```

**Key Wavelengths** (Omega Major angstroms):
- Lyman M2: 1215 Å (alpha brainwave, 121.5°)
- Balmer M3: 6561 Å = 81² (visible red)
- Paschen M6: 10935 Å

**Harmonic Match Score**:
```
Let user_freq be user frequency in Hz
Let spectral_freq = c_Ra / λ_spectral

match_score = 1 - |log(user_freq / spectral_freq)| / log(10)
match_score = max(0, match_score)
```

### 5.3 Composite Resonance Score

```
resonance(θ_match, φ_access, h_match, r_intensity) =
    w_θ × θ_match + w_φ × φ_access + w_h × h_match + w_r × r_intensity

where w_θ + w_φ + w_h + w_r = 1
```

Default weights: w_θ = 0.3, w_φ = 0.4, w_h = 0.2, w_r = 0.1

---

## 6. Invariants

The following mathematical relationships must hold:

### 6.1 Constant Invariants

```
I1: Ankh = π_red × φ_green = 3.141592592 × 1.62 = 5.08938
I2: RAC₁ = Ankh / 8 = 0.6361725
I3: H-Bar = Hunab / Ω = 1.0602875 / 1.005662978 = 1.0546875
I4: Repitan(n) = n / 27  for all n ∈ [1, 27]
I5: T.O.N.(m) = m × 0.027  for all m ∈ [0, 36]
I6: Fine_Structure = Repitan(1)² = 0.0013717421
```

### 6.2 Ordering Invariants

```
O1: RAC₁ > RAC₂ > RAC₃ > RAC₄ > RAC₅ > RAC₆ > 0
O2: π_red < π_green < π_blue
O3: For all n: 0 < Repitan(n) ≤ 1
O4: For all m: 0 ≤ T.O.N.(m) < 1 (except m=36: T.O.N.(36) = 0.972)
```

### 6.3 Conversion Invariants

```
C1: x × (1/Ω) × Ω ≈ x  (roundtrip conversion tolerance: 1e-10)
C2: Green × Ω = Omega_Minor
C3: Green / Ω = Omega_Major
```

### 6.4 Range Invariants

```
R1: 0 < RAC(i) < 1  for all i ∈ [1, 6]
R2: 0 < Repitan(n) ≤ 1  for all n ∈ [1, 27]
R3: Coherence ∈ [0, 1]
R4: Omega format index ∈ {0, 1, 2, 3, 4}
```

---

## 7. Data Types

### 7.1 Primitive Types

| Type | Description | Constraint |
|------|-------------|------------|
| Ankh | Master harmonic constant | value = 5.08938 |
| Repitan | Fractional part of 27 | index ∈ [1, 27] |
| RacLevel | Access restriction level | enum {1, 2, 3, 4, 5, 6} |
| OmegaFormat | Frequency tier | enum {Red, OmegaMajor, Green, OmegaMinor, Blue} |
| Coherence | Normalized coherence score | value ∈ [0, 1] |

### 7.2 Composite Types

```
AccessResult = FullAccess | PartialAccess(α: Float) | Blocked

Coordinate = {
    theta: Repitan,
    phi: RacLevel,
    harmonic: OmegaFormat,
    radius: Float  -- Ankh-normalized
}

ResonanceScore = {
    theta_match: Float,
    phi_access: AccessResult,
    harmonic_match: Float,
    intensity: Float,
    composite: Float
}
```

---

## 8. Reference Values

### 8.1 Quick Reference Table

| Constant | Value | Use |
|----------|-------|-----|
| Ankh | 5.08938 | Master harmonic |
| π_red | 3.141592592 | Circle calculations |
| φ_green | 1.62 | Threshold harmonic |
| Hunab | 1.0602875 | Scalar unit |
| Ω | 1.005662978 | Format conversion |
| Balmer | 364.5 | Spectral reference |

### 8.2 RAC Quick Reference

| Level | Value | Pyramid Div |
|-------|-------|-------------|
| RAC₁ | 0.6361725 | 360 |
| RAC₂ | 0.628318519 | 364.5 |
| RAC₃ | 0.57255525 | 400 |
| RAC₄ | 0.523598765 | 437.4 |
| RAC₅ | 0.4580442 | 500 |
| RAC₆ | 0.3998594565 | 572.76 |

---

## Appendix A: Notation

- `⌈x⌉` - ceiling function
- `⌊x⌋` - floor function
- `≈` - approximately equal (within tolerance)
- `∈` - element of
- `→` - maps to / returns

## Appendix B: Changelog

- **1.0.0** (2025-12-29): Initial specification
