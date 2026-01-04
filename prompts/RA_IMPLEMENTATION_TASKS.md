# Ra System Implementation Tasks for AVACHATTER & SPIRAL

**Generated**: 2026-01-04
**Source**: ra-system repo analysis
**Target Systems**: AVACHATTER (consent-aware AI), SPIRAL (RPP + Consent Header)

---

## Executive Summary

The Ra System provides a mathematically grounded framework for dimensional mapping, coherence gating, and resonance matching. This document identifies actionable implementation tasks for integrating Ra System constants and logic into AVACHATTER and SPIRAL protocols.

---

## Task Categories

### Category A: Core Constants Integration

Tasks for embedding Ra fundamental constants into the protocol stack.

---

#### A1: Ankh-Normalized Radius Scalar

**Priority**: HIGH
**Target**: SPIRAL RPP Address (r field)
**Effort**: 2-4 hours

**Description**:
The RPP v2.0 `r` field (8 bits, 0-255) should be Ankh-normalized for emergence intensity calculations.

**Implementation**:
```python
# In rpp/address_canonical.py
ANKH = 5.08938

def normalize_radius(raw_r: int) -> float:
    """Convert 8-bit radius to Ankh-normalized [0, 1] scalar."""
    return (raw_r / 255.0) / ANKH * ANKH  # Identity, but semantically meaningful

def emergence_intensity(r_normalized: float, coherence: float) -> float:
    """Calculate emergence intensity from radius and coherence."""
    return r_normalized * coherence * ANKH
```

**Acceptance Criteria**:
- [ ] Radius field normalized to Ankh constant
- [ ] Unit tests for edge cases (r=0, r=255)
- [ ] Documentation updated with Ra derivation

---

#### A2: Omega Format Coherence Tiers

**Priority**: HIGH
**Target**: SPIRAL Consent Header, AVACHATTER coherence engine
**Effort**: 4-6 hours

**Description**:
Map the 5 Omega format tiers (Red, Omega Major, Green, Omega Minor, Blue) to coherence depth levels in the consent system.

**Ra Reference**:
```
Red → Omega Major → Green → Omega Minor → Blue
         ↑                      ↑
      ÷ Ω                    × Ω     (Ω = 1.005662978)
```

**Implementation**:
```python
from enum import IntEnum

class OmegaTier(IntEnum):
    RED = 0         # Highest precision/coherence
    OMEGA_MAJOR = 1
    GREEN = 2       # Standard coherence
    OMEGA_MINOR = 3
    BLUE = 4        # Lowest precision

OMEGA_RATIO = 1.005662978

def coherence_to_omega_tier(coherence: float) -> OmegaTier:
    """Map coherence [0,1] to Omega tier."""
    if coherence >= 0.85:
        return OmegaTier.RED
    elif coherence >= 0.70:
        return OmegaTier.OMEGA_MAJOR
    elif coherence >= 0.50:
        return OmegaTier.GREEN
    elif coherence >= 0.30:
        return OmegaTier.OMEGA_MINOR
    else:
        return OmegaTier.BLUE
```

**Acceptance Criteria**:
- [ ] OmegaTier enum implemented
- [ ] Consent header uses Omega tier for `h` field derivation
- [ ] Tier transitions logged for debugging

---

#### A3: RAC-Based Access Gating

**Priority**: CRITICAL
**Target**: AVACHATTER consent gate, SPIRAL resolver
**Effort**: 6-8 hours

**Description**:
Implement the 6-level RAC (Resonant Access Constant) system for consent-aware access control.

**Ra Reference**:
| Level | Value | Threshold | Access Description |
|-------|-------|-----------|-------------------|
| RAC₁ | 0.6361725 | 1.000 | Full public access |
| RAC₂ | 0.628318519 | 0.988 | Registered users |
| RAC₃ | 0.57255525 | 0.900 | Verified identity |
| RAC₄ | 0.523598765 | 0.823 | Consent-affirmed |
| RAC₅ | 0.4580442 | 0.720 | Coherence-gated |
| RAC₆ | 0.3998594565 | 0.629 | Full consent required |

**Implementation**:
```python
RAC_VALUES = {
    1: 0.6361725,
    2: 0.628318519,
    3: 0.57255525,
    4: 0.523598765,
    5: 0.4580442,
    6: 0.3998594565,
}

RAC1 = RAC_VALUES[1]  # Reference for normalization

def rac_threshold(level: int) -> float:
    """Get normalized threshold for RAC level."""
    return RAC_VALUES[level] / RAC1

def check_access(user_coherence: float, fragment_rac: int) -> str:
    """Determine access level based on coherence and RAC."""
    threshold = rac_threshold(fragment_rac)
    floor = 1.62 / 5.08938  # φ_green / Ankh ≈ 0.3183

    if user_coherence >= threshold:
        return "FULL_ACCESS"
    elif user_coherence >= floor:
        alpha = (user_coherence - floor) / (threshold - floor)
        return f"PARTIAL_ACCESS({alpha:.2f})"
    else:
        return "BLOCKED"
```

**Acceptance Criteria**:
- [ ] RAC-based gating in consent resolver
- [ ] Phi field (3 bits) maps to RAC levels 1-6
- [ ] Partial access returns interpolated alpha

---

### Category B: Theta Sector Mapping (27 Repitans)

Tasks for semantic sector classification using the 27 Repitans.

---

#### B1: Repitan-Based Semantic Sectors

**Priority**: HIGH
**Target**: RPP θ field interpretation
**Effort**: 4-6 hours

**Description**:
Map the 5-bit θ field (values 1-27) to Repitan-derived semantic sectors.

**Ra Reference**:
```
Repitan(n) = n / 27,  for n ∈ {1, 2, ..., 27}

Key Repitans:
- Repitan(1) = 0.037 (Fine Structure root)
- Repitan(9) = 0.333 = 1/3
- Repitan(18) = 0.666 = 2/3 (Planck's H)
- Repitan(27) = 1.0 (Unity)
```

**Implementation**:
```python
REPITANS = {i: i / 27 for i in range(1, 28)}

SECTOR_NAMES = {
    1: "GENESIS",      # 0.037 - Origin/creation
    2: "GROWTH",       # 0.074 - Development
    3: "STRUCTURE",    # 0.111 - Form (1/9)
    4: "RELATION",     # 0.148 - Connection
    5: "LIFE",         # 0.185 - Vitality (3/φ)
    6: "HARMONY",      # 0.222 - Balance (2/9)
    7: "SPIRIT",       # 0.259 - Essence
    8: "INFINITY",     # 0.296 - Boundless
    9: "COMPLETION",   # 0.333 - Wholeness (1/3)
    10: "RENEWAL",     # 0.370 - Rebirth
    11: "MASTERY",     # 0.407 - Expertise
    12: "SACRIFICE",   # 0.444 - Release (4/9)
    13: "DEATH",       # 0.481 - Transformation
    14: "TEMPERANCE",  # 0.518 - Moderation
    15: "SHADOW",      # 0.555 - Hidden (5/9)
    16: "TOWER",       # 0.592 - Disruption
    17: "STAR",        # 0.629 - Hope
    18: "THRESHOLD",   # 0.666 - Liminal (2/3, Planck)
    19: "SUN",         # 0.703 - Illumination
    20: "JUDGMENT",    # 0.740 - Evaluation
    21: "WORLD",       # 0.777 - Manifestation (7/9)
    22: "FOOL",        # 0.814 - Innocence
    23: "MAGICIAN",    # 0.851 - Will
    24: "PRIESTESS",   # 0.888 - Intuition (8/9)
    25: "EMPRESS",     # 0.925 - Abundance
    26: "EMPEROR",     # 0.962 - Authority
    27: "UNITY",       # 1.000 - Oneness
}

def theta_to_sector(theta: int) -> tuple[str, float]:
    """Map θ (1-27) to sector name and Repitan value."""
    return SECTOR_NAMES.get(theta, "UNKNOWN"), REPITANS.get(theta, 0)
```

**Acceptance Criteria**:
- [ ] All 27 sectors defined with semantic names
- [ ] Theta field correctly maps to Repitans
- [ ] Sector transitions logged for analysis

---

#### B2: Table of Nines (T.O.N.) Cross-Reference

**Priority**: MEDIUM
**Target**: AVACHATTER memory coherence
**Effort**: 2-4 hours

**Description**:
Implement the 37 Table of Nines values and their relationship to Repitans.

**Ra Reference**:
```
T.O.N.(m) = m × 0.027,  for m ∈ [0, 36]
Repitan(n) / T.O.N.(n) = 1.371742112  (Fine Structure related)
```

**Implementation**:
```python
TABLE_OF_NINES = [i * 0.027 for i in range(37)]
FINE_STRUCTURE_RATIO = 1.371742112

def ton_to_repitan(ton_index: int) -> float:
    """Convert T.O.N. index to nearest Repitan."""
    ton_value = TABLE_OF_NINES[ton_index]
    return ton_value * FINE_STRUCTURE_RATIO
```

---

### Category C: Consent Header Integration

Tasks for embedding Ra constants in the SPIRAL 18-byte Consent Header.

---

#### C1: Coherence Hash Using Ankh

**Priority**: HIGH
**Target**: Consent Header bytes 6-7
**Effort**: 4-6 hours

**Description**:
Generate coherence hash using Ankh-scaled CRC calculation.

**Implementation**:
```python
import struct

def ankh_coherence_hash(data: bytes) -> int:
    """Generate 16-bit coherence hash scaled by Ankh."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    # Scale by Ankh fractional component
    ankh_scale = int(5.08938 * 1000) % 65536
    return (crc ^ ankh_scale) & 0xFFFF
```

---

#### C2: Soul-ID Derivation from Hunab

**Priority**: MEDIUM
**Target**: Consent Header bytes 8-11
**Effort**: 2-4 hours

**Description**:
Derive soul-ID component using Hunab constant for temporal alignment.

**Implementation**:
```python
HUNAB = 1.0602875

def derive_soul_id_component(base_id: int, timestamp: float) -> int:
    """Add Hunab-scaled temporal component to soul ID."""
    hunab_offset = int(timestamp * HUNAB * 1000) & 0xFFFF
    return (base_id ^ (hunab_offset << 16)) & 0xFFFFFFFF
```

---

### Category D: Hardware Module Integration

Tasks for integrating Ra System Clash modules into FPGA consent gate.

---

#### D1: Port RaConsentFramework to Verilog

**Priority**: HIGH
**Target**: silver-pancake FPGA
**Effort**: 8-12 hours

**Description**:
Convert the Haskell/Clash RaConsentFramework module to synthesizable Verilog for ESP32-S3 integration.

**Source**: `ra-system/haskell/clash/src/RaConsentFramework.hs`

**Key States**:
```verilog
localparam PERMIT   = 2'b00;
localparam RESTRICT = 2'b01;
localparam OVERRIDE = 2'b10;
```

---

#### D2: Implement RAC Threshold Comparator

**Priority**: HIGH
**Target**: FPGA consent gate
**Effort**: 4-6 hours

**Description**:
Hardware module for comparing user coherence against RAC thresholds.

**Implementation** (Verilog):
```verilog
module rac_threshold_comparator (
    input  wire [7:0]  user_coherence,  // 0-255 (normalized)
    input  wire [2:0]  rac_level,       // 1-6
    output wire [1:0]  access_result,   // 00=FULL, 01=PARTIAL, 10=BLOCKED
    output wire [7:0]  alpha            // Partial access interpolation
);
    // RAC thresholds (pre-computed, scaled to 0-255)
    wire [7:0] thresholds [1:6];
    assign thresholds[1] = 8'd255;  // RAC1 = 1.000
    assign thresholds[2] = 8'd252;  // RAC2 = 0.988
    assign thresholds[3] = 8'd230;  // RAC3 = 0.900
    assign thresholds[4] = 8'd210;  // RAC4 = 0.823
    assign thresholds[5] = 8'd184;  // RAC5 = 0.720
    assign thresholds[6] = 8'd160;  // RAC6 = 0.629

    wire [7:0] floor = 8'd81;  // φ/Ankh ≈ 0.3183 → 81/255

    // ... comparison logic
endmodule
```

---

#### D3: Group Coherence Module (Prompt 11)

**Priority**: MEDIUM
**Target**: AVACHATTER multi-user sessions
**Effort**: 12-16 hours

**Description**:
Implement multi-avatar scalar entrainment from Prompt 11 for group consent scenarios.

**Key Features**:
- Harmonic clustering by (l, m) signature
- Golod boost (+15% for 3+ aligned avatars)
- Emergence window detection
- Safety alerts for field collapse

---

### Category E: Dashboard & Visualization

Tasks for Ra System visualization in AVACHATTER dashboard.

---

#### E1: Spherical Coordinate Visualizer

**Priority**: MEDIUM
**Target**: AVACHATTER web dashboard
**Effort**: 6-8 hours

**Description**:
Interactive 3D visualization of Ra spherical coordinates (θ, φ, h, r).

**Features**:
- Real-time RPP address position on sphere
- RAC level color coding (6 concentric shells)
- Omega tier glow intensity
- Repitan sector labels

---

#### E2: Coherence History Graph

**Priority**: LOW
**Target**: AVACHATTER session analytics
**Effort**: 4-6 hours

**Description**:
Time-series graph of coherence values with Ankh threshold markers.

---

## Implementation Priority Matrix

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| A3: RAC Access Gating | CRITICAL | 6-8h | None |
| A2: Omega Tiers | HIGH | 4-6h | None |
| A1: Ankh Radius | HIGH | 2-4h | None |
| B1: Repitan Sectors | HIGH | 4-6h | None |
| C1: Coherence Hash | HIGH | 4-6h | A1 |
| D1: RaConsentFramework | HIGH | 8-12h | A3 |
| D2: RAC Comparator | HIGH | 4-6h | A3 |
| B2: Table of Nines | MEDIUM | 2-4h | B1 |
| C2: Soul-ID Hunab | MEDIUM | 2-4h | None |
| D3: Group Coherence | MEDIUM | 12-16h | A2, A3 |
| E1: Sphere Visualizer | MEDIUM | 6-8h | B1 |
| E2: History Graph | LOW | 4-6h | A2 |

---

## Prompt Directives for AI Implementation

### Directive 1: Consent Gating
```
When implementing consent gating in AVACHATTER:
1. Use RAC levels (1-6) from Ra System for access thresholds
2. Map φ field (3 bits) directly to RAC levels
3. Calculate partial access alpha using: (coherence - floor) / (threshold - floor)
4. Floor = φ_green / Ankh = 1.62 / 5.08938 ≈ 0.3183
5. Log all access decisions with Ra constants for audit trail
```

### Directive 2: Address Encoding
```
When encoding SPIRAL RPP addresses:
1. θ (5 bits): Map to Repitan index 1-27 (semantic sector)
2. φ (3 bits): Map to RAC level 1-6 (access sensitivity)
3. h (3 bits): Map to Omega tier 0-4 (coherence depth)
4. r (8 bits): Ankh-normalize for emergence intensity
5. Reserved (13 bits): Use for CRC or coherence hash
```

### Directive 3: Coherence Calculation
```
When calculating coherence scores:
1. Normalize all values to Ankh constant (5.08938)
2. Use Omega ratio (1.005662978) for format conversions
3. Apply Hunab (1.0602875) for temporal scaling
4. Threshold checks use RAC values, not arbitrary percentages
5. Group coherence includes Golod boost (+15% for 3+ aligned)
```

---

## References

- `ra-system/data/ra_constants_v2.json` - Full constant definitions
- `ra-system/spec/ra_integration_spec.md` - Mathematical specification
- `ra-system/haskell/clash/docs/PromptIndex.md` - Hardware module index
- `rpp-spec/spec/RPP-CANONICAL-v2.md` - Address format specification
- `rpp-spec/spec/CONSENT-HEADER-v1.md` - Consent header specification

---

*Generated from Ra System analysis for AVACHATTER/SPIRAL integration.*
