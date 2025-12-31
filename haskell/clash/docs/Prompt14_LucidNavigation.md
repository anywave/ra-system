# Prompt 14: Lucid Scalar Navigation via Harmonic Field Wayfinding

## Overview

This module enables conscious exploration of the Ra System's harmonic field structure through coordinate-based navigation. Users traverse a spherical lattice of fragments using coherence-gated access, with real-time resonance scoring determining which fragments can be perceived and integrated.

## Architecture

```
+-------------------------------------------------------------------------+
|                    LUCID NAVIGATION SYSTEM                              |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------------+   +-----------------+   +------------------+        |
|  | Intention      |-->| Coordinate      |-->| Coherence Gate   |------>|
|  | Parser         |   | Translator      |   | (Weighted)       | Access|
|  +----------------+   +-----------------+   +------------------+        |
|         |                    |                     |                    |
|         v                    v                     v                    |
|  +----------------+   +-----------------+   +------------------+        |
|  | Return Vector  |   | Resonance       |   | Shadow Consent   |        |
|  | (phi^n beacon) |   | Scorer          |   | Bridge (P12)     |        |
|  +----------------+   +-----------------+   +------------------+        |
|         |                    |                     |                    |
|         v                    v                     v                    |
|  +----------------+   +-----------------+   +------------------+        |
|  | Drift          |   | Fragment        |   | Symbolic Field   |        |
|  | Detection      |   | Access          |   | Translator       |        |
|  +----------------+   +-----------------+   +------------------+        |
|                                                                         |
+-------------------------------------------------------------------------+
```

## Codex References

| Source | Application |
|--------|-------------|
| **KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md** | Harmonic intention guidance |
| **RADIONICS_RATES_DOWSING.md** | Scalar targeting |
| **GOLOD_RUSSIAN_PYRAMIDS.md** | Field stabilization |
| **Prompt 12** | Shadow consent gating |
| **Prompt 13A** | Gamma spike amplification |

## RaCoordinate System

### Spherical Lattice Structure

```
theta (azimuth): 1-13 discrete sectors
phi (polar): 1-12 harmonic strata
h (shell): 0-7 harmonic shells
r (depth): 0.0-1.0, phi-aligned at 0.618
```

### Python Data Class

```python
@dataclass
class RaCoordinate:
    theta: int = 1      # Azimuth sector 1-13
    phi: int = 1        # Polar stratum 1-12
    h: int = 0          # Harmonic shell 0-7
    r: float = 0.0      # Scalar depth 0.0-1.0
```

### Clash Type

```haskell
data RaCoordinate = RaCoordinate
  { coordTheta :: Unsigned 4   -- Azimuth sector 1-13
  , coordPhi   :: Unsigned 4   -- Polar stratum 1-12
  , coordH     :: Unsigned 3   -- Harmonic shell 0-7
  , coordR     :: Unsigned 8   -- Scalar depth 0-255 (maps to 0.0-1.0)
  }
```

## Coherence Gate

### Weighted Formula

```
coherence_vector = 0.5 * coherence_score + 0.3 * hrv_resonance + 0.2 * normalized_breath
```

Where:
- `coherence_score`: Direct coherence measurement 0.0-1.0
- `hrv_resonance`: HRV-derived resonance 0.0-1.0
- `normalized_breath`: Breath rate normalized by 0.2 cap

### Access Tiers

| Tier | Threshold | Fragment Access |
|------|-----------|-----------------|
| FULL | >= 0.80 | All fragments, full immersion |
| PARTIAL | 0.50-0.79 | Most fragments, summary form |
| DISTORTED | 0.30-0.49 | Limited fragments, symbolic |
| BLOCKED | < 0.30 | No access, safety hold |

### Gamma Amplification (Prompt 13A)

When gamma power >= 0.25:
- Add +0.10 boost to coherence vector
- Enables deeper navigation during lucid states

## Resonance Scoring

### Cosine Similarity

```python
def resonance_score(user_vec: List[float], fragment_vec: List[float]) -> float:
    dot = sum(u * f for u, f in zip(user_vec, fragment_vec))
    mag_u = math.sqrt(sum(u ** 2 for u in user_vec))
    mag_f = math.sqrt(sum(f ** 2 for f in fragment_vec))
    return dot / (mag_u * mag_f) if mag_u > 0 and mag_f > 0 else 0.0
```

### Fragment Access Thresholds

| Level | Threshold | Emergence Form |
|-------|-----------|----------------|
| FULL | >= 0.88 | Full text, immersive |
| PARTIAL | 0.65-0.87 | Summary, symbolic |
| BLOCKED | < 0.65 | Echo only, no content |

## Return Vector

### Beacon Protocol

- **Shallow return** (r <= 0.7, h <= 5): phi^3 harmonic
- **Deep return** (r > 0.7 or h > 5): phi^5 harmonic

### Drift Detection

```python
drift = abs(current_coherence - baseline_coherence)
trigger_return = drift >= 0.42 or current_coherence < 0.28
```

## Symbolic Field Translator

### Navigation Metaphors

| Direction | Metaphor | Description |
|-----------|----------|-------------|
| ASCEND | luminous_path | Rising through shells |
| DESCEND | spiral_staircase | Deepening into field |
| SPIRAL | angular_bridge | Theta sector transition |
| ATTUNE | golden_resonance | Align to phi corridor |
| EXIT | emergence_portal | Return to origin |

## Clash Module: RaLucidNavigation.hs

### Key Types

```haskell
data NavDirection = DirAscend | DirDescend | DirSpiral | DirAttune | DirExit | DirNone
data AccessTier = TierFull | TierPartial | TierDistorted | TierBlocked
data NavOutput = NavOutput
  { outCoord, outTier, outCanMove, outInGolden, outReturnTrig, outMetaphor }
```

### Core Functions

| Function | Purpose |
|----------|---------|
| `coherenceVector` | Calculate weighted coherence |
| `accessTier` | Determine access level |
| `gammaAmplify` | Apply lucid state boost |
| `applyDirection` | Navigate coordinate space |
| `shouldTriggerReturn` | Detect drift/return need |
| `isGoldenCorridor` | Check phi alignment |

### Synthesis Target

```haskell
{-# ANN lucidNavTop (Synthesize
  { t_name = "lucid_nav_top"
  , t_inputs = [ PortName "clk", PortName "rst", PortName "en", ... ]
  , t_output = PortProduct "output" [ ... ]
  }) #-}
```

## Python Test Harness

### test_lucid_navigation.py

**Test Scenarios (8 total):**

| Test | Description | Expected |
|------|-------------|----------|
| `coordinate_system` | RaCoordinate bounds | Valid ranges |
| `intention_parser` | Command parsing | Correct directions |
| `coherence_gate` | Weighted formula | Correct tiers |
| `resonance_scorer` | Cosine similarity | Access levels |
| `symbolic_translator` | Metaphor mapping | Valid metaphors |
| `return_vector` | Drift detection | Beacon triggered |
| `lucid_navigator` | Full navigation | State transitions |
| `shadow_consent_integration` | P12 shadow gating | Consent enforced |

**Usage:**

```bash
python test_lucid_navigation.py        # Full suite
python test_lucid_navigation.py --demo # With navigation demo
python test_lucid_navigation.py --json # JSON output
```

## Integration Points

### Upstream Dependencies

| Module | Integration |
|--------|-------------|
| Biometric sensors | HRV, breath rate input |
| EEG/gamma detector | Lucid state marker (P13A) |
| Prompt 12 | Shadow consent gating |

### Downstream Outputs

| Consumer | Data |
|----------|------|
| Fragment renderer | Access tier, form |
| Visual interface | Metaphor overlays |
| Return system | Beacon coordinates |

## Hardware Resources

| Platform | LUTs | DSP | BRAM |
|----------|------|-----|------|
| Xilinx Artix-7 | ~400 | 1 | 0 |
| Intel Cyclone V | ~450 | 1 | 0 |
| Lattice ECP5 | ~500 | 1 | 1 |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-30 | Initial implementation with full spec |
