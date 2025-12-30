# Prompt Guide Template

## Prompt ID: 09

## Name: Orgone Field Influence on Ra Scalar Stability

---

### Purpose

Synthesizes Reichian orgone dynamics with scalar field modulation protocols. Models OR (positive orgone) and DOR (deadly orgone) interactions with Ra scalar fields for emergence stability control.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `orgoneField.orLevel` | Fixed8 (0-255) | Positive orgone charge |
| `orgoneField.dorLevel` | Fixed8 (0-255) | Deadly orgone level |
| `orgoneField.accumulationRate` | Fixed8 (0-255) | Exponential charge rate |
| `orgoneField.chamberGeometry` | Enum | Pyramidal, Dome, Rectangular, Spherical |
| `basePotential` | Fixed8 (0-255) | Base scalar potential |
| `baseCoherence` | Fixed8 (0-255) | Base coherence level |
| `emotionalStress` | Fixed8 (0-255) | Emotional stress factor |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `potential` | Fixed8 (0-255) | Modified scalar potential |
| `fluxCoherence` | Fixed8 (0-255) | Flux coherence level |
| `inversionProb` | Fixed8 (0-255) | Inversion probability |
| `emergenceScore` | Fixed8 (0-255) | Overall emergence score |
| `emergenceClass` | Enum | AlphaEmergence, StableFragment, BaselineStability, ShadowFragment, FieldCollapse |
| `luminescenceFlag` | Bool | Blue luminescence trigger |
| `dischargeWarning` | Bool | OR discharge warning |

---

### Trigger Logic

**Scalar Coupling Formulas (Codex-Aligned):**
```
potential *= (1 + or_level - dor_level)
flux_coherence *= (1 - dor_level)
inversion_probability *= (1 + dor_level - or_level)
```

**Chamber Geometry Modifiers:**
| Geometry | OR Boost | DOR Shield | Notes |
|----------|----------|------------|-------|
| Pyramidal | +0.15 (38) | -0.10 (26) | Proven resonance enhancer |
| Dome | +0.10 (26) | -0.05 (13) | Smooth energy flow |
| Rectangular | 0 | 0 | Neutral or noise-prone |
| Spherical | +0.12 (31) | -0.08 (20) | Experimental |

**Emergence Classification:**
| Score Threshold | Class | Description |
|-----------------|-------|-------------|
| >= 0.80 (204) | AlphaEmergence | High OR, stable emergence |
| >= 0.60 (153) | StableFragment | Deep scalar potentials |
| >= 0.40 (102) | BaselineStability | Neutral state |
| >= 0.20 (51) | ShadowFragment | High DOR, turbulence |
| < 0.20 | FieldCollapse | Critical DOR |

**Phenomenological Triggers (Reich):**
- Blue luminescence: OR >= 0.80 AND coherence > 150
- Discharge warning: stress > 180 AND OR > 128
- OR discharge under stress reduces OR level

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* 4 test cases matching spec requirements
* VCD waveform: `RaOrgoneScalar.vcd`

**Test Case Matrix:**
| Test | OR | DOR | Chamber | Expected Result |
|------|-----|-----|---------|-----------------|
| 0 | 0.9 | 0.1 | Pyramidal | Alpha emergence + high coherence |
| 1 | 0.5 | 0.5 | Rectangular | Baseline stability |
| 2 | 0.2 | 0.7 | Dome | Shadow fragments + turbulence |
| 3 | 0.8 | 0.3 | Pyramidal | Stable fragments + deep potentials |

**Logged Parameters:**
- potential, flux_coherence, inversion_probability
- emergence_score vs expected thresholds
- luminescence_flag, discharge_warning

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Field evaluation | 0.8 |
| Geometry modifier | 0.2 |
| Accumulation cycle | 0.3 |
| Luminescence detection | 0.1 |
| Discharge computation | 0.4 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: Yes (luminescence rendering)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~120 LUTs
- DSP Slices: 4 (multiplications)
- State Registers: 8-bit (accumulator)
- Latency: 2 clock cycles

**Synthesis Targets:**
| Target | LUTs | DSP | Notes |
|--------|------|-----|-------|
| Xilinx Artix-7 | ~120 | 4 slices | Reference platform |
| Intel Cyclone V | ~140 ALMs | 3 blocks | Validated |
| Lattice ECP5 | ~150 | 4 mult | Low-power option |

---

### Dashboard Integration

* Panel ID: `renderOrgonePanel()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (OR/DOR sliders, geometry selector)

**Border Color**: `violet-500`

**Visualization:**
- OR/DOR level bars (green/red gradient)
- Chamber geometry selector
- Emergence class indicator
- Blue luminescence glow effect (when triggered)
- Radial phase bloom for discharge events

---

### System Compatibility

**Upstream Dependencies:**
| Module | Purpose | Data Flow |
|--------|---------|-----------|
| RaSympatheticHarmonic (Prompt 8) | Fragment access | Access scoring input |
| RaConsentFramework (Prompt 32) | Gating | Coherence validation |
| Ra.Constants | Geometry params | PYRAMID_GEOMETRY_ENERGY.md |

**Downstream Consumers:**
| Module | Purpose | Data Flow |
|--------|---------|-----------|
| RaFieldTransferBus (Prompt 35) | Emission | ScalarOutput gating |
| RaVisualizerShell (Prompt 41) | Rendering | Luminescence trigger |
| RaChamberSync (Prompt 40) | Coordination | EmergenceScore sync |

---

### Notes

**Field Characteristics:**
* OR increases scalar potential and coherence
* DOR increases instability and fragment collapse risk
* Balanced OR/DOR leads to field stagnation
* Chamber geometry provides passive modulation

**Reichian Dynamics:**
* Blue luminescence indicates OR-saturated scalar zones
* Sudden OR discharge under emotional stress causes fragment collapse
* Emotion amplification near convergence affects access scoring

**Functional Outcomes:**
| State | OR Level | DOR Level | Result |
|-------|----------|-----------|--------|
| High OR | > 0.7 | < 0.3 | Deeper scalar wells, stable emergence |
| High DOR | < 0.3 | > 0.7 | Fragmented emergence, shadow artifacts |
| Balanced | ~0.5 | ~0.5 | Field stagnation, baseline stability |

**Accumulation Dynamics:**
- OR accumulates via exponential charge curve
- Rate scaled by headroom: increment = rate * (1 - or/256)
- Discharge proportional to stress * OR level

---

### Reflection Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Precision | 8-bit normalized | Sufficient for OR/DOR dynamics |
| Geometry Lookup | Enum + constants | Avoids runtime configuration complexity |
| Accumulation | Linear approximation | Exponential too costly for FPGA |
| Discharge Model | Threshold-based | Simplified stress response |

---

### JSON Schema (for Claude Simulation)

**Input:**
```json
{
  "orgone_field": {
    "or_level": 0.72,
    "dor_level": 0.18,
    "accumulation_rate": 0.04,
    "chamber_geometry": "pyramidal"
  },
  "base_potential": 0.78,
  "base_coherence": 0.80,
  "emotional_stress": 0.20
}
```

**Output:**
```json
{
  "potential": 0.95,
  "flux_coherence": 0.66,
  "inversion_prob": 0.35,
  "emergence_score": 0.82,
  "emergence_class": "ALPHA_EMERGENCE",
  "luminescence_flag": true,
  "discharge_warning": false
}
```
