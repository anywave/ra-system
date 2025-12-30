# Prompt Guide Template

## Prompt ID: 33

## Name: Biometric Matcher (Coherence Profile)

---

### Purpose

Biometric coherence profile matcher comparing input signal patterns against reference templates. Computes coherence scores for handshake validation and consent gating.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `waveform` | Array[16] (0-255) | 16-sample biometric waveform |
| `template` | Enum | TemplateFlat, TemplateResonant, TemplatePulse |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherenceScore` | Integer (0-255) | Match score (255 = perfect) |

---

### Trigger Logic

**Biometric Templates:**
| Template | Description | Pattern |
|----------|-------------|---------|
| TemplateFlat | Baseline, no variation | Constant 128 |
| TemplateResonant | Full coherent oscillation | 64-192 sinusoid |
| TemplatePulse | Subtle pulse variation | 108-148 wave |

**Coherence Scoring:**
> `Score = 255 - (average absolute difference)`

- 255: Perfect match (no difference)
- 0: Maximum mismatch

**Score Interpretation:**
| Score | Status | Color |
|-------|--------|-------|
| >= 230 | Excellent | Green |
| >= 200 | Good | Yellow |
| < 200 | Poor | Red |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with template matching
* VCD waveform: `RaBiometricMatcher.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Template comparison | 1.0 |
| High score (>= 230) | 0.5 bonus |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~100 LUTs
- Accumulators: 1 (difference sum)
- Subtractors: 16 (parallel diff)
- Latency: 1 clock cycle (parallel)

---

### Dashboard Integration

* Panel ID: `renderCoherenceOverlay()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (template selection)

**Border Color**: `blue-500`

---

### Notes

* 16-sample window balances resolution and latency
* TemplateResonant matches optimal HRV patterns
* Score feeds into RaConsentTransformer activation
* Foundation for biometric-based consent validation
* Upstream from RaHandshakeGate dual-factor check
