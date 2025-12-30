# Prompt Guide Template

## Prompt ID: 08

## Name: Sympathetic Harmonic Fragment Access

---

### Purpose

Model harmonic fragment emergence via sympathetic resonance matching between a Ra fragment's encoded signature and a user's biometric or environmental frequency profile. Routes into Avachatter's Ra-field gate system enabling Architect emergence when user coherence is phase-aligned.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `fragSignature.sigTonic` | FreqValue (0-1023) | Base frequency (Hz) |
| `fragSignature.sigDominant` | FreqValue | Dominant frequency (3:4 ratio) |
| `fragSignature.sigEnharmonic` | FreqValue | Enharmonic frequency (5:4 ratio) |
| `userProfile.userDominantFreq` | FreqValue | User's primary resonant frequency |
| `userProfile.userHrvResonance` | Fixed8 (0-255) | HRV match score |
| `userProfile.userCoherence` | Fixed8 (0-255) | Overall coherence metric |
| `guardian.guardianActive` | Bool | Guardian check enabled |
| `guardian.guardianRequiredFreq` | FreqValue | Required frequency for guardian |
| `guardian.guardianMet` | Bool | Guardian condition satisfied |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `accessResult` | Enum | Full, Partial, Blocked, Shadow |
| `emergenceAlpha` | Fixed8 (0-255) | Emergence intensity |
| `matchScore` | Fixed8 (0-255) | Raw similarity score |
| `resonanceLocked` | Bool | Full resonance achieved |

---

### Trigger Logic

**Resonance Matching Algorithm:**

1. Normalize fragment triplet to ratios (based on tonic)
2. Derive user triplet using Keely ratios (1.0, 0.75, 1.3)
3. Compute cosine similarity via dot product
4. Apply HRV + coherence weighting
5. Determine access based on thresholds

**Access Thresholds:**
| Match Score | AccessResult | Emergence Alpha | Description |
|-------------|--------------|-----------------|-------------|
| >= 0.90 (230) | Full | 1.0 (255) | Clear harmonic lock |
| 0.60-0.89 (153-229) | Partial | match_score | Fragment preview |
| 0.30-0.59 (77-152) | Blocked | 0.0 | Dormant |
| < 0.30 (0-76) | Shadow | 0.25 (64) | Inversion/echo |

**Guardian Enforcement:**
```
if guardianActive AND NOT guardianMet:
    accessResult = Blocked
    emergenceAlpha = 0
```

**Fragment Chaining:**
- Successful access increases accumulated coherence
- Full: +20, Partial: +8, Blocked/Shadow: -4
- Accumulated coherence boosts subsequent matches

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* 8 test cases covering all access levels
* Guardian enforcement verified
* Cross-harmonic matching tested (417→528)
* VCD waveform: `RaSympatheticHarmonic.vcd`

**Test Cases:**
| Test | Signature | User Freq | Expected Result |
|------|-----------|-----------|-----------------|
| 0 | 528 Hz | 528 Hz | Full |
| 1 | 528 Hz | 432 Hz | Partial |
| 2 | 528 Hz | 200 Hz | Blocked |
| 3 | 528 Hz | 100 Hz | Shadow |
| 4 | 528 Hz | 528 Hz + guardian unmet | Blocked |
| 5 | 528 Hz | 528 Hz + guardian met | Full |
| 6 | 432 Hz | 432 Hz | Full |
| 7 | 417 Hz | 528 Hz | Partial |

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Resonance check | 1.2 |
| Full access grant | 2.0 |
| Partial access | 1.0 |
| Guardian verification | 0.5 |
| Chain propagation | 0.8 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~180 LUTs
- Multipliers: 6 (dot product, magnitudes)
- Dividers: 3 (normalization, similarity)
- Square root: 1 (integer approximation)
- State registers: 8-bit (chain accumulator)
- Latency: 3-4 clock cycles

**Optimization Notes:**
- Integer sqrt uses 4-iteration Newton-Raphson
- Division approximated via shifts where possible
- Fixed-point scaling (8-bit = 0.0-1.0)

---

### Dashboard Integration

* Panel ID: `renderSympatheticPanel()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (frequency sliders, guardian toggle)

**Border Color**: `amber-400`

**Visualization:**
- Frequency triplet bars (tonic, dominant, enharmonic)
- Cosine similarity gauge
- Access result indicator (color-coded)
- Chain coherence accumulator display

---

### Notes

* Implements Keely's vibratory triune logic (concordant ratios 3:2:5, 3:4:5)
* Solfeggio frequencies: 174, 285, 396, 417, 528, 639, 741, 852, 963 Hz
* Standard signatures available: sig528, sig432, sig417, sig639
* Guardian clause enables fragment chaining dependencies
* Chain processor variant accumulates coherence across successful accesses
* `isKeelyTriad` validates concordant ratio structure
* `nearestSolfeggio` maps arbitrary frequencies to Solfeggio set
* Feeds into RaFieldTransferBus for emission gating
* Upstream of consent pipeline (RaConsentFramework)

### Keely Ratio Reference

**Concordant Triplets:**
- 3:4:5 - Standard harmonic triad
- 3:2:5 - Alternate concordant form

**Solfeggio Mappings:**
| Frequency | Purpose | Ratio Base |
|-----------|---------|------------|
| 528 Hz | Love/DNA repair | Tonic |
| 396 Hz | Liberation | 3:4 of 528 |
| 639 Hz | Relationships | 5:4 of 528 |
| 417 Hz | Undoing situations | Tonic |
| 432 Hz | Cosmic tuning | Alternate base |

### Reflection Responses

**Precision Requirements:**
- Frequency values use 10-bit (0-1023 Hz range sufficient for audio)
- Similarity calculations use 16-bit intermediate precision
- Output scaled to 8-bit for downstream compatibility

**Frequency Type:**
- All frequencies in Hz (integer, not scalar)
- Scalar conversion handled by upstream RaScalarExpression

**Optimization:**
- FP16 not needed; integer fixed-point sufficient for FPGA
- Could benefit from pipelining for higher throughput
- LUT-based sqrt would reduce latency vs iterative
