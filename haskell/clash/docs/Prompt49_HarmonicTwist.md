# Prompt Guide Template

## Prompt ID: 49

## Name: Harmonic Inversion Twist

---

### Purpose

Computes twist envelope from harmonic mode pair Y(a,b) and coherence level. Produces twist magnitude and duration for animation/rendering pipelines based on coherence-gated amplification or suppression.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `modeA` | Integer (0-15) | Azimuthal harmonic mode Y(a,b) |
| `modeB` | Integer (0-15) | Polar harmonic mode |
| `coherence` | Integer (0-255) | Coherence level for threshold gating |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `twistVector` | Integer (0-255) | Computed twist magnitude |
| `durationPhiN` | Integer | Duration in phi-N units (4 or 8 cycles) |

---

### Trigger Logic

1. **Inverse Magnitude Calculation**:
   ```
   invMag = modeA * 10 + modeB * 7
   ```

2. **Coherence Threshold Gating**:
   > If `coherence >= 105`, apply amplification: `twistMag = invMag + 15`
   > If `coherence < 105`, apply suppression: `twistMag = invMag - 10`

3. **Duration Selection**:
   > If `twistMag > 80`, set `durationPhiN = 8` cycles
   > Otherwise, set `durationPhiN = 4` cycles

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with 4 test cases
* VCD waveform: `RaHarmonicTwist.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Low coherence (< 105) | 1.2 |
| High coherence (>= 105) | 2.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~50 LUTs
- Multipliers: 2 (for invMag calculation)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderTwistEnvelopePanel()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (editable JSON input)

**Border Color**: `yellow-700`

---

### Notes

* Uses saturating arithmetic (`satAdd`, `satSub`) to prevent overflow
* Twist magnitude directly maps to visual intensity in animation pipeline
* Duration feeds into timing subsystem for frame scheduling
* Edge case: Zero modes (0,0) with high coherence still produce output of 15
