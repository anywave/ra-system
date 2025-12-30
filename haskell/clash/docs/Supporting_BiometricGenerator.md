# Prompt Guide Template

## Prompt ID: Supporting

## Name: Biometric Generator (Waveform Simulation)

---

### Purpose

Generates simulated biometric waveforms for coherence entrainment testing. Produces test patterns for validating biometric matching and consent pipeline without live sensors.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | Enum | Flatline, BreathRise, CoherentPulse, Arrhythmic |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `sample` | Integer (0-255) | Current waveform sample |

---

### Trigger Logic

**Biometric Patterns:**
| Pattern | Description | Waveform |
|---------|-------------|----------|
| Flatline | No variation | Constant 128 |
| BreathRise | Breath cycle | 0→255→0 triangle |
| CoherentPulse | Smooth HRV | 100-180 sinusoid |
| Arrhythmic | Erratic | Random spikes |

**Pattern Selection:**
> Pattern selected once, then cycles indefinitely
> Sample output changes each clock cycle

**Coherence Expectations:**
| Pattern | Expected Match Score |
|---------|---------------------|
| Flatline | High (vs TemplateFlat) |
| CoherentPulse | High (vs TemplateResonant) |
| Arrhythmic | Low (vs any template) |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with pattern generation
* VCD waveform: `RaBiometricGenerator.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Pattern generation | 0.3 per sample |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~40 LUTs
- LUT ROM: Pattern storage
- Counter: 8-bit (phase)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderBiometricVisualizer()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (pattern selector)

**Border Color**: `green-400`

---

### Notes

* Test utility - not for production biometric input
* CoherentPulse simulates ideal HRV pattern
* Arrhythmic tests rejection of poor coherence
* Feeds into RaBiometricMatcher for validation
* Useful for CI/CD testing without hardware
