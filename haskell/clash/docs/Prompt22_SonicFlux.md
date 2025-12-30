# Prompt Guide Template

## Prompt ID: 22

## Name: Sonic Flux Harmonic Driver

---

### Purpose

Real-time harmonic driver mapping coherence levels to audio scalar output states. Converts field coherence into audible frequency domains for sonic entrainment and feedback.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherence` | Float (0-1) | Coherence level from biofield |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `audioState` | Enum | Output: Silence, HarmonicLow, HarmonicMid, HarmonicHigh |
| `amplitude` | Float (0-1) | Audio amplitude scalar |

---

### Trigger Logic

**Audio State Mapping:**
| Coherence | Output State | Amplitude |
|-----------|--------------|-----------|
| < 0.30 | Silence | 0.0 |
| < 0.55 | HarmonicLow | 0.3 |
| < 0.80 | HarmonicMid | 0.6 |
| >= 0.80 | HarmonicHigh | 0.9 |

**PWM Pipeline (RaSonicEmitter):**
> `Coherence → sonicFluxMapper → AudioState → audioEmitter → Amplitude → scalarToPWM → PWM`

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with coherence sweeps
* VCD waveform: `RaSonicFlux.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Silence state | 0.3 |
| Harmonic output | 0.8 |
| High harmonic | 1.2 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No (audio output)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~30 LUTs
- Comparators: 3 (threshold checks)
- Latency: 1 clock cycle

**PWM Driver Specs:**
- Resolution: 8-bit (256 levels)
- Duty cycle: amplitude × 255

---

### Dashboard Integration

* Panel ID: `renderSonicFluxControl()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes

**Border Color**: N/A (Phase 1 core)

---

### Notes

* RaSonicEmitter extends this with full PWM hardware pipeline
* RaPWMDriver converts scalar amplitude to duty cycle
* Applications: LED modulation, haptic vibration, Solfeggio entrainment
* RaPWMMultiFreqTest adds multi-harmonic blend with biometric override
