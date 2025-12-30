# Prompt Guide Template

## Prompt ID: 17

## Name: Biofield Loopback

---

### Purpose

Biofield loopback feedback system implementing closed-loop resonant coupling between biometric input and avatar field output. Computes coherence from breath rate and HRV to drive emergence glow states.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `breathRate` | Float | Breath rate in Hz (optimal: 6.5 Hz) |
| `hrv` | Float (0-1) | Heart rate variability normalized |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `glowState` | Enum | Emergence level: None, Low, Moderate, High |
| `coherence` | Float | Raw coherence value (0-1) |

---

### Trigger Logic

**Coherence Formula:**
> `coherence = (6.5 - abs(6.5 - breathRate)) * hrv`

Optimal breath rate of 6.5 Hz maximizes coherence. HRV scales the result.

**Glow State Thresholds:**
| Coherence | Glow State |
|-----------|------------|
| < 0.3 | None |
| < 0.5 | Low |
| < 0.7 | Moderate |
| >= 0.7 | High |

---

### Testing

* Testbench validated via `Testbench.hs` (separate file)
* Claude JSON prompt tests with biometric scenarios
* VCD waveform: `testBench.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Per coherence calculation | 0.5 |
| High glow state | 1.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: Yes (glow rendering)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~40 LUTs
- Floating point: Requires FP unit or fixed-point conversion
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderLoopbackFeedback()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes

**Border Color**: N/A (core module)

---

### Notes

* 6.5 Hz is the resonant breath frequency for coherence entrainment
* HRV provides stability measure - low HRV reduces coherence
* Mealy state machine maintains glow state across cycles
* Foundation module for all downstream field synthesis
