# Prompt Guide Template

## Prompt ID: 56

## Name: Tactile Control (Haptic Interface)

---

### Purpose

Interfaces with tactile sensors and haptic actuators for bidirectional physical feedback. Maps gesture inputs to haptic patterns with consent-level gating for safe physical interaction.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherence` | Integer (0-255) | Biometric coherence level |
| `gesture` | BitVector (4-bit) | Gesture code (0000-0101) |
| `consentLevel` | Integer (0-2) | Haptic permission level |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `hapticPattern` | Enum | Silent, Pulse, Buzz, Wave, Spiral, DoublePulse |
| `pwmDuty` | Integer (0-255) | PWM duty cycle |
| `duration` | Integer | Pattern duration in cycles |

---

### Trigger Logic

**Gesture Codes:**
| Code | Gesture | Haptic Response |
|------|---------|-----------------|
| 0000 | None | Silent |
| 0001 | Tap | Short pulse |
| 0010 | Hold | Sustained buzz |
| 0011 | Swipe | Wave pattern |
| 0100 | Circle | Spiral ramp |
| 0101 | Pinch | Double pulse |

**Consent Gating:**
| Level | Allowed Patterns |
|-------|------------------|
| 0 | None (all haptics disabled) |
| 1 | Pulse, Buzz only |
| 2 | Full haptic range |

**Haptic Patterns:**
| Pattern | PWM Duty | Duration |
|---------|----------|----------|
| Silent | 0% | 0 |
| Pulse | 80% | 8 cycles |
| Buzz | 50% | Sustained |
| Wave | 0-100% | 16 (ramp) |
| Spiral | 25-75% | 24 (oscillate) |
| DoublePulse | 80% | 4+4+gap |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with gesture/consent combos
* VCD waveform: `RaTactileControl.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Silent (no gesture) | 0.2 |
| Simple pattern | 0.6 |
| Complex pattern | 1.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No (haptic output)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~60 LUTs
- PWM generator: 8-bit
- Pattern sequencer: 5-bit counter
- Latency: 1 clock cycle (pattern select)

---

### Dashboard Integration

* Panel ID: N/A (integrated with consent flow)
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: No (safety-gated)

**Border Color**: N/A

---

### Notes

* Consent level 0 is fail-safe (no haptic output)
* Level 1 allows non-startling feedback only
* Level 2 enables full expressive haptics
* Gesture recognition upstream from touch sensors
* Integrates with RaBiofeedbackHarness for body-aware response
