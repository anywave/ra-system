# Prompt Guide Template

## Prompt ID: 52

## Name: Biofeedback Harness

---

### Purpose

Maps exhale-hold breath transition with high coherence to physical/energetic output signals. Triggers motion intent and haptic feedback when breath phase and coherence align for avatar physical response.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `phase` | String | Breath state: `Inhale`, `Exhale`, `Hold`, `ExhaleHold` |
| `coherence` | Integer (0-255) | Coherence signal level (230 = ~0.9 threshold) |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `motionIntent` | Bool | Whether avatar intends motion |
| `hapticPing` | Bool | Whether to trigger haptic feedback |

---

### Trigger Logic

Single activation rule combining phase and coherence:

> If `phase == "ExhaleHold"` AND `coherence >= 230`, set both `motionIntent` and `hapticPing` to `true`.

**Breath Phase Semantics:**
| Phase | Description | Can Trigger? |
|-------|-------------|--------------|
| Inhale | Breathing in | No |
| Exhale | Breathing out | No |
| Hold | Breath held (post-inhale) | No |
| ExhaleHold | Breath held after exhale | Yes (with coherence) |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with 4 test cases
* VCD waveform: `RaBiofeedbackHarness.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Normal state (inactive) | 0.6 |
| Triggered state (active) | 1.8 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~20 LUTs
- Comparators: 2 (phase match, coherence threshold)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderBiofeedbackPanel()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (editable JSON input)

**Border Color**: `lime-600`

---

### Notes

* ExhaleHold is distinct from Hold - represents post-exhale stillness
* The 230 threshold (~0.9) ensures only highly coherent states trigger
* Both outputs always match (both true or both false)
* Physiological rationale: exhale-hold + high coherence = optimal window for avatar response
