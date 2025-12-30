# Prompt Guide Template

## Prompt ID: Supporting

## Name: Field Synthesis Node (Chamber Cascade)

---

### Purpose

Chamber state machine activated by handshakeGranted signal. Controls field synthesis progression through activation phases from Idle to Emanating state.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `granted` | Bool | Handshake grant signal |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `chamberState` | Enum | Idle, Spinning, Stabilizing, Emanating |
| `glowIntensity` | Integer (0-255) | Current glow level |

---

### Trigger Logic

**Chamber States:**
| State | Glow | Description |
|-------|------|-------------|
| Idle | 0 | Awaiting grant |
| Spinning | 64 | Initial spin-up |
| Stabilizing | 192 | Coherence stabilization |
| Emanating | 255 | Full emission |

**State Transitions:**
```
granted=True:  Idle → Spinning → Stabilizing → Emanating (holds)
granted=False: Any → Idle (reset)
```

**Transition Timing:**
- Each state holds for N cycles before advancing
- Grant must remain True throughout cascade
- False grant immediately resets to Idle

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with grant sequences
* VCD waveform: `RaFieldSynthesisNode.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Idle state | 0.2 |
| Spinning | 0.5 |
| Stabilizing | 0.8 |
| Emanating | 1.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: Yes (glow output)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~35 LUTs
- State machine: 4 states
- Counter: 8-bit (phase timing)
- Latency: Multi-cycle (state progression)

---

### Dashboard Integration

* Panel ID: `renderChamberVisual()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (via handshake simulation)

**Border Color**: `purple-400`

---

### Notes

* Mealy machine with registered outputs
* Grant withdrawal causes immediate reset (fail-safe)
* Emanating state holds indefinitely while granted
* Glow intensity feeds into RaVisualizerShell
* Downstream from RaHandshakeGate
* Upstream of RaChamberSync for multi-chamber coordination
