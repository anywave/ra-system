# Prompt Guide Template

## Prompt ID: 54

## Name: Symbolic Coherence Operations

---

### Purpose

Defines compositional symbolic operations for transforming emergence conditions. Supports phase shifting, angle inversion, and threshold gating with fixed-point arithmetic for DSL-based field manipulation.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `composition` | String | DSL expression: `PhaseShift(fx) ○ InvertAngle ○ GateThreshold(fx)` |
| `coherence` | Integer (0-255) | Input coherence level |
| `angle` | Integer (0-255) | Input phase angle |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherence` | Integer (0-255) | Transformed coherence |
| `angle` | Integer (0-255) | Transformed angle |

---

### Trigger Logic

Three symbolic operations compose left-to-right:

**PhaseShift(fx):**
> Add phase shift to coherence: `coherence' = min(255, coherence + fx * 255)`

**InvertAngle:**
> Mirror angle about midpoint: `angle' = 255 - angle`

**GateThreshold(fx):**
> Zero coherence below threshold: `coherence' = coherence >= fx ? coherence : 0`

**Fixed-Point Values:**
| Decimal | Fixed (0-255) | Use |
|---------|---------------|-----|
| 0.4 | 102 | 40% gate |
| 0.5 | 128 | Half |
| 0.618 | 158 | Golden ratio |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with 4 input conditions
* VCD waveform: `RaSymbolicCoherenceOps.vcd`

---

### Tokenomics

| Operation | Cost Units |
|-----------|------------|
| PhaseShift | 1.2 |
| InvertAngle | 0.8 |
| GateThreshold | 1.0 |

Total cost = sum of operations in composition.

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~60 LUTs (composition dependent)
- Latency: N cycles for N operations (pipelined)

---

### Dashboard Integration

* Panel ID: `renderSymbolicOps()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (editable DSL + JSON input)

**Border Color**: `purple-600`

---

### Notes

* Operations compose left-to-right (first applied first)
* Saturation prevents overflow on PhaseShift
* GateThreshold can zero out low-coherence inputs entirely
* Golden ratio (0.618) is commonly used for harmonic phase shifts
