# Prompt Guide Template

## Prompt ID: 62

## Name: Avatar Field Visualizer (Glow Anchors)

---

### Purpose

Generates AuraPattern (4 glow anchors) from signature vector, chamber state, and emergence level. Produces visual field intensity values for avatar rendering when chamber is in emanating state.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `signature` | Array[4] (0-255) | 4-element signature vector |
| `chamberState` | BitVector (3-bit) | Chamber state code |
| `emergenceLevel` | Integer (0-255) | Field emergence intensity |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `auraPattern` | Array[4] (0-255) | 4 glow anchor intensities |

---

### Trigger Logic

**Chamber State Codes:**
| Code | State | Visualization |
|------|-------|---------------|
| 0b000 | Idle | Inactive |
| 0b001 | Spinning | Inactive |
| 0b010 | Stabilizing | Inactive |
| 0b101 | Emanating | Active |

**Output Calculation:**
> When `chamberState == 0b101`:
> `auraPattern[i] = signature[i] * emergenceLevel / 256`

> Otherwise:
> `auraPattern[i] = 0`

**Glow Anchor Positions:**
| Index | Position | Description |
|-------|----------|-------------|
| 0 | Crown | Top of avatar |
| 1 | Heart | Center chest |
| 2 | Solar | Solar plexus |
| 3 | Root | Base/grounding |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with state activation
* VCD waveform: `RaAvatarFieldVisualizer.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Inactive state | 0.3 |
| Emanating (active) | 1.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: Yes (primary visual output)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~40 LUTs
- Multipliers: 4 (parallel scaling)
- Comparator: 1 (state check)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderAvatarFieldControl()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (signature/state inputs)

**Border Color**: `pink-400`

---

### Notes

* Only emanating state (0b101) produces visible output
* Signature vector defines per-anchor base intensity
* Emergence level scales all anchors uniformly
* 4 anchors map to chakra-inspired visualization points
* Downstream from RaFieldSynthesisNode state machine
* Feeds into LED driver or avatar shader system
