# Prompt Guide Template

## Prompt ID: 41

## Name: Visualizer Shell (RGB Renderer)

---

### Purpose

Renders visual shell feedback from chamber state, coherence, and sync status to RGB color output. Provides real-time visual feedback for field synthesis state via LED/display driver.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `chamberState` | Enum | Idle, Spinning, Stabilizing, Emanating |
| `syncState` | Enum | Desync, Aligning, Locked, Drifting |
| `coherence` | Integer (0-255) | Coherence level for intensity |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `red` | Integer (0-255) | Red channel |
| `green` | Integer (0-255) | Green channel |
| `blue` | Integer (0-255) | Blue channel |

---

### Trigger Logic

**Chamber State Colors:**
| State | Base RGB | Description |
|-------|----------|-------------|
| Idle | (0, 0, 32) | Deep blue (dim) |
| Spinning | (0, 64, 128) | Cyan glow |
| Stabilizing | (128, 64, 255) | Purple pulse |
| Emanating | (255, 128, 64) | Golden radiance |

**Sync State Modulation:**
| Sync State | Effect |
|------------|--------|
| Desync | Flash red overlay (50%) |
| Aligning | Pulse brightness (25-100%) |
| Locked | Steady (100%) |
| Drifting | Subtle fade (75-100%) |

**Coherence Intensity:**
> `finalBrightness = baseColor * syncModulation * (coherence / 255)`

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with state combinations
* VCD waveform: `RaVisualizerShell.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Idle state | 0.3 |
| Active state | 0.6 |
| Emanating | 1.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: Yes (primary visual output)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~50 LUTs
- Multipliers: 3 (RGB scaling)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: Part of `renderChamberVisual()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes

**Border Color**: Phase 1 visual

---

### Notes

* RGB output drives LED strips or display pixels
* Sync state overlay indicates distributed system health
* Desync causes red flash as warning
* Coherence scales overall brightness
* Consumes from RaFieldSynthesisNode and RaChamberSync
