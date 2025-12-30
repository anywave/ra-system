# Prompt Guide Template

## Prompt ID: 64

## Name: Music Chamber Harmonics

---

### Purpose

Maps coherence band levels to Solfeggio overtone frequencies for music chamber harmonics. Produces frequency values for audio synthesis based on coherence modulation of sacred frequencies.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherenceBand` | Array[4] (0-255) | 4-element coherence band vector |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `overtoneFrequencies` | Array[4] (Hz) | 4-element frequency output vector |

---

### Trigger Logic

Harmonic mapping formula for each band:

> `overtoneFreq[i] = baseFreq[i] + (baseFreq[i] * coherenceBand[i]) / 256`

**Solfeggio Base Frequencies:**
| Index | Base (Hz) | Note | Association |
|-------|-----------|------|-------------|
| 0 | 396 | G | Liberation from fear |
| 1 | 417 | G# | Undoing situations |
| 2 | 528 | C | Transformation/DNA |
| 3 | 639 | E | Connecting relations |

**Coherence Modulation:**
- coherence = 0: Output = base frequency
- coherence = 128: Output = 1.5× base
- coherence = 255: Output = ~2× base

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with frequency verification
* VCD waveform: `RaMusicChamberHarmonics.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Per band processed | 0.4 |
| Full 4-band mapping | 1.6 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No (audio output)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~100 LUTs
- Multipliers: 4 (one per band)
- Latency: 1 clock cycle (parallel)

---

### Dashboard Integration

* Panel ID: `renderMusicChamberModule()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (coherence band sliders)

**Border Color**: `yellow-500`

---

### Notes

* Frequencies are 16-bit unsigned (0-65535 Hz range)
* Higher coherence creates richer harmonic overtones
* Downstream feeds into RaSonicEmitter for audible output
* Extended Solfeggio (741, 852 Hz) not yet implemented
