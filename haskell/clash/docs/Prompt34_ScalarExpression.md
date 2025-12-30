# Prompt Guide Template

## Prompt ID: 34

## Name: Scalar Expression (Avatar Mapper)

---

### Purpose

Maps biometric coherence and breath phase to avatar visual expression parameters. Converts physiological state into aura intensity and limb motion vectors for avatar animation.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherence` | Integer (0-255) | Biometric coherence score |
| `breathPhase` | Bool | True = Exhale, False = Inhale |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `auraIntensity` | Integer (0-255) | Visual field luminance |
| `limbVector` | Signed Integer (-128 to +127) | Motion intent scalar |

---

### Trigger Logic

**Expression Mapping:**
| Coherence | Breath | Aura | Limb | Notes |
|-----------|--------|------|------|-------|
| >= 200 | Exhale | coherence | +40 | Full expression |
| >= 150 | Exhale | 128 | +20 | Moderate outward |
| >= 150 | Inhale | 128 | -20 | Moderate inward |
| < 150 | Any | 64 | 0 | Minimal expression |

**Limb Vector Semantics:**
- Positive: Outward/expansive motion
- Negative: Inward/contractive motion
- Zero: Neutral/still

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with expression scenarios
* VCD waveform: `RaScalarExpression.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Minimal expression | 0.4 |
| Moderate expression | 0.8 |
| Full expression | 1.2 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: Yes (avatar rendering)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~35 LUTs
- Comparators: 2 (coherence thresholds)
- MUX: 4-way (expression selection)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderExpressionOverlay()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (coherence/breath buttons)

**Border Color**: `purple-500`

---

### Notes

* Exhale phase promotes outward expression
* Inhale phase promotes inward/receptive state
* High coherence (>200) unlocks full avatar expression
* Limb vector drives skeletal animation system
* Aura intensity controls visual glow effects
