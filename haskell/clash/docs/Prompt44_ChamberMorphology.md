# Prompt Guide Template

## Prompt ID: 44

## Name: Chamber Morphology System

---

### Purpose

Models chamber morphology transitions based on coherence and instability thresholds. When coherence drops and instability rises, triggers rapid collapse to toroidal form for visual feedback and state management.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherence` | Integer (0-255) | Coherence level (~0.39 threshold at 100) |
| `instability` | Integer (0-255) | Instability level (~0.30 threshold at 77) |
| `form` | String | Current form: `Sphere`, `Toroid`, `Cube` |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `newForm` | String | Resulting geometric form |
| `event` | String | Transition event: `NoChange` or `RapidCollapse` |

---

### Trigger Logic

Collapse triggered when BOTH thresholds are crossed:

> If `coherence < 100` AND `instability > 77`, set `newForm = Toroid` and `event = RapidCollapse`.
> Otherwise, maintain current form with `event = NoChange`.

**Chamber Forms:**
| Form | Description |
|------|-------------|
| Sphere | Default stable form |
| Toroid | Collapsed form (low coherence) |
| Cube | Crystallized form (high coherence) |

**Advanced Mode** (optional):
> If `coherence > 200` AND `instability < 30`, crystallize to `Cube`.

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with 6 edge cases
* VCD waveform: `RaChamberMorphology.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| NoChange (stable) | 0.5 |
| RapidCollapse (transition) | 1.5 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: Yes (form rendering)
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~30 LUTs
- Comparators: 2 (coherence, instability thresholds)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderMorphologyFallback()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (editable JSON input)

**Border Color**: `orange-600`

---

### Notes

* Edge case: At exactly threshold (100, 77) - no collapse (requires strict < and >)
* Collapse is one-way until coherence recovers
* Form maintains through transitions when stable
* Future: Add crystallization to Cube for very high coherence
