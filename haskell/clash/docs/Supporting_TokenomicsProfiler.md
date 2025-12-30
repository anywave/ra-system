# Prompt Guide Template

## Prompt ID: Supporting

## Name: Tokenomics Profiler (Cost Analyzer)

---

### Purpose

Tracks token usage and compute costs for Ra system operations. Provides real-time cost analysis for prompt execution and field synthesis operations.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `opTrigger` | Enum | Handshake, BioEmit, ChamberSpin |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `tokensUsed` | Integer | Cumulative token count |
| `computeCost` | Integer | Cumulative compute units |

---

### Trigger Logic

**Operation Costs:**
| Operation | Tokens | Compute |
|-----------|--------|---------|
| Handshake | 128 | 500 |
| BioEmit | 64 | 200 |
| ChamberSpin | 256 | 1000 |

**Accumulation:**
> Costs accumulate across session
> Reset on session clear or explicit reset

**Cost Thresholds:**
| Token Level | Status | Color |
|-------------|--------|-------|
| < 100 | Low | Green |
| < 200 | Medium | Yellow |
| >= 200 | High | Red |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with operation sequences
* VCD waveform: `RaTokenomicsProfiler.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Profile update | 0.1 |
| (Meta: profiling the profiler) |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~30 LUTs
- Accumulators: 2 (tokens, compute)
- LUT ROM: Cost table
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderCostOverlay()`, `renderTokenOverlay()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: No (read-only monitoring)

**Border Color**: `yellow-500`

---

### Notes

* Enables cost-aware field operations
* High token usage may trigger backprop gating
* Compute cost tracks GPU/FPGA utilization
* Export telemetry for offline analysis
* Part of Ra economic model for prompt budgeting
