# Prompt Guide Template

## Prompt ID: 35

## Name: Tesla Coherent Field Transfer

---

### Purpose

Simulates scalar packet transmission from Avatar A to Avatar B with coherence preservation. Validates transfer latency and signal integrity for distributed field synthesis across avatar network.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherence` | Integer (0-255) | Source coherence score |
| `signal` | Array[4] (0-255) | 4-element scalar harmonic vector |
| `send` | Boolean | Transfer trigger (rising edge) |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `signal` | Array[4] (0-255) | Received scalar harmonic vector |
| `latency` | Integer | Transfer time in cycles |
| `ok` | Boolean | True if coherence preserved |
| `tokensUsed` | Integer | Token consumption |
| `computeCost` | Integer | Compute units consumed |

---

### Trigger Logic

**Transfer Constraints:**
> `TransferLatency < 300 cycles` (simulated at 1ms/cycle)
> `|destCoherence - srcCoherence| <= 1` (coherence invariant)

**State Machine:**
> IDLE → (send=True) → TRANSMITTING → (latency reached) → COMPLETE

**Latency Calculation:**
> `latency = baseLatency + coherence-dependent jitter`
> Higher coherence = lower latency

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with integrity verification
* VCD waveform: `RaFieldTransferBus.vcd`

---

### Tokenomics

| Action | Token Cost | Compute Cost |
|--------|------------|--------------|
| Successful transfer | 128 | 500 |
| Failed transfer | 64 | 250 |

**Backprop Gating:**
- If `!ok`: defer with "Signal integrity failure"
- If `tokensUsed > 200`: defer with "Token strain"

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~150 LUTs
- FIFO buffer: 4 × 8-bit
- Counter: 9-bit
- Latency: Variable (30-300 cycles)

---

### Dashboard Integration

* Panel ID: `renderTransferOverlay()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (Simulate Transfer button)

**Border Color**: `blue-400`

---

### Notes

* Signal vector passes through unchanged if coherence preserved
* Low coherence causes more jitter (longer, less predictable latency)
* Integrity check ensures field coherence across distributed systems
* Token/compute tracking enables cost-aware field operations
