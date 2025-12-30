# Prompt Guide Template

## Prompt ID: 31

## Name: Consent Transformer (Multi-Core)

---

### Purpose

Distributed consent logic across multiple avatar threads with quorum voting and biometric gating. Aggregates consent votes from N cores with coherence-based activation thresholds.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `coherenceScore` | Integer (0-255) | Biometric coherence level |
| `auraIntensity` | Integer (0-255) | Avatar aura intensity |
| `votes` | Array[N] Bool | Consent votes from each core |
| `quorum` | Integer (0-100) | Required percentage for consent |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `consentGranted` | Bool | True if quorum % of active votes agree |
| `fieldEntropy` | Integer | Number of dissenting votes |
| `activeVotes` | Integer | Number of qualifying True votes |

---

### Trigger Logic

**Activation Requirements:**
> `coherenceScore >= 180` AND `auraIntensity >= 128`

If either threshold not met, all votes suspended (treated as False).

**Quorum Calculation:**
> `consentGranted = (activeVotes / totalVotes) * 100 >= quorum`

**Field Entropy:**
> `fieldEntropy = totalVotes - activeVotes` (count of False/suspended votes)

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with voting scenarios
* VCD waveform: `RaConsentTransformer.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Vote aggregation | 0.5 per core |
| Consent granted | 1.5 |
| Consent denied | 0.8 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~60 LUTs (scales with N)
- Comparators: 2 + N (thresholds + votes)
- Counters: 2 (active, entropy)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderConsentOverlay()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (vote simulation)

**Border Color**: `red-500`

---

### Notes

* Multi-avatar coordination requires quorum consensus
* Low coherence suspends ALL votes (fail-safe)
* Field entropy indicates system disagreement level
* High entropy may trigger re-synchronization
* Downstream: feeds into RaConsentFramework for state machine
