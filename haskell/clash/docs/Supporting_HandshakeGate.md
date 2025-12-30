# Prompt Guide Template

## Prompt ID: Supporting

## Name: Handshake Gate (Dual-Factor Validation)

---

### Purpose

Dual-factor validation handshake combining symbolic consent triggers with biometric coherence verification. Wired to Phase II dashboard and field trigger cascade for secure consent gating.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `gestureID` | Integer (0-9) | Symbolic gesture identifier |
| `biometricMatch` | Bool | Biometric coherence status |
| `overrideFlag` | Bool | Emergency override signal |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `handshakeGranted` | Bool | Final grant decision |
| `passedBiometric` | Bool | Biometric check status |
| `matchedSymbol` | Bool | Symbolic ID in permitted list |
| `overrideUsed` | Bool | Override flag was enabled |

---

### Trigger Logic

**Handshake Logic:**
> `handshakeGranted = overrideFlag OR (biometricMatch AND symbolOK)`

**Permitted Trigger IDs:** `[3, 4, 7, 9]`

**Decision Matrix:**
| biometricMatch | symbolOK | override | Result |
|----------------|----------|----------|--------|
| True | True | False | Granted |
| True | False | False | Denied |
| False | True | False | Denied |
| False | False | True | Granted (override) |

**Field Cascade:**
> `fieldTriggerFromHandshake` extracts grant for downstream emitters

---

### Testing

* Testbench: Pending (manual validation)
* Claude JSON prompt tests with handshake scenarios
* VCD waveform: N/A

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Handshake check | 0.8 |
| Grant issued | 1.2 |
| Override used | 2.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~30 LUTs
- Comparators: 4 (permitted ID check)
- AND/OR gates: 3
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: `renderHandshakeSimulator()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes (gesture/biometric/override controls)

**Border Color**: `blue-400`

---

### Notes

* Dual-factor: requires BOTH biometric AND symbolic match
* Override bypasses both checks (emergency use only)
* Permitted IDs are configurable (currently [3,4,7,9])
* Feeds into RaFieldSynthesisNode for chamber activation
* Part of Phase 2 consent pipeline
