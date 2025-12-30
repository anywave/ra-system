# Prompt Guide Template

## Prompt ID: Supporting

## Name: Consent Router (Channel Splitter)

---

### Purpose

Routes consent states to downstream trigger channels. Distributes consent decisions from RaConsentFramework to biometric, gesture, and field pathways.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `consentState` | Enum | Permit, Restrict, Override |
| `coherenceGate` | Bool | Additional coherence qualifier |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `bioTrigger` | Bool | Biometric pathway enabled |
| `gestureTrigger` | Bool | Gesture pathway enabled |
| `fieldTrigger` | Bool | Field synthesis enabled |

---

### Trigger Logic

**Trigger Channels:**
| Channel | Activation Condition |
|---------|---------------------|
| bioTrigger | Permit OR Override |
| gestureTrigger | Override only |
| fieldTrigger | Permit AND coherenceGate |

**Truth Table:**
| State | Gate | bioTrigger | gestureTrigger | fieldTrigger |
|-------|------|------------|----------------|--------------|
| Permit | False | True | False | False |
| Permit | True | True | False | True |
| Restrict | False | False | False | False |
| Restrict | True | False | False | False |
| Override | Any | True | True | False |

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with routing scenarios
* VCD waveform: `RaConsentRouter.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Routing decision | 0.3 |
| Multi-channel activation | 0.5 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~20 LUTs
- Comparators: 3 (state checks)
- AND/OR gates: 4
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: Part of consent pipeline visualization
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: No (follows consent state)

**Border Color**: N/A (internal routing)

---

### Notes

* Restrict state blocks ALL pathways (fail-safe)
* Override enables bio + gesture (emergency only)
* Field trigger requires BOTH Permit AND coherenceGate
* Gesture pathway most restricted (Override-only)
* Downstream from RaConsentFramework
* Upstream of individual trigger handlers
