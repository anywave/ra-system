# Prompt Guide Template

## Prompt ID: 32

## Name: Consent Framework (Symbolic Gate)

---

### Purpose

Self-regulating consent framework with scalar-aware symbolic validation. Honors Codex protocols of shadow gating, harmonic override, and coherence memory for consent state management.

---

### Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `isCoherent` | Bool | Biometric coherence status |
| `overrideFlag` | Bool | Emergency override signal |
| `coherenceDur` | Integer | Cycles of maintained coherence |

---

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `consentState` | Enum | Permit, Restrict, or Override |
| `coherenceDur` | Integer | Updated coherence duration |

---

### Trigger Logic

**Consent States:**
| State | Description |
|-------|-------------|
| Permit | Full consent - coherent and authorized |
| Restrict | Limited consent - incoherent or unauthorized |
| Override | Emergency bypass - highest priority |

**State Transitions:**
```
overrideFlag = True  → Override (highest priority)
isCoherent = True    → Permit
isCoherent = False   → Restrict
```

**Coherence Memory:**
> Tracks `coherenceDur` - cycles since coherence established
> Enables graduated consent escalation and shadow gating warmup

---

### Testing

* Testbench validated via `stimuliGenerator` & `outputVerifier'`
* Claude JSON prompt tests with state transitions
* VCD waveform: `RaConsentFramework.vcd`

---

### Tokenomics

| Action | Cost Units |
|--------|------------|
| Permit state | 0.5 |
| Restrict state | 0.3 |
| Override state | 2.0 |

---

### Hardware Notes

* FPGA: Optional - synthesizable via Clash
* GPU/Visual: No
* RPP: Compatible

**Resource Estimate:**
- Logic Cells: ~25 LUTs
- State machine: 3 states
- Counter: 8-bit (coherenceDur)
- Latency: 1 clock cycle

---

### Dashboard Integration

* Panel ID: Part of `renderHandshakeSimulator()`
* Docs Viewer: Linked in dashboard tooltip
* Real-time override: Yes

**Border Color**: Phase 2 core

---

### Notes

* Override bypasses normal coherence checks (emergency use)
* CoherenceDur enables "warmup" period before full Permit
* RaConsentRouter distributes state to downstream triggers
* Shadow gating: gradual permission escalation based on duration
* Part of dual-factor validation with RaHandshakeGate
