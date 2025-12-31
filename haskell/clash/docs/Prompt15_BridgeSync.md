# Prompt 15: Multi-User Harmonic Entanglement via Scalar Bridges

## Overview

This module enables consensual scalar coupling between multiple users through entanglement bridges. Users with compatible harmonics and aligned phi phases can share fragment experiences, with four bridge modes supporting different relationship dynamics.

## Architecture

```
+-------------------------------------------------------------------------+
|                    MULTI-USER ENTANGLEMENT SYSTEM                       |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------------+   +-----------------+   +------------------+        |
|  | ScalarUser A   |-->| Harmonic        |-->| Consent          |------>|
|  | ScalarUser B   |   | Compatibility   |   | Validation       | Bridge|
|  +----------------+   +-----------------+   +------------------+        |
|         |                    |                     |                    |
|         v                    v                     v                    |
|  +----------------+   +-----------------+   +------------------+        |
|  | Entanglement   |   | Phase           |   | Safety           |        |
|  | Score          |   | Alignment       |   | Enforcement      |        |
|  +----------------+   +-----------------+   +------------------+        |
|         |                    |                     |                    |
|         v                    v                     v                    |
|  +----------------+   +-----------------+   +------------------+        |
|  | ScalarBridge   |   | Fragment        |   | RaCohereNet      |        |
|  | (4 modes)      |   | Emergence       |   | (Group Sync)     |        |
|  +----------------+   +-----------------+   +------------------+        |
|                                                                         |
+-------------------------------------------------------------------------+
```

## Codex References

| Source | Application |
|--------|-------------|
| **Ra.Gates** | Consent gating protocols |
| **Ra.Emergence** | Shared fragment emergence |
| **Ra.Identity** | User identity validation |
| **Prompt 11** | Group coherence (RaCohereNet) |
| **Prompt 12** | Shadow consent for shared fragments |
| **Prompt 14** | Shared lucid navigation |

## Consent States

### Unified Consent Model (Prompt 12 + 15)

| State | Value | Description |
|-------|-------|-------------|
| NONE | 0 | No consent given |
| PRIVATE | 1 | Solo operation only |
| THERAPEUTIC | 2 | Operator-assisted only |
| ENTANGLED | 3 | Peer-to-peer bridging allowed |
| WITHDRAWN | 4 | Previously consented, now revoked |
| EMERGENCY | 5 | Immediate safety override |

## Harmonic Compatibility

### Compatibility Check (Stricter AND Rule)

```python
def check_harmonic_compatibility(user_a, user_b):
    l_match = user_a.harmonic[0] == user_b.harmonic[0]
    m_near = abs(user_a.harmonic[1] - user_b.harmonic[1]) <= 1
    return l_match and m_near  # AND, not OR
```

**Requirements:**
- l indices must match exactly
- m indices must differ by at most 1

## Entanglement Score

### Calculation Formula

```python
def calculate_entanglement_score(user_a, user_b):
    coherence_delta = abs(user_a.coherence - user_b.coherence)
    phase_delta = abs(user_a.phi_phase - user_b.phi_phase)
    raw_score = 1.0 - (coherence_delta + phase_delta / 2.0)
    return max(0.0, raw_score)
```

**Thresholds:**
- Minimum stable score: 0.60
- PHI_THRESHOLD: 0.03 (phase alignment)
- COHERENCE_TOLERANCE: 0.08

## Bridge Modes

### Four Entanglement Modes

| Mode | Emergence Type | Description |
|------|----------------|-------------|
| MIRROR | shared | Identical experience for both users |
| COMPLEMENT | dual-reflect | Dual perspectives, complementary views |
| ASYMMETRIC | relay-access | Leader/follower relationship |
| BROADCAST | multicast-sync | One-to-many distribution |

### Bridge State Machine

```
IDLE --> ENTANGLED --> SUSPENDED --> TERMINATED
  ^          |              |
  |__________|______________|
       (re-create)
```

## Safety Enforcement

### Requirements

- Maximum latency: 250ms
- Continuous consent validation
- Override veto support
- Graceful and emergency disentanglement

### Enforcement Logic

```python
def enforce_safety(bridge, user_map):
    if not bridge.active or not bridge.stable:
        return False, "Bridge not active/stable"
    for uid in bridge.users:
        user = user_map[uid]
        if user.override_veto:
            return False, f"User {uid} has veto active"
        if user.consent_state != ConsentState.ENTANGLED:
            return False, f"User {uid} consent changed"
    return True, "Safe"
```

## RaCohereNet (Group Synchronization)

### Group Requirements

- Minimum 2 users, maximum 8
- Group coherence >= 0.72
- Phi deviation <= 0.04

### Group Metrics

```python
def group_stabilized(users):
    avg_coherence = sum(u.coherence for u in users) / len(users)
    phases = [u.phi_phase for u in users]
    phi_deviation = max(phases) - min(phases)

    stable = avg_coherence >= 0.72 and phi_deviation <= 0.04
    return {"status": "STABILIZED" if stable else "UNSTABLE", ...}
```

### Entanglement Matrix

N x N matrix of pairwise viable bridges:
- 1 = bridge possible
- 0 = incompatible

## Shadow Fragment Sharing

### Requirements

- Both users: THERAPEUTIC or ENTANGLED consent
- Combined average coherence >= 0.75

```python
def can_share_shadow(user_a, user_b):
    valid_consent = {ConsentState.THERAPEUTIC, ConsentState.ENTANGLED}
    consent_ok = user_a.consent_state in valid_consent and \
                 user_b.consent_state in valid_consent
    avg_coherence = (user_a.coherence + user_b.coherence) / 2
    return consent_ok and avg_coherence >= 0.75
```

## Clash Module: RaBridgeSync.hs

### Key Types

```haskell
data ConsentState = ConsentNone | ConsentPrivate | ConsentTherapeutic
                  | ConsentEntangled | ConsentWithdrawn | ConsentEmergency

data BridgeMode = ModeMirror | ModeComplement | ModeAsymmetric | ModeBroadcast

data BridgeState = BridgeIdle | BridgeEntangled | BridgeSuspended | BridgeTerminated

data UserSnapshot = UserSnapshot
  { userId, uCoherence, uHarmonicL, uHarmonicM, uConsent, uPhiPhase, uVeto }
```

### Core Functions

| Function | Purpose |
|----------|---------|
| `checkHarmonicCompat` | Verify l-match AND m-near |
| `calculateEntanglementScore` | Score with clamping |
| `entanglementPossible` | Full validation with reason |
| `checkBridgeSafety` | Continuous safety check |
| `emergenceFromMode` | Mode to emergence type |
| `calcFluxBoost` | Emergence flux amplifier |
| `calcGroupCoherence` | N-user average |
| `checkGroupPhiStability` | Max deviation check |

### Synthesis Target

```haskell
{-# ANN bridgeSyncTop (Synthesize
  { t_name = "bridge_sync_top"
  , t_inputs = [ PortName "clk", PortName "rst", PortName "en", ... ]
  , t_output = PortProduct "output" [ ... ]
  }) #-}
```

## Python Test Harness

### test_entanglement.py

**Test Scenarios (10 total):**

| Test | Description | Expected |
|------|-------------|----------|
| `harmonic_compatibility` | l-match AND m-diff<=1 | Correct compatibility |
| `entanglement_score` | Score calculation | Valid clamping |
| `bridge_creation` | Create with consent | ENTANGLED status |
| `safety_enforcement` | Veto and consent changes | Safety detected |
| `emergence_trigger` | Fragment emergence | Flux boost calculated |
| `shadow_sharing` | Dual consent + coherence | Correct gating |
| `group_stabilization` | Group coherence check | STABILIZED status |
| `entanglement_matrix` | Pairwise viability | Matrix generated |
| `group_bridge_creation` | Full group bridge | Leader selected |
| `graceful_disentangle` | Suspend bridge | SUSPENDED status |

**Usage:**

```bash
python test_entanglement.py        # Full suite
python test_entanglement.py --demo # With entanglement demo
python test_entanglement.py --json # JSON output
```

## Integration Points

### Upstream Dependencies

| Module | Integration |
|--------|-------------|
| User identity system | User snapshots |
| Consent manager | Real-time consent state |
| Coherence tracker | Coherence values |

### Downstream Outputs

| Consumer | Data |
|----------|------|
| Fragment sharing | Emergence type, flux |
| Group dashboard | Entanglement matrix |
| Safety monitor | Bridge state, veto status |

## Hardware Resources

| Platform | LUTs | DSP | BRAM |
|----------|------|-----|------|
| Xilinx Artix-7 | ~550 | 1 | 0 |
| Intel Cyclone V | ~600 | 1 | 0 |
| Lattice ECP5 | ~650 | 1 | 1 |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-30 | Initial implementation with full spec |
