# Prompt 12: Consent-Gated Shadow Harmonics (Patch 12B/12C)

## Overview

This module implements consent-gated access to shadow fragment harmonics, ensuring that sensitive psychodynamic patterns are only processed when proper therapeutic consent, licensed operator presence, and coherence thresholds are met. It generates therapeutic feedback to guide safe shadow integration work.

**Patch 12B** adds:
- Session persistence via tokens for multi-session tracking
- Emotional charge decay (5% per cycle, floor at 0.15)
- Integration count tracking
- Crypto override interface stub for Prompt 33

**Patch 12C** adds:
- Multi-fragment priority queue (ShadowQueue)
- Priority ordering by emotional charge, harmonic mismatch, alpha

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SHADOW CONSENT SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐   ┌────────────────┐   ┌─────────────────┐          │
│  │  Shadow        │──▶│   Ra.Gates     │──▶│  Therapeutic    │──▶ Out  │
│  │  Fragment      │   │   Consent      │   │  Feedback       │          │
│  │  Schema        │   │   Gating       │   │  Generator      │          │
│  └────────────────┘   └────────────────┘   └─────────────────┘          │
│         │                    │                    │                      │
│         ▼                    ▼                    ▼                      │
│  ┌────────────────┐   ┌────────────────┐   ┌─────────────────┐          │
│  │   Session      │◀──│   Gating       │   │   Safety        │          │
│  │   State        │   │   Result       │   │   Alert         │          │
│  │   Tracking     │   │   (ALLOW/BLOCK)│   │   Generation    │          │
│  └────────────────┘   └────────────────┘   └─────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Codex References

| Source | Application |
|--------|-------------|
| **REICH_ORGONE_ACCUMULATOR.md** | DOR/armoring detection, shadow inversion states |
| **KAALI_BECK_BIOELECTRICAL.md** | Coherence gating threshold (0.66 safe access) |

## Shadow Fragment Schema

### Data Structure

```haskell
data ShadowFragment = ShadowFragment
  { fragmentId       :: FragmentId       -- Unique identifier
  , fragmentForm     :: Unsigned 8       -- Spherical harmonic l value
  , inversion        :: InversionState   -- Normal/Inverted/Oscillating/Chaotic
  , alpha            :: Fixed8           -- Normalized amplitude 0.0-1.0
  , consentState     :: ConsentState     -- Therapeutic consent status
  , shadowType       :: ShadowType       -- Repressed/Projected/Ancestral/Collective
  , requiresOverride :: Bool             -- Needs emergency override
  , originFragment   :: FragmentId       -- Link to originating fragment
  , harmonicMismatch :: Fixed8           -- Distance from dominant mode
  , emotionalCharge  :: Fixed8           -- Emotional intensity 0.0-1.0
  }
```

### Shadow Types

| Type | Description | Grounding Prompt |
|------|-------------|------------------|
| REPRESSED | Hidden/denied aspects | "Feel your feet on the ground. You are safe here." |
| PROJECTED | Externalized onto others | "Notice what you see in others. It may reflect within." |
| ANCESTRAL | Inherited patterns | "Honor what came before. You are not bound by it." |
| COLLECTIVE | Archetypal/collective | "You are part of something larger. Breathe with it." |

### Consent States

| State | Value | Description |
|-------|-------|-------------|
| NONE | 0 | No therapeutic consent |
| THERAPEUTIC | 1 | Active consent for shadow work |
| WITHDRAWN | 2 | Previously consented, now withdrawn |
| EMERGENCY | 3 | Emergency override active |

## Consent Gating Logic (Ra.Gates)

### Conditions for ALLOW

All four conditions must be met:

1. **Consent State is THERAPEUTIC** - User has explicitly consented
2. **Licensed Operator Present** - Qualified guide available
3. **Override Active** (if required) - Emergency override for deep work
4. **Coherence >= 0.66** - Kaali/Beck safe access threshold

### Gating Results

| Result | Condition Failed | Response |
|--------|------------------|----------|
| ALLOW | None | Emergence permitted |
| BLOCK_NO_CONSENT | Consent != THERAPEUTIC | Request consent |
| BLOCK_NO_OPERATOR | No licensed operator | Require guide |
| BLOCK_LOW_COHERENCE | Coherence < 0.66 | Build coherence first |
| BLOCK_NO_OVERRIDE | Requires override but none | Request authorization |

### Implementation

```haskell
shouldAllowEmergence :: ShadowFragment -> SessionState -> GatingResult
shouldAllowEmergence fragment session
  | consentState fragment /= TherapeuticConsent = BlockNoConsent
  | not (licensedOperator session)              = BlockNoOperator
  | requiresOverride fragment && not (overrideActive session) = BlockNoOverride
  | sessionCoherence session < coherenceSafeAccess = BlockLowCoherence
  | otherwise                                   = Allow
```

## Therapeutic Feedback Generator

### Feedback Components

| Component | Purpose |
|-----------|---------|
| Grounding Prompt | Body awareness, safety anchoring |
| Context Prompt | Explanation of current state |
| Reflection Prompt | Insight question or warning |

### Feedback Intensity Levels

| Level | Trigger | Audio | Visual |
|-------|---------|-------|--------|
| GENTLE | Default blocked state | TONE | MANDALA |
| MODERATE | Allowed emergence | BINAURAL | FLOWER |
| FIRM | High emotional charge | PULSE | SPIRAL |
| URGENT | Safety alert | ALARM | ALERT |

### High Charge Warning

When `emotionalCharge >= 0.75`:
- Intensity escalates to FIRM
- Reflection prompt includes charge warning
- Safety alert triggered
- Visual glyph switches to SPIRAL (recalibration)

## Session State Tracking

### State Structure

```haskell
data SessionState = SessionState
  { sessionCoherence  :: Fixed8      -- Current coherence level
  , licensedOperator  :: Bool        -- Operator present
  , overrideActive    :: Bool        -- Emergency override
  , overrideSource    :: OverrideId  -- Who authorized
  , shadowProgress    :: Fixed8      -- Session progress 0.0-1.0
  , resonanceDelta    :: Signed 16   -- Change since last cycle
  }
```

### Coherence Updates

| Condition | Update |
|-----------|--------|
| ALLOW (low charge) | +5% coherence boost |
| ALLOW (high charge) | +5% × (1 - charge) |
| BLOCK | -2% coherence decay |

### Progress Tracking

- Each successful emergence adds 10% to shadow progress
- Progress capped at 100%
- Coherence capped at 100%, floored at 0%

## Safety Considerations

### Safety Alert Triggers

| Condition | Response |
|-----------|----------|
| Emotional charge >= 75% | Alert + FIRM feedback |
| Coherence < 30% | Alert + grounding |
| Inversion = CHAOTIC | Alert + stabilization |

### Emergency Override

Override can be applied by:
- Licensed therapist
- Emergency system
- Session supervisor

Override records source and reason for audit trail.

## Clash Module: RaShadowConsent.hs

### Key Types

```haskell
data ShadowConsentOutput = ShadowConsentOutput
  { gatingResult     :: GatingResult
  , emergenceAllowed :: Bool
  , feedback         :: TherapeuticFeedback
  , sessionUpdate    :: SessionState
  , safetyAlert      :: Bool
  , cycleCount       :: Unsigned 16
  }
```

### Core Functions

| Function | Purpose |
|----------|---------|
| `shouldAllowEmergence` | Ra.Gates consent gating logic |
| `generateFeedback` | Therapeutic feedback generation |
| `updateSession` | Session state progression |
| `processShadowConsent` | Main processing pipeline |

### Synthesis Target

```haskell
{-# ANN shadowConsentTop (Synthesize
  { t_name = "shadow_consent_unit"
  , t_inputs = [PortName "clk", PortName "rst", PortName "en"
               , PortName "fragment", PortName "session"]
  , t_output = PortProduct "output" [...]
  }) #-}
```

## Python Test Harness

### test_shadow_consent.py

**Test Scenarios:**

| Scenario | Description | Expected |
|----------|-------------|----------|
| `therapeutic_consent_allow` | Full consent + operator | ALLOW, FLOWER glyph |
| `no_consent_blocked` | Missing consent | BLOCK_NO_CONSENT |
| `no_operator_blocked` | No licensed operator | BLOCK_NO_OPERATOR |
| `low_coherence_blocked` | Coherence < 0.66 | BLOCK_LOW_COHERENCE |
| `override_required_blocked` | Needs override, none active | BLOCK_NO_OVERRIDE |
| `override_allows` | Override active | ALLOW |
| `high_charge_warning` | Charge >= 0.75 | ALLOW + safety alert |
| `shadow_type_prompts` | Each type | Unique grounding prompts |

**CLI Dashboard Output:**

```
======================================================================
               CONSENT-GATED SHADOW HARMONICS - Cycle 3
======================================================================

 Shadow Fragment:
   ID: 42
   Type: ANCESTRAL
   Consent: THERAPEUTIC
   Inversion: NORMAL
   Alpha: 0.650
   Emotional Charge: 0.550

 Consent Gating:
   Result: [ALLOW]
   Emergence: PERMITTED

 Session State:
   Coherence: [######################--------] 0.742
   Progress:  [############------------------] 0.400
   Delta: +0.022
   Operator: YES
   Override: NO

 Therapeutic Feedback:
   Intensity: MODERATE
   Audio: BINAURAL
   Glyph: FLOWER

   Grounding: "Honor what came before. You are not bound by it."
   Context: "This shadow aspect is ready for gentle exploration."
   Reflection: "What does this shadow aspect teach you?"
======================================================================
```

**Usage:**

```bash
# Full test suite
python test_shadow_consent.py

# Multi-cycle simulation
python test_shadow_consent.py --sim

# JSON output for integration
python test_shadow_consent.py --json
```

## Integration Points

### Upstream Dependencies

| Module | Integration |
|--------|-------------|
| Prompt 10 (Scalar Resonance) | Coherence input |
| Prompt 11 (Group Coherence) | Multi-avatar context |
| Prompt 9 (Orgone Scalar) | Inversion state detection |

### Downstream Outputs

| Consumer | Data |
|----------|------|
| Prompt 31 (Consent Transformer) | Gating results for quorum |
| Audio synthesis | Feedback audio cues |
| Visual renderer | Glyph type and intensity |
| Session logger | Progress and alerts |

## Hardware Resources

| Platform | LUTs | DSP | BRAM |
|----------|------|-----|------|
| Xilinx Artix-7 | ~450 | 2 | 0 |
| Intel Cyclone V | ~500 | 2 | 0 |
| Lattice ECP5 | ~550 | 1 | 1 |

## Patch 12B: Session Persistence & Emotional Charge Decay

### Session Token

Each session receives a unique token for multi-session tracking:

```haskell
type SessionToken = Unsigned 64
```

The token persists across cycles and is stored on the fragment after processing.

### Emotional Charge Decay

After each successful integration, emotional charge decays by 5%:

```haskell
applyChargeDecay :: Fixed8 -> Fixed8
applyChargeDecay charge =
  let decayed = (resize charge * 243) `shiftR` 8  -- 0.95 * 256 = 243
  in max emotionalChargeFloor decayed
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| Decay factor | 0.95 | 5% reduction per cycle |
| Floor | 0.15 (38/256) | Minimum charge after decay |

### Integration Count

Tracks cumulative successful integrations:

```haskell
integrationCount :: Unsigned 16  -- Increments on each ALLOW
lastSessionToken :: SessionToken -- Links fragment to session
```

## Patch 12C: Multi-Fragment Priority Queue

### ShadowQueue

Manages multiple shadow fragments with priority ordering:

```haskell
data ShadowQueue = ShadowQueue
  { queueEntries :: Vec 8 QueueEntry
  , queueCount   :: Unsigned 4
  , activeIndex  :: Unsigned 4
  }
```

### Priority Calculation

Fragments are ordered by:
1. **Emotional charge** (DESC) - Higher charge = higher priority
2. **Harmonic mismatch** (DESC) - Greater mismatch = higher priority
3. **Alpha** (ASC) - Lower alpha = safer emergence first

```haskell
calculatePriority :: ShadowFragment -> PriorityScore
calculatePriority frag =
  (charge `shiftL` 8) + (mismatch `shiftL` 4) + (256 - alpha)
```

### Queue Operations

| Function | Description |
|----------|-------------|
| `addToQueue` | Insert fragment with computed priority |
| `nextFromQueue` | Pop highest-priority fragment |
| `initQueue` | Create empty queue (8 slots) |

## Prompt 33 Interface: Crypto Override

Stub for signature verification (full implementation in Prompt 33):

```haskell
data ConsentOverride = ConsentOverride
  { overrideSource    :: OverrideSource
  , overrideReason    :: Unsigned 8
  , overrideTimestamp :: Timestamp
  , overrideValid     :: Bool
  , signatureHash     :: Unsigned 64   -- Prompt 33
  , operatorKeyId     :: Unsigned 32   -- Prompt 33
  }

verifySignature :: ConsentOverride -> Bool
verifySignature override =
  signatureHash override /= 0 && overrideValid override
```

Future Prompt 33 will implement:
- ECDSA signature verification
- Authorized keys registry lookup
- Timestamp freshness check (< 24h)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-30 | Initial implementation |
| 1.1.0 | 2025-12-30 | Patch 12B: Session persistence, charge decay |
| 1.2.0 | 2025-12-30 | Patch 12C: Multi-fragment queue, Prompt 33 interface |
