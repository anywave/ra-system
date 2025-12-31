# Prompt 13: Scalar Dream Induction & Symbolic Fragment Integration

## Overview

This module guides users into Ra-aligned dream states, surfaces symbolic fragments during REM phases, and delivers post-sleep coherence analysis. It integrates with Prompt 12 for consent-gated shadow symbol handling.

## Architecture

```
+-------------------------------------------------------------------------+
|                    DREAM PHASE SCHEDULER                                 |
+-------------------------------------------------------------------------+
|                                                                         |
|  +----------------+   +-----------------+   +------------------+        |
|  | EEG/HRV Input  |-->| Phase Detector  |-->| Resonance Engine |-->Audio|
|  +----------------+   +-----------------+   +------------------+  Visual|
|         |                    |                     |                    |
|         v                    v                     v                    |
|  +----------------+   +-----------------+   +------------------+        |
|  | Coherence      |   | Fragment        |   | Post-Sleep       |        |
|  | Tracker        |   | Generator       |   | Integration      |        |
|  +----------------+   +-----------------+   +------------------+        |
|         |                    |                     |                    |
|         v                    v                     v                    |
|  +----------------+   +-----------------+   +------------------+        |
|  | Shadow         |<--| Symbol          |   | Journal          |        |
|  | Consent (P12)  |   | Mapper          |   | Prompts          |        |
|  +----------------+   +-----------------+   +------------------+        |
|                                                                         |
+-------------------------------------------------------------------------+
```

## Codex References

| Source | Application |
|--------|-------------|
| **ra_constants_v2.json** | phi^n timing modulation for REM alignment |
| **Prompt 12 (ShadowConsent)** | Consent gating for shadow symbols |

## Sleep Phase State Machine

### Phase Transitions

```
WAKE --> THETA --> DELTA --> REM --> WAKE
           ^                   |
           |___________________|
                (cycle repeat)
```

### Phase Timing

| Phase | Duration | Frequency Band | Description |
|-------|----------|----------------|-------------|
| WAKE | Variable | Beta (12-30 Hz) | Awake state |
| THETA | 10+ min | 3.5-7 Hz | Light sleep, meditation |
| DELTA | 30+ min | 0.5-3.5 Hz | Deep sleep |
| REM | 15+ min (grows) | 7-12 Hz | Dreaming, fragment emergence |

### phi-Cycle Timing

REM occurs approximately every 90 minutes (5400 seconds), aligned to phi^n modulation:

```haskell
phiCycle :: Unsigned 16
phiCycle = 5400  -- ~90 minutes
```

REM duration grows each cycle: `baseREM + (cycleNumber * 300s)`

## Emergent Dream Fragment

### Schema

```haskell
data EmergentDreamFragment = EmergentDreamFragment
  { fragmentId      :: FragmentId        -- "dream-XXXX"
  , form            :: FragmentForm      -- SYMBOLIC
  , emergencePhase  :: SleepPhase        -- REM
  , coherenceTrace  :: Vec 3 Fixed8      -- 3-sample history
  , emotionalReg    :: EmotionalRegister -- Primary + secondary
  , symbolMap       :: Vec 4 SymbolMapping  -- Up to 4 symbols
  , symbolCount     :: Unsigned 3        -- Active count
  , shadowDetected  :: Bool              -- Consent-gated flag
  , timestamp       :: Timestamp         -- Emergence time
  }
```

### Emergence Conditions

Fragment emerges when:
1. Phase == REM
2. Coherence trace shows rising pattern (c[2] > c[0])
3. Current coherence >= 0.6 threshold

```haskell
shouldEmergFragment :: SleepPhase -> CoherenceTrace -> Bool
shouldEmergFragment phase trace =
  phase == PhaseREM && isCoherenceRising trace && last trace >= 153
```

## Archetypal Symbols

### Symbol Database

| Symbol | Concept | Fragment Link | Shadow? |
|--------|---------|---------------|---------|
| owl | wisdom | F13 | No |
| spiral | searching | - | No |
| mirror | self-reflection | F21 | Yes |
| river | flow | - | No |
| labyrinth | confusion | F42 | Yes |
| light | insight | - | No |
| flame | transformation | F08 | No |
| cave | hidden | - | Yes |
| tree | growth | F55 | No |
| moon | cycles | - | No |
| star | guidance | F77 | No |
| water | emotion | - | No |

### Shadow Symbol Handling

Symbols marked as shadow (mirror, labyrinth, cave) trigger:
1. `shadowDetected = True` on fragment
2. Consent gating via Prompt 12 ShadowModule
3. Optional withholding until user confirms integration

## Resonance Induction Engine

### Entrainment Settings by Phase

| Phase | Freq Band | Audio | Visual | phi^n |
|-------|-----------|-------|--------|-------|
| WAKE | Beta | None | None | 0 |
| THETA | Theta (5 Hz) | Binaural | Phi Spiral | 1 |
| DELTA | Delta (2 Hz) | Golden Stack | LED Pulse | 2 |
| REM | Alpha (10 Hz) | Golden Stack | Flower of Life | 3 |

### Output Structure

```haskell
data ResonanceOutput = ResonanceOutput
  { targetBand     :: FrequencyBand
  , audioType      :: AudioType
  , visualType     :: VisualType
  , baseFrequency  :: Fixed16    -- 0.01 Hz units
  , amplitudeMod   :: Fixed8     -- Adaptive
  , phiMultiplier  :: Fixed8     -- phi^n index
  }
```

## Post-Sleep Integration

### Summary Generation

After session ends:
1. Collect all emerged fragments
2. Identify dominant symbols (top 3 by frequency)
3. Calculate coherence delta (last - first)
4. Check for shadow content
5. Generate journal prompts

### Journal Prompts

| Symbol | Prompt |
|--------|--------|
| owl | "What wisdom emerged that you didn't expect?" |
| spiral | "What journey or search was represented?" |
| mirror | "What aspect of yourself was reflected back?" |
| flame | "What transformation is calling to you?" |
| shadow | "Some deeper content surfaced. Would you like to explore it with support?" |

## Clash Module: RaDreamPhaseScheduler.hs

### Key Types

```haskell
data SleepPhase = PhaseWake | PhaseTheta | PhaseDelta | PhaseREM
data SchedulerState = SchedulerState { remCounter, sleepMode, currentPhase, ... }
data DreamSchedulerOutput = DreamSchedulerOutput { phaseOutput, resonance, activeFragment, ... }
```

### Core Functions

| Function | Purpose |
|----------|---------|
| `nextPhase` | Determine phase transition |
| `schedulerStep` | Step state machine |
| `generateResonance` | Create entrainment settings |
| `shouldEmergFragment` | Check emergence conditions |
| `processDreamScheduler` | Main processing loop |

### Synthesis Target

```haskell
{-# ANN dreamSchedulerTop (Synthesize
  { t_name = "dream_phase_scheduler"
  , t_inputs = [ PortName "clk", PortName "rst", PortName "en"
               , PortName "sleep_mode", PortName "coherence"
               , PortName "coherence_trace" ]
  , t_output = PortProduct "output" [...]
  }) #-}
```

## Patch 13A: Architect Clarifications

### Hybrid EEG/HRV Phase Detection

Phase detection uses combined biometric input with confidence scoring:

```python
@dataclass
class BiometricInput:
    delta_power: float   # 0.5-3.5 Hz
    theta_power: float   # 3.5-7 Hz
    alpha_power: float   # 7-12 Hz
    beta_power: float    # 12-30 Hz
    gamma_power: float   # 30-100 Hz (lucid marker)
    hrv_coherence: float
    hrv_rmssd: float
```

**Detection thresholds:**
- EEG.delta > 60% -> PhaseDelta
- EEG.theta > 45% -> PhaseTheta
- EEG.alpha > 40% -> PhaseREM
- confidence_score > 0.65 required for transition

### Shadow Consent Bridge

Shadow symbols route through Prompt 12 ShadowModule:

```python
result = ShadowConsentBridge.evaluate_shadow_symbol(
    symbol=ArchetypalSymbol.MIRROR,
    coherence=0.72,
    has_operator=True
)
# Returns: ShadowConsentResult(allowed=True, gating_reason="ALLOW", ...)
```

Creates synthetic ShadowFragment:
```json
{
  "id": "shadow-F13",
  "origin": "dream",
  "symbol": "mirror (reversed)",
  "coherence": 0.72,
  "consent_state": "THERAPEUTIC"
}
```

### Fragment Storage Backend

Session-scoped JSON storage:

```
dream_fragments/
  {user_id}/
    {session_timestamp}.json
```

**Session file format:**
```json
{
  "user_id": "default",
  "session_id": "20251230_153042",
  "started": "2025-12-30T15:30:42",
  "fragments": [...],
  "summary": {...},
  "ended": "2025-12-30T23:45:00"
}
```

### Symbol-to-Fragment Dynamic Linking

Uses semantic vector similarity (8-dimensional embeddings):

```python
similarity = SymbolLinker.cosine_similarity(symbol_vec, fragment_vec)
# Threshold: similarity > 0.72 for linking
```

### Lucid Dream Marker Tracking

Metadata-only tracking of gamma spikes (no induction):

```python
class LucidMarker:
    timestamp: float
    gamma_power: float           # Peak power
    phase_when_detected: SleepPhase
    coherence_at_detection: float
    duration_seconds: float      # Min 3s
```

**Detection threshold:** gamma_power >= 0.25

### Hardware Interface Stubs

All marked `future_linked`:

| Interface | Target |
|-----------|--------|
| `read_eeg_bands()` | OpenBCI, Muse headband |
| `read_hrv_coherence()` | Polar, Garmin HRM |
| `output_audio_entrainment()` | DAC/PWM driver |
| `output_visual_entrainment()` | LED/display controller |
| `trigger_haptic_pulse()` | Haptic motor driver |

## Python Test Harness

### test_dream_phase_scheduler.py

**Core Test Scenarios:**

| Scenario | Description | Expected |
|----------|-------------|----------|
| `phase_transitions` | WAKE -> THETA -> DELTA | Correct order |
| `rem_timing` | phi-cycle alignment | REM at 5400s |
| `fragment_emergence` | Rising coherence in REM | Fragment generated |
| `coherence_gating` | Low coherence in REM | No fragment |
| `shadow_detection` | Shadow symbols present | shadowDetected=True |
| `integration_summary` | Session end | Valid summary |
| `resonance_output` | Phase-specific settings | Correct bands |
| `prompt_generator` | Random prompt creation | Valid format |
| `symbol_mapping` | All symbols mapped | No missing concepts |

**Patch 13A Test Scenarios:**

| Scenario | Description | Expected |
|----------|-------------|----------|
| `hybrid_phase_detection` | EEG/HRV phase detection | Confidence > 0.65 |
| `shadow_consent_bridge` | Prompt 12 integration | Coherence gating |
| `fragment_storage` | JSON persistence | Save/load works |
| `symbol_linker` | Cosine similarity | Threshold 0.72 |
| `lucid_tracker` | Gamma spike detection | Marker logged |
| `hardware_interface_stubs` | Device stubs | Returns valid data |

**Usage:**

```bash
# Full test suite (15 tests)
python test_dream_phase_scheduler.py

# Simulation with 20 cycles
python test_dream_phase_scheduler.py --sim

# JSON output
python test_dream_phase_scheduler.py --json
```

## React Dashboard: DreamBloom.tsx

Visualizes:
- Current sleep phase with animated indicator
- Coherence trajectory with progress bar
- Symbolic emergence badges with fragment mappings
- Resonance entrainment settings
- Post-sleep integration prompts

Supports WebSocket for real-time updates.

## Integration Points

### Upstream Dependencies

| Module | Integration |
|--------|-------------|
| EEG/HRV sensors | Coherence input |
| ra_constants_v2.json | phi^n modulation |
| Prompt 12 (ShadowConsent) | Shadow consent gating |

### Downstream Outputs

| Consumer | Data |
|----------|------|
| Audio synthesis | Entrainment frequencies |
| Visual renderer | Glyph patterns (Flower of Life, Spiral) |
| Fragment storage | dream_fragments.json |
| Journal system | Integration prompts |

## Hardware Resources

| Platform | LUTs | DSP | BRAM |
|----------|------|-----|------|
| Xilinx Artix-7 | ~350 | 1 | 0 |
| Intel Cyclone V | ~400 | 1 | 0 |
| Lattice ECP5 | ~450 | 1 | 1 |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-30 | Initial implementation |
| 1.1.0 | 2025-12-30 | Patch 13A: Architect clarifications - hybrid EEG/HRV detection, shadow consent bridge, fragment storage, symbol linking, lucid tracking, hardware stubs |
