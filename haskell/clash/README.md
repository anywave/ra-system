# Ra System - Clash FPGA Synthesis

This directory contains Clash-compatible modules for FPGA synthesis of Ra System components.

## Requirements

- [Clash Compiler](https://clash-lang.org/) (clash-ghc)
- GHC 8.10+ or 9.0+

Install Clash:
```bash
cabal install clash-ghc
```

## Modules

### BiofieldLoopback.hs (Prompt 17)

Biofield loopback feedback system - closed-loop resonant coupling.

### RaSonicFlux.hs (Prompt 22)

Real-time harmonic driver mapping coherence to audio scalar output.

**Audio State Mapping:**
| Coherence | Output State  |
|-----------|---------------|
| < 0.30    | Silence       |
| < 0.55    | HarmonicLow   |
| < 0.80    | HarmonicMid   |
| >= 0.80   | HarmonicHigh  |

### RaSonicEmitter.hs (Full Hardware Pipeline)

Complete pipeline from coherence to PWM output for hardware integration.

**Pipeline:** `Coherence → sonicFluxMapper → OutputAudioScalar → audioEmitter → Amplitude → scalarToPWM → PWM`

**Amplitude Mapping:**
| Scalar State  | Amplitude |
|---------------|-----------|
| Silence       | 0.0       |
| HarmonicLow   | 0.3       |
| HarmonicMid   | 0.6       |
| HarmonicHigh  | 0.9       |

**Key Function:** `sonicPWMOutput :: Signal dom Float -> Signal dom Bool`

### RaPWMDriver.hs (Hardware Output)

Scalar amplitude to PWM duty signal generator for low-frequency outputs.

**Applications:**
- LED intensity modulation
- Haptic vibration control
- Solfeggio/Schumann entrainment

**Specs:**
- Resolution: 8-bit (256 levels)
- Duty cycle: amplitude × 255

### RaPWMMultiFreqTest.hs (Multi-Harmonic Entrainment)

Multi-harmonic PWM scalar pattern generator with runtime weight tuning, visual envelope output, and live biometric override.

**Harmonic Bands:**
| Band       | Frequency Range | Waveform | Default Weight |
|------------|-----------------|----------|----------------|
| Theta      | 4-8 Hz          | Sine     | 40%            |
| Delta      | 0.5-4 Hz        | Cosine   | 30%            |
| Solfeggio  | Sacred tones    | 3x Sine  | 30%            |

**Biometric Override:**
When `bioOverride > 0`, the biometric signal takes priority over synthetic blend. This enables real-time body-sensing to override synthetic waveforms.

**Pipeline with Biometric Override:**
```
Weights ─────────────┐
                     ▼
Theta ─┐          ┌─────────────┐
       ├─────────▶│ blendFields │──▶ envelope ──┐
Delta ─┤          └─────────────┘               ▼
       │                               ┌─────────────────┐
Solfeggio ─┘                           │ mux(bioOverride)│──▶ PWM
                                       └────────┬────────┘
BioOverride ────────────────────────────────────┘
```

**Key Function:** `multiHarmonicBiometric :: Signal dom HarmonicWeights -> Signal dom Float -> Signal dom Float -> Signal dom Float -> Signal dom Float -> (Signal dom Bool, Signal dom Float)`

### RaConsentFramework.hs (Prompt 32 - Consent Gating)

Self-regulating consent framework with scalar-aware symbolic validation. Honors Codex protocols of shadow gating, harmonic override, and coherence memory.

**Consent States:**
| State    | Description                                    |
|----------|------------------------------------------------|
| Permit   | Full consent - coherent and authorized         |
| Restrict | Limited consent - incoherent or unauthorized   |
| Override | Emergency override - bypasses normal gating    |

**State Transitions:**
```
isCoherent = True  ──▶ Permit
isCoherent = False ──▶ Restrict
overrideFlag = True ──▶ Override (highest priority)
```

**Coherence Memory:**
- Tracks `coherenceDur` - how long coherence has been maintained
- Enables graduated consent escalation and shadow gating warmup

**Key Function:** `consentGate :: Signal dom ConsentInput -> Signal dom ConsentState`

### RaConsentRouter.hs (Consent Channel Routing)

Routes consent states to downstream trigger channels. Integrates with RaConsentFramework for complete consent pipeline.

**Trigger Channels:**
| Channel        | Activation Condition                           |
|----------------|------------------------------------------------|
| bioTrigger     | Permit OR Override                             |
| gestureTrigger | Override only                                  |
| fieldTrigger   | Permit AND coherenceGate high                  |

**Truth Table:**
| State    | Gate  | bioTrigger | gestureTrigger | fieldTrigger |
|----------|-------|------------|----------------|--------------|
| Permit   | False | True       | False          | False        |
| Permit   | True  | True       | False          | True         |
| Restrict | False | False      | False          | False        |
| Override | True  | True       | True           | False        |

**Key Function:** `consentRouterTop :: ... -> Signal System ConsentState -> Signal System Bool -> (Signal System Bool, Signal System Bool, Signal System Bool)`

### RaHandshakeGate.hs (Dual-Factor Validation)

Dual-factor validation handshake combining symbolic consent triggers with biometric coherence verification. Wired to Phase II dashboard and field trigger cascade.

**Handshake Logic:**
```
handshakeGranted = overrideFlag OR (biometricMatch AND symbolOK)
```

**Permitted Trigger IDs:** `[3, 4, 7, 9]`

**Output Bundle:**
| Field | Description |
|-------|-------------|
| handshakeGranted | Final grant decision |
| passedBiometric | Biometric coherence status |
| matchedSymbol | Symbolic ID in permitted list |
| overrideUsed | Override flag was enabled |

**Field Cascade:** `fieldTriggerFromHandshake` extracts grant for downstream emitters.

**Key Function:** `handshakeTop :: ... -> Signal System HandshakeIn -> Signal System HandshakeOut`

### RaFieldSynthesisNode.hs (Chamber State Cascade)

Chamber state machine activated by handshakeGranted signal. Controls field synthesis progression through activation phases.

**Chamber States:**
| State | Glow | Description |
|-------|------|-------------|
| Idle | 0 | Awaiting grant |
| Spinning | 64 | Initial spin-up |
| Stabilizing | 192 | Coherence stabilization |
| Emanating | 255 | Full emission |

**State Transitions:**
```
granted=True:  Idle → Spinning → Stabilizing → Emanating (holds)
granted=False: Any → Idle (reset)
```

**Key Function:** `fieldSynthesisNode :: Signal dom Bool -> (Signal dom ChamberState, Signal dom (Unsigned 8))`

### RaBiometricGenerator.hs (Biometric Waveform)

Generates simulated biometric waveforms for coherence entrainment testing.

**Biometric Patterns:**
| Pattern | Description | Waveform |
|---------|-------------|----------|
| Flatline | No variation | Constant 128 |
| BreathRise | Breath cycle | 0→255→0 |
| CoherentPulse | Smooth HRV | 100-180 sinusoid |
| Arrhythmic | Erratic | Random spikes |

**Key Function:** `biometricSignal :: Signal dom BioPattern -> Signal dom (Unsigned 8)`

### RaTokenomicsProfiler.hs (Cost Analysis)

Tracks token usage and compute costs for Ra system operations.

**Operation Costs:**
| Operation | Tokens | Compute |
|-----------|--------|---------|
| Handshake | 128 | 500 |
| BioEmit | 64 | 200 |
| ChamberSpin | 256 | 1000 |

**Key Function:** `profiler :: Signal dom OpTrigger -> Signal dom (Unsigned 16, Unsigned 16)`

### RaBiometricMatcher.hs (Prompt 33 - Coherence Profile Matcher)

Biometric coherence profile matcher. Compares input signal patterns against reference templates to compute coherence scores for handshake validation.

**Biometric Templates:**
| Template | Description | Pattern |
|----------|-------------|---------|
| TemplateFlat | Baseline, no variation | Constant 128 |
| TemplateResonant | Full coherent oscillation | 64-192 sinusoid |
| TemplatePulse | Subtle pulse variation | 108-148 wave |

**Coherence Scoring:**
```
Score = 255 - (average absolute difference)
```
- 255: Perfect match (no difference)
- 0: Maximum mismatch

**Key Function:** `matchCoherence :: Signal dom (Vec 16 (Unsigned 8)) -> Signal dom BioTemplate -> Signal dom (Unsigned 8)`

### RaScalarExpression.hs (Prompt 34 - Avatar Expression Mapper)

Maps biometric coherence and breath phase to avatar visual expression parameters.

**Expression Mapping:**
| Coherence | Breath Phase | Aura Intensity | Limb Vector |
|-----------|--------------|----------------|-------------|
| >= 200    | Exhale       | coherence      | +40         |
| >= 150    | Exhale       | 128            | +20         |
| >= 150    | Inhale       | 128            | -20         |
| < 150     | Any          | 64             | 0           |

**Output Bundle:**
- `auraIntensity` (Unsigned 8): Visual field luminance [0-255]
- `limbVector` (Signed 8): Motion intent scalar [-128 to +127]

**Key Function:** `mapExpression :: Signal dom (Unsigned 8) -> Signal dom Bool -> Signal dom AvatarExpression`

### RaConsentTransformer.hs (Prompt 31 - Multi-Core Consent)

Distributed consent logic across multiple avatar threads with quorum voting and biometric gating.

**Activation Requirements:**
- `coherenceScore >= 180`
- `auraIntensity >= 128`

If either threshold is not met, all votes are suspended (treated as False).

**Outputs:**
| Field | Description |
|-------|-------------|
| consentGranted | True if quorum % of active votes agree |
| fieldEntropy | Number of dissenting votes |
| activeVotes | Number of qualifying True votes |

**Key Function:** `consentTransform :: Signal dom (Unsigned 8) -> Signal dom (Unsigned 8) -> Signal dom (Vec n Bool) -> Signal dom (Unsigned 8) -> Signal dom (Bool, Unsigned 8, Unsigned 8)`

### RaFieldTransferBus.hs (Prompt 35 - Tesla Coherent Field Transfer)

Simulates scalar packet transmission from Avatar A to Avatar B with coherence preservation.

**Transfer Constraints:**
- TransferLatency < 300 cycles (simulated at 1ms per cycle)
- CoherenceInvariant: Score delta <= +/-1 post-transfer

**Inputs:**
| Signal | Description |
|--------|-------------|
| srcCoherence | Coherence score (0-255) |
| srcSignal | 4-element scalar harmonic vector |
| sendPulse | Triggers transfer on rising edge |

**Outputs:**
| Signal | Description |
|--------|-------------|
| destSignal | Received scalar harmonic vector |
| latencyCount | Transfer time in cycles |
| integrityOK | True if coherence preserved within tolerance |

**Key Function:** `fieldTransferBus :: Signal dom (Unsigned 8) -> Signal dom (Vec 4 (Unsigned 8)) -> Signal dom Bool -> Signal dom (Vec 4 (Unsigned 8), Unsigned 9, Bool)`

### RaChamberSync.hs (Prompt 40 - Multi-Chamber Synchronization)

Synchronizes multiple chamber nodes to maintain phase coherence across distributed field synthesis network.

**Sync States:**
| State    | Description                                    |
|----------|------------------------------------------------|
| Desync   | Chambers out of phase, awaiting alignment      |
| Aligning | Active phase correction in progress            |
| Locked   | All chambers synchronized                      |
| Drifting | Minor phase drift detected, auto-correcting   |

**Sync Logic:**
```
syncState = Locked   when all chambers same state AND drift < 8
syncState = Drifting when all chambers same state BUT drift >= 8
syncState = Aligning when majority chambers agree
syncState = Desync   otherwise
```

**Phase Drift:**
- Tracks cycles since last full synchronization
- maxDrift = 32 cycles before quality degrades

**Output Bundle:**
| Field | Description |
|-------|-------------|
| syncState | Current synchronization status |
| syncPulse | Pulse to trigger chamber alignment |
| syncQuality | Sync quality 0-255 (255 = perfect) |
| phaseDrift | Cycles since last full sync |

**Key Function:** `chamberSync :: Signal dom (Vec n ChamberState) -> Signal dom SyncOutput`

### RaBiofeedbackHarness.hs (Prompt 52 - Exhale-Hold Trigger)

Maps exhale-hold breath transition with high coherence to physical/energetic output signals.

**Breath Phases (ADT):**
| Phase      | Description                    |
|------------|--------------------------------|
| Inhale     | Breathing in                   |
| Exhale     | Breathing out                  |
| Hold       | Breath held (post-inhale)      |
| ExhaleHold | Breath held after exhale       |

**Trigger Logic:**
```
Trigger = (breathPhase == ExhaleHold) AND (coherence >= 230)
```

**Outputs:**
| Signal       | Description                    |
|--------------|--------------------------------|
| motionIntent | Triggers limb movement cascade |
| hapticPing   | Triggers haptic feedback pulse |

**Integration:**
- Downstream from RaFieldTransferBus (Prompt 35)
- Links to chamber update via RPP field coherence

**Key Function:** `biofeedbackHarness :: Signal dom BioState -> Signal dom BioOutput`

### RaVisualizerShell.hs (Prompt 41 - Visual Shell Renderer)

Renders visual shell feedback from chamber state, coherence, and sync status to RGB color output.

**Chamber State Colors:**
| State       | Base RGB        | Description       |
|-------------|-----------------|-------------------|
| Idle        | (0, 0, 32)      | Deep blue (dim)   |
| Spinning    | (0, 64, 128)    | Cyan glow         |
| Stabilizing | (128, 64, 255)  | Purple pulse      |
| Emanating   | (255, 128, 64)  | Golden radiance   |

**Sync State Modulation:**
| Sync State | Effect                      |
|------------|-----------------------------|
| Desync     | Flash red overlay (50%)     |
| Aligning   | Pulse brightness (25-100%)  |
| Locked     | Steady (100%)               |
| Drifting   | Subtle fade (75-100%)       |

**Coherence Intensity:**
```
Final brightness = baseColor * syncModulation * (coherence / 255)
```

**Integration:**
- Consumes ChamberState from RaFieldSynthesisNode
- Consumes SyncState from RaChamberSync
- Outputs RGB triplet for LED/display driver

**Key Function:** `visualizerShell :: Signal dom ChamberState -> Signal dom SyncState -> Signal dom (Unsigned 8) -> Signal dom RGBColor`

### RaAvatarFieldVisualizer.hs (Prompt 62 - Avatar Field Glow Anchors)

Generates AuraPattern (4 glow anchors) from signature vector, chamber state, and emergence level.

**Chamber State Codes:**
| Code  | State       | Visualization |
|-------|-------------|---------------|
| 0b000 | Idle        | Inactive      |
| 0b001 | Spinning    | Inactive      |
| 0b010 | Stabilizing | Inactive      |
| 0b101 | Emanating   | Active        |

**Output Calculation:**
```
AuraPattern[i] = signature[i] * emergenceLevel / 256  (when state = 0b101)
AuraPattern[i] = 0                                    (otherwise)
```

**Integration:**
- Downstream from RaFieldSynthesisNode
- Feeds into LED driver or display renderer

**Key Function:** `avatarField :: Signal dom (Vec 4 (Unsigned 8)) -> Signal dom (BitVector 3) -> Signal dom (Unsigned 8) -> Signal dom (Vec 4 (Unsigned 8))`

### RaTactileControl.hs (Prompt 56 - Tactile Haptic Interface)

Interfaces with tactile sensors and haptic actuators for bidirectional physical feedback.

**Gesture Codes:**
| Code | Gesture      | Haptic Response |
|------|--------------|-----------------|
| 0000 | None         | Silent          |
| 0001 | Tap          | Short pulse     |
| 0010 | Hold         | Sustained buzz  |
| 0011 | Swipe        | Wave pattern    |
| 0100 | Circle       | Spiral ramp     |
| 0101 | Pinch        | Double pulse    |

**Consent Gating:**
| Level | Allowed Patterns              |
|-------|-------------------------------|
| 0     | None (all haptics disabled)   |
| 1     | Pulse, Buzz only              |
| 2     | Full haptic range             |

**Haptic Patterns:**
| Pattern     | PWM Duty | Duration    |
|-------------|----------|-------------|
| Silent      | 0%       | 0           |
| Pulse       | 80%      | 8 cycles    |
| Buzz        | 50%      | Sustained   |
| Wave        | 0-100%   | 16 (ramp)   |
| Spiral      | 25-75%   | 24 (osc)    |
| DoublePulse | 80%      | 4+4+gap     |

**Key Function:** `tactileControl :: Signal dom (Unsigned 8) -> Signal dom (BitVector 4) -> Signal dom (Unsigned 2) -> Signal dom TactileOutput`

### RaMusicChamberHarmonics.hs (Prompt 64 - Solfeggio Overtone Mapper)

Maps coherence band levels to Solfeggio overtone frequencies for music chamber harmonics.

**Solfeggio Base Frequencies:**
| Index | Frequency | Note  | Association           |
|-------|-----------|-------|----------------------|
| 0     | 396 Hz    | G     | Liberation from fear |
| 1     | 417 Hz    | G#    | Undoing situations   |
| 2     | 528 Hz    | C     | Transformation/DNA   |
| 3     | 639 Hz    | E     | Connecting relations |

**Harmonic Mapping:**
```
overtoneFreq[i] = baseFreq[i] + (baseFreq[i] * coherenceBand[i]) / 256
```

**Example:**
| Band | Base | Coherence | Overtone |
|------|------|-----------|----------|
| 0    | 396  | 128       | 594 Hz   |
| 1    | 417  | 255       | 832 Hz   |
| 2    | 528  | 64        | 660 Hz   |
| 3    | 639  | 192       | 1117 Hz  |

**Integration:**
- Consumes coherence bands from RaBiometricMatcher
- Outputs frequency values for audio synthesis
- Feeds into RaSonicEmitter for audible output

**Key Function:** `harmonicMapper :: Signal dom (Vec 4 (Unsigned 8)) -> Signal dom (Vec 4 (Unsigned 16))`

### RaSymbolicCoherenceOps.hs (Prompt 54 - Symbolic Coherence Transformations)

Compositional symbolic operations for transforming emergence conditions with fixed-point arithmetic.

**Symbolic Operations:**
| Operation | Formula | Description |
|-----------|---------|-------------|
| PhaseShift(fx) | coherence += fx * 255 | Add phase with saturation |
| InvertAngle | angle = 255 - angle | Mirror angle about midpoint |
| GateThreshold(fx) | coherence = 0 if < fx | Zero below threshold |

**Fixed-Point Encoding:**
| Value | Fraction | Common Use |
|-------|----------|------------|
| 0 | 0.0 | Zero |
| 102 | 0.4 | 40% threshold |
| 128 | 0.5 | Half |
| 158 | 0.618 | Golden ratio |
| 255 | 1.0 | Full |

**Example Composition:**
```
Input: (coherence=100, angle=200)
PhaseShift(0.618): coherence = min(255, 100 + 97) = 197
InvertAngle:       angle = 255 - 200 = 55
GateThreshold(0.4): 197 >= 102, keep
Output: (coherence=197, angle=55)
```

**Integration:**
- Consumes EmergenceCondition from avatar state
- Operations compose left-to-right
- Feeds into field synthesis pipeline

**Key Function:** `symbolicProcessor :: Signal dom (Vec 3 SymbolicOp) -> Signal dom EmergenceCondition -> Signal dom EmergenceCondition`

### RaChamberMorphology.hs (Prompt 44 - Chamber Form Transitions)

Models chamber morphology transitions based on coherence and instability thresholds.

**Chamber Forms:**
| Form    | Description                        |
|---------|------------------------------------|
| Sphere  | Default stable form                |
| Toroid  | Collapsed form (low coherence)     |
| Cube    | Crystallized form (high coherence) |

**Morphology Events:**
| Event         | Trigger Condition                    |
|---------------|--------------------------------------|
| NoChange      | Normal operation, form stable        |
| RapidCollapse | coherence < 0.39 AND instability > 0.30 |

**Thresholds:**
- coherenceThreshold = 100 (~0.39)
- instabilityThreshold = 77 (~0.30)

**State Transition:**
```
if coherence < 100 AND instability > 77:
    result = (Toroid, RapidCollapse)
else:
    result = (currentForm, NoChange)
```

**Integration:**
- Consumes coherence from RaBiometricMatcher
- Consumes instability from field entropy
- Outputs form change events for visualization

**Key Function:** `morphologyProcessor :: Signal dom ChamberState -> Signal dom MorphResult`

### RaHarmonicTwist.hs (Prompt 49 - Harmonic Inversion Twist)

Computes twist envelope from harmonic mode pair Y(a,b) and coherence level for animation/rendering.

**Harmonic Mode Encoding:**
| Parameter | Description | Range |
|-----------|-------------|-------|
| modeA | Azimuthal mode Y(a,b) | 0-15 |
| modeB | Polar mode | 0-15 |
| coherence | Coherence level | 0-255 |

**Twist Calculation:**
```
invMag = a * 10 + b * 7
twistMag = invMag + 15  (if coherence >= 0.41)
         = invMag - 10  (if coherence < 0.41)
duration = 8 cycles     (if twistMag > 80)
         = 4 cycles     (if twistMag <= 80)
```

**Thresholds:**
- coherenceThreshold = 105 (~0.41)
- twistMagThreshold = 80

**Example Calculations:**
| modeA | modeB | coherence | invMag | twistMag | duration |
|-------|-------|-----------|--------|----------|----------|
| 5     | 3     | 120       | 71     | 86       | 8        |
| 2     | 4     | 80        | 48     | 38       | 4        |
| 8     | 6     | 200       | 122    | 137      | 8        |

**Integration:**
- Consumes harmonic mode from field analysis
- Outputs twist envelope for animation/rendering
- Duration feeds into timing subsystem

**Key Function:** `computeTwistEnvelope :: HarmonicInput -> TwistResult`

---

### BiofieldLoopback.hs Details

Biofield loopback feedback system - closed-loop resonant coupling between biometric input and avatar field output.

**Inputs:**
- `breathRate` - Breath rate in Hz (optimal: 6.5 Hz)
- `hrv` - Heart rate variability [0, 1]

**Outputs:**
- `glowState` - Emergence glow level (None | Low | Moderate | High)
- `coherence` - Raw coherence value

**Core Formula:**
```haskell
coherence = (6.5 - abs(6.5 - breathRate)) * hrv
```

## Synthesis Commands

### Generate Verilog
```bash
clash --verilog BiofieldLoopback.hs
```

### Generate VHDL
```bash
clash --vhdl BiofieldLoopback.hs
```

### Generate VCD Waveforms
```bash
clash --vcd Testbench.hs
gtkwave testBench.vcd
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BiofieldLoopback                      │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────┐  │
│  │ BiometricInput│───▶│computeCoherence│───▶│classifyGlow│
│  │ - breathRate  │    │               │    │          │  │
│  │ - hrv         │    │ (6.5-|6.5-br|)│    │ Threshold│  │
│  └──────────────┘    │    * hrv      │    │ Classify │  │
│                      └───────────────┘    └────┬─────┘  │
│                                                 │        │
│                      ┌───────────────┐          │        │
│                      │ AvatarFieldFrame│◀────────┘        │
│                      │ - glowState   │                   │
│                      │ - coherence   │                   │
│                      └───────────────┘                   │
│                             │                            │
│                    ┌────────▼────────┐                   │
│                    │   Mealy State   │                   │
│                    │   Register      │◀──────────────────│
│                    └─────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## Integration with Ra Library

The Clash modules can import types and pure functions from `Ra.Biofield.Loopback`:

```haskell
-- In Clash module:
import Ra.Biofield.Loopback (computeCoherenceSimple, initFrame, testInputs)
```

The main Ra library (`ra-system`) is designed for simulation and standard Haskell use, while these Clash modules are for FPGA synthesis.

## Test Data

| Input # | Breath Rate | HRV  | Expected Glow |
|---------|-------------|------|---------------|
| 1       | 6.2 Hz      | 0.81 | High          |
| 2       | 6.4 Hz      | 0.85 | High          |
| 3       | 6.1 Hz      | 0.65 | Moderate      |
| 4       | 5.7 Hz      | 0.90 | Moderate      |
| 5       | 6.6 Hz      | 0.78 | High          |
