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
