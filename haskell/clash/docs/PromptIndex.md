# Ra System - Clash FPGA Prompt Index

Central reference for all Ra System Clash modules with prompt compliance testing.

## Module Registry

| Prompt | Module | Description | Testbench | Dashboard | Status |
|--------|--------|-------------|-----------|-----------|--------|
| 8 | RaSympatheticHarmonic | Sympathetic resonance fragment access | Yes | Phase 1 | Verified |
| 9 | RaOrgoneScalar | Orgone field scalar stability | Yes | Phase 1 | Verified |
| 10 | RaScalarResonance | Scalar resonance biofeedback loop | Yes | Phase 1 | Verified |
| 11 | RaGroupCoherence | Multi-avatar scalar entrainment | Yes | Phase 1 | Verified |
| 12 | RaShadowConsent | Consent-gated shadow harmonics | Yes | Phase 2 | Verified |
| 13 | RaDreamPhaseScheduler | Scalar dream induction & symbols | Yes | Phase 1 | Verified |
| 17 | BiofieldLoopback | Biofield loopback feedback system | Testbench.hs | Phase 1 | Verified |
| 22 | RaSonicFlux | Real-time harmonic driver | Yes | Phase 1 | Verified |
| 22+ | RaSonicEmitter | Full hardware PWM pipeline | Yes | Phase 1 | Verified |
| - | RaPWMDriver | Scalar to PWM converter | Pending | - | Verified |
| - | RaPWMMultiFreqTest | Multi-harmonic entrainment | Pending | - | Verified |
| 31 | RaConsentTransformer | Multi-core consent quorum | Yes | Phase 2 | Verified |
| 32 | RaConsentFramework | Symbolic consent gating | Yes | Phase 2 | Verified |
| 32+ | RaConsentRouter | Consent channel routing | Yes | Phase 2 | Verified |
| 33 | RaBiometricMatcher | Coherence profile matcher | Yes | Phase 2 | Verified |
| 34 | RaScalarExpression | Avatar expression mapper | Yes | Phase 2 | Verified |
| 35 | RaFieldTransferBus | Tesla coherent field transfer | Yes | Phase 1 | Verified |
| 40 | RaChamberSync | Multi-chamber synchronization | Yes | Phase 1 | Verified |
| 41 | RaVisualizerShell | Visual shell RGB renderer | Yes | Phase 1 | Verified |
| 44 | RaChamberMorphology | Chamber form transitions | Yes | Phase 1 | Verified |
| 49 | RaHarmonicTwist | Harmonic inversion twist | Yes | Phase 1 | Verified |
| 52 | RaBiofeedbackHarness | Exhale-hold trigger | Yes | Phase 1 | Verified |
| 54 | RaSymbolicCoherenceOps | Symbolic coherence DSL | Yes | Phase 1 | Verified |
| 56 | RaTactileControl | Tactile haptic interface | Yes | Phase 1 | Verified |
| 62 | RaAvatarFieldVisualizer | Avatar field glow anchors | Yes | Phase 1 | Verified |
| 64 | RaMusicChamberHarmonics | Solfeggio overtone mapper | Yes | Phase 1 | Verified |
| - | RaHandshakeGate | Dual-factor validation | Pending | Phase 2 | Verified |
| - | RaFieldSynthesisNode | Chamber state cascade | Yes | Phase 2 | Verified |
| - | RaBiometricGenerator | Biometric waveform gen | Yes | Phase 2 | Verified |
| - | RaTokenomicsProfiler | Token cost analyzer | Yes | Phase 2 | Verified |

## Test Status Legend

- **Verified**: Module has testbench + expected outputs validated
- **Pending**: Module functional, testbench to be added
- **WIP**: Work in progress

## Quick Links

### Phase 1 Modules (Field Synthesis)
- [Prompt 08 Guide](./Prompt08_SympatheticHarmonic.md) - Sympathetic resonance
- [Prompt 09 Guide](./Prompt09_OrgoneScalar.md) - Orgone field stability
- [Prompt 10 Guide](./Prompt10_ScalarResonance.md) - Scalar biofeedback
- [Prompt 11 Guide](./Prompt11_GroupCoherence.md) - Group coherence
- [Prompt 13 Guide](./Prompt13_DreamPhaseScheduler.md) - Dream induction
- [Prompt 17 Guide](./Prompt17_BiofieldLoopback.md) - Biofield resonance
- [Prompt 22 Guide](./Prompt22_SonicFlux.md) - Audio scalar output
- [Prompt 35 Guide](./Prompt35_FieldTransferBus.md) - Tesla field transfer
- [Prompt 40 Guide](./Prompt40_ChamberSync.md) - Chamber synchronization
- [Prompt 41 Guide](./Prompt41_VisualizerShell.md) - RGB visual feedback
- [Prompt 44 Guide](./Prompt44_ChamberMorphology.md) - Form transitions
- [Prompt 49 Guide](./Prompt49_HarmonicTwist.md) - Harmonic inversion
- [Prompt 52 Guide](./Prompt52_BiofeedbackHarness.md) - Breath trigger
- [Prompt 54 Guide](./Prompt54_SymbolicOps.md) - Symbolic DSL
- [Prompt 56 Guide](./Prompt56_TactileControl.md) - Haptic interface
- [Prompt 62 Guide](./Prompt62_AvatarFieldVisualizer.md) - Aura patterns
- [Prompt 64 Guide](./Prompt64_MusicChamberHarmonics.md) - Solfeggio mapping

### Phase 2 Modules (Consent Pipeline)
- [Prompt 12 Guide](./Prompt12_ShadowConsent.md) - Shadow consent gating
- [Prompt 31 Guide](./Prompt31_ConsentTransformer.md) - Quorum voting
- [Prompt 32 Guide](./Prompt32_ConsentFramework.md) - Consent gating
- [Prompt 33 Guide](./Prompt33_BiometricMatcher.md) - Template matching
- [Prompt 34 Guide](./Prompt34_ScalarExpression.md) - Avatar expression

### Supporting Modules (Infrastructure)
- [HandshakeGate Guide](./Supporting_HandshakeGate.md) - Dual-factor validation
- [FieldSynthesisNode Guide](./Supporting_FieldSynthesisNode.md) - Chamber state cascade
- [BiometricGenerator Guide](./Supporting_BiometricGenerator.md) - Waveform simulation
- [TokenomicsProfiler Guide](./Supporting_TokenomicsProfiler.md) - Cost analyzer
- [ConsentRouter Guide](./Supporting_ConsentRouter.md) - Channel splitter

## Synthesis Commands

```bash
# Generate Verilog for a module
clash --verilog RaHarmonicTwist.hs

# Generate VHDL
clash --vhdl RaChamberMorphology.hs

# Generate VCD waveforms for GTKWave
clash --vcd Testbench.hs
gtkwave testBench.vcd
```

## JSON Simulation Protocol

All modules support Claude simulation via JSON:

```json
{
  "inputStates": [
    { "field1": value1, "field2": value2 }
  ]
}
```

Response format:

```json
{
  "result": [
    {
      "input": { "field1": value1, "field2": value2 },
      "output": { "outField1": result1 }
    }
  ]
}
```

## Last Updated

- **Date**: 2025-12-30
- **Modules**: 31 total, 25 with testbenches
- **Dashboard Panels**: 13 interactive controls
- **Guides**: 27 prompt guides (22 numbered + 5 supporting)
- **Prompt 13 Tests**: 15 tests (9 core + 6 Patch 13A)
