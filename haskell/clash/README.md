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
