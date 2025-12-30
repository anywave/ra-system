{-|
Module      : RaSonicEmitter
Description : Signal-level audio generator with scalar envelope synthesis
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Expands Prompt 22 into wave-emitting pattern driver for hardware or emulation.
Converts symbolic OutputAudioScalar states into actual waveform amplitudes.

== Pipeline Architecture

@
Coherence (Float) → sonicFluxMapper → OutputAudioScalar → audioEmitter → Float (amplitude)
@

== Amplitude Mapping

| Scalar State  | Amplitude |
|---------------|-----------|
| Silence       | 0.0       |
| HarmonicLow   | 0.3       |
| HarmonicMid   | 0.6       |
| HarmonicHigh  | 0.9       |

== Synthesis Commands

@
clash --verilog RaSonicEmitter.hs
clash --vcd RaSonicEmitter.hs
@
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaSonicEmitter where

import Clash.Prelude
import GHC.Generics (Generic)
import qualified Prelude as P
import RaSonicFlux (OutputAudioScalar(..), sonicFluxMapper, testCoherence)

-- =============================================================================
-- Types
-- =============================================================================

-- | Audio waveform output signal (amplitude 0.0 to 1.0)
type AudioWave dom = Signal dom Float

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Map symbolic scalar output to waveform amplitude
-- Simple envelope mapping: silence (0.0), low (0.3), mid (0.6), high (0.9)
mapScalarToWave :: OutputAudioScalar -> Float
mapScalarToWave Silence      = 0.0
mapScalarToWave HarmonicLow  = 0.3
mapScalarToWave HarmonicMid  = 0.6
mapScalarToWave HarmonicHigh = 0.9

-- | Convert symbolic scalar to waveform amplitude output
-- Pure combinational mapping for synthesis
audioEmitter :: HiddenClockResetEnable dom
             => Signal dom OutputAudioScalar -> AudioWave dom
audioEmitter = fmap mapScalarToWave

-- | Full pipeline: coherence → scalar → waveform output
-- Composes sonicFluxMapper and audioEmitter into single pipeline
sonicPipeline :: HiddenClockResetEnable dom
              => Signal dom Float -> AudioWave dom
sonicPipeline = audioEmitter . sonicFluxMapper

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top entity for waveform output synthesis
-- Input: coherence (0.0-1.0)
-- Output: waveform amplitude (0.0, 0.3, 0.6, or 0.9)
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System Float
  -> Signal System Float
topEntity = exposeClockResetEnable sonicPipeline

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test vector for waveform output verification
-- Maps testCoherence through full pipeline
-- Expected: [0.0, 0.3, 0.6, 0.9, 0.9, 0.0]
testWaveOutput :: Vec 6 Float
testWaveOutput = sampleN @System 6 $ sonicPipeline (fromList testCoherence)

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for waveform inspection
-- Validates full pipeline output against expected amplitudes
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    expected = outputVerifier' clk rst testWaveOutput
    stim = stimuliGenerator clk rst testCoherence
    done = expected (topEntity clk rst enableGen stim)
