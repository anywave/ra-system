{-|
Module      : RaSonicEmitter
Description : Signal-level audio generator with scalar envelope synthesis
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Expanded to drive PWM modulation from symbolic scalar amplitude.
Full pipeline from coherence input to PWM hardware output.

== Complete Pipeline

@
Coherence (Float) → sonicFluxMapper → OutputAudioScalar → audioEmitter → Float → scalarToPWM → Bool (PWM)
@

== Amplitude Mapping

| Scalar State  | Amplitude |
|---------------|-----------|
| Silence       | 0.0       |
| HarmonicLow   | 0.3       |
| HarmonicMid   | 0.6       |
| HarmonicHigh  | 0.9       |

== Hardware Output

Output can drive GPIO or chamber modulator for:
- LED biofeedback
- Haptic entrainment
- Solfeggio/Schumann resonance
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaSonicEmitter where

import Clash.Prelude
import GHC.Generics (Generic)
import qualified Prelude as P
import RaSonicFlux (OutputAudioScalar(..), sonicFluxMapper, testCoherence)
import RaPWMDriver (scalarToPWM)

-- =============================================================================
-- Types
-- =============================================================================

-- | Audio waveform output signal (amplitude 0.0 to 1.0)
type AudioWave dom = Signal dom Float

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Map symbolic scalar output to waveform amplitude
mapScalarToWave :: OutputAudioScalar -> Float
mapScalarToWave Silence      = 0.0
mapScalarToWave HarmonicLow  = 0.3
mapScalarToWave HarmonicMid  = 0.6
mapScalarToWave HarmonicHigh = 0.9

-- | Convert symbolic scalar to waveform amplitude output
audioEmitter :: HiddenClockResetEnable dom
             => Signal dom OutputAudioScalar -> AudioWave dom
audioEmitter = fmap mapScalarToWave

-- | Full pipeline: coherence → scalar → amplitude → PWM output
-- Combines all stages into single hardware-ready pipeline
sonicPWMOutput :: HiddenClockResetEnable dom
               => Signal dom Float -> Signal dom Bool
sonicPWMOutput = scalarToPWM . audioEmitter . sonicFluxMapper

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top entity for scalar-PWM integration
-- Output can drive GPIO or chamber modulator
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System Float
  -> Signal System Bool
topEntity = exposeClockResetEnable sonicPWMOutput

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test vector for PWM pipeline output
-- NOTE: This is mock expected data for shape only — real PWM is dynamic
-- Visualize with GTKWave for proper waveform validation
testPWMOutput :: Vec 6 Bool
testPWMOutput = map (\x -> x > 0.3) testCoherence

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for scalar to PWM waveform validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testCoherence
    pwmOut = topEntity clk rst enableGen stim
    expected = outputVerifier' clk rst testPWMOutput
    done = expected pwmOut
