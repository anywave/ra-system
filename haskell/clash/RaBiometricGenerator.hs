{-|
Module      : RaBiometricGenerator
Description : Simulated biometric signal for loopback feedback
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Generates simulated biometric waveforms for coherence entrainment testing.
Injects dynamic biometric signals for handshake loopback validation.

== Biometric Patterns

| Pattern       | Description                              | Waveform          |
|---------------|------------------------------------------|-------------------|
| Flatline      | No variation, baseline signal            | Constant 128      |
| BreathRise    | Rising/falling breath cycle              | 0→255→0 envelope  |
| CoherentPulse | Smooth coherent HRV pattern              | 100-180 sinusoid  |
| Arrhythmic    | Erratic, non-coherent signal             | Random-like spikes|

== Integration

Feeds into RaHandshakeGate biometric input for coherence validation.
Can be controlled from dashboard via pattern selector.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaBiometricGenerator where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Simulated breath/HRV patterns for coherence entrainment
data BioPattern = Flatline | BreathRise | CoherentPulse | Arrhythmic
  deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Waveform Data
-- =============================================================================

-- | Convert BioPattern to 16-sample waveform vector
toWaveform :: BioPattern -> Vec 16 (Unsigned 8)
toWaveform Flatline      = replicate d16 128
toWaveform BreathRise    = $(listToVecTH [0, 64, 128, 192, 255, 192, 128, 64, 0, 64, 128, 192, 255, 192, 128, 64 :: Unsigned 8])
toWaveform CoherentPulse = $(listToVecTH [128, 140, 150, 160, 170, 180, 170, 160, 150, 140, 130, 120, 110, 100, 110, 120 :: Unsigned 8])
toWaveform Arrhythmic    = $(listToVecTH [128, 180, 90, 200, 50, 220, 80, 160, 40, 255, 70, 100, 200, 60, 90, 180 :: Unsigned 8])

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Pattern selector with cycling sample index
-- Outputs current sample from selected pattern's waveform
biometricSignal
  :: HiddenClockResetEnable dom
  => Signal dom BioPattern
  -> Signal dom (Unsigned 8)
biometricSignal pat = sampleSeq <$> pat <*> counter
  where
    counter = regEn 0 (pure True) (counter + 1)
    sampleSeq p idx = toWaveform p !! (idx `mod` 16)

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level wrapper for Clash synthesis
bioTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System BioPattern
  -> Signal System (Unsigned 8)
bioTop = exposeClockResetEnable biometricSignal

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test pattern sequence
patternVec :: Vec 8 BioPattern
patternVec = $(listToVecTH
  [Flatline, BreathRise, CoherentPulse, Arrhythmic, CoherentPulse, BreathRise, Flatline, Arrhythmic])

-- | Expected first sample from each pattern
expectedSamples :: Vec 8 (Unsigned 8)
expectedSamples = map (\p -> head (toWaveform p)) patternVec

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for biometric signal generation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst patternVec
    output = bioTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedSamples output
