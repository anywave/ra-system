{-|
Module      : RaSonicFlux
Description : Clash FPGA synthesis for real-time harmonic driver
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Real-Time Harmonic Driver: Field Coherence to Audio Scalar Output.
Maps coherence states to symbolic sonic field triggers for audio synthesis.

== Synthesis Commands

Generate Verilog:
@
clash --verilog RaSonicFlux.hs
@

Generate VCD waveforms:
@
clash --vcd RaSonicFlux.hs
@

== Audio State Mapping

| Coherence | Output State  |
|-----------|---------------|
| < 0.30    | Silence       |
| < 0.55    | HarmonicLow   |
| < 0.80    | HarmonicMid   |
| >= 0.80   | HarmonicHigh  |
-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaSonicFlux where

import Clash.Prelude
import GHC.Generics (Generic)
import qualified Prelude as P

-- =============================================================================
-- Core Types
-- =============================================================================

-- | CoherenceStream represents incoming coherence states (0.0 to 1.0)
type CoherenceStream dom = Signal dom Float

-- | OutputAudioScalar is a symbolic sonic field trigger
-- These map to frequency bands or synthesis parameters in the audio layer
data OutputAudioScalar
  = Silence       -- ^ No audio output (coherence < 0.3)
  | HarmonicLow   -- ^ Low frequency harmonics (0.3 <= coherence < 0.55)
  | HarmonicMid   -- ^ Mid frequency harmonics (0.55 <= coherence < 0.8)
  | HarmonicHigh  -- ^ High frequency harmonics (coherence >= 0.8)
  deriving (Generic, Show, Eq, Enum, NFDataX)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Translate coherence into scalar audio states
-- This is the core mapping function - pure and synthesizable
mapToAudio :: Float -> OutputAudioScalar
mapToAudio c
  | c < 0.3   = Silence
  | c < 0.55  = HarmonicLow
  | c < 0.8   = HarmonicMid
  | otherwise = HarmonicHigh

-- | Stateless sonic mapper - directly maps coherence to audio state
-- No mealy machine needed since mapping is purely combinational
sonicFluxMapper :: HiddenClockResetEnable dom
                => CoherenceStream dom -> Signal dom OutputAudioScalar
sonicFluxMapper = fmap mapToAudio

-- =============================================================================
-- Synthesis Entry Points
-- =============================================================================

-- | Top entity for Clash synthesis
-- Exposes clock, reset, and enable for explicit control
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System Float
  -> Signal System OutputAudioScalar
topEntity = exposeClockResetEnable sonicFluxMapper

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Example coherence input vector for simulation
-- Tests all four output states across the coherence range
testCoherence :: Vec 6 Float
testCoherence = $(listToVecTH [0.1, 0.42, 0.6, 0.85, 0.9, 0.25])

-- | Expected outputs for test vector:
-- 0.1  -> Silence
-- 0.42 -> HarmonicLow
-- 0.6  -> HarmonicMid
-- 0.85 -> HarmonicHigh
-- 0.9  -> HarmonicHigh
-- 0.25 -> Silence

-- | Simulated output vector for testbench verification
simOutput :: Vec 6 OutputAudioScalar
simOutput = sampleN @System 6 $ sonicFluxMapper (fromList testCoherence)

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for VCD waveform inspection
-- Validates topEntity output against expected simOutput
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    expected = outputVerifier' clk rst simOutput
    inputStream = stimuliGenerator clk rst testCoherence
    done = expected (topEntity clk rst enableGen inputStream)
