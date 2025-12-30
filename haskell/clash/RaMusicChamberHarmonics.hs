{-|
Module      : RaMusicChamberHarmonics
Description : Music chamber harmonics mapper for Solfeggio overtone generation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 64: Maps coherence band levels to Solfeggio overtone frequencies.
Each band modulates a base sacred frequency proportionally to coherence level.

== Solfeggio Base Frequencies

| Index | Frequency | Note  | Association           |
|-------|-----------|-------|----------------------|
| 0     | 396 Hz    | G     | Liberation from fear |
| 1     | 417 Hz    | G#    | Undoing situations   |
| 2     | 528 Hz    | C     | Transformation/DNA   |
| 3     | 639 Hz    | E     | Connecting relations |

== Harmonic Mapping

@
overtoneFreq[i] = baseFreq[i] * (1 + coherenceBand[i] / 255)
                = baseFreq[i] + (baseFreq[i] * coherenceBand[i]) / 256
@

== Example Calculation

coherenceBand = [128, 255, 64, 192]

| Band | Base | Coherence | Overtone                    |
|------|------|-----------|-----------------------------|
| 0    | 396  | 128       | 396 + (396*128)/256 = 594   |
| 1    | 417  | 255       | 417 + (417*255)/256 = 832   |
| 2    | 528  | 64        | 528 + (528*64)/256 = 660    |
| 3    | 639  | 192       | 639 + (639*192)/256 = 1117  |

== Integration

- Consumes coherence bands from RaBiometricMatcher
- Outputs frequency values for audio synthesis/PWM generation
- Feeds into RaSonicEmitter for audible output
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaMusicChamberHarmonics where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Constants
-- =============================================================================

-- | Solfeggio base frequencies (Hz)
baseFreqs :: Vec 4 (Unsigned 16)
baseFreqs = 396 :> 417 :> 528 :> 639 :> Nil

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Map a single band: freq = base + (base * coherence) / 256
mapBand :: Unsigned 16 -> Unsigned 8 -> Unsigned 16
mapBand base coherence =
  let scaled = (resize base * resize coherence) :: Unsigned 24
  in base + resize (scaled `shiftR` 8)

-- | Map all bands to overtone frequencies
mapHarmonics :: Vec 4 (Unsigned 8) -> Vec 4 (Unsigned 16)
mapHarmonics bands = zipWith mapBand baseFreqs bands

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Harmonic mapper - converts coherence bands to overtone frequencies
harmonicMapper
  :: HiddenClockResetEnable dom
  => Signal dom (Vec 4 (Unsigned 8))      -- ^ Coherence bands
  -> Signal dom (Vec 4 (Unsigned 16))     -- ^ Overtone frequencies
harmonicMapper bands = fmap mapHarmonics bands

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
harmonicTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 4 (Unsigned 8))
  -> Signal System (Vec 4 (Unsigned 16))
harmonicTop = exposeClockResetEnable harmonicMapper

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test coherence bands
testBands :: Vec 4 (Vec 4 (Unsigned 8))
testBands = $(listToVecTH
  [ $(listToVecTH [0 :: Unsigned 8, 0, 0, 0])           -- No modulation
  , $(listToVecTH [128, 128, 128, 128])                 -- 50% modulation
  , $(listToVecTH [255, 255, 255, 255])                 -- Max modulation
  , $(listToVecTH [64, 192, 32, 224])                   -- Mixed modulation
  ])

-- | Expected outputs:
-- Band 0: [0,0,0,0] -> [396, 417, 528, 639] (no change)
-- Band 1: [128,128,128,128] -> [396+198, 417+208, 528+264, 639+319] = [594, 625, 792, 958]
-- Band 2: [255,255,255,255] -> [396+394, 417+415, 528+526, 639+636] = [790, 832, 1054, 1275]
-- Band 3: [64,192,32,224] -> mixed calculations
expectedOutput :: Vec 4 (Vec 4 (Unsigned 16))
expectedOutput = $(listToVecTH
  [ $(listToVecTH [396 :: Unsigned 16, 417, 528, 639])    -- No modulation
  , $(listToVecTH [594, 625, 792, 958])                   -- 50% modulation
  , $(listToVecTH [790, 832, 1054, 1275])                 -- Max modulation
  , $(listToVecTH [495, 729, 594, 1198])                  -- Mixed modulation
  ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for harmonic mapper validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testBands
    out = harmonicTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedOutput out
