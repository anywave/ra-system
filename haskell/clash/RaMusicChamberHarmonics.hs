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

-- | Test coherence bands (0% to 100% in key transitions)
testBands :: Vec 6 (Vec 4 (Unsigned 8))
testBands =
  (0 :> 0 :> 0 :> 0 :> Nil) :>               -- 0% modulation
  (64 :> 64 :> 64 :> 64 :> Nil) :>           -- 25% modulation
  (128 :> 128 :> 128 :> 128 :> Nil) :>       -- 50% modulation
  (192 :> 192 :> 192 :> 192 :> Nil) :>       -- 75% modulation
  (255 :> 255 :> 255 :> 255 :> Nil) :>       -- 100% modulation
  (100 :> 200 :> 50 :> 250 :> Nil) :> Nil    -- Mixed modulation

-- | Expected outputs:
-- 0%:   base frequencies unchanged
-- 25%:  base + base*64/256  = base * 1.25
-- 50%:  base + base*128/256 = base * 1.50
-- 75%:  base + base*192/256 = base * 1.75
-- 100%: base + base*255/256 = base * ~2.0
-- Mixed: per-band calculation
expectedOutput :: Vec 6 (Vec 4 (Unsigned 16))
expectedOutput =
  (396 :> 417 :> 528 :> 639 :> Nil) :>       -- 0%
  (495 :> 521 :> 660 :> 798 :> Nil) :>       -- 25%
  (594 :> 625 :> 792 :> 957 :> Nil) :>       -- 50%
  (693 :> 729 :> 924 :> 1116 :> Nil) :>      -- 75%
  (792 :> 833 :> 1056 :> 1275 :> Nil) :>     -- 100%
  (551 :> 733 :> 631 :> 1264 :> Nil) :> Nil  -- Mixed

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
