{-|
Module      : RaHarmonicTwist
Description : Harmonic inversion twist envelope generator
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 49: Computes twist envelope from harmonic mode pair Y(a,b) and
coherence level. Produces twist magnitude and duration based on
coherence threshold gating.

== Harmonic Mode Encoding

Y(a,b) represents spherical harmonic modes where:
- a = azimuthal mode (0-15)
- b = polar mode (0-15)

== Twist Calculation

@
invMag = a * 10 + b * 7
twistMag = invMag + 15  (if coherence >= 0.41)
         = invMag - 10  (if coherence < 0.41)
duration = 8 cycles     (if twistMag > 80)
         = 4 cycles     (if twistMag <= 80)
@

== Coherence Threshold

coherenceThreshold = 105 (~0.41 on 0-255 scale)

== Example Calculations

| modeA | modeB | coherence | invMag | twistMag | duration |
|-------|-------|-----------|--------|----------|----------|
| 5     | 3     | 120       | 71     | 86       | 8        |
| 2     | 4     | 80        | 48     | 38       | 4        |
| 8     | 6     | 200       | 122    | 137      | 8        |

== Integration

- Consumes harmonic mode from field analysis
- Outputs twist envelope for animation/rendering
- Duration feeds into timing subsystem
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaHarmonicTwist where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Fixed-point representation (0-255 maps to 0.0-1.0)
type Fixed = Unsigned 8

-- | Harmonic input bundle
data HarmonicInput = HarmonicInput
  { modeA     :: Unsigned 4    -- ^ Azimuthal mode Y(a,b)
  , modeB     :: Unsigned 4    -- ^ Polar mode
  , coherence :: Fixed         -- ^ Coherence level (0-255)
  } deriving (Generic, NFDataX, Show, Eq)

-- | Twist result bundle
data TwistResult = TwistResult
  { twistVector  :: Fixed          -- ^ Twist magnitude
  , durationPhiN :: Unsigned 8     -- ^ Duration in phi-N units
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Coherence threshold (~0.41)
coherenceThreshold :: Fixed
coherenceThreshold = 105

-- | Twist magnitude threshold for duration selection
twistMagThreshold :: Fixed
twistMagThreshold = 80

-- | Duration for high twist magnitude
durationHigh :: Unsigned 8
durationHigh = 8

-- | Duration for low twist magnitude
durationLow :: Unsigned 8
durationLow = 4

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Compute inverse magnitude from harmonic modes
-- Formula: invMag = a * 10 + b * 7
harmonicInverseY :: Unsigned 4 -> Unsigned 4 -> Fixed
harmonicInverseY a b = resize (resize a * 10 + resize b * 7 :: Unsigned 8)

-- | Compute twist envelope from harmonic input
computeTwistEnvelope :: HarmonicInput -> TwistResult
computeTwistEnvelope hi =
  let
    invMag = harmonicInverseY (modeA hi) (modeB hi)
    -- Apply coherence-based modulation
    twistMag = if coherence hi >= coherenceThreshold
               then satAdd SatBound invMag 15
               else satSub SatBound invMag 10
    -- Determine duration based on twist magnitude
    duration = if twistMag > twistMagThreshold
               then durationHigh
               else durationLow
  in
    TwistResult
      { twistVector  = twistMag
      , durationPhiN = duration
      }

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Harmonic twist processor
harmonicTwistProcessor
  :: HiddenClockResetEnable dom
  => Signal dom HarmonicInput       -- ^ Input harmonic parameters
  -> Signal dom TwistResult         -- ^ Twist envelope result
harmonicTwistProcessor input = fmap computeTwistEnvelope input

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
harmonicTwistTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System HarmonicInput
  -> Signal System TwistResult
harmonicTwistTop = exposeClockResetEnable harmonicTwistProcessor

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test harmonic inputs
testInputs :: Vec 6 HarmonicInput
testInputs =
  HarmonicInput 5 3 120 :>    -- invMag=71, high coh -> 86, dur=8
  HarmonicInput 2 4 80 :>     -- invMag=48, low coh -> 38, dur=4
  HarmonicInput 8 6 200 :>    -- invMag=122, high coh -> 137, dur=8
  HarmonicInput 0 0 150 :>    -- invMag=0, high coh -> 15, dur=4
  HarmonicInput 10 8 50 :>    -- invMag=156, low coh -> 146, dur=8
  HarmonicInput 4 5 105 :>    -- invMag=75, at threshold -> 90, dur=8
  Nil

-- | Expected outputs
-- Test 0: invMag = 5*10 + 3*7 = 71, coh 120 >= 105 -> 71+15=86, 86>80 -> dur=8
-- Test 1: invMag = 2*10 + 4*7 = 48, coh 80 < 105 -> 48-10=38, 38<=80 -> dur=4
-- Test 2: invMag = 8*10 + 6*7 = 122, coh 200 >= 105 -> 122+15=137, 137>80 -> dur=8
-- Test 3: invMag = 0*10 + 0*7 = 0, coh 150 >= 105 -> 0+15=15, 15<=80 -> dur=4
-- Test 4: invMag = 10*10 + 8*7 = 156, coh 50 < 105 -> 156-10=146, 146>80 -> dur=8
-- Test 5: invMag = 4*10 + 5*7 = 75, coh 105 >= 105 -> 75+15=90, 90>80 -> dur=8
expectedOutput :: Vec 6 TwistResult
expectedOutput =
  TwistResult 86 8 :>
  TwistResult 38 4 :>
  TwistResult 137 8 :>
  TwistResult 15 4 :>
  TwistResult 146 8 :>
  TwistResult 90 8 :>
  Nil

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for harmonic twist validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testInputs
    out = harmonicTwistTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedOutput out
