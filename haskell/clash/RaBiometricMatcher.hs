{-|
Module      : RaBiometricMatcher
Description : Coherence profile matcher for biometric template validation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 33: Biometric coherence profile matcher. Compares input signal
patterns against reference templates to compute coherence scores.

== Biometric Templates

| Template       | Description                    | Pattern         |
|----------------|--------------------------------|-----------------|
| TemplateFlat   | Baseline, no variation         | Constant 128    |
| TemplateResonant | Full coherent oscillation    | 64-192 sinusoid |
| TemplatePulse  | Subtle pulse variation         | 108-148 wave    |

== Coherence Scoring

Score = 255 - (average absolute difference)

- 255: Perfect match (no difference)
- 0: Maximum mismatch

== Integration

Used by RaHandshakeGate for biometric validation before consent grant.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaBiometricMatcher where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Biometric template types for coherence reference
data BioTemplate = TemplateFlat | TemplateResonant | TemplatePulse
  deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Reference Waveforms
-- =============================================================================

-- | Get reference waveform for template type (16 samples)
referenceWaveform :: BioTemplate -> Vec 16 (Unsigned 8)
referenceWaveform TemplateFlat     = replicate d16 128
referenceWaveform TemplateResonant = $(listToVecTH
  [128, 140, 160, 180, 192, 180, 160, 140, 128, 116, 96, 76, 64, 76, 96, 116 :: Unsigned 8])
referenceWaveform TemplatePulse    = $(listToVecTH
  [128, 132, 136, 140, 144, 148, 144, 140, 136, 132, 128, 124, 120, 116, 112, 108 :: Unsigned 8])

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Compute absolute difference between two unsigned values
absDiff :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
absDiff a b = if a >= b then a - b else b - a

-- | Compute coherence score between input and reference
-- Returns 255 for perfect match, lower for worse match
computeScore :: Vec 16 (Unsigned 8) -> BioTemplate -> Unsigned 8
computeScore vIn temp =
  let ref = referenceWaveform temp
      diffs = zipWith absDiff vIn ref
      sumErr = fold (+) diffs
  in 255 - resize (sumErr `div` 16)

-- | Match coherence signal processor
-- Compares input pattern against selected template
matchCoherence
  :: HiddenClockResetEnable dom
  => Signal dom (Vec 16 (Unsigned 8))  -- ^ Input biometric pattern
  -> Signal dom BioTemplate            -- ^ Template to compare against
  -> Signal dom (Unsigned 8)           -- ^ Coherence score (0-255)
matchCoherence inputVec template = liftA2 computeScore inputVec template

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
matcherTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 16 (Unsigned 8))
  -> Signal System BioTemplate
  -> Signal System (Unsigned 8)
matcherTop = exposeClockResetEnable matchCoherence

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test input patterns
patternVec :: Vec 3 (Vec 16 (Unsigned 8))
patternVec = $(listToVecTH
  [ $(listToVecTH [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 :: Unsigned 8])
  , $(listToVecTH [128,140,160,180,192,180,160,140,128,116,96,76,64,76,96,116 :: Unsigned 8])
  , $(listToVecTH [100,120,140,160,180,160,140,120,100,80,60,40,20,40,60,80 :: Unsigned 8])
  ])

-- | Test templates
templates :: Vec 3 BioTemplate
templates = $(listToVecTH [TemplateFlat, TemplateResonant, TemplateResonant])

-- | Expected scores:
-- Pattern 1 vs TemplateFlat: Perfect match (all 128) -> 255
-- Pattern 2 vs TemplateResonant: Perfect match -> 255
-- Pattern 3 vs TemplateResonant: Significant mismatch -> ~180
expectedScores :: Vec 3 (Unsigned 8)
expectedScores = $(listToVecTH [255 :: Unsigned 8, 255, 180])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for biometric matcher validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    vStim = stimuliGenerator clk rst patternVec
    tStim = stimuliGenerator clk rst templates
    output = matcherTop clk rst enableGen vStim tStim
    done = outputVerifier' clk rst expectedScores output
