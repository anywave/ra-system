{-|
Module      : RaScalarExpression
Description : Biometric-to-Avatar scalar expression mapping
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 34: Maps biometric coherence and breath phase to avatar visual
expression parameters (aura intensity and limb motion vector).

== Expression Mapping

| Coherence | Breath Phase | Aura Intensity | Limb Vector |
|-----------|--------------|----------------|-------------|
| >= 200    | Exhale       | coherence      | +40         |
| >= 150    | Exhale       | 128            | +20         |
| >= 150    | Inhale       | 128            | -20         |
| < 150     | Any          | 64             | 0           |

== Integration

Pipeline: RaBiometricMatcher (Prompt 33) -> RaScalarExpression (Prompt 34) -> RaConsentFramework (Prompt 32)

Dashboard outputs:
- Aura ring visualization (0-255 luminance)
- Limb vector bar (-128 to +127 motion intent)
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaScalarExpression where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Avatar expression output bundle
data AvatarExpression = AvatarExpression
  { auraIntensity :: Unsigned 8  -- ^ Visual field luminance [0-255]
  , limbVector    :: Signed 8    -- ^ Motion intent scalar [-128 to +127]
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Compute avatar expression from coherence and breath phase
computeExpression :: Unsigned 8 -> Bool -> AvatarExpression
computeExpression coherence breathExhale
  | coherence >= 200 && breathExhale = AvatarExpression coherence 40
  | coherence >= 150 = AvatarExpression 128 (if breathExhale then 20 else (-20))
  | otherwise = AvatarExpression 64 0

-- | Signal-level expression mapper
mapExpression
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)     -- ^ Coherence score (0-255)
  -> Signal dom Bool             -- ^ Breath phase (True = exhale)
  -> Signal dom AvatarExpression -- ^ Avatar expression output
mapExpression coherence breath = liftA2 computeExpression coherence breath

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
expressionTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 8)
  -> Signal System Bool
  -> Signal System AvatarExpression
expressionTop = exposeClockResetEnable mapExpression

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Coherence input sequence
cohInput :: Vec 3 (Unsigned 8)
cohInput = $(listToVecTH [255 :: Unsigned 8, 180, 100])

-- | Breath phase sequence (True = exhale)
bphInput :: Vec 3 Bool
bphInput = $(listToVecTH [True, False, True])

-- | Expected outputs:
-- (255, True)  -> coherence >= 200, exhale -> AvatarExpression 255 40
-- (180, False) -> coherence >= 150, inhale -> AvatarExpression 128 (-20)
-- (100, True)  -> coherence < 150         -> AvatarExpression 64 0
expected :: Vec 3 AvatarExpression
expected = $(listToVecTH
  [ AvatarExpression 255 40
  , AvatarExpression 128 (-20)
  , AvatarExpression 64 0
  ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for scalar expression validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    cStim = stimuliGenerator clk rst cohInput
    bStim = stimuliGenerator clk rst bphInput
    output = expressionTop clk rst enableGen cStim bStim
    done = outputVerifier' clk rst expected output
