{-|
Module      : RaConsentTransformer
Description : Multi-core consent transformer with quorum voting
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 31: Distributed consent logic across multiple avatar threads.
Consent is granted only when quorum threshold is met AND biometric
context (coherence + aura) is above activation thresholds.

== Activation Requirements

- coherenceScore >= 180
- auraIntensity >= 128

If either threshold is not met, all votes are suspended (treated as False).

== Outputs

| Field | Description |
|-------|-------------|
| consentGranted | True if quorum % of active votes agree |
| fieldEntropy | Number of dissenting votes |
| activeVotes | Number of qualifying True votes |

== Integration

Links to: Prompt 12, 18, 46 for downstream consent unlocks
Depends on: Prompt 33 (BiometricMatcher), Prompt 34 (ScalarExpression)
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module RaConsentTransformer where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Constants
-- =============================================================================

-- | Minimum coherence score to activate voting
coherenceThreshold :: Unsigned 8
coherenceThreshold = 180

-- | Minimum aura intensity to activate voting
auraThreshold :: Unsigned 8
auraThreshold = 128

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Count True values in a vector
countTrue :: KnownNat n => Vec n Bool -> Unsigned 8
countTrue = fold (+) . map (\b -> if b then 1 else 0)

-- | Compute consent state from inputs
computeConsent
  :: forall n. KnownNat n
  => Unsigned 8           -- ^ coherence score
  -> Unsigned 8           -- ^ aura intensity
  -> Vec n Bool           -- ^ avatar votes
  -> Unsigned 8           -- ^ quorum threshold (0-100)
  -> (Bool, Unsigned 8, Unsigned 8)  -- ^ (granted, entropy, activeVotes)
computeConsent coherence aura votes threshold =
  let -- Check if biometric context is active
      contextActive = coherence >= coherenceThreshold && aura >= auraThreshold
      -- Apply context filter to votes
      activeVotes = if contextActive then votes else replicate SNat False
      -- Count agrees and total
      total = fromIntegral (length activeVotes) :: Unsigned 8
      agrees = countTrue activeVotes
      disagrees = total - agrees
      -- Calculate required votes for quorum
      required = (threshold * total) `div` 100
      -- Determine consent
      granted = agrees >= required && agrees > 0
  in (granted, disagrees, agrees)

-- | Signal-level consent transformer
consentTransform
  :: forall n dom. (KnownNat n, HiddenClockResetEnable dom)
  => Signal dom (Unsigned 8)         -- ^ coherenceScore
  -> Signal dom (Unsigned 8)         -- ^ auraIntensity
  -> Signal dom (Vec n Bool)         -- ^ avatarVotes
  -> Signal dom (Unsigned 8)         -- ^ quorumThreshold %
  -> Signal dom (Bool, Unsigned 8, Unsigned 8)  -- ^ (consentGranted, fieldEntropy, activeVotes)
consentTransform coherence aura votes threshold =
  computeConsent <$> coherence <*> aura <*> votes <*> threshold

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis (3 voters)
consentTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 8)
  -> Signal System (Unsigned 8)
  -> Signal System (Vec 3 Bool)
  -> Signal System (Unsigned 8)
  -> Signal System (Bool, Unsigned 8, Unsigned 8)
consentTop = exposeClockResetEnable (consentTransform @3)

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test coherence values
testCoh :: Vec 3 (Unsigned 8)
testCoh = $(listToVecTH [190 :: Unsigned 8, 170, 200])

-- | Test aura values
testAura :: Vec 3 (Unsigned 8)
testAura = $(listToVecTH [140 :: Unsigned 8, 100, 160])

-- | Test vote vectors
testVotes :: Vec 3 (Vec 3 Bool)
testVotes = $(listToVecTH
  [ $(listToVecTH [True, True, True])
  , $(listToVecTH [True, False, False])
  , $(listToVecTH [True, True, False])
  ])

-- | Test thresholds
testThresh :: Vec 3 (Unsigned 8)
testThresh = $(listToVecTH [66 :: Unsigned 8, 50, 75])

-- | Expected outputs:
-- Test 1: (190, 140) active, [T,T,T], 66% -> 3/3 = 100% >= 66% -> granted, 0 entropy, 3 active
-- Test 2: (170, 100) NOT active (aura < 128) -> all votes False -> not granted, 0 entropy, 0 active
-- Test 3: (200, 160) active, [T,T,F], 75% -> 2/3 = 66% < 75% -> not granted, 1 entropy, 2 active
expectedOut :: Vec 3 (Bool, Unsigned 8, Unsigned 8)
expectedOut = $(listToVecTH
  [ (True, 0, 3)
  , (False, 0, 0)
  , (False, 1, 2)
  ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for consent transformer validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    s1 = stimuliGenerator clk rst testCoh
    s2 = stimuliGenerator clk rst testAura
    s3 = stimuliGenerator clk rst testVotes
    s4 = stimuliGenerator clk rst testThresh
    out = consentTop clk rst enableGen s1 s2 s3 s4
    done = outputVerifier' clk rst expectedOut out
