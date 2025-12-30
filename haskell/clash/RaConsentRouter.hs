{-|
Module      : RaConsentRouter
Description : Unified downstream activation routing from ConsentState
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Used to control biometric, gesture, and field triggers based on consent state
and external coherence gate.

== Trigger Channels

| Channel        | Activation Condition                           |
|----------------|------------------------------------------------|
| bioTrigger     | Permit OR Override                             |
| gestureTrigger | Override only                                  |
| fieldTrigger   | Permit AND coherenceGate high                  |

== Pipeline Integration

@
ConsentFramework ──▶ ConsentState ─┬─▶ bioTrigger ──▶ Bio Systems
                                   ├─▶ gestureTrigger ──▶ Gesture Control
CoherenceGate ─────────────────────┴─▶ fieldTrigger ──▶ Field Emitters
@
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaConsentRouter where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | ConsentState encoding
data ConsentState = Permit | Restrict | Override
  deriving (Show, Eq, Generic, NFDataX)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Main router logic: activates based on consent state and external coherence
consentRouter
  :: HiddenClockResetEnable dom
  => Signal dom ConsentState
  -> Signal dom Bool               -- ^ coherenceGate (scalar)
  -> (Signal dom Bool, Signal dom Bool, Signal dom Bool)
     -- ^ (bioTrigger, gestureTrigger, fieldTrigger)
consentRouter stateS coherenceS = (bioTrigger, gestureTrigger, fieldTrigger)
  where
    bioTrigger     = fmap (\s -> s == Permit || s == Override) stateS
    gestureTrigger = fmap (== Override) stateS
    fieldTrigger   = liftA2 (&&) (fmap (== Permit) stateS) coherenceS

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top entity with explicit clock, reset, enable
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System ConsentState
  -> Signal System Bool
  -> (Signal System Bool, Signal System Bool, Signal System Bool)
topEntity = exposeClockResetEnable consentRouter

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test vectors for consent states
testStates :: Vec 6 ConsentState
testStates = $(listToVecTH [Permit, Override, Restrict, Permit, Override, Restrict])

-- | Test vectors for coherence gate
testCoherence :: Vec 6 Bool
testCoherence = $(listToVecTH [True, False, True, False, True, False])

-- | Expected bioTrigger outputs
expectedBio :: Vec 6 Bool
expectedBio = $(listToVecTH [True, True, False, True, True, False])

-- | Expected gestureTrigger outputs
expectedGesture :: Vec 6 Bool
expectedGesture = $(listToVecTH [False, True, False, False, True, False])

-- | Expected fieldTrigger outputs
expectedField :: Vec 6 Bool
expectedField = $(listToVecTH [True, False, False, False, False, False])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for consent router validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stateS = stimuliGenerator clk rst testStates
    coherS = stimuliGenerator clk rst testCoherence
    (bioS, gestS, fieldS) = topEntity clk rst enableGen stateS coherS
    bioCheck = outputVerifier' clk rst expectedBio bioS
    gestCheck = outputVerifier' clk rst expectedGesture gestS
    fieldCheck = outputVerifier' clk rst expectedField fieldS
    done = bioCheck .&&. gestCheck .&&. fieldCheck
