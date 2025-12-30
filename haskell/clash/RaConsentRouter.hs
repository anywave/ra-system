{-|
Module      : RaConsentRouter
Description : Consent State to Channel Router
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Routes consent states to downstream trigger channels based on state values.
Integrates with RaConsentFramework for complete consent pipeline.

== Trigger Channels

| Channel        | Activation Condition                           |
|----------------|------------------------------------------------|
| bioTrigger     | Permit OR Override                             |
| gestureTrigger | Override only                                  |
| fieldTrigger   | Permit AND coherenceGate high                  |

== Truth Table

| ConsentState | coherenceGate | bioTrigger | gestureTrigger | fieldTrigger |
|--------------|---------------|------------|----------------|--------------|
| Permit       | False         | True       | False          | False        |
| Permit       | True          | True       | False          | True         |
| Restrict     | False         | False      | False          | False        |
| Restrict     | True          | False      | False          | False        |
| Override     | False         | True       | True           | False        |
| Override     | True          | True       | True           | False        |

== Pipeline Integration

@
ConsentFramework ─▶ ConsentState ─┬─▶ bioTrigger ──▶ Bio Systems
                                  ├─▶ gestureTrigger ──▶ Gesture Control
CoherenceGate ────────────────────┴─▶ fieldTrigger ──▶ Field Emitters
@
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaConsentRouter where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | ConsentState encodes the user's current permission signature
-- Imported conceptually from RaConsentFramework
data ConsentState = Permit | Restrict | Override
  deriving (Show, Eq, Generic, NFDataX)

-- | Trigger output bundle
type TriggerBundle = (Bool, Bool, Bool)  -- (bioTrigger, gestureTrigger, fieldTrigger)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Route consent state to trigger channels
-- Pure combinational logic - no state required
routeConsent :: ConsentState -> Bool -> TriggerBundle
routeConsent state coherenceGate = (bioTrigger, gestureTrigger, fieldTrigger)
  where
    -- bioTrigger: activates if state is Permit or Override
    bioTrigger = state == Permit || state == Override

    -- gestureTrigger: activates only if state is Override
    gestureTrigger = state == Override

    -- fieldTrigger: activates if state is Permit AND coherenceGate is high
    fieldTrigger = state == Permit && coherenceGate

-- | Signal-level consent router
-- Lifts pure routing logic to signal domain
consentRouter
  :: HiddenClockResetEnable dom
  => Signal dom ConsentState
  -> Signal dom Bool
  -> (Signal dom Bool, Signal dom Bool, Signal dom Bool)
consentRouter stateS gateS = unbundle $ routeConsent <$> stateS <*> gateS

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top entity for Clash synthesis
-- Input: ConsentState signal, coherenceGate signal
-- Output: (bioTrigger, gestureTrigger, fieldTrigger)
consentRouterTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System ConsentState
  -> Signal System Bool
  -> (Signal System Bool, Signal System Bool, Signal System Bool)
consentRouterTop clk rst en stateS gateS =
  withClockResetEnable clk rst en $ consentRouter stateS gateS

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test input states (covers all trigger combinations)
testStates :: Vec 6 ConsentState
testStates = $(listToVecTH
  [ Permit    -- Case 1: Permit + gate low
  , Permit    -- Case 2: Permit + gate high
  , Restrict  -- Case 3: Restrict + gate low
  , Restrict  -- Case 4: Restrict + gate high
  , Override  -- Case 5: Override + gate low
  , Override  -- Case 6: Override + gate high
  ])

-- | Test coherence gate values
testGates :: Vec 6 Bool
testGates = $(listToVecTH [False, True, False, True, False, True])

-- | Expected bioTrigger outputs
expectedBio :: Vec 6 Bool
expectedBio = $(listToVecTH [True, True, False, False, True, True])

-- | Expected gestureTrigger outputs
expectedGesture :: Vec 6 Bool
expectedGesture = $(listToVecTH [False, False, False, False, True, True])

-- | Expected fieldTrigger outputs
expectedField :: Vec 6 Bool
expectedField = $(listToVecTH [False, True, False, False, False, False])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for consent router validation
-- Tests all six input combinations and verifies trigger outputs
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen

    -- Stimuli generators
    stateStim = stimuliGenerator clk rst testStates
    gateStim = stimuliGenerator clk rst testGates

    -- Router under test
    (bioOut, gestureOut, fieldOut) = consentRouterTop clk rst enableGen stateStim gateStim

    -- Output verifiers
    bioVerifier = outputVerifier' clk rst expectedBio
    gestureVerifier = outputVerifier' clk rst expectedGesture
    fieldVerifier = outputVerifier' clk rst expectedField

    -- All verifiers must pass
    done = bioVerifier bioOut .&&. gestureVerifier gestureOut .&&. fieldVerifier fieldOut
