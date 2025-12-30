{-|
Module      : RaBiofeedbackHarness
Description : Breath-hold + coherence trigger for physical/energetic output
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 52: Maps exhale-hold breath transition with high coherence to
MotionIntent and HapticPing outputs for avatar physical response.

== Trigger Logic

@
Trigger = (breathPhase == ExhaleHold) AND (coherence >= 230)
@

== Breath Phases

| Phase      | Description                    |
|------------|--------------------------------|
| Inhale     | Breathing in                   |
| Exhale     | Breathing out                  |
| Hold       | Breath held (post-inhale)      |
| ExhaleHold | Breath held after exhale       |

== Outputs

| Signal       | Description                    |
|--------------|--------------------------------|
| motionIntent | Triggers limb movement cascade |
| hapticPing   | Triggers haptic feedback pulse |

== Integration

- Downstream from RaFieldTransferBus (Prompt 35)
- Links to chamber update via RPP field coherence
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaBiofeedbackHarness where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Fixed-point representation (0-255 maps to 0.0-1.0)
type Fixed = Unsigned 8

-- | Breath phase ADT
data BreathPhase
  = Inhale      -- ^ Breathing in
  | Exhale      -- ^ Breathing out
  | Hold        -- ^ Breath held (post-inhale)
  | ExhaleHold  -- ^ Breath held after exhale (trigger phase)
  deriving (Show, Eq, Generic, NFDataX)

-- | Biofeedback input state
data BioState = BioState
  { phase     :: BreathPhase   -- ^ Current breath phase
  , coherence :: Fixed         -- ^ Coherence level (0-255)
  } deriving (Show, Eq, Generic, NFDataX)

-- | Biofeedback output bundle
data BioOutput = BioOutput
  { motionIntent :: Bool       -- ^ Triggers limb movement cascade
  , hapticPing   :: Bool       -- ^ Triggers haptic feedback pulse
  } deriving (Show, Eq, Generic, NFDataX)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Coherence threshold for trigger activation (~0.9)
coherenceThreshold :: Fixed
coherenceThreshold = 230

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Check if breath phase is the trigger phase (ExhaleHold)
triggerBreath :: BreathPhase -> Bool
triggerBreath ExhaleHold = True
triggerBreath _          = False

-- | Process biofeedback state to generate output signals
processBiofeedback :: BioState -> BioOutput
processBiofeedback bio =
  let active = triggerBreath (phase bio) && coherence bio >= coherenceThreshold
  in BioOutput
     { motionIntent = active
     , hapticPing   = active
     }

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Biofeedback harness - maps breath state + coherence to physical output
biofeedbackHarness
  :: HiddenClockResetEnable dom
  => Signal dom BioState           -- ^ Input biofeedback state
  -> Signal dom BioOutput          -- ^ Output motion/haptic signals
biofeedbackHarness = fmap processBiofeedback

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
biofeedbackTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System BioState
  -> Signal System BioOutput
biofeedbackTop = exposeClockResetEnable biofeedbackHarness

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test biofeedback states
testInputs :: Vec 4 BioState
testInputs =
  BioState ExhaleHold 240 :>   -- Trigger: ExhaleHold + coherence >= 230
  BioState Exhale 240 :>       -- Not trigger phase (wrong phase)
  BioState ExhaleHold 200 :>   -- ExhaleHold but coherence < 230
  BioState Inhale 255 :>       -- Not trigger phase (wrong phase)
  Nil

-- | Expected outputs
-- Test 0: ExhaleHold + 240 >= 230 -> (True, True)
-- Test 1: Exhale, not trigger phase -> (False, False)
-- Test 2: ExhaleHold + 200 < 230, no trigger -> (False, False)
-- Test 3: Inhale, not trigger phase -> (False, False)
testExpected :: Vec 4 BioOutput
testExpected =
  BioOutput True True :>
  BioOutput False False :>
  BioOutput False False :>
  BioOutput False False :>
  Nil

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for biofeedback harness validation
testBench :: Signal System Bool
testBench = done
  where
    testInput = stimuliGenerator testInputs
    expectedOutput = outputVerifier' testExpected
    done = expectedOutput (biofeedbackHarness testInput)
