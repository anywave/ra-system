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
testInputs :: Vec 6 BioState
testInputs =
  BioState Inhale 200 :>       -- Not trigger phase
  BioState Exhale 240 :>       -- Not trigger phase
  BioState ExhaleHold 235 :>   -- Trigger: ExhaleHold + coherence > 230
  BioState Hold 250 :>         -- Not trigger phase (wrong hold type)
  BioState ExhaleHold 220 :>   -- ExhaleHold but coherence < 230
  BioState ExhaleHold 231 :>   -- Trigger: ExhaleHold + coherence > 230
  Nil

-- | Expected outputs
-- Test 0: Inhale, no trigger -> (False, False)
-- Test 1: Exhale, no trigger -> (False, False)
-- Test 2: ExhaleHold + 235 >= 230 -> (True, True)
-- Test 3: Hold (not ExhaleHold), no trigger -> (False, False)
-- Test 4: ExhaleHold + 220 < 230, no trigger -> (False, False)
-- Test 5: ExhaleHold + 231 >= 230 -> (True, True)
expectedOutput :: Vec 6 BioOutput
expectedOutput =
  BioOutput False False :>
  BioOutput False False :>
  BioOutput True True :>
  BioOutput False False :>
  BioOutput False False :>
  BioOutput True True :>
  Nil

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for biofeedback harness validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testInputs
    out = biofeedbackTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedOutput out
