{-|
Module      : RaBiofeedbackHarness
Description : Breath-hold + coherence trigger for physical/energetic output
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 52: Maps exhale-hold breath transition with high coherence to
MotionIntent and HapticPing outputs for avatar physical response.

== Trigger Logic

@
Trigger = (breathPhase transitions exhale→hold) AND (coherence > 230)
@

== Breath Phases

| Code | Phase  |
|------|--------|
| 00   | Inhale |
| 01   | Exhale |
| 10   | Hold   |
| 11   | Rest   |

== Outputs

| Signal | Description |
|--------|-------------|
| MotionIntent | Triggers limb movement cascade |
| HapticPing | Triggers haptic feedback pulse |

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
-- Constants
-- =============================================================================

-- | Coherence threshold for trigger activation
coherenceThreshold :: Unsigned 8
coherenceThreshold = 230

-- | Breath phase codes
phaseInhale, phaseExhale, phaseHold, phaseRest :: BitVector 2
phaseInhale = 0b00
phaseExhale = 0b01
phaseHold   = 0b10
phaseRest   = 0b11

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Check if breath phase matches target
breathMatch :: BitVector 2 -> BitVector 2 -> Bool
breathMatch current target = current == target

-- | Detect exhale-to-hold transition
detectExhaleHold :: BitVector 2 -> BitVector 2 -> Bool
detectExhaleHold prev curr =
  breathMatch prev phaseExhale && breathMatch curr phaseHold

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Biofeedback harness - maps breath + coherence to physical output
biofeedbackHarness
  :: HiddenClockResetEnable dom
  => Signal dom (BitVector 2)        -- ^ breathPhase (2-bit code)
  -> Signal dom (Unsigned 8)         -- ^ coherenceScore (0-255)
  -> Signal dom (Bool, Bool)         -- ^ (MotionIntent, HapticPing)
biofeedbackHarness breath coherence = bundle (motionIntent, hapticPulse)
  where
    -- Track previous breath phase
    prevBreath = register phaseRest breath

    -- Detect exhale→hold transition
    exhaleHoldEvent = liftA2 detectExhaleHold prevBreath breath

    -- Check coherence above threshold
    coherenceHigh = fmap (> coherenceThreshold) coherence

    -- Trigger on both conditions
    trigger = liftA2 (&&) exhaleHoldEvent coherenceHigh

    -- Output pulses (registered for timing)
    motionIntent = register False trigger
    hapticPulse  = register False trigger

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
biofeedbackTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (BitVector 2)
  -> Signal System (Unsigned 8)
  -> Signal System (Bool, Bool)
biofeedbackTop = exposeClockResetEnable biofeedbackHarness

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test breath phases: rest → exhale → hold → inhale → exhale → hold
testBreath :: Vec 6 (BitVector 2)
testBreath = $(listToVecTH [0b11, 0b01, 0b10, 0b00, 0b01, 0b10])

-- | Test coherence values: varying around threshold
testCoherence :: Vec 6 (Unsigned 8)
testCoherence = $(listToVecTH [200 :: Unsigned 8, 240, 235, 220, 250, 231])

-- | Expected outputs:
-- Cycle 0: rest→exhale, not exhale→hold, no trigger
-- Cycle 1: exhale→hold, coherence 235 > 230, TRIGGER (output at cycle 2 due to register)
-- Cycle 2: hold→inhale, not exhale→hold, no trigger
-- Cycle 3: inhale→exhale, not exhale→hold, no trigger
-- Cycle 4: exhale→hold, coherence 231 > 230, TRIGGER (output at cycle 5 due to register)
expectedOutput :: Vec 6 (Bool, Bool)
expectedOutput = $(listToVecTH
  [ (False, False)   -- Cycle 0: no trigger yet
  , (False, False)   -- Cycle 1: trigger detected, not output yet
  , (True, True)     -- Cycle 2: previous trigger output
  , (False, False)   -- Cycle 3: no trigger
  , (False, False)   -- Cycle 4: trigger detected, not output yet
  , (True, True)     -- Cycle 5: previous trigger output
  ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for biofeedback harness validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    s1 = stimuliGenerator clk rst testBreath
    s2 = stimuliGenerator clk rst testCoherence
    out = biofeedbackTop clk rst enableGen s1 s2
    done = outputVerifier' clk rst expectedOutput out
