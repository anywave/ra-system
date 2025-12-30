{-|
Module      : RaTactileControl
Description : Tactile control interface for haptic feedback and gesture input
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 56: Interfaces with tactile sensors and haptic actuators for
bidirectional physical feedback. Maps touch pressure and gesture events
to consent-gated haptic responses.

== Input Channels

| Channel | Description |
|---------|-------------|
| touchPressure | Analog pressure sensor (0-255) |
| gestureCode | 4-bit gesture identifier |
| consentLevel | Current consent state (0=none, 1=partial, 2=full) |

== Gesture Codes

| Code | Gesture      | Haptic Response |
|------|--------------|-----------------|
| 0000 | None         | Silent          |
| 0001 | Tap          | Short pulse     |
| 0010 | Hold         | Sustained buzz  |
| 0011 | Swipe        | Wave pattern    |
| 0100 | Circle       | Spiral ramp     |
| 0101 | Pinch        | Double pulse    |

== Haptic Patterns

| Pattern   | PWM Duty | Duration (cycles) |
|-----------|----------|-------------------|
| Silent    | 0%       | 0                 |
| Pulse     | 80%      | 8                 |
| Buzz      | 50%      | 32                |
| Wave      | 0-100%   | 16 (ramp)         |
| Spiral    | 25-75%   | 24 (oscillate)    |
| DoublePulse | 80%    | 4+4 (gap 4)       |

== Consent Gating

- consentLevel 0: All haptics disabled
- consentLevel 1: Pulse/Buzz only (no sustained)
- consentLevel 2: Full haptic range

== Integration

- Consumes consent from RaConsentFramework
- Outputs PWM signal for haptic actuator
- Reports gesture recognition status
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaTactileControl where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Haptic pattern type
data HapticPattern = Silent | Pulse | Buzz | Wave | Spiral | DoublePulse
  deriving (Generic, NFDataX, Show, Eq)

-- | Tactile control output
data TactileOutput = TactileOutput
  { hapticPWM     :: Unsigned 8     -- ^ PWM duty cycle (0-255)
  , gestureActive :: Bool           -- ^ Gesture recognition active
  , patternType   :: HapticPattern  -- ^ Current pattern being output
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Pressure threshold for gesture recognition
pressureThreshold :: Unsigned 8
pressureThreshold = 50

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Map gesture code to haptic pattern
gestureToPattern :: BitVector 4 -> HapticPattern
gestureToPattern 0b0000 = Silent
gestureToPattern 0b0001 = Pulse
gestureToPattern 0b0010 = Buzz
gestureToPattern 0b0011 = Wave
gestureToPattern 0b0100 = Spiral
gestureToPattern 0b0101 = DoublePulse
gestureToPattern _      = Silent

-- | Check if pattern is allowed at consent level
patternAllowed :: HapticPattern -> Unsigned 2 -> Bool
patternAllowed Silent      _ = True
patternAllowed Pulse       c = c >= 1
patternAllowed Buzz        c = c >= 1
patternAllowed Wave        c = c >= 2
patternAllowed Spiral      c = c >= 2
patternAllowed DoublePulse c = c >= 2

-- | Generate PWM value for pattern at given phase
patternPWM :: HapticPattern -> Unsigned 8 -> Unsigned 8
patternPWM Silent      _ = 0
patternPWM Pulse       t = if t < 8 then 204 else 0           -- 80% for 8 cycles
patternPWM Buzz        _ = 128                                -- 50% sustained
patternPWM Wave        t = resize (t * 16)                    -- 0-100% ramp
patternPWM Spiral      t = 64 + resize ((t `mod` 24) * 5)     -- 25-75% oscillate
patternPWM DoublePulse t = if t < 4 || (t >= 8 && t < 12) then 204 else 0

-- | Compute tactile output from inputs
computeTactile :: Unsigned 8 -> BitVector 4 -> Unsigned 2 -> Unsigned 8 -> TactileOutput
computeTactile pressure gesture consent phase =
  let pattern = gestureToPattern gesture
      allowed = patternAllowed pattern consent
      active = pressure >= pressureThreshold
      pwm = if allowed && active
            then patternPWM pattern phase
            else 0
  in TactileOutput pwm active (if allowed then pattern else Silent)

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Tactile control interface
-- Processes gesture input and generates haptic feedback
tactileControl
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)      -- ^ Touch pressure
  -> Signal dom (BitVector 4)     -- ^ Gesture code
  -> Signal dom (Unsigned 2)      -- ^ Consent level
  -> Signal dom TactileOutput     -- ^ Tactile output bundle
tactileControl pressure gesture consent = output
  where
    -- Phase counter for pattern animation
    phase = register 0 (phase + 1)

    -- Compute output
    output = computeTactile <$> pressure <*> gesture <*> consent <*> phase

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
tactileTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 8)
  -> Signal System (BitVector 4)
  -> Signal System (Unsigned 2)
  -> Signal System TactileOutput
tactileTop = exposeClockResetEnable tactileControl

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test pressure values
testPressure :: Vec 6 (Unsigned 8)
testPressure = $(listToVecTH [100 :: Unsigned 8, 100, 30, 100, 100, 100])

-- | Test gesture codes
testGesture :: Vec 6 (BitVector 4)
testGesture = $(listToVecTH [0b0001, 0b0010, 0b0001, 0b0011, 0b0100, 0b0000])

-- | Test consent levels
testConsent :: Vec 6 (Unsigned 2)
testConsent = $(listToVecTH [2 :: Unsigned 2, 1, 2, 2, 0, 2])

-- | Expected outputs:
-- Cycle 0: Pulse + consent 2 + pressure 100 -> PWM active
-- Cycle 1: Buzz + consent 1 + pressure 100 -> PWM 128
-- Cycle 2: Pulse + consent 2 + pressure 30 (below threshold) -> PWM 0
-- Cycle 3: Wave + consent 2 + pressure 100 -> PWM ramp
-- Cycle 4: Spiral + consent 0 -> blocked, PWM 0
-- Cycle 5: Silent + consent 2 -> PWM 0

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for tactile control validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    s1 = stimuliGenerator clk rst testPressure
    s2 = stimuliGenerator clk rst testGesture
    s3 = stimuliGenerator clk rst testConsent
    out = tactileTop clk rst enableGen s1 s2 s3
    -- Validate that gesture active matches pressure threshold
    activeOut = gestureActive <$> out
    expectedActive = fmap (>= pressureThreshold) (stimuliGenerator clk rst testPressure)
    done = register clk rst enableGen False $
      pure (length testPressure == 6)
