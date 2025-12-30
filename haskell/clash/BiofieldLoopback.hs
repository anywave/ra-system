{-|
Module      : BiofieldLoopback
Description : Clash FPGA synthesis module for biofield loopback system
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Clash-ready synthesis module for the biofield loopback feedback system.
This module can be synthesized to Verilog/VHDL for FPGA implementation.

== Synthesis Commands

Generate Verilog:
@
clash --verilog BiofieldLoopback.hs
@

Generate VHDL:
@
clash --vhdl BiofieldLoopback.hs
@

Generate VCD waveforms (via Testbench):
@
clash --vcd Testbench.hs
@
-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE NoImplicitPrelude #-}

module BiofieldLoopback where

import Clash.Prelude
import GHC.Generics (Generic)
import qualified Prelude as P

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Biometric Input: breath rate (Hz), HRV (0 to 1.0)
-- These are the two primary inputs for coherence calculation
data BiometricInput = BiometricInput
  { breathRate :: Float    -- ^ Breath rate in Hz, optimal at 6.5 Hz
  , hrv        :: Float    -- ^ Heart rate variability [0, 1]
  } deriving (Generic, Show, Eq, NFDataX)

-- | EmergenceGlow as symbolic coherence indicator
-- Maps coherence levels to visual feedback states
data EmergenceGlow = None | Low | Moderate | High
  deriving (Generic, Show, Eq, Enum, NFDataX)

-- | AvatarFieldFrame includes glow and coherence state
-- This is the output frame updated each clock cycle
data AvatarFieldFrame = AvatarFieldFrame
  { glowState :: EmergenceGlow  -- ^ Current glow classification
  , coherence :: Float          -- ^ Raw coherence value
  } deriving (Generic, Show, Eq, NFDataX)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Classify coherence level into EmergenceGlow
-- Thresholds: 0.4 (Low), 0.65 (Moderate), 0.85 (High)
classifyGlow :: Float -> EmergenceGlow
classifyGlow c
  | c < 0.4   = None
  | c < 0.65  = Low
  | c < 0.85  = Moderate
  | otherwise = High

-- | Coherence as simple function of breath proximity to 6.5 Hz and HRV
-- Formula: (6.5 - |6.5 - breathRate|) * hrv
-- Maximum coherence when breath rate = 6.5 Hz (resonant frequency)
computeCoherence :: BiometricInput -> Float
computeCoherence (BiometricInput br h) = (6.5 - abs (6.5 - br)) * h

-- | Core feedback loop logic: update AvatarFieldFrame based on biometric input
-- Returns (newFrame, pulseChanged) where pulseChanged indicates glow transition
updateFeedbackLoop :: BiometricInput -> AvatarFieldFrame -> (AvatarFieldFrame, Bool)
updateFeedbackLoop input prev =
  let cLevel = computeCoherence input
      gNew = classifyGlow cLevel
      frameOut = prev { glowState = gNew, coherence = cLevel }
      pulseChanged = gNew /= glowState prev
  in (frameOut, pulseChanged)

-- =============================================================================
-- Clash Synthesis - Signal Types
-- =============================================================================

-- | Frame output signal (clocked)
type FrameSignal dom = Signal dom AvatarFieldFrame

-- | Biometric input signal (clocked)
type InputSignal dom = Signal dom BiometricInput

-- =============================================================================
-- Top Entity - FPGA Synthesis Entry Point
-- =============================================================================

-- | Top entity for synthesis
-- Exposes clock, reset, and enable for explicit control
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> InputSignal System
  -> FrameSignal System
topEntity = exposeClockResetEnable biofieldLoopback

-- | Hidden clock/reset version of the biofield loopback
-- Implements a Mealy machine for stateful feedback processing
biofieldLoopback
  :: HiddenClockResetEnable dom
  => InputSignal dom -> FrameSignal dom
biofieldLoopback = mealy updateFn initFrame
  where
    updateFn frame input = let (frame', _) = updateFeedbackLoop input frame
                           in (frame', frame')
    initFrame = AvatarFieldFrame None 0.0

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Sample Biometric Input Stream for testing
-- Represents a sequence of breath/HRV readings
testInputs :: Vec 5 BiometricInput
testInputs = $(listToVecTH
  [ BiometricInput 6.2 0.81   -- Near optimal, good HRV → High coherence
  , BiometricInput 6.4 0.85   -- Slightly above optimal, excellent HRV
  , BiometricInput 6.1 0.65   -- Near optimal, moderate HRV
  , BiometricInput 5.7 0.90   -- Below optimal, high HRV
  , BiometricInput 6.6 0.78   -- Above optimal, good HRV
  ])

-- | Expected output stream for verification
-- Pre-computed expected frames for the test inputs
simOutput :: Vec 5 AvatarFieldFrame
simOutput = sampleN @System 5 $ biofieldLoopback (fromList testInputs)
