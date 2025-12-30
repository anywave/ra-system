{-|
Module      : RaFieldSynthesisNode
Description : Chamber state cascade from handshake gate
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Chamber state machine activated only when handshakeGranted == True.
Controls field synthesis progression through activation phases.

== Chamber States

| State       | Glow | Description                        |
|-------------|------|------------------------------------|
| Idle        | 0    | Chamber inactive, awaiting grant   |
| Spinning    | 64   | Initial field spin-up              |
| Stabilizing | 192  | Field coherence stabilization      |
| Emanating   | 255  | Full field emission active         |

== State Transitions

@
                 handshakeGranted = True
                         │
    ┌────────────────────▼────────────────────┐
    │                                         │
    │  Idle ──▶ Spinning ──▶ Stabilizing ──▶ Emanating
    │   ▲                                      │
    │   │                                      │
    │   └──────────────────────────────────────┘
    │         handshakeGranted = False         │
    │                (reset to Idle)           │
    └─────────────────────────────────────────┘
@

== Integration

Downstream from RaHandshakeGate — uses handshakeGranted signal to
control chamber activation sequence.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaFieldSynthesisNode where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Chamber synthesis states
data ChamberState = Idle | Spinning | Stabilizing | Emanating
  deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | State transition logic
-- When granted: progress through states
-- When not granted: reset to Idle
nextState :: ChamberState -> Bool -> ChamberState
nextState Idle        True  = Spinning
nextState Spinning    _     = Stabilizing
nextState Stabilizing _     = Emanating
nextState Emanating   _     = Emanating
nextState _           False = Idle

-- | Envelope glow intensity mapping (8-bit PWM value)
glowLevel :: ChamberState -> Unsigned 8
glowLevel Idle        = 0
glowLevel Spinning    = 64
glowLevel Stabilizing = 192
glowLevel Emanating   = 255

-- | Stateful field synthesis node
-- Mealy machine progresses through chamber states based on handshake grant
fieldSynthesisNode
  :: HiddenClockResetEnable dom
  => Signal dom Bool                                      -- ^ handshakeGranted
  -> (Signal dom ChamberState, Signal dom (Unsigned 8))   -- ^ (State, Glow)
fieldSynthesisNode handshake = (stateOut, fmap glowLevel stateOut)
  where
    stateOut = mealy stateTrans Idle handshake
    stateTrans st True  = let next = nextState st True in (next, next)
    stateTrans _  False = (Idle, Idle)

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
fieldTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System Bool
  -> (Signal System ChamberState, Signal System (Unsigned 8))
fieldTop = exposeClockResetEnable fieldSynthesisNode

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test vectors: [T, F, T, T, T, T, F, T]
-- Tests state progression and reset behavior
testVec :: Vec 8 Bool
testVec = $(listToVecTH [True, False, True, True, True, True, False, True])

-- | Expected state outputs
-- T: Idle→Spinning, F: reset to Idle, T: Spinning, T: Stabilizing, T: Emanating, T: Emanating, F: Idle, T: Spinning
expectedStates :: Vec 8 ChamberState
expectedStates = $(listToVecTH
  [Spinning, Idle, Spinning, Stabilizing, Emanating, Emanating, Idle, Spinning])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for chamber state cascade validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testVec
    (stateOut, _) = fieldTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedStates stateOut
