{-|
Module      : RaAvatarFieldVisualizer
Description : Avatar field visualizer for glow anchor pattern generation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 62: Generates AuraPattern (4 glow anchors) from signature vector,
chamber state, and emergence level. Active only when chamber state = 0b101
(Emanating state in 3-bit encoding).

== Inputs

| Signal | Description |
|--------|-------------|
| signature | 4-element vector of base brightness values (0-255) |
| chamberState | 3-bit chamber state code |
| emergenceLevel | Scalar brightness multiplier (0-255) |

== Chamber State Codes

| Code  | State       |
|-------|-------------|
| 0b000 | Idle        |
| 0b001 | Spinning    |
| 0b010 | Stabilizing |
| 0b101 | Emanating   | <- Active state for visualization

== Output

AuraPattern: 4-element glow anchor vector
- Active (0b101): signature[i] * emergence / 256
- Inactive: all zeros

== Integration

- Downstream from RaFieldSynthesisNode
- Feeds into LED driver or display renderer
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaAvatarFieldVisualizer where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Constants
-- =============================================================================

-- | Chamber state code for Emanating (active visualization)
stateEmanating :: BitVector 3
stateEmanating = 0b101

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Scale brightness by emergence level
scaleBrightness :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
scaleBrightness base emergence =
  let scaled = (resize base * resize emergence) :: Unsigned 16
  in resize (scaled `shiftR` 8)  -- Divide by 256

-- | Compute aura pattern from signature and emergence
computeAura :: Vec 4 (Unsigned 8) -> Unsigned 8 -> Vec 4 (Unsigned 8)
computeAura sig emergence = map (`scaleBrightness` emergence) sig

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Avatar field visualizer
-- Outputs glow anchors when chamber is in emanating state
avatarField
  :: HiddenClockResetEnable dom
  => Signal dom (Vec 4 (Unsigned 8))  -- ^ Signature vector
  -> Signal dom (BitVector 3)         -- ^ Chamber state code
  -> Signal dom (Unsigned 8)          -- ^ Emergence level
  -> Signal dom (Vec 4 (Unsigned 8))  -- ^ AuraPattern output
avatarField sig chamber emergence = output
  where
    -- Check if chamber is in emanating state
    stateMatch = fmap (== stateEmanating) chamber

    -- Compute scaled aura pattern
    scaledAura = computeAura <$> sig <*> emergence

    -- Output pattern only when state matches, else zeros
    output = mux stateMatch scaledAura (pure (repeat 0))

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
avatarFieldTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 4 (Unsigned 8))
  -> Signal System (BitVector 3)
  -> Signal System (Unsigned 8)
  -> Signal System (Vec 4 (Unsigned 8))
avatarFieldTop = exposeClockResetEnable avatarField

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test signatures
testSig :: Vec 4 (Vec 4 (Unsigned 8))
testSig = $(listToVecTH
  [ $(listToVecTH [100 :: Unsigned 8, 150, 200, 255])
  , $(listToVecTH [128, 128, 128, 128])
  , $(listToVecTH [64, 96, 128, 160])
  , $(listToVecTH [200, 200, 200, 200])
  ])

-- | Test chamber states
testChamber :: Vec 4 (BitVector 3)
testChamber = $(listToVecTH [0b101, 0b000, 0b101, 0b010])

-- | Test emergence levels
testEmergence :: Vec 4 (Unsigned 8)
testEmergence = $(listToVecTH [255 :: Unsigned 8, 200, 128, 255])

-- | Expected outputs:
-- Cycle 0: state=101 (emanating), sig*255/256 -> [100,150,200,255] (approx)
-- Cycle 1: state=000 (idle), output zeros
-- Cycle 2: state=101 (emanating), sig*128/256 -> [32,48,64,80]
-- Cycle 3: state=010 (stabilizing), output zeros
expectedOutput :: Vec 4 (Vec 4 (Unsigned 8))
expectedOutput = $(listToVecTH
  [ $(listToVecTH [99 :: Unsigned 8, 149, 199, 254])  -- sig * 255 / 256
  , $(listToVecTH [0, 0, 0, 0])                        -- idle state
  , $(listToVecTH [32, 48, 64, 80])                    -- sig * 128 / 256
  , $(listToVecTH [0, 0, 0, 0])                        -- stabilizing state
  ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for avatar field visualizer validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    s1 = stimuliGenerator clk rst testSig
    s2 = stimuliGenerator clk rst testChamber
    s3 = stimuliGenerator clk rst testEmergence
    out = avatarFieldTop clk rst enableGen s1 s2 s3
    done = outputVerifier' clk rst expectedOutput out
