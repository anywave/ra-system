{-|
Module      : RaChamberSync
Description : Multi-chamber synchronization for distributed field synthesis
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 40: Synchronizes multiple chamber nodes to maintain phase coherence
across distributed field synthesis network. Ensures chambers reach
aligned states for coordinated field emission.

== Sync States

| State        | Description                                    |
|--------------|------------------------------------------------|
| Desync       | Chambers out of phase, awaiting alignment      |
| Aligning     | Active phase correction in progress            |
| Locked       | All chambers synchronized                      |
| Drifting     | Minor phase drift detected, auto-correcting   |

== Sync Logic

@
syncState = Locked   when all chambers in same state AND drift < threshold
syncState = Drifting when all chambers in same state BUT drift >= threshold
syncState = Aligning when majority chambers agree
syncState = Desync   otherwise
@

== Phase Drift

Measures cycles since last full synchronization. If drift exceeds
maxDrift (32 cycles), sync quality degrades.

== Integration

- Coordinates with RaFieldSynthesisNode for chamber state
- Outputs syncPulse to align chamber transitions
- Reports syncQuality (0-255) for dashboard monitoring
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module RaChamberSync where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Chamber state (imported concept from RaFieldSynthesisNode)
data ChamberState = Idle | Spinning | Stabilizing | Emanating
  deriving (Generic, NFDataX, Show, Eq, Enum, Bounded)

-- | Synchronization status
data SyncState = Desync | Aligning | Locked | Drifting
  deriving (Generic, NFDataX, Show, Eq)

-- | Sync output bundle
data SyncOutput = SyncOutput
  { syncState   :: SyncState      -- ^ Current synchronization status
  , syncPulse   :: Bool           -- ^ Pulse to trigger chamber alignment
  , syncQuality :: Unsigned 8     -- ^ Sync quality 0-255 (255 = perfect)
  , phaseDrift  :: Unsigned 8     -- ^ Cycles since last full sync
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Maximum allowed drift before quality degrades
maxDrift :: Unsigned 8
maxDrift = 32

-- | Drift threshold for Drifting vs Locked
driftThreshold :: Unsigned 8
driftThreshold = 8

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Count chambers in a specific state
countInState :: KnownNat n => ChamberState -> Vec n ChamberState -> Unsigned 8
countInState target = fold (+) . map (\s -> if s == target then 1 else 0)

-- | Find majority state among chambers
majorityState :: KnownNat n => Vec n ChamberState -> (ChamberState, Unsigned 8)
majorityState chambers =
  let countIdle = countInState Idle chambers
      countSpin = countInState Spinning chambers
      countStab = countInState Stabilizing chambers
      countEman = countInState Emanating chambers
      maxCount = maximum (countIdle :> countSpin :> countStab :> countEman :> Nil)
  in if countIdle == maxCount then (Idle, countIdle)
     else if countSpin == maxCount then (Spinning, countSpin)
     else if countStab == maxCount then (Stabilizing, countStab)
     else (Emanating, countEman)

-- | Check if all chambers are in same state
allSameState :: KnownNat n => Vec n ChamberState -> Bool
allSameState chambers =
  let first = head chambers
  in fold (&&) (map (== first) chambers)

-- | Compute sync quality from drift
computeQuality :: Unsigned 8 -> Unsigned 8
computeQuality drift
  | drift == 0  = 255
  | drift < 8   = 224
  | drift < 16  = 192
  | drift < 32  = 128
  | otherwise   = 64

-- | Determine sync state from chamber states and drift
determineSyncState :: KnownNat n => Vec n ChamberState -> Unsigned 8 -> SyncState
determineSyncState chambers drift
  | allSameState chambers && drift < driftThreshold = Locked
  | allSameState chambers = Drifting
  | snd (majorityState chambers) >= (fromIntegral (length chambers) `div` 2 + 1) = Aligning
  | otherwise = Desync

-- | State transition for drift counter
driftCounter :: Unsigned 8 -> Bool -> (Unsigned 8, Unsigned 8)
driftCounter drift allSame
  | allSame   = (0, drift)           -- Reset on sync
  | drift < 255 = (drift + 1, drift) -- Increment drift
  | otherwise = (255, drift)         -- Saturate at max

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Chamber synchronization node
-- Monitors multiple chambers and outputs sync status
chamberSync
  :: forall n dom. (KnownNat n, HiddenClockResetEnable dom)
  => Signal dom (Vec n ChamberState)    -- ^ Chamber states input
  -> Signal dom SyncOutput              -- ^ Sync output bundle
chamberSync chambers = syncOut
  where
    -- Check if all chambers same state
    allSame = allSameState <$> chambers

    -- Track phase drift (cycles since last full sync)
    drift = mealy driftCounter 0 allSame

    -- Determine sync state
    state = determineSyncState <$> chambers <*> drift

    -- Generate sync pulse when transitioning to Locked
    prevState = register Desync state
    pulse = liftA2 (\curr prev -> curr == Locked && prev /= Locked) state prevState

    -- Compute quality from drift
    quality = computeQuality <$> drift

    -- Bundle outputs
    syncOut = SyncOutput <$> state <*> pulse <*> quality <*> drift

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis (4 chambers)
chamberSyncTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 4 ChamberState)
  -> Signal System SyncOutput
chamberSyncTop = exposeClockResetEnable (chamberSync @4)

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test chamber state vectors
-- Cycle 0: All Idle (should lock)
-- Cycle 1: Mixed (should desync/align)
-- Cycle 2: All Spinning (should lock)
-- Cycle 3: All Spinning (should stay locked)
-- Cycle 4: One drifts (should align)
-- Cycle 5: All Emanating (should lock)
testChambers :: Vec 6 (Vec 4 ChamberState)
testChambers = $(listToVecTH
  [ $(listToVecTH [Idle, Idle, Idle, Idle])
  , $(listToVecTH [Idle, Spinning, Idle, Idle])
  , $(listToVecTH [Spinning, Spinning, Spinning, Spinning])
  , $(listToVecTH [Spinning, Spinning, Spinning, Spinning])
  , $(listToVecTH [Stabilizing, Spinning, Spinning, Spinning])
  , $(listToVecTH [Emanating, Emanating, Emanating, Emanating])
  ])

-- | Expected sync states
-- Note: First cycle after reset starts with drift=0
expectedSync :: Vec 6 SyncState
expectedSync = $(listToVecTH [Locked, Aligning, Locked, Locked, Aligning, Locked])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for chamber sync validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testChambers
    out = chamberSyncTop clk rst enableGen stim
    -- Extract just syncState for verification
    stateOut = syncState <$> out
    done = outputVerifier' clk rst expectedSync stateOut
