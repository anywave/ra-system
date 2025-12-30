{-|
Module      : RaTokenomicsProfiler
Description : Compute cost + token analysis for Claude prompt execution
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Tracks token usage and compute costs for Ra system operations.
Enables cost analysis for prompt compliance testing.

== Operation Costs

| Operation    | Tokens | Compute |
|--------------|--------|---------|
| Handshake    | 128    | 500     |
| BioEmit      | 64     | 200     |
| ChamberSpin  | 256    | 1000    |

== Integration

Dashboard calls tokenomics endpoint after each operation to
accumulate total token + compute spend for session tracking.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaTokenomicsProfiler where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Token + compute cost per operation type
data OpType = Handshake | BioEmit | ChamberSpin
  deriving (Show, Eq, Generic, NFDataX)

-- | Cost profile for an operation
data CostProfile = CostProfile
  { op      :: OpType
  , tokens  :: Unsigned 16
  , compute :: Unsigned 16
  } deriving (Show, Generic, NFDataX)

-- | Operation trigger with fired flag
data OpTrigger = OpTrigger
  { opType :: OpType
  , fired  :: Bool
  } deriving (Generic, NFDataX)

-- =============================================================================
-- Cost Table
-- =============================================================================

-- | Cost lookup table per operation type
costLookup :: OpType -> CostProfile
costLookup Handshake   = CostProfile Handshake   128 500
costLookup BioEmit     = CostProfile BioEmit     64  200
costLookup ChamberSpin = CostProfile ChamberSpin 256 1000

-- | Get token cost for operation
tokenCost :: OpType -> Unsigned 16
tokenCost = tokens . costLookup

-- | Get compute cost for operation
computeCost :: OpType -> Unsigned 16
computeCost = compute . costLookup

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Accumulating profiler - sums costs when operations fire
-- Returns (tokenSum, computeSum) tuple
profiler
  :: HiddenClockResetEnable dom
  => Signal dom OpTrigger
  -> Signal dom (Unsigned 16, Unsigned 16)
profiler trig = mealy accumulate (0, 0) trig
  where
    accumulate (accT, accC) t =
      if fired t
        then let c = costLookup (opType t)
             in ((accT + tokens c, accC + compute c), (accT + tokens c, accC + compute c))
        else ((accT, accC), (accT, accC))

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
profilerTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System OpTrigger
  -> Signal System (Unsigned 16, Unsigned 16)
profilerTop = exposeClockResetEnable profiler

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test operation sequence
testVec :: Vec 6 OpTrigger
testVec = $(listToVecTH
  [ OpTrigger Handshake   True
  , OpTrigger BioEmit     True
  , OpTrigger ChamberSpin True
  , OpTrigger Handshake   False  -- Not fired, no cost added
  , OpTrigger BioEmit     True
  , OpTrigger ChamberSpin False  -- Not fired, no cost added
  ])

-- | Expected cumulative costs
-- (128, 500) -> (192, 700) -> (448, 1700) -> (448, 1700) -> (512, 1900) -> (512, 1900)
expectedCosts :: Vec 6 (Unsigned 16, Unsigned 16)
expectedCosts = $(listToVecTH
  [ (128, 500)
  , (192, 700)
  , (448, 1700)
  , (448, 1700)
  , (512, 1900)
  , (512, 1900)
  ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for tokenomics profiler validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testVec
    out = profilerTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedCosts out
