{-|
Module      : RaSymbolicCoherenceOps
Description : Symbolic coherence operations for emergence transformation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 54: Defines compositional symbolic operations for transforming
emergence conditions. Supports phase shifting, angle inversion, and
threshold gating with fixed-point arithmetic.

== Symbolic Operations

| Operation | Formula | Description |
|-----------|---------|-------------|
| PhaseShift(fx) | coherence += fx * 255 | Add phase shift with saturation |
| InvertAngle | angle = 255 - angle | Mirror angle about midpoint |
| GateThreshold(fx) | coherence = 0 if < fx | Zero below threshold |

== Fixed-Point Representation

FixedPoint values 0-255 represent fractions 0.0-1.0:
- 0 = 0.0
- 128 = 0.5
- 158 = 0.618 (golden ratio)
- 255 = 1.0

== Example Composition

@
PhaseShift(0.618) compose InvertAngle compose GateThreshold(0.4)
@

With input (coherence=100, angle=200):
1. PhaseShift(158): coherence = min(255, 100 + 97) = 197
2. InvertAngle: angle = 255 - 200 = 55
3. GateThreshold(102): coherence = 197 >= 102, keep 197

Result: (coherence=197, angle=55)

== Integration

- Consumes EmergenceCondition from avatar state
- Outputs transformed condition for field synthesis
- Compositional: operations chain left-to-right
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaSymbolicCoherenceOps where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Fixed-point representation (0-255 maps to 0.0-1.0)
type FixedPoint = Unsigned 8

-- | Emergence condition bundle
data EmergenceCondition = EmergenceCondition
  { coherence :: Unsigned 8    -- ^ Coherence level (0-255)
  , angle     :: Unsigned 8    -- ^ Phase angle (0-255)
  } deriving (Generic, NFDataX, Show, Eq)

-- | Symbolic operation type
-- Encoded as: (opCode, parameter)
-- opCode 0 = PhaseShift, 1 = InvertAngle, 2 = GateThreshold
data SymbolicOp = SymbolicOp
  { opCode :: Unsigned 2       -- ^ Operation type
  , param  :: FixedPoint       -- ^ Operation parameter
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Operation Constructors
-- =============================================================================

-- | Create PhaseShift operation
phaseShift :: FixedPoint -> SymbolicOp
phaseShift fx = SymbolicOp 0 fx

-- | Create InvertAngle operation
invertAngle :: SymbolicOp
invertAngle = SymbolicOp 1 0

-- | Create GateThreshold operation
gateThreshold :: FixedPoint -> SymbolicOp
gateThreshold fx = SymbolicOp 2 fx

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Fixed-point multiplication: (fx * 255) / 256
fxMul :: FixedPoint -> Unsigned 8
fxMul fx =
  let product = (resize fx * 255) :: Unsigned 16
  in resize (product `shiftR` 8)

-- | Apply a single symbolic operation
applyOp :: SymbolicOp -> EmergenceCondition -> EmergenceCondition
applyOp (SymbolicOp code p) e = case code of
  0 -> e { coherence = satAdd SatBound (coherence e) (fxMul p) }  -- PhaseShift
  1 -> e { angle = 255 - angle e }                                 -- InvertAngle
  2 -> e { coherence = if coherence e < p then 0 else coherence e } -- GateThreshold
  _ -> e  -- No-op for invalid codes

-- | Compose operations (apply left-to-right)
applyOps :: KnownNat n => Vec n SymbolicOp -> EmergenceCondition -> EmergenceCondition
applyOps Nil e = e
applyOps (h :> t) e = applyOps t (applyOp h e)

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Symbolic coherence processor (3-operation pipeline)
symbolicProcessor
  :: HiddenClockResetEnable dom
  => Signal dom (Vec 3 SymbolicOp)      -- ^ Operation sequence
  -> Signal dom EmergenceCondition      -- ^ Input condition
  -> Signal dom EmergenceCondition      -- ^ Transformed condition
symbolicProcessor ops input = fmap (uncurry applyOps) (bundle (ops, input))

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
symbolicTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Vec 3 SymbolicOp)
  -> Signal System EmergenceCondition
  -> Signal System EmergenceCondition
symbolicTop = exposeClockResetEnable symbolicProcessor

-- =============================================================================
-- Example Compositions
-- =============================================================================

-- | Golden ratio phase shift + invert + gate at 40%
-- PhaseShift(0.618) ○ InvertAngle ○ GateThreshold(0.4)
goldenComposition :: Vec 3 SymbolicOp
goldenComposition = phaseShift 158 :> invertAngle :> gateThreshold 102 :> Nil

-- | Neutral composition (no-ops)
neutralComposition :: Vec 3 SymbolicOp
neutralComposition = phaseShift 0 :> SymbolicOp 3 0 :> SymbolicOp 3 0 :> Nil

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test operation sequences
testOps :: Vec 4 (Vec 3 SymbolicOp)
testOps =
  goldenComposition :>
  neutralComposition :>
  (phaseShift 128 :> invertAngle :> gateThreshold 200 :> Nil) :>
  (gateThreshold 150 :> phaseShift 64 :> invertAngle :> Nil) :> Nil

-- | Test input conditions
testInputs :: Vec 4 EmergenceCondition
testInputs =
  EmergenceCondition 100 200 :>
  EmergenceCondition 50 100 :>
  EmergenceCondition 180 50 :>
  EmergenceCondition 100 128 :> Nil

-- | Expected outputs:
-- Test 0: (100,200) -> PhaseShift(158): 100+97=197 -> Invert: angle=55 -> Gate(102): 197>=102, keep
--         Result: (197, 55)
-- Test 1: (50,100) -> PhaseShift(0): 50 -> NoOp -> NoOp
--         Result: (50, 100)
-- Test 2: (180,50) -> PhaseShift(128): 180+127=255(sat) -> Invert: angle=205 -> Gate(200): 255>=200
--         Result: (255, 205)
-- Test 3: (100,128) -> Gate(150): 100<150, zero -> PhaseShift(64): 0+63=63 -> Invert: angle=127
--         Result: (63, 127)
expectedOutput :: Vec 4 EmergenceCondition
expectedOutput =
  EmergenceCondition 197 55 :>
  EmergenceCondition 50 100 :>
  EmergenceCondition 255 205 :>
  EmergenceCondition 63 127 :> Nil

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for symbolic coherence ops validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    ops = stimuliGenerator clk rst testOps
    inp = stimuliGenerator clk rst testInputs
    out = symbolicTop clk rst enableGen ops inp
    done = outputVerifier' clk rst expectedOutput out
