{-|
Module      : RaChamberMorphology
Description : Chamber morphology system for form transitions
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 44: Models chamber morphology transitions based on coherence and
instability thresholds. When coherence drops and instability rises,
triggers rapid collapse to toroidal form.

== Chamber Forms

| Form    | Description                        |
|---------|------------------------------------|
| Sphere  | Default stable form                |
| Toroid  | Collapsed form (low coherence)     |
| Cube    | Crystallized form (high coherence) |

== Morphology Events

| Event         | Trigger Condition                              |
|---------------|------------------------------------------------|
| NoChange      | Normal operation, form stable                  |
| RapidCollapse | coherence < 0.39 AND instability > 0.30        |

== Thresholds

- coherenceThreshold = 100 (~0.39 on 0-255 scale)
- instabilityThreshold = 77 (~0.30 on 0-255 scale)

== State Transition Logic

@
if coherence < 100 AND instability > 77:
    newForm = Toroid, event = RapidCollapse
else:
    newForm = currentForm, event = NoChange
@

== Integration

- Consumes coherence from RaBiometricMatcher
- Consumes instability from field entropy measurements
- Outputs form change events for visualization
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaChamberMorphology where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Fixed-point representation (0-255 maps to 0.0-1.0)
type FixedPoint = Unsigned 8

-- | Chamber geometric forms
data ChamberForm = Sphere | Toroid | Cube
  deriving (Generic, NFDataX, Show, Eq)

-- | Morphology transition events
data MorphEvent = NoChange | RapidCollapse
  deriving (Generic, NFDataX, Show, Eq)

-- | Chamber state bundle
data ChamberState = ChamberState
  { coherence   :: FixedPoint    -- ^ Coherence level (0-255)
  , instability :: FixedPoint    -- ^ Instability level (0-255)
  , form        :: ChamberForm   -- ^ Current geometric form
  } deriving (Generic, NFDataX, Show, Eq)

-- | Morphology transition result
data MorphResult = MorphResult
  { newForm :: ChamberForm       -- ^ Resulting form
  , event   :: MorphEvent        -- ^ Transition event type
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Coherence threshold (~0.39)
-- Below this threshold, chamber becomes unstable
coherenceThreshold :: FixedPoint
coherenceThreshold = 100

-- | Instability threshold (~0.30)
-- Above this threshold with low coherence triggers collapse
instabilityThreshold :: FixedPoint
instabilityThreshold = 77

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Check morphology and determine fallback form
checkMorphologyFallback :: ChamberState -> MorphResult
checkMorphologyFallback s
  | coherence s < coherenceThreshold && instability s > instabilityThreshold =
      MorphResult { newForm = Toroid, event = RapidCollapse }
  | otherwise =
      MorphResult { newForm = form s, event = NoChange }

-- | Advanced morphology with crystallization check
-- High coherence + low instability -> Cube formation
checkMorphologyAdvanced :: ChamberState -> MorphResult
checkMorphologyAdvanced s
  | coherence s < coherenceThreshold && instability s > instabilityThreshold =
      MorphResult { newForm = Toroid, event = RapidCollapse }
  | coherence s > 200 && instability s < 30 =
      MorphResult { newForm = Cube, event = NoChange }
  | otherwise =
      MorphResult { newForm = form s, event = NoChange }

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Morphology processor - monitors state and triggers transitions
morphologyProcessor
  :: HiddenClockResetEnable dom
  => Signal dom ChamberState         -- ^ Input chamber state
  -> Signal dom MorphResult          -- ^ Morphology result
morphologyProcessor state = fmap checkMorphologyFallback state

-- | Advanced morphology processor with crystallization
morphologyAdvanced
  :: HiddenClockResetEnable dom
  => Signal dom ChamberState
  -> Signal dom MorphResult
morphologyAdvanced state = fmap checkMorphologyAdvanced state

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
morphologyTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System ChamberState
  -> Signal System MorphResult
morphologyTop = exposeClockResetEnable morphologyProcessor

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test chamber states
testStates :: Vec 6 ChamberState
testStates =
  ChamberState 150 50 Sphere :>    -- Stable: high coherence, low instability
  ChamberState 80 100 Sphere :>    -- Collapse: low coherence, high instability
  ChamberState 120 60 Toroid :>    -- Stable: above threshold
  ChamberState 50 200 Cube :>      -- Collapse: very low coherence, very high instability
  ChamberState 99 78 Sphere :>     -- Edge case: just below both thresholds -> collapse
  ChamberState 100 77 Sphere :>    -- Edge case: at thresholds -> no change
  Nil

-- | Expected outputs
-- Test 0: 150 >= 100, no collapse -> (Sphere, NoChange)
-- Test 1: 80 < 100 AND 100 > 77 -> (Toroid, RapidCollapse)
-- Test 2: 120 >= 100, no collapse -> (Toroid, NoChange)
-- Test 3: 50 < 100 AND 200 > 77 -> (Toroid, RapidCollapse)
-- Test 4: 99 < 100 AND 78 > 77 -> (Toroid, RapidCollapse)
-- Test 5: 100 >= 100 (not <), no collapse -> (Sphere, NoChange)
expectedOutput :: Vec 6 MorphResult
expectedOutput =
  MorphResult Sphere NoChange :>
  MorphResult Toroid RapidCollapse :>
  MorphResult Toroid NoChange :>
  MorphResult Toroid RapidCollapse :>
  MorphResult Toroid RapidCollapse :>
  MorphResult Sphere NoChange :>
  Nil

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for morphology processor validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testStates
    out = morphologyTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedOutput out
