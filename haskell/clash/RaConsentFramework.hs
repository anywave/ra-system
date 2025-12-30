{-|
Module      : RaConsentFramework
Description : Self-Regulating Consent Framework (Prompt 32)
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Scalar-aware symbolic consent validator honoring Codex protocols
of shadow gating, harmonic override, and coherence memory.

== Consent States

| State    | Description                                    |
|----------|------------------------------------------------|
| Permit   | Full consent - coherent and authorized         |
| Restrict | Limited consent - incoherent or unauthorized   |
| Override | Emergency override - bypasses normal gating    |

== State Transitions

@
                    ┌─────────────┐
                    │   Permit    │◀──── isCoherent = True
                    └──────┬──────┘
                           │
        overrideFlag ──────┼────────▶ Override
                           │
                           ▼
                    ┌─────────────┐
                    │  Restrict   │◀──── isCoherent = False
                    └─────────────┘
@

== Coherence Memory

The framework tracks how long coherence has been maintained via `coherenceDur`.
This enables:
- Graduated consent escalation
- Shadow gating with warmup periods
- Harmonic lock-in detection

== Phase II Dashboard Integration

Can be integrated into dashboard gate controller for real-time consent visualization.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaConsentFramework where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | ConsentState encodes the user's current permission signature
data ConsentState = Permit | Restrict | Override
  deriving (Show, Eq, Generic, NFDataX)

-- | Symbolic input trigger from avatar or chamber
-- Could be a gesture ID, biometric spike, or field signature
type ConsentTrigger = Unsigned 8

-- | Input bundle for consent validation
data ConsentInput = ConsentInput
  { trigger      :: ConsentTrigger  -- ^ Symbolic trigger ID
  , isCoherent   :: Bool            -- ^ Current coherence status
  , overrideFlag :: Bool            -- ^ Emergency override flag
  } deriving (Generic, NFDataX)

-- | Output memory state for future filtering decisions
data ConsentMemory = ConsentMemory
  { lastState    :: ConsentState    -- ^ Previous consent state
  , coherenceDur :: Unsigned 8      -- ^ How long coherence held (cycles)
  } deriving (Generic, NFDataX)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Top-level consent validation module
-- Emits updated state based on symbolic and scalar coherence patterns
consentGate
  :: HiddenClockResetEnable dom
  => Signal dom ConsentInput
  -> Signal dom ConsentState
consentGate = mealy updateConsent initMem
  where
    initMem = ConsentMemory Permit 0

    updateConsent :: ConsentMemory -> ConsentInput -> (ConsentMemory, ConsentState)
    updateConsent ConsentMemory{..} ConsentInput{..} =
      let
        newState
          | overrideFlag = Override
          | isCoherent   = Permit
          | otherwise    = Restrict

        newDuration = if isCoherent then coherenceDur + 1 else 0

        updated = ConsentMemory newState newDuration
      in
        (updated, newState)

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test Inputs (Gesture ID, Coherence, Override)
testInputs :: Vec 6 ConsentInput
testInputs =
  $(listToVecTH
    [ ConsentInput 3 True  False   -- Permit
    , ConsentInput 4 True  False   -- Permit
    , ConsentInput 6 False False   -- Restrict
    , ConsentInput 9 False True    -- Override
    , ConsentInput 1 True  False   -- Permit again
    , ConsentInput 2 False False   -- Restrict
    ])

-- | Test expected outputs
testOutputs :: Vec 6 ConsentState
testOutputs =
  $(listToVecTH [Permit, Permit, Restrict, Override, Permit, Restrict])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench simulation for consent framework validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testInputs
    expected = outputVerifier' clk rst testOutputs
    done = expected (consentGate stim)
