{-|
Module      : RaHandshakeGate
Description : Symbolic ↔ Biometric dual-factor handshake logic
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Ensures coherence + identity match, or explicit override.
Now wired to Phase II dashboard and field trigger cascade.

== Handshake Logic

@
handshakeGranted = overrideFlag OR (biometricMatch AND symbolOK)
@

== Permitted Trigger IDs

Only gestures with IDs [3, 4, 7, 9] pass symbolic validation.

== Output Bundle

| Field            | Description                              |
|------------------|------------------------------------------|
| handshakeGranted | Final grant decision                     |
| passedBiometric  | Biometric coherence status               |
| matchedSymbol    | Symbolic ID in permitted list            |
| overrideUsed     | Override flag was enabled                |

== Field Cascade

The `fieldTriggerFromHandshake` function extracts `handshakeGranted` for
downstream field emitter activation.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaHandshakeGate where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Symbolic input ID (8-bit gesture/field identifier)
type ConsentTrigger = Unsigned 8

-- | Dual-factor handshake input bundle
data HandshakeIn = HandshakeIn
  { gestureID      :: ConsentTrigger  -- ^ Symbolic gesture/field ID
  , biometricMatch :: Bool            -- ^ Biometric coherence status
  , overrideFlag   :: Bool            -- ^ Scalar override permission
  } deriving (Generic, NFDataX)

-- | Handshake output bundle with diagnostic flags
data HandshakeOut = HandshakeOut
  { handshakeGranted :: Bool  -- ^ Final grant decision
  , passedBiometric  :: Bool  -- ^ Biometric coherence passed
  , matchedSymbol    :: Bool  -- ^ Symbolic ID was permitted
  , overrideUsed     :: Bool  -- ^ Override flag was enabled
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Permitted symbolic gesture/field IDs
permittedIDs :: Vec 4 ConsentTrigger
permittedIDs = $(listToVecTH [3, 4, 7, 9])

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Handshake validation circuit
-- Outputs rich diagnostic bundle for dashboard visualization
handshakeGate
  :: HiddenClockResetEnable dom
  => Signal dom HandshakeIn
  -> Signal dom HandshakeOut
handshakeGate = fmap validate
  where
    validate HandshakeIn{..} =
      let
        symbolOK = gestureID `elem` permittedIDs
        grant = overrideFlag || (biometricMatch && symbolOK)
      in
        HandshakeOut
          { handshakeGranted = grant
          , passedBiometric = biometricMatch
          , matchedSymbol = symbolOK
          , overrideUsed = overrideFlag
          }

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level wiring for Clash synthesis
handshakeTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System HandshakeIn
  -> Signal System HandshakeOut
handshakeTop = exposeClockResetEnable handshakeGate

-- =============================================================================
-- Field Cascade
-- =============================================================================

-- | Field Cascade: Only output true if handshake granted
-- Extracts grant decision for downstream field emitter activation
fieldTriggerFromHandshake
  :: HiddenClockResetEnable dom
  => Signal dom HandshakeOut
  -> Signal dom Bool
fieldTriggerFromHandshake = fmap handshakeGranted

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test input vectors covering all logic paths
inputVec :: Vec 6 HandshakeIn
inputVec =
  $(listToVecTH
    [ HandshakeIn 3 True False   -- Permitted + bio = Grant
    , HandshakeIn 5 True False   -- Invalid ID = Deny
    , HandshakeIn 9 False True   -- Override = Grant
    , HandshakeIn 2 True False   -- Invalid ID = Deny
    , HandshakeIn 7 False False  -- No bio, no override = Deny
    , HandshakeIn 4 True False   -- Permitted + bio = Grant
    ])

-- | Expected output vectors
expectedOut :: Vec 6 HandshakeOut
expectedOut =
  $(listToVecTH
    [ HandshakeOut True True True False    -- Grant: permitted + bio
    , HandshakeOut False True False False  -- Deny: invalid ID
    , HandshakeOut True False True True    -- Grant: override
    , HandshakeOut False True False False  -- Deny: invalid ID
    , HandshakeOut False False True False  -- Deny: no bio/override
    , HandshakeOut True True True False    -- Grant: permitted + bio
    ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for handshake validation
handshakeBench :: Signal System Bool
handshakeBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    inputStim = stimuliGenerator clk rst inputVec
    outCheck = outputVerifier' clk rst expectedOut (handshakeTop clk rst enableGen inputStim)
    done = outCheck
