{-|
Module      : RaHandshakeGate
Description : Symbolic ↔ Biometric dual-factor handshake logic
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Ensures coherence + identity match, or explicit override.
Dual-factor validation combining symbolic consent triggers with biometric coherence.

== Handshake Logic

@
handshakeGranted = overrideEnable OR (biometricCoherence AND isPermittedID)
@

== Permitted Trigger IDs

Only triggers with IDs [3, 4, 7, 9] pass symbolic validation.

== Test Coverage

| # | Trigger | Bio   | Override | Expected | Reason                    |
|---|---------|-------|----------|----------|---------------------------|
| 1 | 3       | True  | False    | True     | Permitted + coherent      |
| 2 | 5       | True  | False    | False    | Invalid ID                |
| 3 | 9       | False | True     | True     | Override enabled          |
| 4 | 2       | True  | False    | False    | Invalid ID                |
| 5 | 7       | False | False    | False    | No coherence, no override |
| 6 | 4       | True  | False    | True     | Permitted + coherent      |
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

-- | Symbolic input ID (8-bit gesture/trigger identifier)
type ConsentTrigger = Unsigned 8

-- =============================================================================
-- Constants
-- =============================================================================

-- | Permitted symbolic IDs (gesture or field)
permittedIDs :: Vec 4 ConsentTrigger
permittedIDs = $(listToVecTH [3, 4, 7, 9])

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Top-level handshake gate
-- Output is True if: override OR (biometric AND permitted ID)
handshakeGate
  :: HiddenClockResetEnable dom
  => Signal dom ConsentTrigger
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
handshakeGate trigS bioS overrideS = output
  where
    isAllowed trig = any (== trig) permittedIDs
    output = liftA3 (\t b o -> o || (b && isAllowed t)) trigS bioS overrideS

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top entity for Clash synthesis
handshakeTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System ConsentTrigger
  -> Signal System Bool
  -> Signal System Bool
  -> Signal System Bool
handshakeTop = exposeClockResetEnable handshakeGate

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test trigger IDs
trigVec :: Vec 6 ConsentTrigger
trigVec = $(listToVecTH [3, 5, 9, 2, 7, 4])

-- | Test biometric coherence values
biometricVec :: Vec 6 Bool
biometricVec = $(listToVecTH [True, True, False, True, False, True])

-- | Test override flags
overrideVec :: Vec 6 Bool
overrideVec = $(listToVecTH [False, False, True, False, False, False])

-- | Expected handshake results
expected :: Vec 6 Bool
expected = $(listToVecTH [True, False, True, False, False, True])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench validating all logic paths
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stimTrig = stimuliGenerator clk rst trigVec
    stimBio = stimuliGenerator clk rst biometricVec
    stimOver = stimuliGenerator clk rst overrideVec
    out = handshakeTop clk rst enableGen stimTrig stimBio stimOver
    done = outputVerifier' clk rst expected out
