{-|
Module      : RaFieldTransferBus
Description : Tesla coherent field transfer bus for scalar packet transmission
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 35: Simulates scalar packet transmission from Avatar A to Avatar B.
Transfer must preserve coherenceScore and complete within latency constraint.

== Transfer Constraints

- TransferLatency < 300 cycles (simulated at 1ms per cycle)
- CoherenceInvariant: Score delta <= +/-1 post-transfer

== Inputs

| Signal | Description |
|--------|-------------|
| srcCoherence | Coherence score (0-255) |
| srcSignal | 4-element scalar harmonic vector |
| sendPulse | Triggers transfer on rising edge |

== Outputs

| Signal | Description |
|--------|-------------|
| destSignal | Received scalar harmonic vector |
| latencyCount | Transfer time in cycles |
| integrityOK | True if coherence preserved within tolerance |

== Integration

Claude drops packets if latency > threshold OR coherence integrity fails.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaFieldTransferBus where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Constants
-- =============================================================================

-- | Maximum allowed transfer latency (cycles)
maxLatency :: Unsigned 9
maxLatency = 300

-- | Maximum coherence delta for integrity check
coherenceTolerance :: Unsigned 8
coherenceTolerance = 1

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Latency counter state machine
countLatency :: Unsigned 9 -> Bool -> (Unsigned 9, Unsigned 9)
countLatency s True  = (s + 1, s)
countLatency _ False = (0, 0)

-- | Field transfer bus - transfers scalar packets between avatars
fieldTransferBus
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)           -- ^ Source coherence score
  -> Signal dom (Vec 4 (Unsigned 8))   -- ^ Source scalar harmonics
  -> Signal dom Bool                   -- ^ Send pulse trigger
  -> Signal dom (Vec 4 (Unsigned 8), Unsigned 9, Bool)  -- ^ (destSignal, latency, integrityOK)
fieldTransferBus srcCoh srcSig send = bundle (destOut, latency, ok)
  where
    -- Transfer in progress flag (sticky until reset)
    transferReady = register False (send .||. transferReady)

    -- Registered buffer for signal and coherence
    regBuffer = regEn (repeat 0) send srcSig
    regCoh    = regEn 0 send srcCoh

    -- Latency counter
    latency = mealy countLatency 0 transferReady

    -- Output is the registered buffer
    destOut = regBuffer

    -- Integrity check: coherence preserved within tolerance
    ok = fmap (\(a, b) -> abs (a - b) <= 1) (bundle (srcCoh, regCoh))

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
transferTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 8)
  -> Signal System (Vec 4 (Unsigned 8))
  -> Signal System Bool
  -> Signal System (Vec 4 (Unsigned 8), Unsigned 9, Bool)
transferTop = exposeClockResetEnable fieldTransferBus

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test coherence values
testCoh :: Vec 3 (Unsigned 8)
testCoh = $(listToVecTH [200 :: Unsigned 8, 205, 198])

-- | Test signal vectors
testSig :: Vec 3 (Vec 4 (Unsigned 8))
testSig = $(listToVecTH
  [ $(listToVecTH [60 :: Unsigned 8, 90, 120, 180])
  , $(listToVecTH [61, 91, 121, 181])
  , $(listToVecTH [62, 92, 122, 182])
  ])

-- | Test send pulses
testSend :: Vec 3 Bool
testSend = $(listToVecTH [True, True, True])

-- | Expected outputs
-- Cycle 0: send=True, buffer captures [60,90,120,180], latency=1, ok=True
-- Cycle 1: send=True, buffer captures [61,91,121,181], latency=2, ok=True
-- Cycle 2: send=True, buffer captures [62,92,122,182], latency=3, ok=True
expected :: Vec 3 (Vec 4 (Unsigned 8), Unsigned 9, Bool)
expected = $(listToVecTH
  [ ($(listToVecTH [60 :: Unsigned 8, 90, 120, 180]), 1, True)
  , ($(listToVecTH [61 :: Unsigned 8, 91, 121, 181]), 2, True)
  , ($(listToVecTH [62 :: Unsigned 8, 92, 122, 182]), 3, True)
  ])

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for field transfer bus validation (VCD waveform compatible)
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    s1 = stimuliGenerator clk rst testCoh
    s2 = stimuliGenerator clk rst testSig
    s3 = stimuliGenerator clk rst testSend
    out = transferTop clk rst enableGen s1 s2 s3
    done = outputVerifier' clk rst expected out
