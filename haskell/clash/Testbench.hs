{-|
Module      : Testbench
Description : Clash simulation testbench for BiofieldLoopback
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Testbench for validating the BiofieldLoopback FPGA design.
Generates VCD waveform output for analysis.

== Usage

Generate VCD waveforms:
@
clash --vcd Testbench.hs
@

View waveforms with GTKWave:
@
gtkwave testBench.vcd
@
-}

{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE NoImplicitPrelude #-}

module Testbench where

import Clash.Prelude
import BiofieldLoopback
import qualified Prelude as P

-- =============================================================================
-- Clock and Reset Generators
-- =============================================================================

-- | System clock generator
-- Runs until 'done' signal is asserted
clk :: Clock System
clk = tbSystemClockGen (not <$> done)

-- | System reset generator
rst :: Reset System
rst = systemResetGen

-- =============================================================================
-- Stimulus Generation
-- =============================================================================

-- | Stimulus signal from biometric vector
-- Feeds test inputs to the design under test
stimuli :: Signal System BiometricInput
stimuli = stimuliGenerator clk rst testInputs

-- =============================================================================
-- Testbench Top
-- =============================================================================

-- | Testbench top-level
-- Validates 'topEntity' output against expected 'simOutput'
-- Generates VCD output for waveform viewing
--
-- The testbench:
-- 1. Applies 'testInputs' as stimulus
-- 2. Compares output against pre-computed 'simOutput'
-- 3. Asserts 'done' when verification completes
testBench :: Signal System Bool
testBench = done
  where
    expectedOutput = outputVerifier' clk rst simOutput
    done = expectedOutput (topEntity clk rst enableGen stimuli)
