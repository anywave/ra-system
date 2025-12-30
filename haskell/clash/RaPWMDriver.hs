{-|
Module      : RaPWMDriver
Description : Scalar amplitude to PWM duty signal generator
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

PWM driver for low-frequency scalar outputs (e.g. Solfeggio, Schumann entrainment).
Converts amplitude values (0.0-1.0) to PWM duty cycle signals for hardware output.

== Applications

* LED intensity modulation for biofeedback
* Haptic vibration control
* Low-frequency audio entrainment
* Schumann resonance (7.83 Hz) generation
* Solfeggio frequency output (396-852 Hz)

== PWM Specifications

* Resolution: 8-bit (256 levels)
* Period: 255 counts
* Duty cycle: amplitude * 255

== Pipeline

@
Amplitude (Float 0.0-1.0) → amplitudeToThreshold → PWMWidth → pwmMealy → Bool (PWM output)
@
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaPWMDriver where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Width of PWM resolution — 8 bits (0 to 255)
type PWMWidth = Unsigned 8

-- =============================================================================
-- Constants
-- =============================================================================

-- | PWM period — can be slowed by sampling rate externally
-- Full period is 256 counts (0-255)
pwmPeriod :: PWMWidth
pwmPeriod = 255

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Convert amplitude float (0.0 to 1.0) into PWM threshold
-- Maps continuous amplitude to 8-bit duty cycle value
amplitudeToThreshold :: Float -> PWMWidth
amplitudeToThreshold f = truncateF (f * 255)

-- | Mealy machine for PWM toggle signal
-- Compares counter against threshold to generate PWM output
-- Output is HIGH when counter < threshold
pwmMealy :: PWMWidth -> PWMWidth -> (PWMWidth, Bool)
pwmMealy threshold counter = (nextCounter, counter < threshold)
  where
    nextCounter = if counter == pwmPeriod then 0 else counter + 1

-- | Top-level PWM wrapper
-- Takes amplitude signal, outputs PWM boolean signal
scalarToPWM
  :: HiddenClockResetEnable dom
  => Signal dom Float          -- ^ Input amplitude (0.0 to 1.0)
  -> Signal dom Bool           -- ^ Output PWM signal
scalarToPWM amp = mealy pwmMealy 0 counterInput
  where
    counterInput = amplitudeToThreshold <$> amp

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top entity for synthesis
-- Input: amplitude (0.0-1.0)
-- Output: PWM boolean signal
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System Float
  -> Signal System Bool
topEntity = exposeClockResetEnable scalarToPWM

-- =============================================================================
-- Test Data
-- =============================================================================

-- | For testing: sinusoidal ramp input
-- Generates smooth 0.0-1.0 wave for duty cycle testing
testWave :: Vec 64 Float
testWave = map (\x -> (sin x + 1.0) / 2.0) (iterateI (+ (pi / 32)) 0.0)

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for PWM output waveform validation
pwmTestBench :: Signal System Bool
pwmTestBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testWave
    pwmOut = scalarToPWM stim
    expected = outputVerifier' clk rst (map (\x -> x > 0.5) testWave)
    done = expected pwmOut
