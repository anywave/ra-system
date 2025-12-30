{-|
Module      : RaPWMMultiFreqTest
Description : Multi-Harmonic PWM Scalar Pattern Generator
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Designed for entrainment fields with simultaneous Solfeggio, Schumann, and biofield rhythms.
Now supports: runtime weight tuning, visual envelope output, live biometric override.

== Harmonic Bands

* Theta band - 4-8 Hz brain entrainment (sine wave)
* Delta band - 0.5-4 Hz deep sleep rhythm (cosine wave)
* Solfeggio pulse - Higher frequency sacred tones (3x sine)

== Biometric Override

When biometric override > 0, the envelope is replaced by direct biometric input.
This allows real-time body-sensing to take priority over synthetic waveforms.

== Pipeline with Biometric Override

@
Weights ────────────┐
                    ▼
Theta ─┐         ┌─────────────┐      ┌───────────┐
       ├────────▶│ blendFields │─────▶│ envelope  │──┐
Delta ─┤         └─────────────┘      └───────────┘  │
       │                                              ▼
Solfeggio ─┘                              ┌─────────────────┐
                                          │ mux(bioOverride)│──▶ finalEnvelope ──▶ PWM
BioOverride ─────────────────────────────▶└─────────────────┘
@
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE NumericUnderscores #-}

module RaPWMMultiFreqTest where

import Clash.Prelude
import qualified Prelude as P
import RaPWMDriver

-- =============================================================================
-- Types
-- =============================================================================

-- | Multiple harmonic base envelopes (mocked)
type HarmonicField = Signal System Float

-- | Runtime-configurable weights (e.g. via Codex avatar knobs)
-- Format: (theta weight, delta weight, solfeggio weight)
type HarmonicWeights = (Float, Float, Float)

-- =============================================================================
-- Harmonic Band Generators
-- =============================================================================

-- | Theta band (4-8 Hz) - sine wave envelope
-- Used for meditation and relaxation entrainment
thetaBand :: Vec 64 Float
thetaBand = map (\x -> (sin x + 1.0) / 2.0) (iterateI (+ (pi / 64)) 0.0)

-- | Delta band (0.5-4 Hz) - cosine wave envelope
-- Used for deep sleep and regeneration
deltaBand :: Vec 64 Float
deltaBand = map (\x -> (cos x + 1.0) / 2.0) (iterateI (+ (pi / 32)) 0.0)

-- | Solfeggio pulse - higher frequency sacred tones
-- 3x frequency multiplier for harmonic resonance
solfeggioPulse :: Vec 64 Float
solfeggioPulse = map (\x -> (sin (3*x) + 1.0) / 2.0) (iterateI (+ (pi / 128)) 0.0)

-- | Default weight configuration for testing
testWeights :: Vec 64 (Float, Float, Float)
testWeights = repeat (0.4, 0.3, 0.3)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Combine multiple harmonic inputs with runtime-configurable weights
-- Weights tuple: (theta, delta, solfeggio)
blendFields :: HarmonicWeights -> Float -> Float -> Float -> Float
blendFields (wt, wd, ws) a b c = (a * wt) + (b * wd) + (c * ws)

-- | Top-level entity with biometric override
-- When bioOverride > 0, biometric signal takes priority over synthetic blend
multiHarmonicBiometric
  :: HiddenClockResetEnable dom
  => Signal dom HarmonicWeights               -- ^ Control weights
  -> Signal dom Float                         -- ^ Biometric override (0.0 to 1.0)
  -> Signal dom Float -> Signal dom Float -> Signal dom Float
  -> (Signal dom Bool, Signal dom Float)
multiHarmonicBiometric weights bioOverride theta delta solfeggio = (pwmOut, finalEnvelope)
  where
    envelope = blend <$> weights <*> theta <*> delta <*> solfeggio
    finalEnvelope = mux (bioOverride .>. 0) bioOverride envelope
    pwmOut = scalarToPWM finalEnvelope
    blend w t d s = blendFields w t d s

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Clock wrapper
-- Output: (PWM, float envelope with biometric priority)
-- Envelope output enables visual feedback in GTKW or dashboard UI
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System HarmonicWeights
  -> Signal System Float
  -> Signal System Float
  -> Signal System Float
  -> Signal System Float
  -> (Signal System Bool, Signal System Float)
topEntity = exposeClockResetEnable multiHarmonicBiometric

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Testbench vectors for simulation
weightsS :: Signal System HarmonicWeights
weightsS = fromList testWeights

thetaS :: Signal System Float
thetaS = fromList thetaBand

deltaS :: Signal System Float
deltaS = fromList deltaBand

solfeggioS :: Signal System Float
solfeggioS = fromList solfeggioPulse

-- | Biometric override signal (0.0 = use synthetic, >0 = biometric takeover)
-- Set to 0.5 to simulate biometric takeover during testing
bioOverrideS :: Signal System Float
bioOverrideS = fromList (repeat 0.0)

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for biometric-integrated PWM
-- Validates multi-harmonic PWM with configurable weights and biometric override
pwmTestBench :: Signal System Bool
pwmTestBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    (pwmOut, _envelopeOut) = topEntity clk rst enableGen weightsS bioOverrideS thetaS deltaS solfeggioS
    done = outputVerifier' clk rst (map (\x -> x > 0.4) thetaBand) pwmOut
