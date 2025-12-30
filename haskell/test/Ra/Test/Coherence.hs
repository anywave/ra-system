{-|
Module      : Ra.Test.Coherence
Description : Tests for coherence thresholds, shadow inversion, and emergence
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Comprehensive test coverage for:
* Edge-case coherence values (boundaries)
* Null emergence conditions
* Shadow inversion logic
* Loopback feedback patterns
* Consent state transitions
-}
module Ra.Test.Coherence
  ( coherenceTests
  ) where

import Test.Hspec
import Test.QuickCheck

import Ra.Constants.Extended
  ( phi, piRed, coherenceEmergence, coherenceMax )
import Ra.Pipeline

-- =============================================================================
-- Test Suite
-- =============================================================================

coherenceTests :: Spec
coherenceTests = describe "Ra.Test.Coherence" $ do

  describe "Coherence Thresholds" $ do
    it "DOR floor is 1/pi (approx 0.3183)" $
      abs (coherenceFloorDOR - (1.0 / piRed)) < 0.0001

    it "POR floor is 1/phi (approx 0.618)" $
      abs (coherenceFloorPOR - (1.0 / phi)) < 0.0001

    it "DOR < POR (proper ordering)" $
      coherenceFloorDOR < coherenceFloorPOR

    it "POR < Emergence threshold" $
      coherenceFloorPOR < coherenceEmergence

  describe "Edge-Case Coherence" $ do
    it "Zero coherence is below DOR floor" $
      0.0 < coherenceFloorDOR

    it "Max coherence (1.0) is above emergence" $
      coherenceMax > coherenceEmergence

    it "Exactly at DOR boundary" $
      let rv = mkTestVector coherenceFloorDOR
      in resonanceCoherence rv `shouldSatisfy` (>= coherenceFloorDOR - 0.01)

    it "Exactly at POR boundary" $
      let rv = mkTestVector coherenceFloorPOR
      in resonanceCoherence rv `shouldSatisfy` (>= coherenceFloorPOR - 0.01)

  describe "Consent State Transitions" $ do
    it "Full consent maintained above POR" $
      let rv = mkTestVector 0.7
      in computeConsentTransition FullConsent rv `shouldBe` FullConsent

    it "Full to Diminished when below POR but above DOR" $
      let rv = mkTestVector 0.5
      in computeConsentTransition FullConsent rv `shouldBe` DiminishedConsent

    it "Full to Suspended when below DOR" $
      let rv = mkTestVector 0.2
      in computeConsentTransition FullConsent rv `shouldBe` SuspendedConsent

    it "Suspended to Diminished requires exceeding POR" $
      let rv = mkTestVector 0.65
      in computeConsentTransition SuspendedConsent rv `shouldBe` DiminishedConsent

    it "Hysteresis prevents oscillation" $
      let rvLow = mkTestVector 0.61  -- Just below POR + hysteresis
          rvHigh = mkTestVector 0.73  -- Above POR + hysteresis
      in do
        computeConsentTransition DiminishedConsent rvLow `shouldBe` DiminishedConsent
        computeConsentTransition DiminishedConsent rvHigh `shouldBe` FullConsent

  describe "Shadow Inversion Logic" $ do
    it "High GSR + low HRV triggers inversion" $
      let rv = ResonanceVector 0.3 0.8 0.5 0.5 0.5  -- Low HRV, high GSR
      in isInvertedState rv `shouldBe` True

    it "High HRV prevents inversion even with high GSR" $
      let rv = ResonanceVector 0.8 0.8 0.7 0.6 0.7  -- High HRV
      in isInvertedState rv `shouldBe` False

    it "Inversion requires coherence below POR" $
      let rv = ResonanceVector 0.7 0.7 0.7 0.6 0.7  -- High overall coherence
      in isInvertedState rv `shouldBe` False

  describe "Null Emergence Conditions" $ do
    it "Zero HRV blocks emergence" $
      let rv = ResonanceVector 0.0 0.5 0.5 0.5 0.5
      in resonanceCoherence rv < coherenceEmergence

    it "Zero breathing blocks emergence" $
      let rv = ResonanceVector 0.5 0.5 0.0 0.5 0.5
      in resonanceCoherence rv < coherenceEmergence

    it "All zeros gives zero coherence" $
      let rv = ResonanceVector 0.0 0.0 0.0 0.0 0.0
      in resonanceCoherence rv `shouldBe` 0.0

  describe "Fragment Reluctance" $ do
    it "High GSR increases reluctance" $
      let rvLow = ResonanceVector 0.5 0.2 0.5 0.5 0.8
          rvHigh = ResonanceVector 0.5 0.9 0.5 0.5 0.8
      in computeFragmentReluctance rvHigh > computeFragmentReluctance rvLow

    it "Low motion stability increases reluctance" $
      let rvStable = ResonanceVector 0.5 0.5 0.5 0.5 0.9
          rvUnstable = ResonanceVector 0.5 0.5 0.5 0.5 0.1
      in computeFragmentReluctance rvUnstable > computeFragmentReluctance rvStable

    it "Maximum reluctance is capped at 1.0" $
      let rv = ResonanceVector 0.0 1.0 0.0 0.0 0.0  -- Worst case
      in computeFragmentReluctance rv <= 1.0

  describe "Emergency Override" $ do
    it "HRV collapse triggers suspension" $
      let prev = ResonanceVector 0.5 0.5 0.5 0.5 0.5
          curr = ResonanceVector 0.1 0.5 0.5 0.5 0.5  -- HRV dropped
      in shouldSuspendConsent prev curr `shouldBe` True

    it "GSR panic triggers suspension" $
      let prev = ResonanceVector 0.5 0.3 0.5 0.5 0.5
          curr = ResonanceVector 0.5 0.8 0.5 0.5 0.5  -- GSR spiked
      in shouldSuspendConsent prev curr `shouldBe` True

    it "Gradual changes don't trigger emergency" $
      let prev = ResonanceVector 0.6 0.5 0.5 0.5 0.5
          curr = ResonanceVector 0.55 0.55 0.5 0.5 0.5  -- Small changes
      in shouldSuspendConsent prev curr `shouldBe` False

  describe "QuickCheck Properties" $ do
    it "Coherence is always in [0,1]" $ property $
      \h g b t m -> let rv = mkBoundedVector h g b t m
                    in resonanceCoherence rv >= 0.0 && resonanceCoherence rv <= 1.0

    it "Reluctance is always in [0,1]" $ property $
      \h g b t m -> let rv = mkBoundedVector h g b t m
                    in computeFragmentReluctance rv >= 0.0 && computeFragmentReluctance rv <= 1.0

-- =============================================================================
-- Test Helpers
-- =============================================================================

-- | Create a test resonance vector with uniform values
mkTestVector :: Double -> ResonanceVector
mkTestVector v = ResonanceVector v v v v v

-- | Create bounded resonance vector from arbitrary values
mkBoundedVector :: Double -> Double -> Double -> Double -> Double -> ResonanceVector
mkBoundedVector h g b t m = ResonanceVector
  (clamp01 h) (clamp01 g) (clamp01 b) (clamp01 t) (clamp01 m)
  where
    clamp01 x = max 0.0 (min 1.0 (abs x))
