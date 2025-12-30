{-|
Module      : Ra.Constants.Extended
Description : Centralized numerical constants for Ra System
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Unified constant repository including:

* Golden ratio (φ) and derivatives
* π-based symmetry thresholds
* Rife / Keely / Solfeggio harmonic bands
* Coherence floors and caps
* Ankh-derived invariants
* Kozyrev temporal constants

== Design Principle

All numerical constants used across Ra System modules should be
imported from this module to ensure consistency. Constants are
organized by source tradition and mathematical relationship.

== Precision Policy

* Golden ratio: 10 decimal places (1.6180339887)
* Pi values: As defined in Ra.Constants (varying precision per tradition)
* Frequency constants: Hz with 1 decimal precision
* Ratios: 4 decimal places minimum

== Source References

* Bateman: "The Rods of Amon Ra"
* Keely: "40 Laws of Sympathetic Vibration"
* Reich: Orgone Accumulator research
* Rife: Mortal Oscillatory Rates (MORs)
* Kozyrev: Causal mechanics time constants
-}
module Ra.Constants.Extended
  ( -- * Golden Ratio Family
    phi
  , phiSquared
  , phiCubed
  , phiInverse
  , phiInverseSquared
  , phiPower

    -- * Pi Family (from Ra.Constants)
  , piRed
  , piGreen
  , piBlue
  , piInverse

    -- * Coherence Thresholds
  , coherenceFloorDOR
  , coherenceFloorPOR
  , coherenceEmergence
  , coherenceShadow
  , coherenceMax

    -- * Symmetry Thresholds
  , symmetryMinFold
  , symmetryMaxFold
  , symmetryGoldenFold

    -- * Keely Constants
  , keelyOctaveBandWidth
  , keelyTotalOctaves
  , keelyAtomolicDensity
  , keelyTripleRatio

    -- * Rife Frequencies (Hz)
  , rifeCoreFrequencies
  , rifeCarcinoma
  , rifeSarcoma
  , rifeVitality

    -- * Solfeggio Frequencies (Hz)
  , solfeggio174
  , solfeggio285
  , solfeggio396
  , solfeggio417
  , solfeggio528
  , solfeggio639
  , solfeggio741
  , solfeggio852
  , solfeggio963
  , solfeggioAll

    -- * Healing Frequencies (Hz)
  , freqSedation
  , freqBalance
  , freqRepair
  , freqSchumann

    -- * Kozyrev Temporal Constants
  , kozyrevC2
  , kozyrevPseudoscalar

    -- * Reich Orgone Constants
  , reichLayerCount
  , reichAccumulationMax
  , reichChargeDecay
  , reichTemperatureDelta

    -- * Ankh-Derived Constants
  , ankhValue
  , ankhHalfLife
  , ankhResonance

    -- * Repitan Constants
  , repitanCount
  , repitanFineStructure

    -- * Time Constants (seconds)
  , phiWindowBase
  , emergenceWindowMin
  , calibrationPeriod

    -- * Utility Functions
  , phiWindow
  , isGoldenRatio
  , nearestSolfeggio
  ) where

import Data.List (minimumBy)
import Data.Ord (comparing)

-- =============================================================================
-- Golden Ratio Family
-- =============================================================================

-- | Golden ratio φ = (1 + √5) / 2
--
-- The fundamental proportion underlying Ra System geometry.
phi :: Double
phi = 1.6180339887

-- | φ² = φ + 1 ≈ 2.618
phiSquared :: Double
phiSquared = phi * phi

-- | φ³ ≈ 4.236
phiCubed :: Double
phiCubed = phi * phi * phi

-- | 1/φ = φ - 1 ≈ 0.618
--
-- Also the coherence floor for POR (Positive Orgone).
phiInverse :: Double
phiInverse = 1.0 / phi

-- | 1/φ² ≈ 0.382
phiInverseSquared :: Double
phiInverseSquared = 1.0 / phiSquared

-- | Compute φ^n for any integer n
phiPower :: Int -> Double
phiPower n
  | n >= 0    = phi ** fromIntegral n
  | otherwise = phiInverse ** fromIntegral (abs n)

-- =============================================================================
-- Pi Family
-- =============================================================================

-- | Red Trac Pi (from Bateman)
piRed :: Double
piRed = 3.141592592

-- | Green Trac Pi = √9.876543210
piGreen :: Double
piGreen = 3.142696806

-- | Blue Trac Pi = 1/0.318086250
piBlue :: Double
piBlue = 3.143801408

-- | 1/π ≈ 0.3183
--
-- Also the coherence floor for DOR (Deadly Orgone).
piInverse :: Double
piInverse = 1.0 / piRed

-- =============================================================================
-- Coherence Thresholds
-- =============================================================================

-- | DOR floor: Below this, field becomes life-annulling (1/π)
coherenceFloorDOR :: Double
coherenceFloorDOR = piInverse  -- ≈ 0.3183

-- | POR floor: Above this, life-enhancing effects emerge (1/φ)
coherenceFloorPOR :: Double
coherenceFloorPOR = phiInverse  -- ≈ 0.618

-- | Emergence threshold: Strong coherence for fragment emergence
coherenceEmergence :: Double
coherenceEmergence = 0.8

-- | Shadow threshold: Below this, shadow/inverted work activates
coherenceShadow :: Double
coherenceShadow = 0.4

-- | Maximum coherence (unity)
coherenceMax :: Double
coherenceMax = 1.0

-- =============================================================================
-- Symmetry Thresholds
-- =============================================================================

-- | Minimum visual symmetry (triangle)
symmetryMinFold :: Int
symmetryMinFold = 3

-- | Maximum visual symmetry (dodecahedron)
symmetryMaxFold :: Int
symmetryMaxFold = 12

-- | Golden symmetry (pentagon)
symmetryGoldenFold :: Int
symmetryGoldenFold = 5

-- =============================================================================
-- Keely Constants
-- =============================================================================

-- | Keely's 21-octave force transformation band
keelyOctaveBandWidth :: Int
keelyOctaveBandWidth = 21

-- | Total octaves in Keely's system (5 bands × 21)
keelyTotalOctaves :: Int
keelyTotalOctaves = 105

-- | Atomolic density multiplier (986,000× steel)
keelyAtomolicDensity :: Double
keelyAtomolicDensity = 986000.0

-- | Triple ratio for molecular structure (3 atoms per molecule)
keelyTripleRatio :: Int
keelyTripleRatio = 3

-- =============================================================================
-- Rife Frequencies (Hz)
-- =============================================================================

-- | Rife's 5 core pleomorphic targeting frequencies
rifeCoreFrequencies :: [Double]
rifeCoreFrequencies = [666.0, 690.0, 740.0, 1840.0, 1998.0]

-- | Carcinoma frequency (all forms)
rifeCarcinoma :: Double
rifeCarcinoma = 2128.0

-- | Sarcoma frequency (all forms)
rifeSarcoma :: Double
rifeSarcoma = 2008.0

-- | General vitality and energy
rifeVitality :: Double
rifeVitality = 9999.0

-- =============================================================================
-- Solfeggio Frequencies (Hz)
-- =============================================================================

-- | 174 Hz - Foundation, pain reduction
solfeggio174 :: Double
solfeggio174 = 174.0

-- | 285 Hz - Tissue repair, cellular memory
solfeggio285 :: Double
solfeggio285 = 285.0

-- | 396 Hz - Liberation from fear/guilt
solfeggio396 :: Double
solfeggio396 = 396.0

-- | 417 Hz - Facilitating change
solfeggio417 :: Double
solfeggio417 = 417.0

-- | 528 Hz - DNA repair, transformation (MI)
solfeggio528 :: Double
solfeggio528 = 528.0

-- | 639 Hz - Connection, relationships
solfeggio639 :: Double
solfeggio639 = 639.0

-- | 741 Hz - Expression, solutions
solfeggio741 :: Double
solfeggio741 = 741.0

-- | 852 Hz - Intuition, spiritual order
solfeggio852 :: Double
solfeggio852 = 852.0

-- | 963 Hz - Divine consciousness, pineal activation
solfeggio963 :: Double
solfeggio963 = 963.0

-- | All 9 Solfeggio frequencies
solfeggioAll :: [Double]
solfeggioAll =
  [ solfeggio174, solfeggio285, solfeggio396
  , solfeggio417, solfeggio528, solfeggio639
  , solfeggio741, solfeggio852, solfeggio963
  ]

-- =============================================================================
-- Healing Frequencies (Hz)
-- =============================================================================

-- | Sedation/pain relief frequency
freqSedation :: Double
freqSedation = 304.0

-- | Balance/grounding frequency (A=432)
freqBalance :: Double
freqBalance = 432.0

-- | Repair/regeneration frequency (Solfeggio 528)
freqRepair :: Double
freqRepair = 528.0

-- | Schumann resonance (Earth's fundamental)
freqSchumann :: Double
freqSchumann = 7.83

-- =============================================================================
-- Kozyrev Temporal Constants
-- =============================================================================

-- | Kozyrev's C₂ time pattern velocity (km/sec)
--
-- The pseudo-scalar velocity relating cause to effect.
-- C₂ = α × e²/h where e = electron charge, h = Planck
kozyrevC2 :: Double
kozyrevC2 = 350.0

-- | Pseudoscalar property: sign change under mirror reflection
--
-- +1 for laevorotary (left-hand) coordinate system
-- -1 for dextrorotary (right-hand) system
kozyrevPseudoscalar :: Int -> Double
kozyrevPseudoscalar handedness = if handedness > 0 then 1.0 else (-1.0)

-- =============================================================================
-- Reich Orgone Constants
-- =============================================================================

-- | Traditional orgone accumulator layer count
reichLayerCount :: Int
reichLayerCount = 7

-- | Maximum accumulation factor
reichAccumulationMax :: Double
reichAccumulationMax = 10.0

-- | Charge decay rate per second
reichChargeDecay :: Double
reichChargeDecay = 0.05

-- | Einstein experiment temperature differential (°C)
reichTemperatureDelta :: Double
reichTemperatureDelta = 0.35  -- 0.3-0.4°C observed

-- =============================================================================
-- Ankh-Derived Constants
-- =============================================================================

-- | Ankh master constant (π_red × φ_green)
ankhValue :: Double
ankhValue = 5.08938

-- | Ankh half-life (RAC₁ = Ankh/8)
ankhHalfLife :: Double
ankhHalfLife = ankhValue / 8.0

-- | Ankh resonance frequency (Hz)
ankhResonance :: Double
ankhResonance = ankhValue * 100.0  -- 508.938 Hz

-- =============================================================================
-- Repitan Constants
-- =============================================================================

-- | Number of Repitans (27-fold division)
repitanCount :: Int
repitanCount = 27

-- | Fine structure from Repitan(1)² = (1/27)²
repitanFineStructure :: Double
repitanFineStructure = (1.0 / 27.0) ** 2  -- ≈ 0.00137

-- =============================================================================
-- Time Constants
-- =============================================================================

-- | Base φ window duration (seconds)
phiWindowBase :: Double
phiWindowBase = phi  -- ~1.618 seconds

-- | Minimum emergence window (seconds)
emergenceWindowMin :: Double
emergenceWindowMin = 0.1

-- | Calibration period for baseline (seconds)
calibrationPeriod :: Double
calibrationPeriod = 300.0  -- 5 minutes

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Compute φ^n window duration
phiWindow :: Int -> Double
phiWindow depth = phiPower depth

-- | Check if a ratio is approximately golden
isGoldenRatio :: Double -> Double -> Bool
isGoldenRatio ratio tolerance =
  abs (ratio - phi) < tolerance ||
  abs (ratio - phiInverse) < tolerance

-- | Find nearest Solfeggio frequency to a given frequency
nearestSolfeggio :: Double -> Double
nearestSolfeggio freq =
  minimumBy (comparing (\s -> abs (s - freq))) solfeggioAll
