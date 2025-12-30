{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : Ra.Pipeline
Description : Biometric Resonance Tuning Pipeline for Chamber Personalization
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements the full biometric resonance tuning pipeline to personalize
coherence chambers based on user physiological data and Ra system parameters.

== Pipeline Stages

1. **Signal Normalization**: Map raw biometric signals to [0,1] per user baseline
2. **Coherence Vector**: Merge signals into multidimensional ResonanceVector
3. **Phase Alignment**: Compute golden-phase alignment (φ^n window scheduler)
4. **Threshold Mapping**: Map resonance vector to scalar system variables
5. **Consent Modulation**: Auto-suspend on coherence collapse

== Reference Materials

* /RA_CONSTANTS_V2.json/: Scalar types and thresholds
* /ELECTROMAGNETIC_HEALING_FREQUENCIES.md/: Field frequencies for chamber base tones
* /REICH_ORGONE_ACCUMULATOR.md/: Emotional-field accumulation and release

== Orgone Accumulator Principles

From Reich's work:

* Organic layers attract and hold orgone (life energy)
* Inorganic layers conduct and reflect orgone
* Alternating layers create multiplicative accumulation
* DOR (Deadly Orgone) = incoherent state (φ < 0.3183)
* POR (Positive Orgone) = coherent state (φ ≥ 0.618)

== Safety Controls

* Emergency override on sudden HRV collapse
* Consent state transitions (FULL → DIMINISHED → SUSPENDED)
* Fragment reluctance modulation via GSR
-}
module Ra.Pipeline
  ( -- * Core Types
    BiometricSample(..)
  , UserBaseline(..)
  , ResonanceVector(..)
  , PipelineState(..)
  , PipelineOutput(..)
  , ConsentState(..)

    -- * Normalization
  , createBaseline
  , updateBaseline
  , normalizeSample
  , clampNormalized

    -- * Coherence Vector
  , computeResonanceVector
  , resonanceCoherence
  , resonanceStability

    -- * Phase Alignment (φ^n Window)
  , PhaseWindow(..)
  , computePhaseAlignment
  , isEmergenceWindow
  , nextEmergenceWindow

    -- * Threshold Mapping
  , ThresholdConfig(..)
  , defaultThresholds
  , mapToScalarParams
  , computeFluxCoherence
  , computeInversionPotential
  , computeRadialWeight

    -- * Consent Modulation
  , shouldSuspendConsent
  , computeConsentTransition
  , applyConsentRestriction

    -- * Chamber Personalization
  , ChamberConfig(..)
  , personalizeChamber
  , computeAmbientResonance
  , computeSymmetryMode
  , computeTimbreLayers
  , computeAccessRange

    -- * Fragment Reluctance
  , computeFragmentReluctance
  , modulateReluctanceByGSR

    -- * Inversion Handling
  , isInvertedState
  , computeInversionColor
  , applyInversionTransform

    -- * Full Pipeline
  , runPipeline
  , stepPipeline
  , initPipelineState

    -- * Constants
  , coherenceFloorDOR
  , coherenceFloorPOR
  , emergencyHRVThreshold
  , maxPhaseDepth
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)

import Ra.Tuning (phiGolden)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Coherence floor for DOR (Deadly Orgone) - incoherent state
--
-- From Reich: below this threshold, the field becomes life-annulling.
coherenceFloorDOR :: Double
coherenceFloorDOR = 0.3183  -- 1/π ≈ φ⁻²

-- | Coherence floor for POR (Positive Orgone) - coherent state
--
-- From Reich: above this threshold, life-enhancing effects emerge.
coherenceFloorPOR :: Double
coherenceFloorPOR = 0.618   -- φ⁻¹

-- | Emergency HRV threshold for consent suspension
--
-- If HRV drops below this suddenly, trigger emergency override.
emergencyHRVThreshold :: Double
emergencyHRVThreshold = 0.15

-- | Maximum phase depth for φ^n windows
maxPhaseDepth :: Int
maxPhaseDepth = 9

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Raw biometric sample with timestamp
data BiometricSample = BiometricSample
  { bsTimestamp      :: !Double     -- ^ Seconds since epoch
  , bsHeartRate      :: !Double     -- ^ BPM
  , bsHRV            :: !Double     -- ^ Heart Rate Variability (ms)
  , bsGSR            :: !Double     -- ^ Galvanic Skin Response (μS)
  , bsBreathingRate  :: !Double     -- ^ Breaths per minute
  , bsTemperature    :: !Double     -- ^ Skin temperature (°C)
  , bsAccelerometer  :: !(Double, Double, Double)  -- ^ (x, y, z) acceleration
  } deriving (Eq, Show, Generic, NFData)

-- | User baseline for normalization (calibrated over 5-10 minutes)
data UserBaseline = UserBaseline
  { ubHeartRateMin   :: !Double     -- ^ Minimum observed HR
  , ubHeartRateMax   :: !Double     -- ^ Maximum observed HR
  , ubHRVMin         :: !Double     -- ^ Minimum observed HRV
  , ubHRVMax         :: !Double     -- ^ Maximum observed HRV
  , ubGSRMin         :: !Double     -- ^ Minimum observed GSR
  , ubGSRMax         :: !Double     -- ^ Maximum observed GSR
  , ubBreathMin      :: !Double     -- ^ Minimum breathing rate
  , ubBreathMax      :: !Double     -- ^ Maximum breathing rate
  , ubTempMin        :: !Double     -- ^ Minimum temperature
  , ubTempMax        :: !Double     -- ^ Maximum temperature
  , ubCalibrationTime :: !Double    -- ^ Total calibration time (seconds)
  , ubSampleCount    :: !Int        -- ^ Number of calibration samples
  } deriving (Eq, Show, Generic, NFData)

-- | Normalized resonance vector [0,1] for each dimension
data ResonanceVector = ResonanceVector
  { rvHRV          :: !Double       -- ^ Normalized HRV (stress indicator)
  , rvGSR          :: !Double       -- ^ Normalized GSR (arousal)
  , rvBreath       :: !Double       -- ^ Normalized breathing coherence
  , rvTemperature  :: !Double       -- ^ Normalized thermal regulation
  , rvMotion       :: !Double       -- ^ Normalized motion/stability
  } deriving (Eq, Show, Generic, NFData)

-- | Consent state (ACSP protocol)
data ConsentState
  = FullConsent           -- ^ All operations permitted
  | DiminishedConsent     -- ^ Limited dynamics, no extremes
  | SuspendedConsent      -- ^ Drone/ambient only
  | EmergencyOverride     -- ^ System-initiated suspension
  deriving (Eq, Ord, Enum, Bounded, Show, Generic, NFData)

-- | Pipeline state between processing steps
data PipelineState = PipelineState
  { psBaseline       :: !UserBaseline    -- ^ Current baseline
  , psLastVector     :: !ResonanceVector -- ^ Previous resonance vector
  , psConsentState   :: !ConsentState    -- ^ Current consent level
  , psPhaseDepth     :: !Int             -- ^ Current φ^n depth
  , psEmergenceReady :: !Bool            -- ^ Is emergence window open?
  , psInverted       :: !Bool            -- ^ Is field inverted?
  , psReluctance     :: !Double          -- ^ Fragment reluctance [0,1]
  , psAccumulatedCharge :: !Double       -- ^ Orgone charge level
  , psTimestamp      :: !Double          -- ^ Last update time
  } deriving (Eq, Show, Generic, NFData)

-- | Pipeline output for chamber personalization
data PipelineOutput = PipelineOutput
  { poFluxCoherence    :: !Double        -- ^ Combined coherence from HRV + breath
  , poInversionPotential :: !Double      -- ^ Potential for field inversion
  , poRadialWeight     :: !Double        -- ^ Radial depth weighting
  , poConsentState     :: !ConsentState  -- ^ Recommended consent state
  , poAmbientFrequency :: !Double        -- ^ Chamber base frequency (Hz)
  , poSymmetryMode     :: !Int           -- ^ Visual symmetry (3-12 fold)
  , poTimbreLayers     :: ![Double]      -- ^ Audio layer intensities
  , poAccessTheta      :: !(Double, Double) -- ^ θ access range
  , poAccessPhi        :: !(Double, Double) -- ^ φ access range
  , poIsInverted       :: !Bool          -- ^ Is inverted/shadow state?
  , poReluctance       :: !Double        -- ^ Fragment reluctance
  } deriving (Eq, Show, Generic, NFData)

-- =============================================================================
-- Normalization
-- =============================================================================

-- | Create initial baseline from first sample
createBaseline :: BiometricSample -> UserBaseline
createBaseline BiometricSample{..} = UserBaseline
  { ubHeartRateMin = bsHeartRate - 10
  , ubHeartRateMax = bsHeartRate + 10
  , ubHRVMin = bsHRV * 0.8
  , ubHRVMax = bsHRV * 1.2
  , ubGSRMin = bsGSR * 0.8
  , ubGSRMax = bsGSR * 1.2
  , ubBreathMin = bsBreathingRate - 4
  , ubBreathMax = bsBreathingRate + 4
  , ubTempMin = bsTemperature - 0.5
  , ubTempMax = bsTemperature + 0.5
  , ubCalibrationTime = 0
  , ubSampleCount = 1
  }

-- | Update baseline with new sample (moving window)
updateBaseline :: BiometricSample -> UserBaseline -> UserBaseline
updateBaseline BiometricSample{..} ub = ub
  { ubHeartRateMin = min (ubHeartRateMin ub) bsHeartRate
  , ubHeartRateMax = max (ubHeartRateMax ub) bsHeartRate
  , ubHRVMin = min (ubHRVMin ub) bsHRV
  , ubHRVMax = max (ubHRVMax ub) bsHRV
  , ubGSRMin = min (ubGSRMin ub) bsGSR
  , ubGSRMax = max (ubGSRMax ub) bsGSR
  , ubBreathMin = min (ubBreathMin ub) bsBreathingRate
  , ubBreathMax = max (ubBreathMax ub) bsBreathingRate
  , ubTempMin = min (ubTempMin ub) bsTemperature
  , ubTempMax = max (ubTempMax ub) bsTemperature
  , ubSampleCount = ubSampleCount ub + 1
  }

-- | Normalize a sample to [0,1] range using baseline
normalizeSample :: UserBaseline -> BiometricSample -> ResonanceVector
normalizeSample ub BiometricSample{..} =
  let
    -- Normalize each dimension
    normalizeRange :: Double -> Double -> Double -> Double
    normalizeRange minV maxV val
      | maxV <= minV = 0.5  -- Avoid division by zero
      | otherwise = (val - minV) / (maxV - minV)

    hrv = normalizeRange (ubHRVMin ub) (ubHRVMax ub) bsHRV
    gsr = normalizeRange (ubGSRMin ub) (ubGSRMax ub) bsGSR
    breath = normalizeRange (ubBreathMin ub) (ubBreathMax ub) bsBreathingRate
    temp = normalizeRange (ubTempMin ub) (ubTempMax ub) bsTemperature

    -- Compute motion from accelerometer (magnitude normalized)
    (ax, ay, az) = bsAccelerometer
    motionMag = sqrt (ax*ax + ay*ay + az*az)
    -- Motion is inverse: low motion = high stability
    motion = 1.0 - min 1.0 (motionMag / 10.0)

  in clampNormalized $ ResonanceVector hrv gsr breath temp motion

-- | Clamp all resonance vector values to [0,1]
clampNormalized :: ResonanceVector -> ResonanceVector
clampNormalized ResonanceVector{..} = ResonanceVector
  { rvHRV = clamp01 rvHRV
  , rvGSR = clamp01 rvGSR
  , rvBreath = clamp01 rvBreath
  , rvTemperature = clamp01 rvTemperature
  , rvMotion = clamp01 rvMotion
  }
  where
    clamp01 x = max 0.0 (min 1.0 x)

-- =============================================================================
-- Coherence Vector
-- =============================================================================

-- | Compute resonance vector from normalized sample
computeResonanceVector :: BiometricSample -> UserBaseline -> ResonanceVector
computeResonanceVector = flip normalizeSample

-- | Compute overall coherence from resonance vector
--
-- Weighted combination per Prompt #7:
-- * 40% HRV (stress/relaxation)
-- * 30% Breathing (coherence frequency)
-- * 20% GSR (arousal)
-- * 10% Motion (stability)
resonanceCoherence :: ResonanceVector -> Double
resonanceCoherence ResonanceVector{..} =
  rvHRV * 0.4 + rvBreath * 0.3 + rvGSR * 0.2 + rvMotion * 0.1

-- | Compute stability of resonance (low variance = high stability)
resonanceStability :: [ResonanceVector] -> Double
resonanceStability [] = 0.5
resonanceStability [_] = 0.5
resonanceStability rvs =
  let
    coherences = map resonanceCoherence rvs
    mean = sum coherences / fromIntegral (length coherences)
    variance = sum (map (\c -> (c - mean) ^ (2 :: Int)) coherences) / fromIntegral (length coherences)
  in 1.0 - min 1.0 (sqrt variance * 2.0)  -- Invert: low variance = high stability

-- =============================================================================
-- Phase Alignment (φ^n Window)
-- =============================================================================

-- | Phase window for golden ratio timing
data PhaseWindow = PhaseWindow
  { pwDepth        :: !Int          -- ^ φ^n depth (0-9)
  , pwDuration     :: !Double       -- ^ Window duration (seconds)
  , pwAlignment    :: !Double       -- ^ How aligned is current time [0,1]
  , pwNextWindow   :: !Double       -- ^ Seconds until next window
  } deriving (Eq, Show, Generic, NFData)

-- | Compute phase alignment for current time
--
-- Uses golden ratio windows: φ^n for n ∈ {0..maxPhaseDepth}
computePhaseAlignment :: Double -> Int -> PhaseWindow
computePhaseAlignment timestamp depth =
  let
    -- φ^n period
    period = phiGolden ** fromIntegral depth

    -- Current phase within period
    phase = timestamp `mod'` period
    alignment = 1.0 - abs (2.0 * phase / period - 1.0)

    -- Time to next alignment peak
    nextPeak = period - phase

  in PhaseWindow depth period alignment nextPeak

-- | Check if we're in an emergence-favorable window
--
-- Emergence is favored when alignment > 0.8 across multiple depths
isEmergenceWindow :: Double -> Bool
isEmergenceWindow timestamp =
  let
    alignments = map (pwAlignment . computePhaseAlignment timestamp) [1..5]
    avgAlignment = sum alignments / fromIntegral (length alignments)
  in avgAlignment > 0.8

-- | Compute seconds until next emergence window
nextEmergenceWindow :: Double -> Double
nextEmergenceWindow timestamp =
  let
    windows = map (computePhaseAlignment timestamp) [1..5]
  in minimum (map pwNextWindow windows)

-- | Modular float (helper)
mod' :: Double -> Double -> Double
mod' x y = x - y * fromIntegral (floor (x / y) :: Int)

-- =============================================================================
-- Threshold Mapping
-- =============================================================================

-- | Configuration for threshold mapping
data ThresholdConfig = ThresholdConfig
  { tcFluxWeight     :: !Double     -- ^ HRV weight for flux coherence
  , tcBreathWeight   :: !Double     -- ^ Breathing weight for flux
  , tcGSRInvWeight   :: !Double     -- ^ GSR weight for inversion
  , tcStressInvWeight :: !Double    -- ^ Stress weight for inversion
  , tcRadialDecay    :: !Double     -- ^ Decay rate for radial weight
  } deriving (Eq, Show, Generic, NFData)

-- | Default threshold configuration
defaultThresholds :: ThresholdConfig
defaultThresholds = ThresholdConfig
  { tcFluxWeight = 0.6      -- 60% HRV
  , tcBreathWeight = 0.4    -- 40% breathing
  , tcGSRInvWeight = 0.5    -- 50% GSR for inversion
  , tcStressInvWeight = 0.5 -- 50% stress for inversion
  , tcRadialDecay = 0.1     -- 10% decay per shell
  }

-- | Map resonance vector to scalar system parameters
mapToScalarParams
  :: ThresholdConfig
  -> ResonanceVector
  -> Double              -- ^ Time stability
  -> (Double, Double, Double)  -- ^ (FluxCoherence, InversionPotential, RadialWeight)
mapToScalarParams tc rv stability =
  ( computeFluxCoherence tc rv
  , computeInversionPotential tc rv
  , computeRadialWeight tc stability
  )

-- | Compute FluxCoherence from HRV + breath
computeFluxCoherence :: ThresholdConfig -> ResonanceVector -> Double
computeFluxCoherence ThresholdConfig{..} ResonanceVector{..} =
  tcFluxWeight * rvHRV + tcBreathWeight * rvBreath

-- | Compute InversionPotential from GSR + stress
--
-- High GSR + low HRV = high inversion potential (shadow state)
computeInversionPotential :: ThresholdConfig -> ResonanceVector -> Double
computeInversionPotential ThresholdConfig{..} ResonanceVector{..} =
  let
    stress = 1.0 - rvHRV  -- Invert HRV for stress
    arousal = rvGSR
  in tcGSRInvWeight * arousal + tcStressInvWeight * stress

-- | Compute RadialWeight from time stability
computeRadialWeight :: ThresholdConfig -> Double -> Double
computeRadialWeight ThresholdConfig{..} stability =
  stability * (1.0 - tcRadialDecay)

-- =============================================================================
-- Consent Modulation
-- =============================================================================

-- | Check if consent should be suspended
--
-- Triggers on:
-- * HRV collapse below emergency threshold
-- * Sudden GSR spike (panic response)
shouldSuspendConsent :: ResonanceVector -> ResonanceVector -> Bool
shouldSuspendConsent prev curr =
  let
    -- HRV collapse: current < threshold AND dropped significantly
    hrvCollapse = rvHRV curr < emergencyHRVThreshold &&
                  (rvHRV prev - rvHRV curr) > 0.2

    -- GSR panic: sudden large spike
    gsrPanic = (rvGSR curr - rvGSR prev) > 0.4

  in hrvCollapse || gsrPanic

-- | Compute consent state transition
computeConsentTransition
  :: ConsentState      -- ^ Current state
  -> ResonanceVector   -- ^ Current resonance
  -> ConsentState      -- ^ New state
computeConsentTransition current rv =
  let coherence = resonanceCoherence rv
  in case current of
    FullConsent
      | coherence < coherenceFloorDOR -> SuspendedConsent
      | coherence < coherenceFloorPOR -> DiminishedConsent
      | otherwise -> FullConsent

    DiminishedConsent
      | coherence < coherenceFloorDOR -> SuspendedConsent
      | coherence > coherenceFloorPOR + 0.1 -> FullConsent  -- Hysteresis
      | otherwise -> DiminishedConsent

    SuspendedConsent
      | coherence > coherenceFloorPOR -> DiminishedConsent
      | otherwise -> SuspendedConsent

    EmergencyOverride
      | coherence > coherenceFloorPOR + 0.2 -> DiminishedConsent
      | otherwise -> EmergencyOverride

-- | Apply consent restrictions to output
applyConsentRestriction :: ConsentState -> PipelineOutput -> PipelineOutput
applyConsentRestriction state output = case state of
  FullConsent -> output

  DiminishedConsent -> output
    { poTimbreLayers = map (* 0.7) (poTimbreLayers output)  -- Reduce dynamics
    , poSymmetryMode = max 3 (min 6 (poSymmetryMode output)) -- Limit complexity
    }

  SuspendedConsent -> output
    { poTimbreLayers = [0.3]  -- Drone only
    , poSymmetryMode = 3      -- Minimal symmetry
    , poAccessTheta = (0, 0.1)  -- Minimal access
    , poAccessPhi = (0, 0.1)
    }

  EmergencyOverride -> output
    { poTimbreLayers = []     -- Silence
    , poSymmetryMode = 0      -- No visuals
    , poAccessTheta = (0, 0)
    , poAccessPhi = (0, 0)
    , poReluctance = 1.0      -- Maximum reluctance
    }

-- =============================================================================
-- Chamber Personalization
-- =============================================================================

-- | Chamber configuration derived from biometrics
data ChamberConfig = ChamberConfig
  { ccAmbientHz      :: !Double     -- ^ Base resonance frequency
  , ccSymmetry       :: !Int        -- ^ Visual symmetry mode
  , ccTimbreLayers   :: ![Double]   -- ^ Audio layer intensities
  , ccColorShift     :: !Double     -- ^ Color palette shift (-1 blue, +1 red)
  , ccFieldStrength  :: !Double     -- ^ Overall field intensity
  } deriving (Eq, Show, Generic, NFData)

-- | Personalize chamber from pipeline state
personalizeChamber :: PipelineState -> ResonanceVector -> ChamberConfig
personalizeChamber PipelineState{..} rv = ChamberConfig
  { ccAmbientHz = computeAmbientResonance rv
  , ccSymmetry = computeSymmetryMode rv psLastVector
  , ccTimbreLayers = computeTimbreLayers rv psConsentState
  , ccColorShift = if psInverted then (-0.7) else 0.3  -- Blue-shifted if inverted
  , ccFieldStrength = resonanceCoherence rv * (1.0 - psReluctance)
  }

-- | Compute ambient resonance frequency from coherence
--
-- Maps coherence to healing frequency range (per Rife/Priore research):
-- * Low coherence (< 0.3): 304 Hz (sedation/pain relief)
-- * Medium (0.3-0.6): 432 Hz (balancing)
-- * High (> 0.6): 528 Hz (repair/regeneration)
computeAmbientResonance :: ResonanceVector -> Double
computeAmbientResonance rv =
  let coherence = resonanceCoherence rv
  in if coherence < 0.3 then 304.0
     else if coherence < 0.6 then 432.0
     else 528.0

-- | Compute visual symmetry mode from radial stability
--
-- Stable state = higher-fold symmetry (12-fold dodecahedron)
-- Unstable = lower (3-fold triangle)
computeSymmetryMode :: ResonanceVector -> ResonanceVector -> Int
computeSymmetryMode curr prev =
  let
    stability = 1.0 - abs (resonanceCoherence curr - resonanceCoherence prev)
    -- Map [0,1] to [3,12]
    symmetry = round (3.0 + stability * 9.0)
  in max 3 (min 12 symmetry)

-- | Compute timbre layers from GSR fluctuations
--
-- Returns layer intensities for harmonic stacking:
-- * Layer 0: Fundamental (always present)
-- * Layer 1: 3rd harmonic (adds warmth)
-- * Layer 2: 5th harmonic (adds brightness)
-- * Layer 3: 7th harmonic (adds tension/release)
computeTimbreLayers :: ResonanceVector -> ConsentState -> [Double]
computeTimbreLayers rv consent = case consent of
  EmergencyOverride -> []
  SuspendedConsent -> [0.5]  -- Just fundamental, quiet
  _ ->
    let
      base = 0.8  -- Fundamental always strong
      third = rvGSR rv * 0.6  -- GSR modulates warmth
      fifth = rvBreath rv * 0.4  -- Breathing modulates brightness
      seventh = (1.0 - rvHRV rv) * 0.3  -- Stress adds tension
    in [base, third, fifth, seventh]

-- | Compute access range for θ (angular) dimension
--
-- Based on breathing phase: deeper breath = wider access
computeAccessRange :: ResonanceVector -> ((Double, Double), (Double, Double))
computeAccessRange rv =
  let
    -- θ range: breathing controls width
    thetaWidth = rvBreath rv * 0.8  -- Max 80% of full range
    thetaMin = (1.0 - thetaWidth) / 2.0
    thetaMax = thetaMin + thetaWidth

    -- φ range: HRV controls depth
    phiDepth = rvHRV rv * 0.9  -- Max 90% of full depth
    phiMin = 0.0
    phiMax = phiDepth

  in ((thetaMin, thetaMax), (phiMin, phiMax))

-- =============================================================================
-- Fragment Reluctance
-- =============================================================================

-- | Compute fragment reluctance from biometric resistance
--
-- High GSR + low motion stability = high reluctance (locked fragments)
computeFragmentReluctance :: ResonanceVector -> Double
computeFragmentReluctance ResonanceVector{..} =
  let
    arousalResistance = rvGSR * 0.6
    instabilityResistance = (1.0 - rvMotion) * 0.4
  in min 1.0 (arousalResistance + instabilityResistance)

-- | Modulate reluctance by GSR changes
--
-- Sudden GSR spike = temporary lock
-- GSR decrease = unlock
modulateReluctanceByGSR :: Double -> ResonanceVector -> ResonanceVector -> Double
modulateReluctanceByGSR baseReluctance prev curr =
  let
    gsrDelta = rvGSR curr - rvGSR prev

    -- Spike = increase, decrease = decrease
    modulation = if gsrDelta > 0.1
                 then 0.2  -- Increase reluctance
                 else if gsrDelta < (-0.1)
                 then (-0.1)  -- Decrease reluctance
                 else 0.0

  in max 0.0 (min 1.0 (baseReluctance + modulation))

-- =============================================================================
-- Inversion Handling
-- =============================================================================

-- | Check if state is inverted (shadow work)
--
-- Inversion occurs when:
-- * GSR is high (arousal)
-- * HRV is low (stress)
-- * Coherence is below POR threshold
isInvertedState :: ResonanceVector -> Bool
isInvertedState rv =
  let coherence = resonanceCoherence rv
  in coherence < coherenceFloorPOR && rvGSR rv > 0.6

-- | Compute color shift for inverted state
--
-- Normal: +1.0 (red-shifted, warm)
-- Inverted: -1.0 (blue-shifted, cool)
computeInversionColor :: Bool -> Double
computeInversionColor isInverted = if isInverted then (-0.7) else 0.3

-- | Apply inversion transform to output
applyInversionTransform :: Bool -> PipelineOutput -> PipelineOutput
applyInversionTransform False output = output
applyInversionTransform True output = output
  { poIsInverted = True
  , poTimbreLayers = map (* 0.8) (poTimbreLayers output)  -- Soften for shadow
  , poSymmetryMode = max 3 (poSymmetryMode output - 2)    -- Reduce complexity
  }

-- =============================================================================
-- Full Pipeline
-- =============================================================================

-- | Initialize pipeline state from first sample
initPipelineState :: BiometricSample -> PipelineState
initPipelineState sample =
  let
    baseline = createBaseline sample
    rv = normalizeSample baseline sample
  in PipelineState
    { psBaseline = baseline
    , psLastVector = rv
    , psConsentState = FullConsent
    , psPhaseDepth = 1
    , psEmergenceReady = False
    , psInverted = False
    , psReluctance = 0.0
    , psAccumulatedCharge = 0.5  -- Start at mid-charge
    , psTimestamp = bsTimestamp sample
    }

-- | Run full pipeline step
stepPipeline :: BiometricSample -> PipelineState -> (PipelineOutput, PipelineState)
stepPipeline sample state =
  let
    -- Update baseline
    newBaseline = updateBaseline sample (psBaseline state)

    -- Normalize current sample
    rv = normalizeSample newBaseline sample

    -- Check for consent suspension
    shouldSuspend = shouldSuspendConsent (psLastVector state) rv

    -- Compute consent transition
    baseConsent = if shouldSuspend
                  then EmergencyOverride
                  else computeConsentTransition (psConsentState state) rv

    -- Check inversion
    isInv = isInvertedState rv

    -- Compute phase alignment
    phaseAligned = isEmergenceWindow (bsTimestamp sample)

    -- Compute reluctance
    newReluctance = modulateReluctanceByGSR
                      (computeFragmentReluctance rv)
                      (psLastVector state)
                      rv

    -- Compute access ranges
    ((thetaMin, thetaMax), (phiMin, phiMax)) = computeAccessRange rv

    -- Build output
    output = PipelineOutput
      { poFluxCoherence = computeFluxCoherence defaultThresholds rv
      , poInversionPotential = computeInversionPotential defaultThresholds rv
      , poRadialWeight = computeRadialWeight defaultThresholds (rvMotion rv)
      , poConsentState = baseConsent
      , poAmbientFrequency = computeAmbientResonance rv
      , poSymmetryMode = computeSymmetryMode rv (psLastVector state)
      , poTimbreLayers = computeTimbreLayers rv baseConsent
      , poAccessTheta = (thetaMin, thetaMax)
      , poAccessPhi = (phiMin, phiMax)
      , poIsInverted = isInv
      , poReluctance = newReluctance
      }

    -- Apply consent restrictions
    restrictedOutput = applyConsentRestriction baseConsent output

    -- Apply inversion transform
    finalOutput = applyInversionTransform isInv restrictedOutput

    -- Update state
    newState = state
      { psBaseline = newBaseline
      , psLastVector = rv
      , psConsentState = baseConsent
      , psEmergenceReady = phaseAligned
      , psInverted = isInv
      , psReluctance = newReluctance
      , psTimestamp = bsTimestamp sample
      }

  in (finalOutput, newState)

-- | Run pipeline on sequence of samples
runPipeline :: [BiometricSample] -> PipelineState -> ([PipelineOutput], PipelineState)
runPipeline [] state = ([], state)
runPipeline (s:ss) state =
  let
    (out, newState) = stepPipeline s state
    (outs, finalState) = runPipeline ss newState
  in (out : outs, finalState)
