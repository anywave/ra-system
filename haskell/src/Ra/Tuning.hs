{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : Ra.Tuning
Description : Biometric Resonance Tuning for Scalar Fields
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements real-time biometric resonance tuning that modifies active scalar
field configuration based on live biometric input. This creates a dynamic
feedback loop between physiological state and field emergence dynamics.

== Theoretical Background

Drawing from multiple resonance traditions:

* /Kaali-Beck Blood Electrification/: Biological resonance via electric stimulus
  establishes the principle that subtle electrical signals can modulate
  biological coherence patterns.

* /Reich Orgone/: Layered energy fields and accumulation principles inform
  how shell-based structures concentrate and direct scalar potential.

* /Keely Sympathetic Vibration/: Resonant field modulation through harmonic
  coupling enables coherence between disparate systems (bio ↔ scalar).

* /Golod Russian Pyramids/: Environmental field effects on organisms suggest
  geometric field configurations can enhance biological coherence.

== Physiological-to-Physics Mapping

The core insight is that human physiological rhythms (heart rate variability,
respiration, brainwave coherence) naturally couple to golden-ratio harmonics
when the organism is in coherent states. This module exploits that coupling
to tune scalar field parameters:

@
  Biometric Signal      Effect on Scalar Field
  ─────────────────────────────────────────────
  HRV < baseline    →   Reduce emergence potential
  EEG α coherence ↑ →   Boost deep-shell harmonic weight
  GSR ↑ (stress)    →   Increase flux instability, decay rate
  Respiration sync  →   Align phase to φⁿ temporal windows
  Heart rate        →   Modulate base frequency coupling
@

== Safety Constraints

All field modifications are normalized and bounded. The tuning process
cannot produce unbounded growth or invalid field configurations.
-}
module Ra.Tuning
  ( -- * Core Types
    BiometricInput(..)
  , ResonanceTuningProfile(..)
  , TuningResult(..)

    -- * Phi Windows (DSP)
  , PhiWindow(..)
  , mkPhiWindow
  , DSPTuningProfile(..)

    -- * Smart Constructors
  , mkBiometricInput
  , defaultTuningProfile
  , therapeuticProfile
  , meditativeProfile

    -- * Main Tuning Function
  , tuneScalarField
  , tuneScalarComponent

    -- * DSP Profile Generation
  , generateTuningProfile
  , smoothProfile
  , computeRespiratoryPhaseRaw

    -- * Derived Metrics
  , computeCoherenceIndex
  , computeStressIndex
  , computeRespiratoryPhase
  , computePhiAlignment

    -- * Calibration Helpers
  , calibrateProfile
  , adjustForBaseline

    -- * Constants
  , phiGolden
  , ankh_coherence_threshold
  , hrv_baseline_default
  , gsr_neutral_default
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)

import Ra.Scalar
  ( ScalarField(..)
  , ScalarComponent(..)
  , RadialProfile(..)
  )
import Ra.Constants (Ankh(..), ankh, GreenPhi(..), greenPhi)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Biometric input from physiological sensors
--
-- All values are normalized or in standard units:
--
--   * @heartRate@: Beats per minute (typical range: 50-100 BPM at rest)
--   * @hrv@: Heart rate variability, normalized [0,1] where 1 = high variability
--   * @gsr@: Galvanic skin response, normalized [0,1] where 0 = calm, 1 = aroused
--   * @respirationRate@: Breaths per minute (typical: 12-20 at rest)
--   * @eegAlphaCoherence@: EEG alpha wave coherence [0,1], 1 = highly coherent
--
-- Note: HRV is a key marker of autonomic nervous system balance.
-- High HRV correlates with parasympathetic dominance (rest/digest).
-- Low HRV correlates with sympathetic dominance (fight/flight).
data BiometricInput = BiometricInput
  { biHeartRate        :: !Double  -- ^ Beats per minute
  , biHRV              :: !Double  -- ^ Normalized HRV [0,1]
  , biGSR              :: !Double  -- ^ Galvanic skin response [0,1]
  , biRespirationRate  :: !Double  -- ^ Breaths per minute
  , biEEGAlphaCoherence :: !Double -- ^ EEG alpha wave coherence [0,1]
  } deriving (Eq, Show, Generic, NFData)

-- | Smart constructor with input validation and clamping
mkBiometricInput
  :: Double  -- ^ Heart rate (BPM)
  -> Double  -- ^ HRV [0,1]
  -> Double  -- ^ GSR [0,1]
  -> Double  -- ^ Respiration rate
  -> Double  -- ^ EEG alpha coherence [0,1]
  -> BiometricInput
mkBiometricInput hr hrv gsr resp eeg = BiometricInput
  { biHeartRate = clampPositive hr
  , biHRV = clamp01' hrv
  , biGSR = clamp01' gsr
  , biRespirationRate = clampPositive resp
  , biEEGAlphaCoherence = clamp01' eeg
  }
  where
    clamp01' x = max 0.0 (min 1.0 x)
    clampPositive x = max 0.0 x

-- | Per-user calibration profile for resonance tuning
--
-- This allows individual tuning based on:
--   * Baseline physiological values (varies per person)
--   * Sensitivity coefficients (how strongly biometrics affect field)
--   * Safety overrides (for therapeutic contexts)
data ResonanceTuningProfile = ResonanceTuningProfile
  { rtpHRVBaseline       :: !Double  -- ^ User's resting HRV baseline [0,1]
  , rtpGSRSensitivity    :: !Double  -- ^ How strongly GSR affects flux (0 = ignore)
  , rtpShellGain         :: !Double  -- ^ Amplification for shell depth effects
  , rtpCoherenceWeight   :: !Double  -- ^ Weight for EEG coherence influence
  , rtpOverrideThresholds :: !Bool   -- ^ If True, allow emergence below threshold
  , rtpPhiAlignment      :: !Double  -- ^ Target phi-alignment window [0,1]
  , rtpHeartRateTarget   :: !Double  -- ^ Target heart rate for resonance (BPM)
  } deriving (Eq, Show, Generic, NFData)

-- | Default tuning profile (balanced, general-purpose)
defaultTuningProfile :: ResonanceTuningProfile
defaultTuningProfile = ResonanceTuningProfile
  { rtpHRVBaseline = hrv_baseline_default
  , rtpGSRSensitivity = 0.5
  , rtpShellGain = 1.0
  , rtpCoherenceWeight = 0.7
  , rtpOverrideThresholds = False
  , rtpPhiAlignment = 0.618  -- Golden ratio inverse
  , rtpHeartRateTarget = 60.0  -- Calm resting rate
  }

-- | Therapeutic profile (gentler, more protective)
therapeuticProfile :: ResonanceTuningProfile
therapeuticProfile = ResonanceTuningProfile
  { rtpHRVBaseline = 0.6
  , rtpGSRSensitivity = 0.3      -- Reduced stress sensitivity
  , rtpShellGain = 0.7           -- Gentler shell effects
  , rtpCoherenceWeight = 0.5     -- Moderate coherence influence
  , rtpOverrideThresholds = False
  , rtpPhiAlignment = 0.5        -- More relaxed alignment
  , rtpHeartRateTarget = 65.0
  }

-- | Meditative profile (deep coherence focus)
meditativeProfile :: ResonanceTuningProfile
meditativeProfile = ResonanceTuningProfile
  { rtpHRVBaseline = 0.7         -- Expect high HRV in meditation
  , rtpGSRSensitivity = 0.2      -- Low stress sensitivity
  , rtpShellGain = 1.5           -- Enhanced shell depth access
  , rtpCoherenceWeight = 0.9     -- Strong coherence weighting
  , rtpOverrideThresholds = True -- Allow deeper emergence
  , rtpPhiAlignment = 0.618      -- Strict phi alignment
  , rtpHeartRateTarget = 55.0    -- Deep relaxation target
  }

-- | Result of tuning operation with diagnostic information
data TuningResult = TuningResult
  { trField          :: !ScalarField     -- ^ Tuned field
  , trCoherenceIndex :: !Double          -- ^ Computed coherence [0,1]
  , trStressIndex    :: !Double          -- ^ Computed stress [0,1]
  , trPhiAlignment   :: !Double          -- ^ How well aligned to phi
  , trEmergenceMod   :: !Double          -- ^ Emergence potential modifier
  } deriving (Eq, Show, Generic, NFData)

-- =============================================================================
-- Phi-Scaled Temporal Windows (Prompt #3)
-- =============================================================================

-- | Phi-aligned temporal window for entrainment
--
-- Represents a golden-ratio scaled time window for field resonance.
-- The window depth determines the phi exponent (φ^depth).
data PhiWindow = PhiWindow
  { pwDepth          :: !Int     -- ^ Window depth (φ^n exponent)
  , pwAlignmentScore :: !Double  -- ^ Alignment quality [0,1]
  , pwDuration       :: !Double  -- ^ Window duration in seconds
  } deriving (Eq, Show, Generic, NFData)

-- | Create a phi window at given depth
mkPhiWindow :: Int -> Double -> PhiWindow
mkPhiWindow depth alignment = PhiWindow
  { pwDepth = max 0 depth
  , pwAlignmentScore = clamp01 alignment
  , pwDuration = phiGolden ** fromIntegral depth  -- φ^n seconds
  }

-- | DSP-style tuning profile for field resonance
--
-- Maps biometric state to frequency-domain control parameters.
-- Used for real-time field modulation.
data DSPTuningProfile = DSPTuningProfile
  { dspTargetFrequencies   :: ![Double]        -- ^ Target frequencies (Hz)
  , dspEntrainmentWindow   :: !PhiWindow       -- ^ Phi-scaled entrainment window
  , dspCoherenceScalar     :: !Double          -- ^ Normalized coherence [0,1]
  , dspFeedbackSensitivity :: !Double          -- ^ Field reactivity [0,1]
  , dspIsOverdriven        :: !Bool            -- ^ Emergency override state
  , dspBioSignature        :: !BiometricInput  -- ^ Source biometric snapshot
  } deriving (Eq, Show, Generic, NFData)

-- | Generate DSP tuning profile from biometric snapshot
--
-- This is the core DSP mapping function that transforms physiological
-- signals into frequency-resonant control parameters.
--
-- === Frequency Derivation
--
-- Target frequencies are derived from:
--
--   * HRV → 0.1-2 Hz range (therapeutic resonance)
--   * Respiration → 0.1-0.3 Hz (breathing rate)
--   * Heart rate → base rhythm coupling
--
-- === Override Logic
--
-- @isOverdriven@ is set when:
--   * Low HRV + High GSR (sympathetic overload)
--   * EEG coherence drops while stress rises
--
-- This triggers emergency field stabilization.
generateTuningProfile :: BiometricInput -> DSPTuningProfile
generateTuningProfile input =
  let
    -- Derive target frequencies from biometrics
    -- HRV maps to therapeutic range (0.1-2 Hz)
    hrvFreq = 0.1 + biHRV input * 1.9

    -- Respiration rate to Hz (breaths/min → Hz)
    respFreq = biRespirationRate input / 60.0

    -- Heart rate coupling (BPM → Hz, then halved for subharmonic)
    hrBaseFreq = biHeartRate input / 60.0
    hrSubFreq = hrBaseFreq / 2.0

    -- EEG alpha band influence (8-12 Hz range, scaled)
    alphaFreq = 8.0 + biEEGAlphaCoherence input * 4.0

    targetFreqs = [hrvFreq, respFreq, hrBaseFreq, hrSubFreq, alphaFreq]

    -- Compute coherence from HRV and EEG
    coherence = (biHRV input * 0.4 + biEEGAlphaCoherence input * 0.6)

    -- Compute stress index
    stress = biGSR input * 0.6 + (1.0 - biHRV input) * 0.4

    -- Determine if overdriven (emergency state)
    -- Low HRV + High GSR = sympathetic overload
    isOverdriven = biHRV input < 0.3 && biGSR input > 0.7

    -- Feedback sensitivity: inversely related to stress
    -- High stress = lower sensitivity (protective)
    sensitivity = 1.0 - stress * 0.7

    -- Entrainment window from respiratory phase
    respPhase = computeRespiratoryPhaseRaw input
    windowDepth = round (respPhase * 5)  -- Map to phi window depth 0-5
    phiWindow = mkPhiWindow windowDepth coherence

  in DSPTuningProfile
    { dspTargetFrequencies = targetFreqs
    , dspEntrainmentWindow = phiWindow
    , dspCoherenceScalar = coherence
    , dspFeedbackSensitivity = clamp01 sensitivity
    , dspIsOverdriven = isOverdriven
    , dspBioSignature = input
    }

-- | Raw respiratory phase computation (0-1)
computeRespiratoryPhaseRaw :: BiometricInput -> Double
computeRespiratoryPhaseRaw input =
  let
    optimalRate = 6.0  -- Optimal coherent breathing
    rate = biRespirationRate input
    deviation = abs (rate - optimalRate) / optimalRate
  in clamp01 (1.0 - deviation * 0.5)

-- | Smooth a DSP profile over time (temporal smoothing)
smoothProfile
  :: Double            -- ^ Smoothing factor [0,1], 1 = no smoothing
  -> DSPTuningProfile  -- ^ Previous profile
  -> DSPTuningProfile  -- ^ Current profile
  -> DSPTuningProfile
smoothProfile alpha prev curr =
  let
    -- Interpolate scalar values
    smoothScalar :: Double -> Double -> Double
    smoothScalar p c = p * (1.0 - alpha) + c * alpha

    -- Smooth frequencies
    smoothedFreqs = zipWith smoothScalar
                      (dspTargetFrequencies prev)
                      (dspTargetFrequencies curr)

    -- Keep the shorter list if mismatched
    finalFreqs = if length (dspTargetFrequencies prev) == length (dspTargetFrequencies curr)
                 then smoothedFreqs
                 else dspTargetFrequencies curr

  in curr
    { dspTargetFrequencies = finalFreqs
    , dspCoherenceScalar = smoothScalar (dspCoherenceScalar prev) (dspCoherenceScalar curr)
    , dspFeedbackSensitivity = smoothScalar (dspFeedbackSensitivity prev) (dspFeedbackSensitivity curr)
    }

-- =============================================================================
-- Constants
-- =============================================================================

-- | Golden ratio (φ = 1.618...)
phiGolden :: Double
phiGolden = unGreenPhi greenPhi

-- | Ankh-derived coherence threshold
-- Below this, emergence potential is significantly reduced
ankh_coherence_threshold :: Double
ankh_coherence_threshold = unAnkh ankh / 10.0  -- ~0.318

-- | Default HRV baseline for average adult
hrv_baseline_default :: Double
hrv_baseline_default = 0.5

-- | Default GSR neutral point
gsr_neutral_default :: Double
gsr_neutral_default = 0.3

-- =============================================================================
-- Main Tuning Function
-- =============================================================================

-- | Tune a scalar field based on biometric input
--
-- This is the core function that creates biometric-to-field coupling.
-- The tuning process:
--
--   1. Compute coherence index from HRV and EEG
--   2. Compute stress index from GSR and heart rate
--   3. Derive phi-alignment from respiration phase
--   4. Modify each field component based on these indices
--
-- The resulting field maintains normalization and safety bounds.
--
-- === Physiological Mapping Details
--
-- * /HRV below baseline/: Reduces scalar potential. Low HRV indicates
--   sympathetic nervous system activation (stress), which constrains
--   the field's emergence capacity.
--
-- * /EEG alpha coherence up/: Boosts harmonic weight for deep shells.
--   High alpha coherence indicates a relaxed, focused state conducive
--   to accessing deeper field structures.
--
-- * /GSR elevated (stress)/: Increases flux instability and decay rate.
--   Stress creates "noise" in the field, reducing coherent emergence.
--
-- * /Respiration synchronized/: Aligns phase timing to φⁿ windows.
--   Coherent breathing patterns naturally entrain to golden-ratio
--   harmonics, enhancing field resonance.
tuneScalarField
  :: ResonanceTuningProfile
  -> BiometricInput
  -> ScalarField
  -> TuningResult
tuneScalarField profile input (ScalarField components) =
  let
    -- Step 1: Compute derived metrics
    coherence = computeCoherenceIndex profile input
    stress = computeStressIndex profile input
    _respPhase = computeRespiratoryPhase profile input  -- Used in phiAlign
    phiAlign = computePhiAlignment profile input

    -- Step 2: Compute emergence modifier
    -- Base: coherence weighted, stress reduces, phi alignment boosts
    emergenceMod = computeEmergenceModifier profile coherence stress phiAlign

    -- Step 3: Tune each component
    tunedComponents = map (tuneScalarComponent profile coherence stress emergenceMod) components

    -- Step 4: Normalize the field (ensure bounded output)
    normalizedField = normalizeField (ScalarField tunedComponents)

  in TuningResult
    { trField = normalizedField
    , trCoherenceIndex = coherence
    , trStressIndex = stress
    , trPhiAlignment = phiAlign
    , trEmergenceMod = emergenceMod
    }

-- | Tune a single scalar component
--
-- Modifies the component's weight and radial profile based on biometric state.
tuneScalarComponent
  :: ResonanceTuningProfile
  -> Double  -- ^ Coherence index
  -> Double  -- ^ Stress index
  -> Double  -- ^ Emergence modifier
  -> ScalarComponent
  -> ScalarComponent
tuneScalarComponent profile coherence stress emergMod (SC l m rp weight) =
  let
    -- Shell depth effect: higher coherence allows deeper shells
    -- High l values (deeper harmonics) are boosted by coherence
    shellFactor = 1.0 + (rtpShellGain profile) * coherence * (fromIntegral l / 10.0)

    -- Stress reduces weight (field "contracts" under stress)
    stressFactor = 1.0 - (rtpGSRSensitivity profile) * stress * 0.5

    -- Coherence weight boosts harmonic contribution
    coherenceFactor = 1.0 + (rtpCoherenceWeight profile) * coherence * 0.3

    -- Combined weight modification
    newWeight = weight * emergMod * shellFactor * stressFactor * coherenceFactor

    -- Radial profile modification: stress increases decay
    newProfile = tuneRadialProfile profile stress rp

  in SC l m newProfile (clampWeight newWeight)

-- | Tune radial profile based on stress
--
-- Stress increases decay rate (field "dissipates" faster under stress)
tuneRadialProfile :: ResonanceTuningProfile -> Double -> RadialProfile -> RadialProfile
tuneRadialProfile profile stress (RP typ scale decay) =
  let
    -- Stress increases decay rate
    stressDecayMod = 1.0 + stress * (rtpGSRSensitivity profile) * 0.5

    -- But coherence can counteract (via override thresholds)
    decayMod = if rtpOverrideThresholds profile
               then stressDecayMod * 0.5  -- Dampen stress effect
               else stressDecayMod

    newDecay = decay * decayMod

  in RP typ scale (max 0.01 newDecay)  -- Ensure minimum decay

-- | Compute emergence modifier from biometric state
computeEmergenceModifier
  :: ResonanceTuningProfile
  -> Double  -- ^ Coherence
  -> Double  -- ^ Stress
  -> Double  -- ^ Phi alignment
  -> Double
computeEmergenceModifier profile coherence stress phiAlign =
  let
    -- Base: phi-alignment is primary driver
    base = 0.5 + 0.5 * phiAlign

    -- Coherence boost (up to 1.5x at full coherence)
    coherenceBoost = 1.0 + coherence * 0.5

    -- Stress reduction (down to 0.5x at full stress)
    stressReduction = 1.0 - stress * 0.5

    -- Override can bypass stress reduction
    stressMod = if rtpOverrideThresholds profile
                then max 0.7 stressReduction  -- Floor at 0.7
                else stressReduction

    raw = base * coherenceBoost * stressMod

  in clamp01 raw  -- Normalize to [0,1]

-- =============================================================================
-- Derived Metrics
-- =============================================================================

-- | Compute coherence index from HRV and EEG alpha
--
-- High HRV + high alpha coherence = high overall coherence
-- This represents the degree of autonomic-neural synchronization.
computeCoherenceIndex :: ResonanceTuningProfile -> BiometricInput -> Double
computeCoherenceIndex profile input =
  let
    -- HRV contribution: compare to baseline
    hrvDelta = biHRV input - rtpHRVBaseline profile
    hrvContrib = 0.5 + hrvDelta * 0.5  -- Maps [-baseline, 1-baseline] to [0.5±...]

    -- EEG contribution (direct)
    eegContrib = biEEGAlphaCoherence input

    -- Weighted combination (favor EEG slightly)
    weighted = (hrvContrib * 0.4 + eegContrib * 0.6)

  in clamp01 weighted

-- | Compute stress index from GSR and heart rate
--
-- High GSR + elevated heart rate = high stress
-- This represents sympathetic nervous system activation.
computeStressIndex :: ResonanceTuningProfile -> BiometricInput -> Double
computeStressIndex profile input =
  let
    -- GSR contribution (direct: high GSR = stress)
    gsrContrib = biGSR input

    -- Heart rate contribution: deviation from target
    hrDelta = abs (biHeartRate input - rtpHeartRateTarget profile) / 40.0
    hrContrib = clamp01 hrDelta

    -- Weighted combination
    weighted = gsrContrib * 0.6 + hrContrib * 0.4

  in clamp01 weighted

-- | Compute respiratory phase for phi-alignment
--
-- Maps respiration rate to a phase value [0,1] representing
-- alignment with golden-ratio temporal windows.
computeRespiratoryPhase :: ResonanceTuningProfile -> BiometricInput -> Double
computeRespiratoryPhase _profile input =
  let
    -- Optimal breathing rate is ~6 breaths/min for HRV coherence
    -- This creates a ~10-second respiratory cycle
    optimalRate = 6.0
    rate = biRespirationRate input

    -- Compute phase alignment
    -- Closest to optimal = highest alignment
    deviation = abs (rate - optimalRate) / optimalRate
    alignment = 1.0 - min 1.0 (deviation * 0.5)

  in clamp01 alignment

-- | Compute phi-alignment score
--
-- Combines respiratory phase with overall coherence to determine
-- how well the organism is aligned to golden-ratio harmonics.
computePhiAlignment :: ResonanceTuningProfile -> BiometricInput -> Double
computePhiAlignment profile input =
  let
    respPhase = computeRespiratoryPhase profile input
    coherence = computeCoherenceIndex profile input

    -- Phi alignment is respiratory-coherence resonance
    -- High coherence + good breathing = phi alignment
    phiTarget = rtpPhiAlignment profile
    currentPhi = respPhase * coherence

    -- Distance from target phi alignment
    distance = abs (currentPhi - phiTarget)
    alignment = 1.0 - min 1.0 (distance / 0.5)

  in clamp01 alignment

-- =============================================================================
-- Calibration Helpers
-- =============================================================================

-- | Calibrate profile based on user's current biometric baseline
--
-- Takes a "resting" biometric sample and adjusts the profile
-- to use those values as the baseline reference.
calibrateProfile
  :: BiometricInput        -- ^ Resting/baseline sample
  -> ResonanceTuningProfile
  -> ResonanceTuningProfile
calibrateProfile baseline profile = profile
  { rtpHRVBaseline = biHRV baseline
  , rtpHeartRateTarget = biHeartRate baseline
  }

-- | Adjust tuning for baseline drift
--
-- If the user's baseline has shifted (e.g., due to fatigue),
-- this function compensates to maintain consistent tuning behavior.
adjustForBaseline
  :: BiometricInput  -- ^ Current baseline
  -> BiometricInput  -- ^ Original calibration baseline
  -> BiometricInput  -- ^ Input to adjust
  -> BiometricInput
adjustForBaseline current original input =
  let
    -- Compute drift
    hrvDrift = biHRV current - biHRV original
    hrDrift = biHeartRate current - biHeartRate original

    -- Compensate input
    adjustedHRV = biHRV input - hrvDrift * 0.5  -- Partial compensation
    adjustedHR = biHeartRate input - hrDrift * 0.5

  in input
    { biHRV = clamp01 adjustedHRV
    , biHeartRate = max 40.0 adjustedHR
    }

-- =============================================================================
-- Field Normalization
-- =============================================================================

-- | Normalize field to ensure bounded output
--
-- Prevents unbounded growth from accumulated tuning.
normalizeField :: ScalarField -> ScalarField
normalizeField (ScalarField components) =
  let
    -- Compute total weight magnitude
    totalWeight = sum [abs (scWeight sc) | sc <- components]

    -- Normalize if exceeds threshold
    normFactor = if totalWeight > 10.0
                 then 10.0 / totalWeight
                 else 1.0

    normalizedComponents = map (normalizeComponent normFactor) components

  in ScalarField normalizedComponents

normalizeComponent :: Double -> ScalarComponent -> ScalarComponent
normalizeComponent factor (SC l m rp weight) =
  SC l m rp (weight * factor)

-- =============================================================================
-- Utility Functions
-- =============================================================================

clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)

clampWeight :: Double -> Double
clampWeight x = max (-10.0) (min 10.0 x)
