{-|
Module      : Ra.Healing
Description : Scalar resonance biofeedback loop for healing
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Closed-loop scalar healing interface integrating biometric input,
Ra scalar alignment, and harmonic frequency output to restore coherence.

== Healing Feedback Loop

1. Biometrics detect instability (HRV, EEG, GSR, breath)
2. System tunes Ra coordinate + chamber parameters
3. Harmonic feedback delivered (tones, lights, tactile)
4. Coherence improves
5. System adapts based on response

== Electromagnetic Healing Frequencies

Key frequencies from healing research:

* 7.83 Hz - Schumann resonance (Earth grounding)
* 432 Hz - Natural tuning (A=432 Hz)
* 528 Hz - DNA repair, transformation (Solfeggio MI)
* Delta (0.5-3.5 Hz) - Deep sleep, regeneration
* Theta (3.5-7 Hz) - Meditation, healing trance
* Alpha (8-12 Hz) - Relaxation, calm focus

== Chakra Frequency Mapping

* Root (Muladhara): 256 Hz, C note
* Sacral (Svadhisthana): 288 Hz, D note
* Solar Plexus (Manipura): 320 Hz, E note
* Heart (Anahata): 341 Hz, F note
* Throat (Vishuddha): 384 Hz, G note
* Third Eye (Ajna): 426 Hz, A note
* Crown (Sahasrara): 480 Hz, B note
-}
module Ra.Healing
  ( -- * Biometric Input Layer
    BiometricInput(..)
  , FeedbackCondition(..)
  , biometricsToCondition
  , detectImbalance

    -- * Chakra System
  , Chakra(..)
  , chakraFrequency
  , chakraFromDrift
  , ChakraDrift(..)
  , computeChakraDrift

    -- * Scalar Alignment Controller
  , AlignmentTarget(..)
  , ScalarAlignment(..)
  , computeAlignment
  , targetCoordinate
  , inverseAnkhBalance

    -- * Harmonic Output Generator
  , HarmonicOutput(..)
  , OutputModality(..)
  , generateOutput
  , frequencyToColor
  , frequencyToTactile

    -- * Feedback Adaptation Loop
  , FeedbackState(..)
  , AdaptationResult(..)
  , adaptFeedback
  , coherenceDerivative
  , shouldReinforce
  , shouldAdjust

    -- * Healing Session
  , HealingSession(..)
  , SessionPhase(..)
  , initSession
  , updateSession
  , sessionRecommendation

    -- * Healing Frequencies
  , freqSchumann
  , freqNaturalA
  , freqDNARepair
  , freqDelta
  , freqTheta
  , freqAlpha
  ) where

import Data.List (sortBy)
import Data.Ord (comparing)

import Ra.Constants.Extended
  ( phi, freqSchumann, freqBalance, freqRepair
  , coherenceEmergence, coherenceFloorPOR
  )

-- =============================================================================
-- Biometric Input Layer
-- =============================================================================

-- | Real-time biometric input
data BiometricInput = BiometricInput
  { biHRV           :: !Double  -- ^ Heart rate variability [0,1]
  , biEEG           :: !Double  -- ^ Dominant EEG frequency (Hz)
  , biGSR           :: !Double  -- ^ Galvanic skin response [0,1]
  , biBreathRate    :: !Double  -- ^ Breaths per minute
  , biBreathDepth   :: !Double  -- ^ Respiratory depth [0,1]
  } deriving (Eq, Show)

-- | Feedback condition derived from biometrics
data FeedbackCondition = FeedbackCondition
  { fcCoherence       :: !Double      -- ^ Overall coherence [0,1]
  , fcChakraDrift     :: ![Double]    -- ^ Drift per chakra (7 values)
  , fcEmotionalTension :: !Double     -- ^ Tension level [0,1]
  , fcTargetChakra    :: !(Maybe Chakra) -- ^ Primary imbalance
  } deriving (Eq, Show)

-- | Transform biometrics to feedback condition
biometricsToCondition :: BiometricInput -> FeedbackCondition
biometricsToCondition bi =
  let -- Compute overall coherence from HRV and breath
      coherence = (biHRV bi + biBreathDepth bi) / 2

      -- Chakra drift based on biometric patterns
      drift = computeChakraDrifts bi

      -- Emotional tension from GSR
      tension = biGSR bi

      -- Find chakra with maximum drift
      targetChakra = chakraFromDrift drift
  in FeedbackCondition
      { fcCoherence = clamp01 coherence
      , fcChakraDrift = drift
      , fcEmotionalTension = clamp01 tension
      , fcTargetChakra = targetChakra
      }

-- | Compute chakra drift values from biometrics
computeChakraDrifts :: BiometricInput -> [Double]
computeChakraDrifts bi =
  let hrv = biHRV bi
      eeg = biEEG bi / 40.0  -- Normalize to [0,1]
      gsr = biGSR bi
      breath = biBreathDepth bi

      -- Map biometrics to chakras
      root = 1.0 - breath      -- Grounding
      sacral = gsr * 0.7       -- Emotional reactivity
      solar = 1.0 - hrv        -- Power/control
      heart = 1.0 - (hrv + breath) / 2  -- Love/compassion
      throat = abs (eeg - 0.3) -- Expression
      thirdEye = abs (eeg - 0.5)  -- Intuition
      crown = 1.0 - min 1.0 (eeg * 2)  -- Spiritual connection
  in map (\x -> clamp01 (x - 0.5) * 2 - 1) [root, sacral, solar, heart, throat, thirdEye, crown]

-- | Detect primary imbalance from condition
detectImbalance :: FeedbackCondition -> Maybe (Chakra, Double)
detectImbalance fc =
  case fcTargetChakra fc of
    Nothing -> Nothing
    Just chakra ->
      let idx = fromEnum chakra
          drift = fcChakraDrift fc !! idx
      in Just (chakra, drift)

-- =============================================================================
-- Chakra System
-- =============================================================================

-- | Seven major chakras
data Chakra
  = Root       -- ^ Muladhara - grounding, survival
  | Sacral     -- ^ Svadhisthana - creativity, emotion
  | SolarPlexus -- ^ Manipura - power, will
  | Heart      -- ^ Anahata - love, compassion
  | Throat     -- ^ Vishuddha - expression, truth
  | ThirdEye   -- ^ Ajna - intuition, insight
  | Crown      -- ^ Sahasrara - spiritual connection
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Chakra resonant frequency (Hz)
chakraFrequency :: Chakra -> Double
chakraFrequency Root = 256.0       -- C
chakraFrequency Sacral = 288.0     -- D
chakraFrequency SolarPlexus = 320.0 -- E
chakraFrequency Heart = 341.33     -- F
chakraFrequency Throat = 384.0     -- G
chakraFrequency ThirdEye = 426.67  -- A
chakraFrequency Crown = 480.0      -- B

-- | Find chakra with maximum absolute drift
chakraFromDrift :: [Double] -> Maybe Chakra
chakraFromDrift drifts
  | length drifts /= 7 = Nothing
  | all (< 0.1) (map abs drifts) = Nothing  -- No significant drift
  | otherwise =
      let indexed = zip [Root ..] drifts
          sorted = sortBy (comparing (negate . abs . snd)) indexed
      in Just (fst (head sorted))

-- | Chakra drift measurement
data ChakraDrift = ChakraDrift
  { cdChakra    :: !Chakra
  , cdMagnitude :: !Double  -- ^ Drift amount [-1, 1]
  , cdDirection :: !Bool    -- ^ True = excess, False = deficient
  } deriving (Eq, Show)

-- | Compute drift for a single chakra
computeChakraDrift :: Chakra -> [Double] -> ChakraDrift
computeChakraDrift chakra drifts =
  let idx = fromEnum chakra
      drift = if idx < length drifts then drifts !! idx else 0.0
  in ChakraDrift
      { cdChakra = chakra
      , cdMagnitude = abs drift
      , cdDirection = drift > 0
      }

-- =============================================================================
-- Scalar Alignment Controller
-- =============================================================================

-- | Alignment target for healing
data AlignmentTarget = AlignmentTarget
  { atPotentialDepth   :: !Double  -- ^ Target scalar potential [0,1]
  , atTemporalWindow   :: !Int     -- ^ φ^n window index
  , atHarmonicIndex    :: !(Int, Int) -- ^ (l, m) harmonic
  , atFrequency        :: !Double  -- ^ Target frequency (Hz)
  } deriving (Eq, Show)

-- | Scalar alignment computation result
data ScalarAlignment = ScalarAlignment
  { saTarget      :: !AlignmentTarget
  , saCoherence   :: !Double  -- ^ Required coherence
  , saAnkhBalance :: !Double  -- ^ Δ(ankh) for balance
  , saPhaseOffset :: !Double  -- ^ Phase adjustment [0, 2π)
  } deriving (Eq, Show)

-- | Compute optimal alignment for healing
computeAlignment :: FeedbackCondition -> ScalarAlignment
computeAlignment fc =
  let -- Determine target based on imbalance
      (targetL, targetM, freq) = case fcTargetChakra fc of
        Nothing -> (0, 0, freqBalance)  -- Default: 432 Hz balance
        Just chakra ->
          let l = fromEnum chakra + 1
              m = if fcEmotionalTension fc > 0.5 then 0 else 1
              f = chakraFrequency chakra
          in (l, m, f)

      -- Potential depth: deeper for lower coherence
      depth = 1.0 - fcCoherence fc

      -- Temporal window: larger φ^n for more time to entrain
      window = ceiling (3.0 * (1.0 - fcCoherence fc))

      -- Ankh balance: counter-cohere to fragment
      ankhBalance = inverseAnkhBalance (fcChakraDrift fc)

      -- Phase offset based on emotional tension
      phase = fcEmotionalTension fc * 2 * pi
  in ScalarAlignment
      { saTarget = AlignmentTarget
          { atPotentialDepth = depth
          , atTemporalWindow = window
          , atHarmonicIndex = (targetL, targetM)
          , atFrequency = freq
          }
      , saCoherence = coherenceFloorPOR + 0.1  -- Target just above POR floor
      , saAnkhBalance = ankhBalance
      , saPhaseOffset = phase
      }

-- | Target coordinate from alignment
targetCoordinate :: ScalarAlignment -> (Int, Int, Double)
targetCoordinate sa =
  let (l, m) = atHarmonicIndex (saTarget sa)
  in (l, m, atPotentialDepth (saTarget sa))

-- | Compute inverse Δ(ankh) to balance field
inverseAnkhBalance :: [Double] -> Double
inverseAnkhBalance drifts =
  let totalDrift = sum (map abs drifts) / fromIntegral (length drifts)
  in negate totalDrift * phi  -- Golden-scaled counter-balance

-- =============================================================================
-- Harmonic Output Generator
-- =============================================================================

-- | Output modality for feedback
data OutputModality
  = AudioTone    -- ^ Sound frequency
  | BinauralBeat -- ^ Binaural beat (requires L/R difference)
  | LEDColor     -- ^ Visual color pattern
  | Tactile      -- ^ Haptic pulse
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Harmonic output specification
data HarmonicOutput = HarmonicOutput
  { hoModality   :: !OutputModality
  , hoFrequency  :: !Double  -- ^ Primary frequency (Hz)
  , hoAmplitude  :: !Double  -- ^ Intensity [0,1]
  , hoDuration   :: !Double  -- ^ Duration (seconds)
  , hoWaveform   :: !String  -- ^ Waveform type (sine, triangle, etc.)
  } deriving (Eq, Show)

-- | Generate output from alignment
generateOutput :: ScalarAlignment -> OutputModality -> HarmonicOutput
generateOutput sa modality =
  let freq = atFrequency (saTarget sa)
      amp = 0.5 + saCoherence sa * 0.5  -- Amplitude from coherence
      dur = phi ** fromIntegral (atTemporalWindow (saTarget sa))  -- φ^n duration
      waveform = case modality of
        AudioTone -> "sine"
        BinauralBeat -> "sine"
        LEDColor -> "pulse"
        Tactile -> "square"
  in HarmonicOutput
      { hoModality = modality
      , hoFrequency = freq
      , hoAmplitude = clamp01 amp
      , hoDuration = dur
      , hoWaveform = waveform
      }

-- | Convert frequency to color (visible light mapping)
--
-- Maps audio frequencies to visible spectrum via octave folding.
-- 432 Hz -> Green, 528 Hz -> Blue-Green, etc.
frequencyToColor :: Double -> (Int, Int, Int)
frequencyToColor freq =
  let -- Fold to visible spectrum (380-780 nm -> 384-768 THz)
      -- Use 2^40 octaves from audio to light
      visibleHz = freq * (2 ** 40)
      -- Map to wavelength (nm)
      wavelength = 3e17 / visibleHz  -- c / f

      -- Simplified RGB from wavelength
      (r, g, b) = wavelengthToRGB wavelength
  in (r, g, b)

-- | Wavelength (nm) to RGB
wavelengthToRGB :: Double -> (Int, Int, Int)
wavelengthToRGB wl
  | wl < 380 = (128, 0, 128)   -- Violet
  | wl < 440 = (75, 0, 130)    -- Indigo
  | wl < 490 = (0, 0, 255)     -- Blue
  | wl < 510 = (0, 255, 255)   -- Cyan
  | wl < 540 = (0, 255, 0)     -- Green
  | wl < 560 = (173, 255, 47)  -- Yellow-green
  | wl < 590 = (255, 255, 0)   -- Yellow
  | wl < 620 = (255, 165, 0)   -- Orange
  | wl < 700 = (255, 0, 0)     -- Red
  | otherwise = (139, 0, 0)    -- Deep red

-- | Convert frequency to tactile pulse parameters
frequencyToTactile :: Double -> (Double, Double)
frequencyToTactile freq =
  let -- Tactile range: 1-500 Hz perceptible
      tactileFreq = if freq > 500 then freq / 10 else freq
      -- Pulse width inversely proportional to frequency
      pulseWidth = 1.0 / (tactileFreq + 1)
  in (tactileFreq, pulseWidth)

-- =============================================================================
-- Feedback Adaptation Loop
-- =============================================================================

-- | Current feedback state
data FeedbackState = FeedbackState
  { fsCoherenceHistory :: ![Double]  -- ^ Recent coherence values
  , fsCurrentOutput    :: !HarmonicOutput
  , fsAlignment        :: !ScalarAlignment
  , fsCycleCount       :: !Int       -- ^ Feedback cycles completed
  } deriving (Eq, Show)

-- | Result of adaptation evaluation
data AdaptationResult
  = Reinforce     -- ^ Continue current approach
  | Adjust        -- ^ Modify alignment parameters
  | Escalate      -- ^ Increase intensity
  | DeEscalate    -- ^ Decrease intensity
  | Reset         -- ^ Return to baseline
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Adapt feedback based on biometric response
adaptFeedback :: FeedbackState -> Double -> AdaptationResult
adaptFeedback fs newCoherence =
  let history = fsCoherenceHistory fs
      derivative = coherenceDerivative (history ++ [newCoherence])
  in if derivative > 0.05
     then Reinforce      -- Rising coherence
     else if derivative > 0
     then Reinforce      -- Slowly rising
     else if derivative > -0.05
     then Adjust         -- Stagnant
     else if derivative > -0.1
     then DeEscalate     -- Falling slowly
     else Reset          -- Falling fast

-- | Compute coherence derivative (rate of change)
coherenceDerivative :: [Double] -> Double
coherenceDerivative [] = 0.0
coherenceDerivative [_] = 0.0
coherenceDerivative history =
  let recent = take 5 (reverse history)  -- Last 5 values
      weights = [0.4, 0.3, 0.15, 0.1, 0.05]  -- Decay weights
      weightedDiffs = zipWith3 (\w a b -> w * (a - b))
                               weights
                               recent
                               (tail recent ++ [0])
  in sum weightedDiffs

-- | Should reinforce current approach?
shouldReinforce :: Double -> Bool
shouldReinforce derivative = derivative > 0

-- | Should adjust alignment?
shouldAdjust :: Double -> Bool
shouldAdjust derivative = derivative <= 0 && derivative > -0.1

-- =============================================================================
-- Healing Session
-- =============================================================================

-- | Session phase
data SessionPhase
  = Calibration  -- ^ Initial baseline measurement
  | Assessment   -- ^ Evaluating imbalances
  | Treatment    -- ^ Active feedback delivery
  | Integration  -- ^ Consolidating gains
  | Completion   -- ^ Session wrap-up
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Complete healing session state
data HealingSession = HealingSession
  { hsPhase          :: !SessionPhase
  , hsCondition      :: !FeedbackCondition
  , hsAlignment      :: !ScalarAlignment
  , hsOutput         :: !(Maybe HarmonicOutput)
  , hsHistory        :: ![FeedbackCondition]
  , hsDuration       :: !Double  -- ^ Elapsed seconds
  , hsRecommendation :: !String
  } deriving (Eq, Show)

-- | Initialize healing session
initSession :: BiometricInput -> HealingSession
initSession bi =
  let condition = biometricsToCondition bi
      alignment = computeAlignment condition
  in HealingSession
      { hsPhase = Calibration
      , hsCondition = condition
      , hsAlignment = alignment
      , hsOutput = Nothing
      , hsHistory = [condition]
      , hsDuration = 0.0
      , hsRecommendation = "Calibrating baseline..."
      }

-- | Update session with new biometric input
updateSession :: BiometricInput -> Double -> HealingSession -> HealingSession
updateSession bi dt session =
  let newCondition = biometricsToCondition bi
      oldCondition = hsCondition session
      coherenceDelta = fcCoherence newCondition - fcCoherence oldCondition

      -- Determine phase transition
      newPhase = determinePhase session newCondition

      -- Update alignment if needed
      newAlignment = if newPhase /= hsPhase session
                     then computeAlignment newCondition
                     else hsAlignment session

      -- Generate output for treatment phase
      newOutput = if newPhase == Treatment
                  then Just (generateOutput newAlignment AudioTone)
                  else Nothing

      -- Generate recommendation
      rec = sessionRecommendation newPhase newCondition coherenceDelta
  in session
      { hsPhase = newPhase
      , hsCondition = newCondition
      , hsAlignment = newAlignment
      , hsOutput = newOutput
      , hsHistory = newCondition : take 100 (hsHistory session)
      , hsDuration = hsDuration session + dt
      , hsRecommendation = rec
      }

-- | Determine appropriate session phase
determinePhase :: HealingSession -> FeedbackCondition -> SessionPhase
determinePhase session condition =
  let duration = hsDuration session
      coherence = fcCoherence condition
      currentPhase = hsPhase session
  in case currentPhase of
      Calibration ->
        if duration > 30 then Assessment else Calibration
      Assessment ->
        if duration > 60 then Treatment else Assessment
      Treatment ->
        if coherence > coherenceEmergence
        then Integration
        else if duration > 600  -- 10 minutes max treatment
        then Integration
        else Treatment
      Integration ->
        if duration > 660 then Completion else Integration
      Completion -> Completion

-- | Generate session recommendation
sessionRecommendation :: SessionPhase -> FeedbackCondition -> Double -> String
sessionRecommendation phase condition delta =
  case phase of
    Calibration -> "Breathe naturally while we establish baseline..."
    Assessment ->
      case fcTargetChakra condition of
        Nothing -> "Your energy centers appear balanced."
        Just chakra -> "Primary focus: " ++ show chakra ++ " center"
    Treatment ->
      if delta > 0
      then "Coherence rising... maintain this rhythm"
      else "Adjusting frequencies... breathe slowly and deeply"
    Integration -> "Excellent progress. Allowing integration..."
    Completion -> "Session complete. Rest and hydrate."

-- =============================================================================
-- Healing Frequencies (Constants)
-- =============================================================================

-- | Schumann resonance - Earth's fundamental frequency
-- Re-exported from Ra.Constants.Extended
-- freqSchumann = 7.83 Hz

-- | Natural tuning (A=432 Hz)
freqNaturalA :: Double
freqNaturalA = freqBalance  -- 432.0 Hz

-- | DNA repair frequency (Solfeggio 528 Hz)
freqDNARepair :: Double
freqDNARepair = freqRepair  -- 528.0 Hz

-- | Delta wave range (deep sleep, regeneration)
freqDelta :: (Double, Double)
freqDelta = (0.5, 3.5)

-- | Theta wave range (meditation, healing trance)
freqTheta :: (Double, Double)
freqTheta = (3.5, 7.0)

-- | Alpha wave range (relaxation, calm focus)
freqAlpha :: (Double, Double)
freqAlpha = (8.0, 12.0)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
