{-|
Module      : Ra.Sympathetic
Description : Sympathetic harmonic fragment access modeling
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Models sympathetic resonance between scalar fields and user-specific
frequencies to modulate fragment access and emergence strength.

== Keely's Sympathetic Vibratory Physics

From John W. Keely's 40 Laws of Sympathetic Vibration:

* Law 14 (Sympathetic Concordance): "When two or more bodies vibrate
  at the same frequency, they will mutually attract each other."

* Law 17 (Sympathetic Transmission): "Vibrations of a specific frequency
  can be transmitted through a medium only when the medium's mass
  structure is sympathetically attuned."

== Fragment Harmonic Signature

Each Ra fragment carries a harmonic signature consisting of:

* Tonic: Primary resonant frequency (Hz)
* Dominant: Secondary harmonic (typically 3:2 or 5:4 ratio)
* Enharmonic: Tertiary color tone for emotional/semantic context

== Access Modulation Table

@
match_score >= 0.90  -> FULL_ACCESS     (direct sympathetic lock)
0.60 <= match < 0.90 -> PARTIAL_ACCESS  (entrainment possible)
0.30 <= match < 0.60 -> BLOCKED         (dissonance rejection)
match_score < 0.30   -> SHADOW_ACCESS   (inverted/mirror path)
@

== Guardian Harmonics

Fragments can specify guardian harmonics that must be present in the
user's resonance profile for chaining access. This enables therapeutic
sequencing where Fragment A unlocks access to Fragment B.
-}
module Ra.Sympathetic
  ( -- * Harmonic Signature Types
    FrequencyHz(..)
  , HarmonicTriad(..)
  , mkHarmonicTriad
  , triadToList
  , triadCentroid
  , triadSpread

    -- * User Resonance Profile
  , ResonanceProfile(..)
  , mkResonanceProfile
  , profileDominantFreq
  , profileHarmonicRichness

    -- * Matching Algorithm
  , MatchScore(..)
  , harmonicSimilarity
  , frequencyDeviation
  , intervalMatch
  , overallMatchScore

    -- * Access Modulation
  , SympatheticAccess(..)
  , modulateAccess
  , accessToAlpha

    -- * Guardian Harmonics
  , GuardianRequirement(..)
  , GuardianChain(..)
  , checkGuardian
  , evaluateChain
  , chainAccessible

    -- * Emergence Strength
  , EmergenceStrength(..)
  , computeEmergenceStrength
  , strengthToIntensity

    -- * Sympathetic Resonance Computation
  , ResonanceResult(..)
  , evaluateResonance
  , resonanceToAccess

    -- * Keely Integration
  , keelyOctave
  , keelyTripleStructure
  , sympatheticConcordance
  , transmissionEfficiency
  ) where

import Data.List (minimumBy)
import Data.Ord (comparing)

import Ra.Constants.Extended
  ( phi, solfeggioAll, keelyOctaveBandWidth )

-- =============================================================================
-- Frequency and Harmonic Types
-- =============================================================================

-- | Frequency in Hertz (positive real)
newtype FrequencyHz = FrequencyHz { unFrequencyHz :: Double }
  deriving (Eq, Ord, Show)

-- | Harmonic triad: tonic, dominant, enharmonic frequencies
--
-- The three frequencies form a "harmonic fingerprint" for a fragment
-- or user profile. The relationship between them encodes semantic
-- and emotional information.
data HarmonicTriad = HarmonicTriad
  { triadTonic      :: !FrequencyHz  -- ^ Primary resonant frequency
  , triadDominant   :: !FrequencyHz  -- ^ Secondary harmonic (3:2 typical)
  , triadEnharmonic :: !FrequencyHz  -- ^ Tertiary color tone
  } deriving (Eq, Show)

-- | Smart constructor ensuring frequencies are positive and ordered
mkHarmonicTriad :: Double -> Double -> Double -> Maybe HarmonicTriad
mkHarmonicTriad t d e
  | t > 0 && d > 0 && e > 0 = Just $ HarmonicTriad
      (FrequencyHz t) (FrequencyHz d) (FrequencyHz e)
  | otherwise = Nothing

-- | Convert triad to frequency list
triadToList :: HarmonicTriad -> [FrequencyHz]
triadToList (HarmonicTriad t d e) = [t, d, e]

-- | Geometric centroid frequency of the triad
triadCentroid :: HarmonicTriad -> FrequencyHz
triadCentroid (HarmonicTriad t d e) =
  let ft = unFrequencyHz t
      fd = unFrequencyHz d
      fe = unFrequencyHz e
  in FrequencyHz $ (ft * fd * fe) ** (1/3)

-- | Spread measure: ratio of highest to lowest frequency
triadSpread :: HarmonicTriad -> Double
triadSpread triad =
  let freqs = map unFrequencyHz (triadToList triad)
      maxF = maximum freqs
      minF = minimum freqs
  in if minF > 0 then maxF / minF else 1.0

-- =============================================================================
-- User Resonance Profile
-- =============================================================================

-- | User's resonance profile derived from biometric data
--
-- Represents the user's current sympathetic frequency state,
-- aggregated from HRV, EEG, and other biometric inputs.
data ResonanceProfile = ResonanceProfile
  { profileSignature    :: !HarmonicTriad  -- ^ User's harmonic triad
  , profileCoherence    :: !Double         -- ^ Overall coherence [0,1]
  , profileVariance     :: !Double         -- ^ Frequency stability [0,1]
  , profileHistory      :: ![FrequencyHz]  -- ^ Recent dominant frequencies
  } deriving (Eq, Show)

-- | Create profile from biometric-derived frequencies
mkResonanceProfile :: HarmonicTriad -> Double -> Double -> ResonanceProfile
mkResonanceProfile sig coh var = ResonanceProfile
  { profileSignature = sig
  , profileCoherence = clamp01 coh
  , profileVariance = clamp01 var
  , profileHistory = []
  }

-- | Extract dominant frequency from profile
profileDominantFreq :: ResonanceProfile -> FrequencyHz
profileDominantFreq = triadTonic . profileSignature

-- | Compute harmonic richness (spread * coherence)
profileHarmonicRichness :: ResonanceProfile -> Double
profileHarmonicRichness prof =
  let spread = triadSpread (profileSignature prof)
      coh = profileCoherence prof
  in coh * log spread / log 2  -- Normalized to octaves

-- =============================================================================
-- Matching Algorithm
-- =============================================================================

-- | Match score result with component breakdown
data MatchScore = MatchScore
  { scoreTotal        :: !Double  -- ^ Overall match [0,1]
  , scoreFrequency    :: !Double  -- ^ Frequency alignment component
  , scoreInterval     :: !Double  -- ^ Interval structure component
  , scoreCoherence    :: !Double  -- ^ Coherence weighting factor
  } deriving (Eq, Show)

-- | Compute harmonic similarity between fragment and user triads
--
-- Uses weighted combination of:
-- * Frequency deviation (40%)
-- * Interval matching (40%)
-- * Coherence factor (20%)
harmonicSimilarity :: HarmonicTriad -> ResonanceProfile -> MatchScore
harmonicSimilarity fragTriad profile =
  let userTriad = profileSignature profile
      freqScore = frequencyDeviation fragTriad userTriad
      intScore = intervalMatch fragTriad userTriad
      cohScore = profileCoherence profile

      -- Weighted combination (0.4, 0.4, 0.2)
      total = 0.4 * freqScore + 0.4 * intScore + 0.2 * cohScore
  in MatchScore
      { scoreTotal = clamp01 total
      , scoreFrequency = freqScore
      , scoreInterval = intScore
      , scoreCoherence = cohScore
      }

-- | Frequency deviation score (1 = perfect match, 0 = octave+ apart)
frequencyDeviation :: HarmonicTriad -> HarmonicTriad -> Double
frequencyDeviation frag user =
  let fragFreqs = map unFrequencyHz (triadToList frag)
      userFreqs = map unFrequencyHz (triadToList user)

      -- For each fragment frequency, find closest user frequency
      deviations = [minDeviation f userFreqs | f <- fragFreqs]

      -- Convert cents deviation to score
      avgDeviation = sum deviations / fromIntegral (length deviations)
  in deviationToScore avgDeviation

-- | Find minimum deviation from a frequency to a list (in cents)
minDeviation :: Double -> [Double] -> Double
minDeviation f targets =
  let deviations = [abs (centsDiff f t) | t <- targets]
  in minimum deviations

-- | Difference in cents between two frequencies
centsDiff :: Double -> Double -> Double
centsDiff f1 f2 = 1200 * logBase 2 (f1 / f2)

-- | Convert cents deviation to match score
-- 0 cents = 1.0, 50 cents = 0.5, 100+ cents = near 0
deviationToScore :: Double -> Double
deviationToScore cents =
  let normalized = abs cents / 100.0
  in max 0 (1 - normalized)

-- | Interval structure matching
--
-- Compares the intervallic relationships within each triad,
-- regardless of absolute pitch.
intervalMatch :: HarmonicTriad -> HarmonicTriad -> Double
intervalMatch frag user =
  let fragIntervals = extractIntervals frag
      userIntervals = extractIntervals user

      -- Compare interval ratios
      diffs = zipWith (\a b -> abs (a - b) / max a b)
                      fragIntervals userIntervals
  in 1 - (sum diffs / fromIntegral (length diffs))

-- | Extract interval ratios from triad
extractIntervals :: HarmonicTriad -> [Double]
extractIntervals (HarmonicTriad t d e) =
  let ft = unFrequencyHz t
      fd = unFrequencyHz d
      fe = unFrequencyHz e
  in [fd / ft, fe / ft, fe / fd]  -- Dominant/Tonic, Enharmonic/Tonic, Enharmonic/Dominant

-- | Compute overall match score with custom weights
overallMatchScore
  :: Double  -- ^ Frequency weight
  -> Double  -- ^ Interval weight
  -> Double  -- ^ Coherence weight
  -> HarmonicTriad
  -> ResonanceProfile
  -> Double
overallMatchScore wf wi wc fragTriad profile =
  let ms = harmonicSimilarity fragTriad profile
      total = wf * scoreFrequency ms + wi * scoreInterval ms + wc * scoreCoherence ms
      normalizer = wf + wi + wc
  in if normalizer > 0 then total / normalizer else 0

-- =============================================================================
-- Access Modulation
-- =============================================================================

-- | Sympathetic access levels based on match score
data SympatheticAccess
  = FullSympathetic      -- ^ >= 0.90: Direct sympathetic lock
  | PartialSympathetic !Double  -- ^ 0.60-0.89: Entrainment possible
  | BlockedSympathetic   -- ^ 0.30-0.59: Dissonance rejection
  | ShadowSympathetic    -- ^ < 0.30: Inverted/mirror path
  deriving (Eq, Show)

-- | Modulate access level based on match score
modulateAccess :: MatchScore -> SympatheticAccess
modulateAccess ms =
  let score = scoreTotal ms
  in if score >= 0.90
     then FullSympathetic
     else if score >= 0.60
     then PartialSympathetic ((score - 0.60) / 0.30)  -- Normalized to [0,1]
     else if score >= 0.30
     then BlockedSympathetic
     else ShadowSympathetic

-- | Convert access to alpha intensity [0,1]
accessToAlpha :: SympatheticAccess -> Double
accessToAlpha FullSympathetic = 1.0
accessToAlpha (PartialSympathetic a) = 0.5 + 0.5 * a  -- [0.5, 1.0]
accessToAlpha BlockedSympathetic = 0.0
accessToAlpha ShadowSympathetic = -0.3  -- Negative for shadow work

-- =============================================================================
-- Guardian Harmonics
-- =============================================================================

-- | Guardian requirement for fragment chaining
data GuardianRequirement = GuardianRequirement
  { guardianFrequency  :: !FrequencyHz   -- ^ Required frequency
  , guardianTolerance  :: !Double        -- ^ Tolerance in cents
  , guardianMinHistory :: !Int           -- ^ Minimum history length
  } deriving (Eq, Show)

-- | Chain of guardian requirements
data GuardianChain = GuardianChain
  { chainRequirements :: ![GuardianRequirement]
  , chainStrict       :: !Bool  -- ^ All must pass vs any can pass
  } deriving (Eq, Show)

-- | Check if user profile satisfies a guardian requirement
checkGuardian :: GuardianRequirement -> ResonanceProfile -> Bool
checkGuardian req profile =
  let targetFreq = unFrequencyHz (guardianFrequency req)
      tolerance = guardianTolerance req
      minHist = guardianMinHistory req
      history = profileHistory profile

      -- Check if any frequency in profile is close enough
      userFreqs = map unFrequencyHz (triadToList (profileSignature profile))
      freqMatch = any (\f -> abs (centsDiff f targetFreq) <= tolerance) userFreqs

      -- Check history requirement
      historyMatch = length history >= minHist
  in freqMatch && historyMatch

-- | Evaluate entire guardian chain
evaluateChain :: GuardianChain -> ResonanceProfile -> Double
evaluateChain chain profile =
  let reqs = chainRequirements chain
      results = map (`checkGuardian` profile) reqs
      passing = length (filter id results)
      total = length reqs
  in if total == 0
     then 1.0
     else fromIntegral passing / fromIntegral total

-- | Check if chain access is granted (depends on strict mode)
chainAccessible :: GuardianChain -> ResonanceProfile -> Bool
chainAccessible chain profile =
  let score = evaluateChain chain profile
  in if chainStrict chain
     then score >= 1.0  -- All must pass
     else score >= 0.5  -- Majority must pass

-- =============================================================================
-- Emergence Strength
-- =============================================================================

-- | Emergence strength combining match score and coherence
data EmergenceStrength = EmergenceStrength
  { strengthValue      :: !Double  -- ^ Combined strength [0,1]
  , strengthConfidence :: !Double  -- ^ Confidence in measurement
  , strengthTrend      :: !Double  -- ^ Rate of change
  } deriving (Eq, Show)

-- | Compute emergence strength from sympathetic access
computeEmergenceStrength :: SympatheticAccess -> Double -> EmergenceStrength
computeEmergenceStrength access coherence =
  let alpha = accessToAlpha access
      value = if alpha >= 0
              then alpha * coherence
              else abs alpha * (1 - coherence)  -- Shadow inverts
      confidence = abs alpha
      trend = 0.0  -- Requires historical data
  in EmergenceStrength
      { strengthValue = clamp01 value
      , strengthConfidence = confidence
      , strengthTrend = trend
      }

-- | Convert strength to intensity for output
strengthToIntensity :: EmergenceStrength -> Double
strengthToIntensity es =
  strengthValue es * strengthConfidence es

-- =============================================================================
-- Sympathetic Resonance Computation
-- =============================================================================

-- | Full resonance evaluation result
data ResonanceResult = ResonanceResult
  { resMatchScore       :: !MatchScore
  , resAccess           :: !SympatheticAccess
  , resEmergence        :: !EmergenceStrength
  , resGuardianPassed   :: !Bool
  , resRecommendedFreq  :: !FrequencyHz
  } deriving (Eq, Show)

-- | Complete resonance evaluation
evaluateResonance
  :: HarmonicTriad       -- ^ Fragment signature
  -> Maybe GuardianChain -- ^ Optional guardian requirements
  -> ResonanceProfile    -- ^ User profile
  -> ResonanceResult
evaluateResonance fragSig mChain profile =
  let -- Compute match score
      matchScore = harmonicSimilarity fragSig profile

      -- Check guardian chain
      guardianOK = case mChain of
        Nothing -> True
        Just chain -> chainAccessible chain profile

      -- Modulate access (blocked if guardian fails)
      rawAccess = modulateAccess matchScore
      access = if guardianOK then rawAccess else BlockedSympathetic

      -- Compute emergence strength
      emergence = computeEmergenceStrength access (profileCoherence profile)

      -- Recommend frequency based on fragment + user
      recommended = recommendFrequency fragSig profile
  in ResonanceResult
      { resMatchScore = matchScore
      , resAccess = access
      , resEmergence = emergence
      , resGuardianPassed = guardianOK
      , resRecommendedFreq = recommended
      }

-- | Recommend bridging frequency between fragment and user
recommendFrequency :: HarmonicTriad -> ResonanceProfile -> FrequencyHz
recommendFrequency frag profile =
  let fragCentroid = unFrequencyHz (triadCentroid frag)
      userCentroid = unFrequencyHz (triadCentroid (profileSignature profile))

      -- Geometric mean as bridge frequency
      bridgeFreq = sqrt (fragCentroid * userCentroid)

      -- Snap to nearest Solfeggio if close
      nearest = nearestSolfeggioFreq bridgeFreq
  in if abs (centsDiff bridgeFreq nearest) < 50
     then FrequencyHz nearest
     else FrequencyHz bridgeFreq

-- | Find nearest Solfeggio frequency
nearestSolfeggioFreq :: Double -> Double
nearestSolfeggioFreq freq =
  minimumBy (comparing (\s -> abs (centsDiff freq s))) solfeggioAll

-- | Convert resonance result to access decision
resonanceToAccess :: ResonanceResult -> SympatheticAccess
resonanceToAccess = resAccess

-- =============================================================================
-- Keely Integration
-- =============================================================================

-- | Compute Keely octave band for a frequency
--
-- Maps frequency to one of 21 octave bands in Keely's system.
keelyOctave :: FrequencyHz -> Int
keelyOctave (FrequencyHz f) =
  let baseOctave = floor (logBase 2 (f / 1.0)) :: Int
  in baseOctave `mod` keelyOctaveBandWidth

-- | Extract Keely triple structure from triad
--
-- Keely's triple structure: Mass, Vibration, Oscillation
-- Maps to: Tonic (mass/inertia), Dominant (vibration), Enharmonic (oscillation)
keelyTripleStructure :: HarmonicTriad -> (Double, Double, Double)
keelyTripleStructure (HarmonicTriad t d e) =
  ( unFrequencyHz t  -- Mass analog (inertial frequency)
  , unFrequencyHz d  -- Vibration analog (harmonic frequency)
  , unFrequencyHz e  -- Oscillation analog (enharmonic modulation)
  )

-- | Compute sympathetic concordance per Keely Law 14
--
-- Returns 1.0 when frequencies are in perfect sympathetic lock,
-- decreasing as frequencies diverge.
sympatheticConcordance :: HarmonicTriad -> HarmonicTriad -> Double
sympatheticConcordance triad1 triad2 =
  let (m1, v1, o1) = keelyTripleStructure triad1
      (m2, v2, o2) = keelyTripleStructure triad2

      -- Concordance for each triple component
      massConcord = concordance m1 m2
      vibConcord = concordance v1 v2
      oscConcord = concordance o1 o2

      -- Weight: mass 0.5, vibration 0.3, oscillation 0.2
  in 0.5 * massConcord + 0.3 * vibConcord + 0.2 * oscConcord
  where
    -- Concordance decreases with interval distance
    concordance f1 f2 =
      let ratio = if f1 > f2 then f1 / f2 else f2 / f1
          -- Perfect unison = 1.0, octave = 0.8, fifth = 0.9
      in if ratio < 1.01 then 1.0
         else if ratio < 1.51 then 0.9  -- Perfect fifth
         else if ratio < 2.01 then 0.8  -- Octave
         else 0.5 / ratio               -- Decay with distance

-- | Transmission efficiency per Keely Law 17
--
-- Efficiency depends on medium attunement (represented by coherence)
-- and frequency match.
transmissionEfficiency :: Double -> MatchScore -> Double
transmissionEfficiency mediumCoherence matchScore =
  let freqMatch = scoreFrequency matchScore
      intMatch = scoreInterval matchScore

      -- Base efficiency from frequency match
      baseEff = (freqMatch + intMatch) / 2

      -- Modulated by medium coherence
  in baseEff * mediumCoherence * phi  -- Golden scaling

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
