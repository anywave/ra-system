{-|
Module      : Ra.Biofield.TuningMap
Description : Live biofield tuning map with frequency resonance tables
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Constructs a live-tuning map integrating electromagnetic healing frequencies,
Rife/Lakhovsky specifications, and blood electrification parameters to create
per-user biometric resonance wave tables.

== Tuning Map Architecture

=== Frequency Bands

* Rife frequencies: Pathogen-specific resonance
* Lakhovsky MWO: Multi-wave oscillation bands
* Solfeggio: Harmonic healing frequencies
* Schumann: Earth resonance coupling

=== Per-User Adaptation

The tuning map adapts to individual biometric patterns, creating personalized
resonance profiles that optimize coherence and healing response.
-}
module Ra.Biofield.TuningMap
  ( -- * Core Types
    TuningMap(..)
  , FrequencyBand(..)
  , ResonanceEntry(..)
  , UserProfile(..)

    -- * Map Creation
  , createTuningMap
  , defaultTuningMap
  , personalizeMap

    -- * Frequency Lookup
  , lookupFrequency
  , findResonance
  , optimalBand

    -- * Band Management
  , addBand
  , removeBand
  , adjustBand

    -- * User Profiling
  , createProfile
  , updateProfile
  , profileResonance

    -- * Integration
  , mapToChamber
  , chamberFeedback
  , biometricSync
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete biofield tuning map
data TuningMap = TuningMap
  { tmBands      :: ![FrequencyBand]        -- ^ Available frequency bands
  , tmProfile    :: !(Maybe UserProfile)    -- ^ User-specific profile
  , tmResonances :: !(Map String ResonanceEntry)  -- ^ Named resonance entries
  , tmBaseline   :: !Double                 -- ^ Baseline frequency (Hz)
  , tmRange      :: !(Double, Double)       -- ^ Frequency range (min, max)
  } deriving (Eq, Show)

-- | Frequency band definition
data FrequencyBand = FrequencyBand
  { fbName       :: !String         -- ^ Band name
  , fbCenter     :: !Double         -- ^ Center frequency (Hz)
  , fbBandwidth  :: !Double         -- ^ Bandwidth (Hz)
  , fbType       :: !BandType       -- ^ Band classification
  , fbIntensity  :: !Double         -- ^ Default intensity [0, 1]
  , fbHarmonics  :: ![Int]          -- ^ Harmonic multipliers
  } deriving (Eq, Show)

-- | Band type classification
data BandType
  = BandRife          -- ^ Rife frequency
  | BandLakhovsky     -- ^ Lakhovsky MWO
  | BandSolfeggio     -- ^ Solfeggio healing
  | BandSchumann      -- ^ Earth resonance
  | BandBiometric     -- ^ User biometric derived
  | BandCustom        -- ^ Custom defined
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Single resonance entry
data ResonanceEntry = ResonanceEntry
  { reFrequency   :: !Double        -- ^ Primary frequency (Hz)
  , reTarget      :: !String        -- ^ Target (organ, system, etc.)
  , reIntensity   :: !Double        -- ^ Recommended intensity
  , reDuration    :: !Int           -- ^ Duration (seconds)
  , reNotes       :: !String        -- ^ Usage notes
  } deriving (Eq, Show)

-- | User-specific profile
data UserProfile = UserProfile
  { upUserId        :: !String          -- ^ User identifier
  , upBaseFrequency :: !Double          -- ^ Personal base frequency
  , upSensitivity   :: !Double          -- ^ Sensitivity factor [0, 1]
  , upPreferredBands :: ![String]       -- ^ Preferred band names
  , upAvoidBands    :: ![String]        -- ^ Bands to avoid
  , upHistory       :: ![(Double, Double)]  -- ^ (frequency, response) history
  } deriving (Eq, Show)

-- =============================================================================
-- Map Creation
-- =============================================================================

-- | Create tuning map with specified bands
createTuningMap :: [FrequencyBand] -> TuningMap
createTuningMap bands = TuningMap
  { tmBands = bands
  , tmProfile = Nothing
  , tmResonances = Map.empty
  , tmBaseline = 7.83  -- Schumann resonance
  , tmRange = (0.1, 10000)
  }

-- | Default tuning map with standard bands
defaultTuningMap :: TuningMap
defaultTuningMap = createTuningMap
  [ FrequencyBand "Schumann" 7.83 0.5 BandSchumann 0.5 [1, 2, 3]
  , FrequencyBand "UT-396" 396 10 BandSolfeggio 0.6 [1]
  , FrequencyBand "RE-417" 417 10 BandSolfeggio 0.6 [1]
  , FrequencyBand "MI-528" 528 10 BandSolfeggio 0.7 [1]
  , FrequencyBand "FA-639" 639 10 BandSolfeggio 0.6 [1]
  , FrequencyBand "SOL-741" 741 10 BandSolfeggio 0.6 [1]
  , FrequencyBand "LA-852" 852 10 BandSolfeggio 0.6 [1]
  , FrequencyBand "SI-963" 963 10 BandSolfeggio 0.5 [1]
  , FrequencyBand "Rife-Cancer" 2128 50 BandRife 0.4 [1, 2]
  , FrequencyBand "Lakhovsky-MWO" 750000 100000 BandLakhovsky 0.3 [1]
  , FrequencyBand "Golden" (528 * phi) 20 BandCustom 0.8 [1]
  ]

-- | Personalize map for user
personalizeMap :: TuningMap -> UserProfile -> TuningMap
personalizeMap tmap profile =
  let adjustedBands = map (adjustForUser profile) (tmBands tmap)
  in tmap { tmBands = adjustedBands, tmProfile = Just profile }

-- =============================================================================
-- Frequency Lookup
-- =============================================================================

-- | Look up frequency by band name
lookupFrequency :: TuningMap -> String -> Maybe Double
lookupFrequency tmap bandName =
  case filter ((== bandName) . fbName) (tmBands tmap) of
    (band:_) -> Just (fbCenter band)
    [] -> Nothing

-- | Find best resonance for target
findResonance :: TuningMap -> String -> Maybe ResonanceEntry
findResonance tmap target = Map.lookup target (tmResonances tmap)

-- | Find optimal band for coherence level
optimalBand :: TuningMap -> Double -> Maybe FrequencyBand
optimalBand tmap coherence =
  let suitable = filter (bandSuitable coherence) (tmBands tmap)
  in case suitable of
    [] -> Nothing
    bands -> Just $ head $ sortByResonance bands coherence

-- =============================================================================
-- Band Management
-- =============================================================================

-- | Add frequency band
addBand :: TuningMap -> FrequencyBand -> TuningMap
addBand tmap band = tmap { tmBands = band : tmBands tmap }

-- | Remove frequency band by name
removeBand :: TuningMap -> String -> TuningMap
removeBand tmap bandName =
  tmap { tmBands = filter ((/= bandName) . fbName) (tmBands tmap) }

-- | Adjust band parameters
adjustBand :: TuningMap -> String -> (FrequencyBand -> FrequencyBand) -> TuningMap
adjustBand tmap bandName f =
  let adjusted = map (\b -> if fbName b == bandName then f b else b) (tmBands tmap)
  in tmap { tmBands = adjusted }

-- =============================================================================
-- User Profiling
-- =============================================================================

-- | Create user profile
createProfile :: String -> Double -> UserProfile
createProfile userId baseFreq = UserProfile
  { upUserId = userId
  , upBaseFrequency = baseFreq
  , upSensitivity = 0.5
  , upPreferredBands = []
  , upAvoidBands = []
  , upHistory = []
  }

-- | Update profile with response data
updateProfile :: UserProfile -> Double -> Double -> UserProfile
updateProfile profile freq response =
  let newHistory = (freq, response) : take 99 (upHistory profile)
      newSensitivity = calculateSensitivity newHistory
  in profile { upHistory = newHistory, upSensitivity = newSensitivity }

-- | Calculate profile resonance score for frequency
profileResonance :: UserProfile -> Double -> Double
profileResonance profile freq =
  let baseDiff = abs (freq - upBaseFrequency profile)
      baseScore = 1 / (1 + baseDiff / 100)
      historyScore = averageResponse profile freq
  in baseScore * 0.6 + historyScore * 0.4

-- =============================================================================
-- Integration
-- =============================================================================

-- | Chamber settings from tuning map
data ChamberTuning = ChamberTuning
  { ctPrimaryFreq   :: !Double
  , ctSecondaryFreq :: !(Maybe Double)
  , ctIntensity     :: !Double
  , ctDuration      :: !Int
  } deriving (Eq, Show)

-- | Convert tuning map selection to chamber settings
mapToChamber :: TuningMap -> String -> Maybe ChamberTuning
mapToChamber tmap bandName =
  case filter ((== bandName) . fbName) (tmBands tmap) of
    (band:_) -> Just ChamberTuning
      { ctPrimaryFreq = fbCenter band
      , ctSecondaryFreq = if length (fbHarmonics band) > 1
                          then Just (fbCenter band * fromIntegral (fbHarmonics band !! 1))
                          else Nothing
      , ctIntensity = fbIntensity band
      , ctDuration = 300  -- Default 5 minutes
      }
    [] -> Nothing

-- | Process chamber feedback
chamberFeedback :: TuningMap -> String -> Double -> TuningMap
chamberFeedback tmap bandName response =
  case tmProfile tmap of
    Nothing -> tmap
    Just profile ->
      case lookupFrequency tmap bandName of
        Nothing -> tmap
        Just freq ->
          let newProfile = updateProfile profile freq response
          in tmap { tmProfile = Just newProfile }

-- | Synchronize with biometric data
biometricSync :: TuningMap -> Double -> Double -> TuningMap
biometricSync tmap hrv coherence =
  let syncFactor = hrv * coherence * phiInverse
      newBaseline = tmBaseline tmap * (1 + syncFactor * 0.1)
  in tmap { tmBaseline = newBaseline }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Adjust band for user profile
adjustForUser :: UserProfile -> FrequencyBand -> FrequencyBand
adjustForUser profile band =
  let sensitivity = upSensitivity profile
      newIntensity = fbIntensity band * sensitivity
  in band { fbIntensity = newIntensity }

-- | Check if band is suitable for coherence
bandSuitable :: Double -> FrequencyBand -> Bool
bandSuitable coherence band =
  let minCoherence = case fbType band of
        BandRife -> 0.5
        BandLakhovsky -> 0.6
        BandSolfeggio -> 0.3
        BandSchumann -> 0.1
        BandBiometric -> 0.4
        BandCustom -> 0.3
  in coherence >= minCoherence

-- | Sort bands by resonance match
sortByResonance :: [FrequencyBand] -> Double -> [FrequencyBand]
sortByResonance bands coherence =
  let score b = fbIntensity b * coherence * (if fbType b == BandSolfeggio then phi else 1)
  in reverse $ map snd $ sortPairs $ zip (map score bands) bands
  where
    sortPairs = foldr insertSorted []
    insertSorted x [] = [x]
    insertSorted x@(s1, _) (y@(s2, _):ys)
      | s1 <= s2 = x : y : ys
      | otherwise = y : insertSorted x ys

-- | Calculate sensitivity from history
calculateSensitivity :: [(Double, Double)] -> Double
calculateSensitivity [] = 0.5
calculateSensitivity history =
  let responses = map snd history
      avg = sum responses / fromIntegral (length responses)
  in min 1.0 (max 0.1 avg)

-- | Average historical response near frequency
averageResponse :: UserProfile -> Double -> Double
averageResponse profile targetFreq =
  let nearby = filter (\(f, _) -> abs (f - targetFreq) < 50) (upHistory profile)
      responses = map snd nearby
  in if null responses then 0.5 else sum responses / fromIntegral (length responses)
