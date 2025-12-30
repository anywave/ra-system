{-|
Module      : Ra.Shell.LimbLearning
Description : Adaptive gesture personalization engine
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Enables per-user learning and refinement of gesture patterns. Improves
responsiveness and inclusivity for neurodiverse or physically atypical
users. Augments Ra.Shell.LimbGestures recognizer.

== Adaptive Learning Theory

=== Personalization Goals

* Adapt gesture classification to individual motor signatures
* Provide user-calibrated mappings from small samples
* Fuse ML heuristics with Ra coherence & torsion analysis
* Enable custom gesture programming

=== Learning Process

1. Capture user's motion path for labeled gesture
2. Store template in per-user library
3. Match incoming signals against personalized templates
4. Refine templates with continued use
-}
module Ra.Shell.LimbLearning
  ( -- * Core Types
    UserID
  , GestureModel(..)
  , UserGestureLibrary
  , UserLibraryStore

    -- * Learning Functions
  , learnGesture
  , learnFromSamples
  , updateModel

    -- * Matching Functions
  , matchUserGesture
  , matchConfidence
  , findBestMatch

    -- * Override System
  , overrideGestureIntent
  , getOverride
  , clearOverride

    -- * Library Management
  , emptyLibrary
  , addGestureModel
  , removeGestureModel
  , getModel

    -- * Template Analysis
  , TemplateStats(..)
  , computeTemplateStats
  , templateSimilarity

    -- * Calibration
  , CalibrationSession(..)
  , startCalibration
  , addCalibrationSample
  , finishCalibration

    -- * Persistence
  , serializeLibrary
  , deserializeLibrary
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

import Ra.Constants.Extended
  ( phiInverse )

-- Import gesture types
import Ra.Shell.LimbGestures
  ( Gesture(..)
  , GestureEvent(..)
  , ScalarVectorTrack(..)
  , ControlIntent(..)
  , trackDirection
  , trackMagnitude
  , trackCurvature
  )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | User identifier
type UserID = String

-- | Gesture model learned from user
data GestureModel = GestureModel
  { gmGestureTemplate :: ![ScalarVectorTrack]   -- ^ Known motion path
  , gmTolerance       :: !Double                -- ^ Deviation tolerance
  , gmOverrideIntent  :: !(Maybe ControlIntent) -- ^ Optional mapping override
  , gmSampleCount     :: !Int                   -- ^ Number of training samples
  , gmAvgCoherence    :: !Double                -- ^ Average coherence during training
  } deriving (Eq, Show)

-- | Per-user gesture library
type UserGestureLibrary = Map Gesture GestureModel

-- | User library store (all users)
type UserLibraryStore = Map UserID UserGestureLibrary

-- =============================================================================
-- Learning Functions
-- =============================================================================

-- | Learn gesture from samples
learnGesture :: Gesture -> [ScalarVectorTrack] -> UserGestureLibrary -> UserGestureLibrary
learnGesture gesture tracks library =
  let model = createModel tracks
  in Map.insert gesture model library

-- | Learn from multiple sample sets
learnFromSamples :: Gesture -> [[ScalarVectorTrack]] -> UserGestureLibrary -> UserGestureLibrary
learnFromSamples gesture sampleSets library =
  let -- Merge all samples into averaged template
      mergedTracks = mergeSamples sampleSets
      model = GestureModel
        { gmGestureTemplate = mergedTracks
        , gmTolerance = computeTolerance sampleSets
        , gmOverrideIntent = Nothing
        , gmSampleCount = length sampleSets
        , gmAvgCoherence = averageCoherence sampleSets
        }
  in Map.insert gesture model library

-- | Update existing model with new sample
updateModel :: Gesture -> [ScalarVectorTrack] -> UserGestureLibrary -> UserGestureLibrary
updateModel gesture newTracks library =
  case Map.lookup gesture library of
    Nothing -> learnGesture gesture newTracks library
    Just existing ->
      let -- Blend new sample with existing template
          blendFactor = 1.0 / fromIntegral (gmSampleCount existing + 1)
          blendedTemplate = blendTemplates
            (gmGestureTemplate existing)
            newTracks
            blendFactor
          updated = existing
            { gmGestureTemplate = blendedTemplate
            , gmSampleCount = gmSampleCount existing + 1
            , gmTolerance = gmTolerance existing * 0.95  -- Tighten with more samples
            }
      in Map.insert gesture updated library

-- Create model from tracks
createModel :: [ScalarVectorTrack] -> GestureModel
createModel tracks = GestureModel
  { gmGestureTemplate = tracks
  , gmTolerance = 0.3  -- Default tolerance
  , gmOverrideIntent = Nothing
  , gmSampleCount = 1
  , gmAvgCoherence = averageTrackCoherence tracks
  }

-- Merge multiple sample sets
mergeSamples :: [[ScalarVectorTrack]] -> [ScalarVectorTrack]
mergeSamples [] = []
mergeSamples sets =
  let -- Use length of first sample as reference
      refLength = length (head sets)
      -- Pad/truncate all sets to same length
      normalized = map (normalizeLength refLength) sets
      -- Average corresponding tracks
  in zipWithN averageTracks normalized

-- Normalize sample length
normalizeLength :: Int -> [ScalarVectorTrack] -> [ScalarVectorTrack]
normalizeLength n tracks
  | length tracks >= n = take n tracks
  | otherwise = tracks ++ replicate (n - length tracks) (last tracks)

-- Average multiple tracks
averageTracks :: [ScalarVectorTrack] -> ScalarVectorTrack
averageTracks [] = error "Cannot average empty track list"
averageTracks tracks =
  let n = fromIntegral (length tracks)
      avgPos = avgVec3 (map svtPosition tracks)
      avgVel = avgVec3 (map svtVelocity tracks)
      avgTor = sum (map svtTorsion tracks) / n
      avgCoh = sum (map svtCoherence tracks) / n
      avgTime = sum (map svtTimestamp tracks) `div` length tracks
  in ScalarVectorTrack avgPos avgVel avgTor avgCoh avgTime

-- Compute tolerance from sample variance
computeTolerance :: [[ScalarVectorTrack]] -> Double
computeTolerance sampleSets
  | length sampleSets < 2 = 0.3
  | otherwise =
      let -- Compute variance of track directions
          directions = map trackDirection sampleSets
          (xs, ys, zs) = transpose3 directions
          variances = [directionVariance xs, directionVariance ys, directionVariance zs]
          avgVariance = sum variances / 3.0
      in min 0.5 (max 0.1 (sqrt avgVariance + 0.1))

-- Blend two templates
blendTemplates :: [ScalarVectorTrack] -> [ScalarVectorTrack] -> Double -> [ScalarVectorTrack]
blendTemplates old new factor =
  let len = max (length old) (length new)
      oldNorm = normalizeLength len old
      newNorm = normalizeLength len new
  in zipWith (blendTrack factor) oldNorm newNorm

-- Blend single track
blendTrack :: Double -> ScalarVectorTrack -> ScalarVectorTrack -> ScalarVectorTrack
blendTrack f old new = ScalarVectorTrack
  { svtPosition = lerpVec3 f (svtPosition old) (svtPosition new)
  , svtVelocity = lerpVec3 f (svtVelocity old) (svtVelocity new)
  , svtTorsion = lerp f (svtTorsion old) (svtTorsion new)
  , svtCoherence = lerp f (svtCoherence old) (svtCoherence new)
  , svtTimestamp = svtTimestamp new
  }

-- =============================================================================
-- Matching Functions
-- =============================================================================

-- | Match incoming track against user's personalized gestures
matchUserGesture :: UserGestureLibrary -> [ScalarVectorTrack] -> Maybe GestureEvent
matchUserGesture library tracks
  | null tracks = Nothing
  | Map.null library = Nothing
  | otherwise =
      let -- Score each gesture model
          scores = [(gesture, matchScore model tracks)
                   | (gesture, model) <- Map.toList library]
          -- Find best match above threshold
          best = maximumByScore scores
      in if snd best >= phiInverse
         then Just $ GestureEvent
           { geGestureType = fst best
           , geConfidenceScore = snd best
           , geTimestampPhiN = svtTimestamp (last tracks)
           , geSpatialVector = svtPosition (last tracks)
           , geCoherence = averageTrackCoherence tracks
           }
         else Nothing
  where
    maximumByScore = foldr1 (\a b -> if snd a > snd b then a else b)

-- | Get match confidence for specific gesture
matchConfidence :: GestureModel -> [ScalarVectorTrack] -> Double
matchConfidence = matchScore

-- | Find best matching gesture
findBestMatch :: UserGestureLibrary -> [ScalarVectorTrack] -> Maybe (Gesture, Double)
findBestMatch library tracks
  | null tracks || Map.null library = Nothing
  | otherwise =
      let scores = [(gesture, matchScore model tracks)
                   | (gesture, model) <- Map.toList library]
          best = foldr1 (\a b -> if snd a > snd b then a else b) scores
      in if snd best >= 0.3 then Just best else Nothing

-- Compute match score between model and tracks
matchScore :: GestureModel -> [ScalarVectorTrack] -> Double
matchScore model tracks =
  let template = gmGestureTemplate model
      tolerance = gmTolerance model

      -- Shape similarity
      shapeSim = shapeSimilarity template tracks

      -- Angle similarity
      angleSim = angleSimilarity template tracks

      -- Coherence similarity
      cohSim = coherenceSimilarity (gmAvgCoherence model) tracks

      -- Combined score with tolerance adjustment
      rawScore = shapeSim * 0.4 + angleSim * 0.4 + cohSim * 0.2
      adjustedScore = rawScore + (tolerance * 0.5)  -- Higher tolerance = more forgiving

  in clamp01 adjustedScore

-- Shape similarity (position trajectory matching)
shapeSimilarity :: [ScalarVectorTrack] -> [ScalarVectorTrack] -> Double
shapeSimilarity template tracks =
  let dir1 = trackDirection template
      dir2 = trackDirection tracks
      mag1 = trackMagnitude template
      mag2 = trackMagnitude tracks

      dirSim = 1.0 - vecDistance3 dir1 dir2 / 2.0
      magSim = 1.0 - abs (mag1 - mag2) / max mag1 mag2

  in (dirSim + magSim) / 2.0

-- Angle similarity (curvature matching)
angleSimilarity :: [ScalarVectorTrack] -> [ScalarVectorTrack] -> Double
angleSimilarity template tracks =
  let curv1 = trackCurvature template
      curv2 = trackCurvature tracks
  in 1.0 - abs (curv1 - curv2)

-- Coherence similarity
coherenceSimilarity :: Double -> [ScalarVectorTrack] -> Double
coherenceSimilarity modelCoh tracks =
  let trackCoh = averageTrackCoherence tracks
  in 1.0 - abs (modelCoh - trackCoh)

-- =============================================================================
-- Override System
-- =============================================================================

-- | Override gesture intent for user
overrideGestureIntent :: Gesture -> ControlIntent -> UserGestureLibrary -> UserGestureLibrary
overrideGestureIntent gesture intent library =
  case Map.lookup gesture library of
    Nothing ->
      -- Create placeholder model with override
      Map.insert gesture (emptyModelWithOverride intent) library
    Just model ->
      Map.insert gesture (model { gmOverrideIntent = Just intent }) library

-- | Get override intent for gesture
getOverride :: Gesture -> UserGestureLibrary -> Maybe ControlIntent
getOverride gesture library =
  case Map.lookup gesture library of
    Nothing -> Nothing
    Just model -> gmOverrideIntent model

-- | Clear override for gesture
clearOverride :: Gesture -> UserGestureLibrary -> UserGestureLibrary
clearOverride gesture library =
  case Map.lookup gesture library of
    Nothing -> library
    Just model -> Map.insert gesture (model { gmOverrideIntent = Nothing }) library

-- Empty model with just override
emptyModelWithOverride :: ControlIntent -> GestureModel
emptyModelWithOverride intent = GestureModel
  { gmGestureTemplate = []
  , gmTolerance = 0.5
  , gmOverrideIntent = Just intent
  , gmSampleCount = 0
  , gmAvgCoherence = 0.5
  }

-- =============================================================================
-- Library Management
-- =============================================================================

-- | Empty gesture library
emptyLibrary :: UserGestureLibrary
emptyLibrary = Map.empty

-- | Add gesture model to library
addGestureModel :: Gesture -> GestureModel -> UserGestureLibrary -> UserGestureLibrary
addGestureModel = Map.insert

-- | Remove gesture model from library
removeGestureModel :: Gesture -> UserGestureLibrary -> UserGestureLibrary
removeGestureModel = Map.delete

-- | Get model for gesture
getModel :: Gesture -> UserGestureLibrary -> Maybe GestureModel
getModel = Map.lookup

-- =============================================================================
-- Template Analysis
-- =============================================================================

-- | Template statistics
data TemplateStats = TemplateStats
  { tsLength       :: !Int              -- ^ Template length
  , tsAvgMagnitude :: !Double           -- ^ Average motion magnitude
  , tsAvgCurvature :: !Double           -- ^ Average curvature
  , tsAvgCoherence :: !Double           -- ^ Average coherence
  , tsTorsionRange :: !(Double, Double) -- ^ Min/max torsion
  } deriving (Eq, Show)

-- | Compute statistics for template
computeTemplateStats :: [ScalarVectorTrack] -> TemplateStats
computeTemplateStats [] = TemplateStats 0 0 0 0 (0, 0)
computeTemplateStats tracks =
  let torsions = map svtTorsion tracks
  in TemplateStats
      { tsLength = length tracks
      , tsAvgMagnitude = trackMagnitude tracks
      , tsAvgCurvature = trackCurvature tracks
      , tsAvgCoherence = averageTrackCoherence tracks
      , tsTorsionRange = (minimum torsions, maximum torsions)
      }

-- | Compare template similarity
templateSimilarity :: [ScalarVectorTrack] -> [ScalarVectorTrack] -> Double
templateSimilarity t1 t2 =
  let shapeSim = shapeSimilarity t1 t2
      angleSim = angleSimilarity t1 t2
  in (shapeSim + angleSim) / 2.0

-- =============================================================================
-- Calibration
-- =============================================================================

-- | Calibration session for learning gestures
data CalibrationSession = CalibrationSession
  { csUserId     :: !UserID
  , csGesture    :: !Gesture
  , csSamples    :: ![[ScalarVectorTrack]]
  , csMinSamples :: !Int
  , csComplete   :: !Bool
  } deriving (Eq, Show)

-- | Start calibration session
startCalibration :: UserID -> Gesture -> Int -> CalibrationSession
startCalibration uid gesture minSamples = CalibrationSession
  { csUserId = uid
  , csGesture = gesture
  , csSamples = []
  , csMinSamples = minSamples
  , csComplete = False
  }

-- | Add sample to calibration
addCalibrationSample :: [ScalarVectorTrack] -> CalibrationSession -> CalibrationSession
addCalibrationSample sample session =
  let newSamples = csSamples session ++ [sample]
      complete = length newSamples >= csMinSamples session
  in session
      { csSamples = newSamples
      , csComplete = complete
      }

-- | Finish calibration and create model
finishCalibration :: CalibrationSession -> UserGestureLibrary -> UserGestureLibrary
finishCalibration session library
  | not (csComplete session) = library
  | otherwise = learnFromSamples (csGesture session) (csSamples session) library

-- =============================================================================
-- Persistence
-- =============================================================================

-- | Serialize library to string (simplified)
serializeLibrary :: UserGestureLibrary -> String
serializeLibrary library =
  show [(gesture, gmSampleCount model, gmTolerance model)
       | (gesture, model) <- Map.toList library]

-- | Deserialize library from string (placeholder)
deserializeLibrary :: String -> Maybe UserGestureLibrary
deserializeLibrary _ = Just emptyLibrary  -- Would parse properly

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Average track coherence
averageTrackCoherence :: [ScalarVectorTrack] -> Double
averageTrackCoherence [] = 0.5
averageTrackCoherence tracks =
  sum (map svtCoherence tracks) / fromIntegral (length tracks)

-- Average coherence across sample sets
averageCoherence :: [[ScalarVectorTrack]] -> Double
averageCoherence sets =
  let cohs = map averageTrackCoherence sets
  in sum cohs / fromIntegral (max 1 (length cohs))

-- | Linear interpolation
lerp :: Double -> Double -> Double -> Double
lerp t a b = a + t * (b - a)

-- | Vector3 linear interpolation
lerpVec3 :: Double -> (Double, Double, Double) -> (Double, Double, Double) -> (Double, Double, Double)
lerpVec3 t (x1, y1, z1) (x2, y2, z2) =
  (lerp t x1 x2, lerp t y1 y2, lerp t z1 z2)

-- | Average multiple vectors
avgVec3 :: [(Double, Double, Double)] -> (Double, Double, Double)
avgVec3 [] = (0, 0, 0)
avgVec3 vecs =
  let n = fromIntegral (length vecs)
      (xs, ys, zs) = unzip3 vecs
  in (sum xs / n, sum ys / n, sum zs / n)

-- | Distance between vectors
vecDistance3 :: (Double, Double, Double) -> (Double, Double, Double) -> Double
vecDistance3 (x1, y1, z1) (x2, y2, z2) =
  sqrt ((x1-x2)^(2::Int) + (y1-y2)^(2::Int) + (z1-z2)^(2::Int))

-- | Direction variance
directionVariance :: [Double] -> Double
directionVariance [] = 0.0
directionVariance xs =
  let n = fromIntegral (length xs)
      mean = sum xs / n
      sqDiffs = map (\x -> (x - mean) ** 2) xs
  in sum sqDiffs / n

-- | Transpose list of 3D vectors
transpose3 :: [(Double, Double, Double)] -> ([Double], [Double], [Double])
transpose3 vecs = unzip3 vecs

-- | Zip with n lists
zipWithN :: ([a] -> b) -> [[a]] -> [b]
zipWithN _ [] = []
zipWithN _ ([]:_) = []
zipWithN f lists = f (map head lists) : zipWithN f (map tail lists)

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
