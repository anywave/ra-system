{-|
Module      : RaShellLimbLearning
Description : Adaptive Gesture Personalization Engine
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 58: Adaptive gesture learning and personalization engine.
Learns gesture patterns per user, allows gesture→intent override,
and uses hybrid matching (DTW + angle cosine + coherence).

Based on incremental model blending with configurable tolerance.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaShellLimbLearning where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Matching weights (scaled to sum to 256)
dtwWeight :: Unsigned 8
dtwWeight = 128    -- 0.5 * 256

cosineWeight :: Unsigned 8
cosineWeight = 77  -- 0.3 * 256

coherenceWeight :: Unsigned 8
coherenceWeight = 51  -- 0.2 * 256

-- | Default tolerance (0.15 * 255)
defaultTolerance :: Unsigned 8
defaultTolerance = 38

-- | Default learning rate (0.3 * 255)
defaultLearningRate :: Unsigned 8
defaultLearningRate = 77

-- | Match confidence threshold (0.80 * 255)
matchConfidenceThreshold :: Unsigned 8
matchConfidenceThreshold = 204

-- | Max gestures per user
maxGesturesPerUser :: Unsigned 4
maxGesturesPerUser = 9

-- | Gesture types (matching P57)
data GestureType
  = GestNone
  | GestReachForward
  | GestPullBack
  | GestPushOut
  | GestGraspClose
  | GestReleaseOpen
  | GestSwipeLeft
  | GestSwipeRight
  | GestSwipeUp
  | GestSwipeDown
  | GestCircleCW
  | GestCircleCCW
  | GestHoldSteady
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Control intent types
data ControlIntent
  = IntentNone
  | IntentReach
  | IntentPull
  | IntentPush
  | IntentGrasp
  | IntentRelease
  | IntentMoveTo
  | IntentHoverAt
  | IntentOpenGate
  | IntentCloseGate
  | IntentActivate
  | IntentDeactivate
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Override condition types
data OverrideCondType
  = CondAlways
  | CondCoherenceAbove
  | CondPhaseAligned
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Scalar vector track point (fixed point)
data TrackPoint = TrackPoint
  { tpX         :: Signed 16
  , tpY         :: Signed 16
  , tpZ         :: Signed 16
  , tpTimestamp :: Unsigned 16
  , tpCoherence :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Override condition
data OverrideCondition = OverrideCondition
  { ocCondType  :: OverrideCondType
  , ocThreshold :: Unsigned 8  -- For coherence condition
  } deriving (Generic, NFDataX)

-- | Biometric state for condition checks
data BiometricState = BiometricState
  { bsCoherence    :: Unsigned 8
  , bsPhaseAligned :: Bool
  } deriving (Generic, NFDataX)

-- | Gesture model for a single gesture type
data GestureModel = GestureModel
  { gmTemplate      :: Vec 16 TrackPoint  -- Template points
  , gmPointCount    :: Unsigned 4         -- Valid points in template
  , gmTolerance     :: Unsigned 8         -- Match tolerance
  , gmOverrideIntent :: ControlIntent     -- Custom intent (IntentNone if no override)
  , gmOverrideCond  :: OverrideCondition  -- Override condition
  , gmSampleCount   :: Unsigned 8         -- Number of samples blended
  , gmIsValid       :: Bool               -- Model is initialized
  } deriving (Generic, NFDataX)

-- | Gesture event result
data GestureEvent = GestureEvent
  { geGesture    :: GestureType
  , geConfidence :: Unsigned 8
  , geIntent     :: ControlIntent
  , geIsOverride :: Bool
  , geIsValid    :: Bool
  } deriving (Generic, NFDataX)

-- | User gesture library (up to 9 gestures)
data UserGestureLibrary = UserGestureLibrary
  { uglModels      :: Vec 13 GestureModel  -- One per GestureType
  , uglUserId      :: Unsigned 8
  , uglGestureCount :: Unsigned 4
  } deriving (Generic, NFDataX)

-- | Initial empty track point
emptyTrackPoint :: TrackPoint
emptyTrackPoint = TrackPoint 0 0 0 0 128

-- | Initial empty model
emptyGestureModel :: GestureModel
emptyGestureModel = GestureModel
  (repeat emptyTrackPoint)
  0
  defaultTolerance
  IntentNone
  (OverrideCondition CondAlways 0)
  0
  False

-- | Initial empty library
emptyUserLibrary :: UserGestureLibrary
emptyUserLibrary = UserGestureLibrary
  (repeat emptyGestureModel)
  0
  0

-- | Compute Euclidean distance squared between two points
distanceSquared :: TrackPoint -> TrackPoint -> Unsigned 32
distanceSquared p1 p2 =
  let dx = tpX p1 - tpX p2
      dy = tpY p1 - tpY p2
      dz = tpZ p1 - tpZ p2
      dxSq = resize (dx * dx) :: Unsigned 32
      dySq = resize (dy * dy) :: Unsigned 32
      dzSq = resize (dz * dz) :: Unsigned 32
  in dxSq + dySq + dzSq

-- | Compute DTW similarity (simplified for FPGA)
-- Uses sum of point-wise distances normalized
computeDTWSimilarity
  :: Vec 16 TrackPoint -> Unsigned 4  -- Template
  -> Vec 16 TrackPoint -> Unsigned 4  -- Candidate
  -> Unsigned 8
computeDTWSimilarity template tCount candidate cCount
  | tCount == 0 || cCount == 0 = 0
  | otherwise =
      let -- Sum minimum distances
          sumDist :: Unsigned 32 -> Unsigned 4 -> Unsigned 32
          sumDist acc i
            | i >= cCount = acc
            | otherwise =
                let cPoint = candidate !! i
                    -- Find minimum distance to any template point
                    minDist = foldl
                      (\m j -> if j < tCount
                               then min m (distanceSquared cPoint (template !! j))
                               else m)
                      maxBound
                      $(listToVecTH [0..15 :: Unsigned 4])
                in acc + (minDist `shiftR` 8)  -- Scale down

          totalDist = foldl sumDist 0 $(listToVecTH [0..15 :: Unsigned 4])

          -- Normalize (lower distance = higher similarity)
          maxDist = 65535 :: Unsigned 32
          normalized = if totalDist > maxDist then 0 else maxDist - totalDist
          similarity = resize (normalized `shiftR` 8) :: Unsigned 8

      in similarity

-- | Compute angle cosine similarity
computeAngleCosineSimilarity
  :: Vec 16 TrackPoint -> Unsigned 4
  -> Vec 16 TrackPoint -> Unsigned 4
  -> Unsigned 8
computeAngleCosineSimilarity template tCount candidate cCount
  | tCount < 2 || cCount < 2 = 128  -- Neutral
  | otherwise =
      let -- Get direction vectors
          getDir pts cnt =
            let p0 = pts !! 0
                pN = pts !! (cnt - 1)
            in (tpX pN - tpX p0, tpY pN - tpY p0, tpZ pN - tpZ p0)

          (tx, ty, tz) = getDir template tCount
          (cx, cy, cz) = getDir candidate cCount

          -- Dot product
          dot = resize tx * resize cx + resize ty * resize cy + resize tz * resize cz :: Signed 32

          -- Approximate normalization (just use sign and magnitude)
          similarity = if dot > 0
                       then 192 + resize (min 63 (dot `shiftR` 10)) :: Unsigned 8
                       else resize (max 0 (128 + resize (dot `shiftR` 10))) :: Unsigned 8

      in similarity

-- | Compute average coherence of track
computeAverageCoherence :: Vec 16 TrackPoint -> Unsigned 4 -> Unsigned 8
computeAverageCoherence track count
  | count == 0 = 0
  | otherwise =
      let sumCoh = foldl
            (\s i -> if i < count then s + resize (tpCoherence (track !! i)) else s)
            (0 :: Unsigned 16)
            $(listToVecTH [0..15 :: Unsigned 4])
      in resize (sumCoh `div` resize count)

-- | Compute hybrid match score
computeMatchScore
  :: Vec 16 TrackPoint -> Unsigned 4  -- Template
  -> Vec 16 TrackPoint -> Unsigned 4  -- Candidate
  -> Unsigned 8
computeMatchScore template tCount candidate cCount =
  let dtwSim = computeDTWSimilarity template tCount candidate cCount
      cosineSim = computeAngleCosineSimilarity template tCount candidate cCount
      coherence = computeAverageCoherence candidate cCount

      -- Weighted sum
      weighted = (resize dtwWeight * resize dtwSim +
                  resize cosineWeight * resize cosineSim +
                  resize coherenceWeight * resize coherence) `shiftR` 8 :: Unsigned 16

  in resize $ min 255 weighted

-- | Blend two track points
blendPoint :: TrackPoint -> TrackPoint -> Unsigned 8 -> TrackPoint
blendPoint old new lr =
  let invLr = 255 - lr

      blendVal :: Signed 16 -> Signed 16 -> Signed 16
      blendVal o n = resize ((resize invLr * resize o + resize lr * resize n) `shiftR` 8 :: Signed 32)

      blendU8 :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
      blendU8 o n = resize ((resize invLr * resize o + resize lr * resize n) `shiftR` 8 :: Unsigned 16)

  in TrackPoint
       (blendVal (tpX old) (tpX new))
       (blendVal (tpY old) (tpY new))
       (blendVal (tpZ old) (tpZ new))
       (tpTimestamp new)
       (blendU8 (tpCoherence old) (tpCoherence new))

-- | Blend templates
blendTemplates
  :: Vec 16 TrackPoint -> Unsigned 4  -- Old
  -> Vec 16 TrackPoint -> Unsigned 4  -- New
  -> Unsigned 8                       -- Learning rate
  -> (Vec 16 TrackPoint, Unsigned 4)
blendTemplates old oldCount new newCount lr =
  let minCount = min oldCount newCount
      blended = zipWith (\o n -> blendPoint o n lr) old new
  in (blended, minCount)

-- | Check override condition
checkOverrideCondition :: OverrideCondition -> BiometricState -> Bool
checkOverrideCondition cond bio =
  case ocCondType cond of
    CondAlways -> True
    CondCoherenceAbove -> bsCoherence bio >= ocThreshold cond
    CondPhaseAligned -> bsPhaseAligned bio

-- | Get default intent for gesture
getDefaultIntent :: GestureType -> ControlIntent
getDefaultIntent gesture = case gesture of
  GestNone         -> IntentNone
  GestReachForward -> IntentReach
  GestPullBack     -> IntentPull
  GestPushOut      -> IntentPush
  GestGraspClose   -> IntentGrasp
  GestReleaseOpen  -> IntentRelease
  GestSwipeLeft    -> IntentMoveTo
  GestSwipeRight   -> IntentMoveTo
  GestSwipeUp      -> IntentPush
  GestSwipeDown    -> IntentPull
  GestCircleCW     -> IntentHoverAt
  GestCircleCCW    -> IntentHoverAt
  GestHoldSteady   -> IntentHoverAt

-- | Learn gesture (update or create model)
learnGesture
  :: UserGestureLibrary
  -> GestureType
  -> Vec 16 TrackPoint
  -> Unsigned 4
  -> UserGestureLibrary
learnGesture library gesture track count =
  let gestIdx = resize (pack gesture) :: Unsigned 4
      model = uglModels library !! gestIdx

      (newTemplate, newCount) =
        if gmIsValid model
        then blendTemplates (gmTemplate model) (gmPointCount model) track count defaultLearningRate
        else (track, count)

      newModel = model
        { gmTemplate = newTemplate
        , gmPointCount = newCount
        , gmSampleCount = gmSampleCount model + 1
        , gmIsValid = True
        }

      newModels = replace gestIdx newModel (uglModels library)
      newGestCount = if gmIsValid model
                     then uglGestureCount library
                     else uglGestureCount library + 1

  in library { uglModels = newModels, uglGestureCount = newGestCount }

-- | Match gesture
matchUserGesture
  :: UserGestureLibrary
  -> Vec 16 TrackPoint
  -> Unsigned 4
  -> BiometricState
  -> GestureEvent
matchUserGesture library track count bio =
  let -- Find best matching gesture
      findBest :: (GestureType, Unsigned 8) -> Unsigned 4 -> (GestureType, Unsigned 8)
      findBest (bestG, bestS) i =
        let gesture = unpack (resize i) :: GestureType
            model = uglModels library !! i
        in if gmIsValid model
           then
             let score = computeMatchScore (gmTemplate model) (gmPointCount model) track count
                 threshold = 255 - gmTolerance model
             in if score >= threshold && score > bestS
                then (gesture, score)
                else (bestG, bestS)
           else (bestG, bestS)

      (bestGesture, bestScore) = foldl findBest (GestNone, 0)
                                       $(listToVecTH [0..12 :: Unsigned 4])

      isValid = bestScore >= matchConfidenceThreshold && bestGesture /= GestNone

      -- Determine intent
      gestIdx = resize (pack bestGesture) :: Unsigned 4
      model = uglModels library !! gestIdx

      defaultInt = getDefaultIntent bestGesture

      (finalIntent, isOverride) =
        if gmOverrideIntent model /= IntentNone
        then
          if checkOverrideCondition (gmOverrideCond model) bio
          then (gmOverrideIntent model, True)
          else (defaultInt, False)
        else (defaultInt, False)

  in GestureEvent bestGesture bestScore finalIntent isOverride isValid

-- | Override gesture intent
overrideGestureIntent
  :: UserGestureLibrary
  -> GestureType
  -> ControlIntent
  -> OverrideCondition
  -> UserGestureLibrary
overrideGestureIntent library gesture intent cond =
  let gestIdx = resize (pack gesture) :: Unsigned 4
      model = uglModels library !! gestIdx
      newModel = model { gmOverrideIntent = intent, gmOverrideCond = cond }
      newModels = replace gestIdx newModel (uglModels library)
  in library { uglModels = newModels }

-- | Learning pipeline state
data LearningState = LearningState
  { lsLibrary :: UserGestureLibrary
  } deriving (Generic, NFDataX)

-- | Initial learning state
initialLearningState :: LearningState
initialLearningState = LearningState emptyUserLibrary

-- | Learning input
data LearningInput = LearningInput
  { liTrack     :: Vec 16 TrackPoint
  , liCount     :: Unsigned 4
  , liGesture   :: GestureType
  , liBio       :: BiometricState
  , liLearnMode :: Bool  -- True = learn, False = match
  } deriving (Generic, NFDataX)

-- | Learning pipeline
learningPipeline
  :: HiddenClockResetEnable dom
  => Signal dom LearningInput
  -> Signal dom GestureEvent
learningPipeline input = mealy learnMealy initialLearningState input
  where
    learnMealy state inp =
      let lib = lsLibrary state
          event = matchUserGesture lib (liTrack inp) (liCount inp) (liBio inp)
          newLib = if liLearnMode inp && liGesture inp /= GestNone
                   then learnGesture lib (liGesture inp) (liTrack inp) (liCount inp)
                   else lib
      in (state { lsLibrary = newLib }, event)
