{-|
Module      : Ra.Shell.LimbGestures
Description : Torsion-based gesture recognition for scalar control
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Detects and interprets gesture patterns from torsion-influenced sensory
input (biometric, intent vectors, and field resonance). Outputs gesture
intents for coherent, field-aware movements of appendages.

== Gesture Recognition Theory

=== Scalar Vector Tracking

Gestures are recognized via:

* Displacement vectors over φ^n ticks
* Short-term memory of spatial motion
* Directional continuity patterns
* Harmonic alignment context

=== Recognition Patterns

* OpenHand: Small jitter then outward expansion
* CloseHand: Inward collapse motion
* SwipeRight: Consistent x-axis movement
* CircleMotion: Circular torsion pattern
-}
module Ra.Shell.LimbGestures
  ( -- * Core Types
    Gesture(..)
  , GestureEvent(..)
  , ScalarVectorTrack(..)

    -- * Gesture Detection
  , detectGesture
  , detectWithThreshold
  , gestureConfidence

    -- * Intent Mapping
  , ControlIntent(..)
  , mapGestureToIntent
  , intentFromGesture

    -- * Pattern Recognition
  , PatternMatcher(..)
  , defaultMatcher
  , matchPattern
  , patternScore

    -- * Vector Analysis
  , analyzeTrack
  , trackDirection
  , trackMagnitude
  , trackCurvature

    -- * Temporal Tracking
  , TrackHistory(..)
  , updateHistory
  , historyToGesture

    -- * Coherence Gating
  , coherenceGate
  , minimumCoherence
  , filterByCoherence
  ) where

import Ra.Constants.Extended
  ( phiInverse )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Gesture classification
data Gesture
  = OpenHand        -- ^ Hand opening/releasing
  | CloseHand       -- ^ Hand closing/grasping
  | Point           -- ^ Pointing gesture
  | SwipeLeft       -- ^ Leftward swipe
  | SwipeRight      -- ^ Rightward swipe
  | RaiseArm        -- ^ Arm raising
  | LowerArm        -- ^ Arm lowering
  | CircleMotion    -- ^ Circular movement
  | HoldStill       -- ^ Stationary hold
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Gesture event with metadata
data GestureEvent = GestureEvent
  { geGestureType     :: !Gesture
  , geConfidenceScore :: !Double          -- ^ [0, 1]
  , geTimestampPhiN   :: !Int             -- ^ φ^n tick index
  , geSpatialVector   :: !(Double, Double, Double)
  , geCoherence       :: !Double          -- ^ Field coherence at detection
  } deriving (Eq, Show)

-- | Scalar vector track (motion sample)
data ScalarVectorTrack = ScalarVectorTrack
  { svtPosition   :: !(Double, Double, Double)  -- ^ Current position
  , svtVelocity   :: !(Double, Double, Double)  -- ^ Velocity vector
  , svtTorsion    :: !Double                    -- ^ Torsion component
  , svtCoherence  :: !Double                    -- ^ Coherence at sample
  , svtTimestamp  :: !Int                       -- ^ φ^n tick
  } deriving (Eq, Show)

-- =============================================================================
-- Gesture Detection
-- =============================================================================

-- | Detect gesture from vector track series
detectGesture :: [ScalarVectorTrack] -> Maybe GestureEvent
detectGesture = detectWithThreshold 0.5

-- | Detect gesture with custom confidence threshold
detectWithThreshold :: Double -> [ScalarVectorTrack] -> Maybe GestureEvent
detectWithThreshold threshold tracks
  | length tracks < 3 = Nothing
  | avgCoherence < phiInverse = Nothing  -- Coherence gate
  | otherwise =
      let -- Analyze track characteristics
          direction = trackDirection tracks
          magnitude = trackMagnitude tracks
          curvature = trackCurvature tracks

          -- Try to match patterns
          candidates =
            [ (OpenHand, matchOpenHand tracks)
            , (CloseHand, matchCloseHand tracks)
            , (Point, matchPoint tracks)
            , (SwipeLeft, matchSwipeLeft direction magnitude)
            , (SwipeRight, matchSwipeRight direction magnitude)
            , (RaiseArm, matchRaiseArm direction)
            , (LowerArm, matchLowerArm direction)
            , (CircleMotion, matchCircle curvature)
            , (HoldStill, matchHoldStill magnitude)
            ]

          -- Find best match
          best = maximumByScore candidates

      in if snd best >= threshold
         then Just $ GestureEvent
           { geGestureType = fst best
           , geConfidenceScore = snd best
           , geTimestampPhiN = svtTimestamp (last tracks)
           , geSpatialVector = finalPosition tracks
           , geCoherence = avgCoherence
           }
         else Nothing
  where
    avgCoherence = sum (map svtCoherence tracks) / fromIntegral (length tracks)
    finalPosition ts = svtPosition (last ts)
    maximumByScore = foldr1 (\a b -> if snd a > snd b then a else b)

-- | Get confidence of gesture detection
gestureConfidence :: GestureEvent -> Double
gestureConfidence = geConfidenceScore

-- =============================================================================
-- Pattern Matching
-- =============================================================================

-- Match open hand (small jitter then outward)
matchOpenHand :: [ScalarVectorTrack] -> Double
matchOpenHand tracks =
  let vels = map svtVelocity tracks
      mags = map vecMagnitude vels
      -- Look for increasing magnitude (expansion)
      expanding = isIncreasing mags
      -- Check for initial jitter (low magnitude start)
      jitterStart = head mags < 0.3
  in if expanding && jitterStart then 0.8 else 0.2

-- Match close hand (inward collapse)
matchCloseHand :: [ScalarVectorTrack] -> Double
matchCloseHand tracks =
  let vels = map svtVelocity tracks
      mags = map vecMagnitude vels
      -- Look for inward motion (velocity toward center)
      inward = all (\v -> vecMagnitude v > 0.1 && isInward v) vels
      -- Decreasing spread
      collapsing = isDecreasing mags || all (< 0.5) mags
  in if inward && collapsing then 0.75 else 0.15

-- Match pointing
matchPoint :: [ScalarVectorTrack] -> Double
matchPoint tracks =
  let vels = map svtVelocity tracks
      -- Consistent direction
      consistent = directionalConsistency vels > 0.8
      -- High torsion stability
      torsionStable = variance (map svtTorsion tracks) < 0.1
  in if consistent && torsionStable then 0.7 else 0.2

-- Match swipe left
matchSwipeLeft :: (Double, Double, Double) -> Double -> Double
matchSwipeLeft (dx, _, _) mag
  | dx < -0.5 && mag > 0.3 = 0.85
  | dx < -0.2 && mag > 0.2 = 0.5
  | otherwise = 0.1

-- Match swipe right
matchSwipeRight :: (Double, Double, Double) -> Double -> Double
matchSwipeRight (dx, _, _) mag
  | dx > 0.5 && mag > 0.3 = 0.85
  | dx > 0.2 && mag > 0.2 = 0.5
  | otherwise = 0.1

-- Match raise arm
matchRaiseArm :: (Double, Double, Double) -> Double
matchRaiseArm (_, dy, _)
  | dy > 0.6 = 0.9
  | dy > 0.3 = 0.6
  | otherwise = 0.1

-- Match lower arm
matchLowerArm :: (Double, Double, Double) -> Double
matchLowerArm (_, dy, _)
  | dy < -0.6 = 0.9
  | dy < -0.3 = 0.6
  | otherwise = 0.1

-- Match circle motion
matchCircle :: Double -> Double
matchCircle curvature
  | curvature > 0.7 = 0.9
  | curvature > 0.4 = 0.6
  | otherwise = 0.2

-- Match hold still
matchHoldStill :: Double -> Double
matchHoldStill magnitude
  | magnitude < 0.05 = 0.95
  | magnitude < 0.15 = 0.7
  | otherwise = 0.1

-- =============================================================================
-- Intent Mapping
-- =============================================================================

-- | Control intent from gesture
data ControlIntent
  = Release                               -- ^ Release/let go
  | Grasp                                 -- ^ Grab/hold
  | MoveTo !(Double, Double, Double)      -- ^ Move to position
  | MoveBy !(Double, Double, Double)      -- ^ Move by offset
  | HoverAt !(Double, Double, Double)     -- ^ Hover at position
  | PointAt !(Double, Double, Double)     -- ^ Point toward
  | Rotate !Double                        -- ^ Rotate by angle
  | NoAction                              -- ^ No action
  deriving (Eq, Show)

-- | Map gesture to control intent
mapGestureToIntent :: GestureEvent -> ControlIntent
mapGestureToIntent event =
  let pos = geSpatialVector event
  in intentFromGesture (geGestureType event) pos

-- | Get intent from gesture type
intentFromGesture :: Gesture -> (Double, Double, Double) -> ControlIntent
intentFromGesture gesture pos = case gesture of
  OpenHand -> Release
  CloseHand -> Grasp
  Point -> PointAt pos
  SwipeLeft -> MoveBy (-1.0, 0.0, 0.0)
  SwipeRight -> MoveBy (1.0, 0.0, 0.0)
  RaiseArm -> MoveBy (0.0, 1.0, 0.0)
  LowerArm -> MoveBy (0.0, -1.0, 0.0)
  CircleMotion -> HoverAt pos
  HoldStill -> NoAction

-- =============================================================================
-- Pattern Matcher
-- =============================================================================

-- | Pattern matcher configuration
data PatternMatcher = PatternMatcher
  { pmMinSamples    :: !Int             -- ^ Minimum samples required
  , pmCoherenceFloor :: !Double         -- ^ Minimum coherence
  , pmConfidenceMin :: !Double          -- ^ Minimum confidence
  , pmTimeWindow    :: !Int             -- ^ φ^n ticks to consider
  } deriving (Eq, Show)

-- | Default pattern matcher
defaultMatcher :: PatternMatcher
defaultMatcher = PatternMatcher
  { pmMinSamples = 3
  , pmCoherenceFloor = phiInverse
  , pmConfidenceMin = 0.5
  , pmTimeWindow = 5
  }

-- | Match pattern with configuration
matchPattern :: PatternMatcher -> [ScalarVectorTrack] -> Maybe GestureEvent
matchPattern matcher tracks
  | length tracks < pmMinSamples matcher = Nothing
  | otherwise = detectWithThreshold (pmConfidenceMin matcher) tracks

-- | Get pattern score (0-1)
patternScore :: [ScalarVectorTrack] -> Gesture -> Double
patternScore tracks gesture =
  let direction = trackDirection tracks
      magnitude = trackMagnitude tracks
      curvature = trackCurvature tracks
  in case gesture of
       OpenHand -> matchOpenHand tracks
       CloseHand -> matchCloseHand tracks
       Point -> matchPoint tracks
       SwipeLeft -> matchSwipeLeft direction magnitude
       SwipeRight -> matchSwipeRight direction magnitude
       RaiseArm -> matchRaiseArm direction
       LowerArm -> matchLowerArm direction
       CircleMotion -> matchCircle curvature
       HoldStill -> matchHoldStill magnitude

-- =============================================================================
-- Vector Analysis
-- =============================================================================

-- | Analyze track for motion characteristics
analyzeTrack :: [ScalarVectorTrack] -> (Double, Double, Double, Double)
analyzeTrack tracks =
  let dir = trackDirection tracks
      mag = trackMagnitude tracks
      curv = trackCurvature tracks
  in (fst3 dir, snd3 dir, mag, curv)
  where
    fst3 (a, _, _) = a
    snd3 (_, b, _) = b

-- | Get overall direction of track
trackDirection :: [ScalarVectorTrack] -> (Double, Double, Double)
trackDirection tracks
  | length tracks < 2 = (0, 0, 0)
  | otherwise =
      let first = svtPosition (head tracks)
          lastP = svtPosition (last tracks)
      in vecNormalize (vecSub lastP first)

-- | Get overall magnitude of motion
trackMagnitude :: [ScalarVectorTrack] -> Double
trackMagnitude tracks
  | length tracks < 2 = 0.0
  | otherwise =
      let displacements = zipWith (\a b -> vecMagnitude (vecSub (svtPosition b) (svtPosition a)))
                                  tracks (tail tracks)
      in sum displacements / fromIntegral (length displacements)

-- | Get curvature (circularity) of track
trackCurvature :: [ScalarVectorTrack] -> Double
trackCurvature tracks
  | length tracks < 4 = 0.0
  | otherwise =
      let vels = map svtVelocity tracks
          angles = zipWith vecAngle vels (tail vels)
          totalTurn = sum angles
      in min 1.0 (totalTurn / (2 * pi))

-- =============================================================================
-- Temporal Tracking
-- =============================================================================

-- | Track history for gesture detection
data TrackHistory = TrackHistory
  { thTracks     :: ![ScalarVectorTrack]
  , thMaxLength  :: !Int
  , thLastGesture :: !(Maybe GestureEvent)
  } deriving (Eq, Show)

-- | Update history with new track
updateHistory :: ScalarVectorTrack -> TrackHistory -> TrackHistory
updateHistory track history =
  let newTracks = take (thMaxLength history) (track : thTracks history)
  in history { thTracks = newTracks }

-- | Convert history to gesture (if detected)
historyToGesture :: TrackHistory -> Maybe GestureEvent
historyToGesture history =
  detectGesture (reverse $ thTracks history)

-- =============================================================================
-- Coherence Gating
-- =============================================================================

-- | Apply coherence gate to detection
coherenceGate :: Double -> [ScalarVectorTrack] -> [ScalarVectorTrack]
coherenceGate threshold = filter (\t -> svtCoherence t >= threshold)

-- | Minimum required coherence
minimumCoherence :: Double
minimumCoherence = phiInverse

-- | Filter tracks by coherence
filterByCoherence :: [ScalarVectorTrack] -> [ScalarVectorTrack]
filterByCoherence = coherenceGate minimumCoherence

-- =============================================================================
-- Vector Utilities
-- =============================================================================

-- | Vector subtraction
vecSub :: (Double, Double, Double) -> (Double, Double, Double) -> (Double, Double, Double)
vecSub (x1, y1, z1) (x2, y2, z2) = (x1 - x2, y1 - y2, z1 - z2)

-- | Vector magnitude
vecMagnitude :: (Double, Double, Double) -> Double
vecMagnitude (x, y, z) = sqrt (x*x + y*y + z*z)

-- | Normalize vector
vecNormalize :: (Double, Double, Double) -> (Double, Double, Double)
vecNormalize v@(x, y, z) =
  let m = vecMagnitude v
  in if m < 0.0001 then (0, 0, 0)
     else (x/m, y/m, z/m)

-- | Angle between vectors
vecAngle :: (Double, Double, Double) -> (Double, Double, Double) -> Double
vecAngle v1 v2 =
  let m1 = vecMagnitude v1
      m2 = vecMagnitude v2
  in if m1 < 0.0001 || m2 < 0.0001 then 0.0
     else acos (clamp11 (vecDot v1 v2 / (m1 * m2)))

-- | Dot product
vecDot :: (Double, Double, Double) -> (Double, Double, Double) -> Double
vecDot (x1, y1, z1) (x2, y2, z2) = x1*x2 + y1*y2 + z1*z2

-- | Check if velocity is inward
isInward :: (Double, Double, Double) -> Bool
isInward (x, y, z) = x*x + y*y + z*z > 0.01 && (x < 0 || y < 0 || z < 0)

-- | Check if list is increasing
isIncreasing :: [Double] -> Bool
isIncreasing xs = and $ zipWith (<) xs (tail xs)

-- | Check if list is decreasing
isDecreasing :: [Double] -> Bool
isDecreasing xs = and $ zipWith (>) xs (tail xs)

-- | Directional consistency (how aligned velocities are)
directionalConsistency :: [(Double, Double, Double)] -> Double
directionalConsistency vels
  | length vels < 2 = 0.0
  | otherwise =
      let normalized = map vecNormalize vels
          dots = zipWith vecDot normalized (tail normalized)
      in sum dots / fromIntegral (length dots)

-- | Variance of list
variance :: [Double] -> Double
variance [] = 0.0
variance xs =
  let n = fromIntegral (length xs)
      mean = sum xs / n
      sqDiffs = map (\x -> (x - mean) ** 2) xs
  in sum sqDiffs / n

-- | Clamp to [-1, 1]
clamp11 :: Double -> Double
clamp11 x = max (-1.0) (min 1.0 x)
