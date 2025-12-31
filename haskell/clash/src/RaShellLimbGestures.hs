{-|
Module      : RaShellLimbGestures
Description : Gesture Recognition from Limb Motion Data
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 57: Gesture recognition from limb motion data for avatar control.
Uses adaptive frame windows (8-21 frames) based on motion speed,
with 0.60 confidence threshold for gesture classification.

Based on motion trajectory analysis and intent mapping.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaShellLimbGestures where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Frame window bounds
minFrameWindow :: Unsigned 5
minFrameWindow = 8

maxFrameWindow :: Unsigned 5
maxFrameWindow = 21

-- | Confidence threshold (0.60 * 255)
confidenceThreshold :: Unsigned 8
confidenceThreshold = 153

-- | Speed thresholds (scaled)
speedSlow :: Unsigned 8
speedSlow = 13    -- 0.05 * 255

speedFast :: Unsigned 8
speedFast = 77    -- 0.30 * 255

-- | Motion epsilon
motionEpsilon :: Unsigned 8
motionEpsilon = 3

-- | Gesture types
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

-- | Control intent (matching P56)
data ControlIntent
  = IntentNone
  | IntentReach
  | IntentPull
  | IntentPush
  | IntentGrasp
  | IntentRelease
  | IntentMoveTo
  | IntentHoverAt
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | 3D position (fixed point, 8.8 format)
data LimbPosition = LimbPosition
  { lpX         :: Signed 16
  , lpY         :: Signed 16
  , lpZ         :: Signed 16
  , lpTimestamp :: Unsigned 16
  } deriving (Generic, NFDataX)

-- | Motion frame
data MotionFrame = MotionFrame
  { mfPosition   :: LimbPosition
  , mfVelocityX  :: Signed 8
  , mfVelocityY  :: Signed 8
  , mfVelocityZ  :: Signed 8
  , mfFrameIndex :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Gesture result
data GestureResult = GestureResult
  { grGesture         :: GestureType
  , grConfidence      :: Unsigned 8
  , grFrameCount      :: Unsigned 5
  , grMotionMagnitude :: Unsigned 16
  , grIsValid         :: Bool
  } deriving (Generic, NFDataX)

-- | Motion buffer state
data MotionBufferState = MotionBufferState
  { mbFrames       :: Vec 24 MotionFrame  -- Max window + margin
  , mbWriteIdx     :: Unsigned 5
  , mbFrameCount   :: Unsigned 5
  , mbTargetWindow :: Unsigned 5
  } deriving (Generic, NFDataX)

-- | Initial buffer state
initialBufferState :: MotionBufferState
initialBufferState = MotionBufferState
  (repeat (MotionFrame (LimbPosition 0 0 0 0) 0 0 0 0))
  0
  0
  13  -- Default window (~φ * 8)

-- | Compute distance between positions
distanceSquared :: LimbPosition -> LimbPosition -> Unsigned 32
distanceSquared p1 p2 =
  let dx = lpX p1 - lpX p2
      dy = lpY p1 - lpY p2
      dz = lpZ p1 - lpZ p2
      dxSq = resize (dx * dx) :: Unsigned 32
      dySq = resize (dy * dy) :: Unsigned 32
      dzSq = resize (dz * dz) :: Unsigned 32
  in dxSq + dySq + dzSq

-- | Approximate square root (for distance)
approxSqrt :: Unsigned 32 -> Unsigned 16
approxSqrt x
  | x == 0 = 0
  | otherwise =
      let -- Newton-Raphson iteration (simplified for FPGA)
          initial = resize (x `shiftR` 1) :: Unsigned 16
          step y = (y + resize (x `div` resize y)) `shiftR` 1
          r1 = step initial
          r2 = step r1
      in r2

-- | Compute adaptive window based on speed
computeAdaptiveWindow :: Unsigned 8 -> Unsigned 5
computeAdaptiveWindow speed
  | speed <= speedSlow = maxFrameWindow
  | speed >= speedFast = minFrameWindow
  | otherwise =
      let range = speedFast - speedSlow
          pos = speed - speedSlow
          windowRange = maxFrameWindow - minFrameWindow
          scaled = resize pos * resize windowRange `div` resize range :: Unsigned 8
      in maxFrameWindow - resize scaled

-- | Compute motion vector (normalized direction)
computeMotionVector :: Vec 24 MotionFrame -> Unsigned 5 -> (Signed 8, Signed 8, Signed 8)
computeMotionVector frames count
  | count < 2 = (0, 0, 0)
  | otherwise =
      let -- Sum velocities
          sumVel :: (Signed 16, Signed 16, Signed 16) -> MotionFrame -> (Signed 16, Signed 16, Signed 16)
          sumVel (sx, sy, sz) frame =
            (sx + resize (mfVelocityX frame),
             sy + resize (mfVelocityY frame),
             sz + resize (mfVelocityZ frame))

          (totalVx, totalVy, totalVz) = foldl sumVel (0, 0, 0) frames

          -- Compute magnitude
          absVx = if totalVx < 0 then -totalVx else totalVx
          absVy = if totalVy < 0 then -totalVy else totalVy
          absVz = if totalVz < 0 then -totalVz else totalVz
          mag = absVx + absVy + absVz

      in if mag < resize motionEpsilon
         then (0, 0, 0)
         else
           let normX = (totalVx * 127) `div` mag
               normY = (totalVy * 127) `div` mag
               normZ = (totalVz * 127) `div` mag
           in (resize normX, resize normY, resize normZ)

-- | Compute motion magnitude
computeMotionMagnitude :: Vec 24 MotionFrame -> Unsigned 5 -> Unsigned 16
computeMotionMagnitude frames count
  | count < 2 = 0
  | otherwise =
      let sumDist :: Unsigned 32 -> Unsigned 5 -> Unsigned 32
          sumDist acc idx
            | idx >= count - 1 = acc
            | otherwise =
                let f1 = frames !! idx
                    f2 = frames !! (idx + 1)
                    distSq = distanceSquared (mfPosition f1) (mfPosition f2)
                in acc + resize (approxSqrt distSq)
      in resize $ foldl sumDist 0 $(listToVecTH [0..22 :: Unsigned 5])

-- | Detect circular motion
detectCircularMotion :: Vec 24 MotionFrame -> Unsigned 5 -> (Bool, Bool)
detectCircularMotion frames count
  | count < 8 = (False, False)
  | otherwise =
      let -- Compute cross products of successive velocity pairs
          crossSum :: Signed 32 -> Unsigned 5 -> Signed 32
          crossSum acc idx
            | idx >= count - 2 = acc
            | otherwise =
                let f1 = frames !! idx
                    f2 = frames !! (idx + 1)
                    v1x = resize (mfVelocityX f1) :: Signed 16
                    v1y = resize (mfVelocityY f1) :: Signed 16
                    v2x = resize (mfVelocityX f2) :: Signed 16
                    v2y = resize (mfVelocityY f2) :: Signed 16
                    cross = v1x * v2y - v1y * v2x
                in acc + resize cross

          totalCross = foldl crossSum 0 $(listToVecTH [0..21 :: Unsigned 5])
          avgCross = totalCross `div` resize (count - 2)

          isCircular = abs avgCross > 256
          isClockwise = avgCross < 0

      in (isCircular, isClockwise)

-- | Detect grasp motion (decelerating)
detectGraspMotion :: Vec 24 MotionFrame -> Unsigned 5 -> Bool
detectGraspMotion frames count
  | count < 4 = False
  | otherwise =
      let halfCount = count `shiftR` 1

          -- First half magnitude
          firstHalfMag = computeMotionMagnitude frames halfCount

          -- Second half magnitude (shift frames)
          secondHalfMag = computeMotionMagnitude
            (map (\i -> frames !! (i + halfCount)) $(listToVecTH [0..23 :: Unsigned 5]))
            (count - halfCount)

      in secondHalfMag < (firstHalfMag `shiftR` 1)

-- | Detect release motion (accelerating)
detectReleaseMotion :: Vec 24 MotionFrame -> Unsigned 5 -> Bool
detectReleaseMotion frames count
  | count < 4 = False
  | otherwise =
      let halfCount = count `shiftR` 1
          firstHalfMag = computeMotionMagnitude frames halfCount
          secondHalfMag = computeMotionMagnitude
            (map (\i -> frames !! (i + halfCount)) $(listToVecTH [0..23 :: Unsigned 5]))
            (count - halfCount)
      in secondHalfMag > firstHalfMag + (firstHalfMag `shiftR` 1)

-- | Classify gesture from motion
classifyGesture :: Vec 24 MotionFrame -> Unsigned 5 -> (GestureType, Unsigned 8)
classifyGesture frames count
  | count < minFrameWindow = (GestNone, 0)
  | otherwise =
      let motionMag = computeMotionMagnitude frames count
          (vx, vy, vz) = computeMotionVector frames count
          (isCircle, isCW) = detectCircularMotion frames count

          absVx = if vx < 0 then -vx else vx
          absVy = if vy < 0 then -vy else vy
          absVz = if vz < 0 then -vz else vz

      in if motionMag < resize motionEpsilon
         then (GestHoldSteady, 204)  -- 0.80 * 255
         else if isCircle
         then (if isCW then GestCircleCW else GestCircleCCW, 191)  -- 0.75
         else if detectGraspMotion frames count
         then (GestGraspClose, 204)
         else if detectReleaseMotion frames count
         then (GestReleaseOpen, 204)
         else if absVz > absVx && absVz > absVy
         then (if vz > 0 then GestReachForward else GestPullBack, 217)  -- 0.85
         else if absVy > absVx
         then (if vy > 0 then GestSwipeUp else GestSwipeDown, 204)
         else (if vx > 0 then GestSwipeRight else GestSwipeLeft, 204)

-- | Recognize gesture from buffer
recognizeGesture :: MotionBufferState -> GestureResult
recognizeGesture state
  | mbFrameCount state < minFrameWindow =
      GestureResult GestNone 0 (mbFrameCount state) 0 False
  | otherwise =
      let (gesture, confidence) = classifyGesture (mbFrames state) (mbFrameCount state)
          motionMag = computeMotionMagnitude (mbFrames state) (mbFrameCount state)
          isValid = confidence >= confidenceThreshold
      in GestureResult gesture confidence (mbFrameCount state) motionMag isValid

-- | Map gesture to control intent
gestureToIntent :: GestureType -> ControlIntent
gestureToIntent gesture = case gesture of
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

-- | Create motion frame from position
createMotionFrame
  :: LimbPosition      -- Current position
  -> LimbPosition      -- Previous position
  -> Unsigned 8        -- Frame index
  -> MotionFrame
createMotionFrame current prev idx =
  let dt = lpTimestamp current - lpTimestamp prev
      dtSafe = if dt == 0 then 1 else dt

      vx = resize ((lpX current - lpX prev) `div` resize dtSafe) :: Signed 8
      vy = resize ((lpY current - lpY prev) `div` resize dtSafe) :: Signed 8
      vz = resize ((lpZ current - lpZ prev) `div` resize dtSafe) :: Signed 8

  in MotionFrame current vx vy vz idx

-- | Add frame to buffer
addFrameToBuffer :: MotionBufferState -> MotionFrame -> MotionBufferState
addFrameToBuffer state frame =
  let newFrames = replace (mbWriteIdx state) frame (mbFrames state)
      newWriteIdx = if mbWriteIdx state >= 23 then 0 else mbWriteIdx state + 1
      newCount = if mbFrameCount state < mbTargetWindow
                 then mbFrameCount state + 1
                 else mbTargetWindow
  in state
       { mbFrames = newFrames
       , mbWriteIdx = newWriteIdx
       , mbFrameCount = newCount
       }

-- | Update adaptive window
updateAdaptiveWindow :: MotionBufferState -> MotionBufferState
updateAdaptiveWindow state =
  let -- Compute average speed
      totalMag = computeMotionMagnitude (mbFrames state) (mbFrameCount state)
      avgSpeed = if mbFrameCount state > 1
                 then resize (totalMag `div` resize (mbFrameCount state - 1)) :: Unsigned 8
                 else 0
      newWindow = computeAdaptiveWindow avgSpeed
  in state { mbTargetWindow = newWindow }

-- | Gesture recognition pipeline
gesturePipeline
  :: HiddenClockResetEnable dom
  => Signal dom LimbPosition
  -> Signal dom GestureResult
gesturePipeline input = mealy gestureMealy (initialBufferState, LimbPosition 0 0 0 0, 0) input
  where
    gestureMealy (state, prevPos, frameIdx) pos =
      let frame = createMotionFrame pos prevPos frameIdx
          newState = updateAdaptiveWindow $ addFrameToBuffer state frame
          result = recognizeGesture newState
          newFrameIdx = frameIdx + 1
      in ((newState, pos, newFrameIdx), result)

-- | Gesture to intent pipeline
gestureIntentPipeline
  :: HiddenClockResetEnable dom
  => Signal dom LimbPosition
  -> Signal dom (GestureResult, ControlIntent)
gestureIntentPipeline input =
  let results = gesturePipeline input
  in fmap (\r -> (r, gestureToIntent (grGesture r))) results
