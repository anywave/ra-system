{-|
Module      : RaInterfaceTactileControl
Description : Intent Detection from Biometric Signals
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 56: Intent detection from biometric signals for tactile interface
control. Converts coherence patterns, HRV spikes, and breath rate into
ControlIntents for digital twin manipulation and interface navigation.

Based on biometric combo detection (coherence + HRV + breath).
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaInterfaceTactileControl where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Coherence thresholds (scaled 0-255)
coherenceHigh :: Unsigned 8
coherenceHigh = 184   -- 0.72 * 255

coherenceMed :: Unsigned 8
coherenceMed = 128    -- 0.50 * 255

coherenceLow :: Unsigned 8
coherenceLow = 77     -- 0.30 * 255

-- | HRV spike threshold (15% = 38/255)
hrvSpikeThreshold :: Unsigned 8
hrvSpikeThreshold = 38

-- | Breath rate thresholds (breaths per minute)
breathFast :: Unsigned 8
breathFast = 18

breathSlow :: Unsigned 8
breathSlow = 8

-- | Intent confidence minimum (0.60 * 255)
intentConfidenceMin :: Unsigned 8
intentConfidenceMin = 153

-- | Control intent types
data ControlIntent
  = IntentNone
  | IntentReach     -- Extend/approach
  | IntentPull      -- Attract/gather
  | IntentPush      -- Repel/send
  | IntentGrasp     -- Hold/acquire
  | IntentRelease   -- Let go/drop
  | IntentMoveTo    -- Navigate to position
  | IntentHoverAt   -- Maintain position
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Coherence level classification
data CoherenceLevel
  = CohNone
  | CohLow
  | CohMedium
  | CohHigh
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Breath pattern classification
data BreathPattern
  = BreathFast
  | BreathNormal
  | BreathSlow
  | BreathModerate
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Biometric state
data BiometricState = BiometricState
  { bsCoherence   :: Unsigned 8   -- 0-255 coherence level
  , bsHRVCurrent  :: Unsigned 16  -- Current HRV (ms * 10)
  , bsHRVPrevious :: Unsigned 16  -- Previous HRV (ms * 10)
  , bsBreathRate  :: Unsigned 8   -- Breaths per minute
  , bsTimestamp   :: Unsigned 32  -- Time in ms
  } deriving (Generic, NFDataX)

-- | Intent result
data IntentResult = IntentResult
  { irIntent            :: ControlIntent
  , irConfidence        :: Unsigned 8   -- 0-255
  , irCoherenceComp     :: Unsigned 8   -- Coherence contribution
  , irHRVComp           :: Unsigned 8   -- HRV contribution
  , irBreathComp        :: Unsigned 8   -- Breath contribution
  , irIsValid           :: Bool         -- Meets confidence threshold
  } deriving (Generic, NFDataX)

-- | Intent history entry
data HistoryEntry = HistoryEntry
  { heIntent     :: ControlIntent
  , heConfidence :: Unsigned 8
  , heIsValid    :: Bool
  } deriving (Generic, NFDataX)

-- | Intent direction vector
data IntentVector = IntentVector
  { ivX :: Signed 8
  , ivY :: Signed 8
  , ivZ :: Signed 8
  } deriving (Generic, NFDataX)

-- | Compute HRV delta (as percentage * 255)
computeHRVDelta :: Unsigned 16 -> Unsigned 16 -> Unsigned 8
computeHRVDelta current previous
  | previous == 0 = 0
  | otherwise =
      let diff = if current > previous
                 then current - previous
                 else previous - current
          delta = (diff * 255) `div` previous
      in resize $ min 255 delta

-- | Check for HRV spike
isHRVSpike :: Unsigned 8 -> Bool
isHRVSpike delta = delta > hrvSpikeThreshold

-- | Classify coherence level
classifyCoherence :: Unsigned 8 -> CoherenceLevel
classifyCoherence coh
  | coh >= coherenceHigh = CohHigh
  | coh >= coherenceMed  = CohMedium
  | coh >= coherenceLow  = CohLow
  | otherwise            = CohNone

-- | Classify breath pattern
classifyBreath :: Unsigned 8 -> BreathPattern
classifyBreath rate
  | rate > breathFast  = BreathFast
  | rate < breathSlow  = BreathSlow
  | rate >= 10 && rate <= 16 = BreathNormal
  | otherwise          = BreathModerate

-- | Compute coherence component
-- Returns (weight, primary intent)
computeCoherenceComponent :: Unsigned 8 -> (Unsigned 8, ControlIntent)
computeCoherenceComponent coh =
  case classifyCoherence coh of
    CohHigh   -> (230, IntentGrasp)    -- 0.9 * 255
    CohMedium -> (153, IntentHoverAt)  -- 0.6 * 255
    CohLow    -> (77, IntentRelease)   -- 0.3 * 255
    CohNone   -> (0, IntentNone)

-- | Compute HRV component
-- Returns (weight, primary intent)
computeHRVComponent :: Unsigned 16 -> Unsigned 16 -> (Unsigned 8, ControlIntent)
computeHRVComponent current previous =
  let delta = computeHRVDelta current previous
  in if not (isHRVSpike delta)
     then (77, IntentHoverAt)   -- Stable -> maintain
     else if current > previous
          then (179, IntentRelease)  -- Increasing HRV -> calming
          else (179, IntentPush)     -- Decreasing HRV -> energizing

-- | Compute breath component
-- Returns (weight, primary intent)
computeBreathComponent :: Unsigned 8 -> (Unsigned 8, ControlIntent)
computeBreathComponent rate =
  case classifyBreath rate of
    BreathFast     -> (153, IntentPush)    -- Urgent actions
    BreathSlow     -> (153, IntentPull)    -- Receptive actions
    BreathNormal   -> (102, IntentHoverAt) -- Balanced
    BreathModerate -> (77, IntentHoverAt)  -- Transitional

-- | Combine intent signals
-- Picks highest weighted intent, boosts for multi-signal agreement
combineIntents
  :: (Unsigned 8, ControlIntent)  -- Coherence
  -> (Unsigned 8, ControlIntent)  -- HRV
  -> (Unsigned 8, ControlIntent)  -- Breath
  -> (ControlIntent, Unsigned 8)  -- (best intent, confidence)
combineIntents (cohW, cohI) (hrvW, hrvI) (brW, brI) =
  let -- Score each intent
      scoreIntent intent =
        let cohMatch = if cohI == intent then cohW else 0
            hrvMatch = if hrvI == intent then hrvW else 0
            brMatch  = if brI == intent then brW else 0
            baseScore = resize cohMatch + resize hrvMatch + resize brMatch :: Unsigned 16

            -- Count matches for boost
            matchCount = (if cohI == intent then 1 else 0)
                       + (if hrvI == intent then 1 else 0)
                       + (if brI == intent then 1 else 0) :: Unsigned 2

            -- Boost for multi-signal agreement (20% per extra signal)
            boost = if matchCount > 1
                    then baseScore + (baseScore `shiftR` 2) * resize (matchCount - 1)
                    else baseScore

        in resize $ min 255 boost :: Unsigned 8

      -- Score all intents
      scores = $(listToVecTH
        [ IntentNone, IntentReach, IntentPull, IntentPush
        , IntentGrasp, IntentRelease, IntentMoveTo, IntentHoverAt ])

      -- Find max (simplified: check each)
      findMax :: (ControlIntent, Unsigned 8) -> ControlIntent -> (ControlIntent, Unsigned 8)
      findMax (bestI, bestS) i =
        let s = scoreIntent i
        in if s > bestS && i /= IntentNone
           then (i, s)
           else (bestI, bestS)

      (bestIntent, maxScore) = foldl findMax (IntentNone, 0) scores

      -- Normalize confidence (max possible ~640 -> scale to 255)
      confidence = (resize maxScore * 100) `shiftR` 8 :: Unsigned 16

  in (bestIntent, resize $ min 255 confidence)

-- | Detect intent from biometric state
detectIntent :: BiometricState -> IntentResult
detectIntent bio =
  let cohComp = computeCoherenceComponent (bsCoherence bio)
      hrvComp = computeHRVComponent (bsHRVCurrent bio) (bsHRVPrevious bio)
      brComp = computeBreathComponent (bsBreathRate bio)

      (intent, confidence) = combineIntents cohComp hrvComp brComp

      isValid = confidence >= intentConfidenceMin

  in IntentResult
       intent
       confidence
       (fst cohComp)
       (fst hrvComp)
       (fst brComp)
       isValid

-- | Convert intent to direction vector
intentToVector :: ControlIntent -> IntentVector
intentToVector intent = case intent of
  IntentNone    -> IntentVector 0 0 0
  IntentReach   -> IntentVector 127 0 0      -- Forward
  IntentPull    -> IntentVector (-127) 0 0   -- Backward
  IntentPush    -> IntentVector 0 127 0      -- Outward
  IntentGrasp   -> IntentVector 0 0 127      -- Inward/up
  IntentRelease -> IntentVector 0 0 (-127)   -- Outward/down
  IntentMoveTo  -> IntentVector 64 64 0      -- Diagonal forward
  IntentHoverAt -> IntentVector 0 0 0        -- Stationary

-- | Compute intent magnitude
-- Scales with confidence and φ
computeMagnitude :: IntentResult -> Unsigned 8
computeMagnitude result
  | not (irIsValid result) = 0
  | otherwise =
      let conf = irConfidence result
          -- Scale by φ/2 ≈ 0.809
          scaled = (resize conf * phi16) `shiftR` 11 :: Unsigned 16
      in resize $ min 255 scaled

-- | Intent history state
data IntentHistoryState = IntentHistoryState
  { ihEntries   :: Vec 8 HistoryEntry
  , ihCount     :: Unsigned 4
  , ihWriteIdx  :: Unsigned 3
  } deriving (Generic, NFDataX)

-- | Initial history state
initialHistoryState :: IntentHistoryState
initialHistoryState = IntentHistoryState
  (repeat (HistoryEntry IntentNone 0 False))
  0
  0

-- | Add result to history
addToHistory :: IntentHistoryState -> IntentResult -> IntentHistoryState
addToHistory state result =
  let entry = HistoryEntry (irIntent result) (irConfidence result) (irIsValid result)
      newEntries = replace (ihWriteIdx state) entry (ihEntries state)
      newCount = if ihCount state < 8 then ihCount state + 1 else 8
      newIdx = ihWriteIdx state + 1
  in state
       { ihEntries = newEntries
       , ihCount = newCount
       , ihWriteIdx = resize newIdx
       }

-- | Get dominant intent from history
getDominantIntent :: IntentHistoryState -> ControlIntent
getDominantIntent state =
  let -- Count valid intents
      countIntent :: Vec 8 (Unsigned 4) -> HistoryEntry -> Vec 8 (Unsigned 4)
      countIntent counts entry
        | not (heIsValid entry) = counts
        | otherwise =
            let idx = resize (pack (heIntent entry)) :: Unsigned 3
                oldCount = counts !! idx
            in replace idx (oldCount + 1) counts

      counts = foldl countIntent (repeat 0) (ihEntries state)

      -- Find max (skip NONE at index 0)
      findMax :: (ControlIntent, Unsigned 4) -> Unsigned 3 -> (ControlIntent, Unsigned 4)
      findMax (bestI, bestC) idx =
        let c = counts !! idx
        in if c > bestC && idx > 0
           then (unpack (resize idx), c)
           else (bestI, bestC)

      (dominant, _) = foldl findMax (IntentNone, 0) $(listToVecTH [0..7 :: Unsigned 3])

  in dominant

-- | Intent detection pipeline
intentPipeline
  :: HiddenClockResetEnable dom
  => Signal dom BiometricState
  -> Signal dom IntentResult
intentPipeline input = detectIntent <$> input

-- | Intent pipeline with history
intentPipelineWithHistory
  :: HiddenClockResetEnable dom
  => Signal dom BiometricState
  -> Signal dom (IntentResult, ControlIntent)  -- (current, dominant)
intentPipelineWithHistory input = mealy historyMealy initialHistoryState input
  where
    historyMealy state bio =
      let result = detectIntent bio
          newState = addToHistory state result
          dominant = getDominantIntent newState
      in (newState, (result, dominant))

-- | Vector output pipeline
vectorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom IntentResult
  -> Signal dom (IntentVector, Unsigned 8)  -- (direction, magnitude)
vectorPipeline input = convert <$> input
  where
    convert result =
      let vec = intentToVector (irIntent result)
          mag = computeMagnitude result
      in (vec, mag)
