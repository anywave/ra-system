{-|
Module      : RaChamberSync
Description : Multi-Chamber Resonance Synchronization
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 40: Synchronizes coherence across linked resonance chambers
with phi-based distance falloff and temporal phase propagation.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaChamberSync where

import Clash.Prelude

-- | Phi constant scaled to 16-bit fixed point (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Sync threshold (0.72 * 255 = 184)
syncThreshold :: Unsigned 8
syncThreshold = 184

-- | Maximum propagation distance
maxPropDistance :: Unsigned 4
maxPropDistance = 7

-- | Sync state
data SyncState = Isolated | Seeking | Syncing | Synchronized | Desync
  deriving (Generic, NFDataX, Eq, Show)

-- | Link type
data LinkType = DirectLink | ResonantLink | ScalarLink | TemporalLink
  deriving (Generic, NFDataX, Eq, Show)

-- | Chamber node
data ChamberNode = ChamberNode
  { cnChamberId     :: Unsigned 8
  , cnBaseFrequency :: Unsigned 16    -- Hz
  , cnCoherence     :: Unsigned 8     -- 0-255 (scaled 0.0-1.0)
  , cnPhaseAngle    :: Unsigned 16    -- 0-65535 (0-2π scaled)
  , cnSyncState     :: SyncState
  , cnHarmonicOrder :: Unsigned 4     -- 1-15
  , cnPositionX     :: Signed 16
  , cnPositionY     :: Signed 16
  , cnPositionZ     :: Signed 16
  } deriving (Generic, NFDataX)

-- | Sync link between chambers
data SyncLink = SyncLink
  { slSourceId     :: Unsigned 8
  , slTargetId     :: Unsigned 8
  , slLinkType     :: LinkType
  , slLinkStrength :: Unsigned 8     -- 0-255
  , slDistance     :: Unsigned 4     -- Hop distance
  , slPhaseOffset  :: Signed 16      -- Phase delay
  , slLastSyncTime :: Unsigned 32    -- Timestamp
  } deriving (Generic, NFDataX)

-- | Phase propagation event
data PhaseEvent = PhaseEvent
  { peSourceId       :: Unsigned 8
  , peTargetId       :: Unsigned 8
  , pePhaseDelta     :: Signed 16
  , peCoherencePulse :: Unsigned 8
  , peTimestamp      :: Unsigned 32
  , peHopsRemaining  :: Unsigned 4
  } deriving (Generic, NFDataX)

-- | Compute resonance score with phi^(-distance) falloff
-- Returns 0-255 score
computeResonanceScore :: ChamberNode -> ChamberNode -> Unsigned 4 -> Unsigned 8
computeResonanceScore n1 n2 hopDistance =
  let -- Harmonic factor (closer orders = higher resonance)
      harmonicDiff = if cnHarmonicOrder n1 > cnHarmonicOrder n2
                     then cnHarmonicOrder n1 - cnHarmonicOrder n2
                     else cnHarmonicOrder n2 - cnHarmonicOrder n1
      harmonicFactor = 255 `div` max 1 (1 + resize harmonicDiff * 64) :: Unsigned 8

      -- Coherence product (both need high coherence)
      coherenceProduct = (resize (cnCoherence n1) * resize (cnCoherence n2)) `shiftR` 8 :: Unsigned 8

      -- Phi falloff: phi^(-d) ≈ (1/phi)^d
      -- Approximate 1/phi = 0.618 as 158/256
      phiFalloff = case hopDistance of
        0 -> 255
        1 -> 158  -- 0.618
        2 -> 98   -- 0.382
        3 -> 60   -- 0.236
        4 -> 37   -- 0.146
        5 -> 23   -- 0.090
        _ -> 14   -- ~0.056

      -- Combined score
      score = (resize harmonicFactor * resize coherenceProduct * resize phiFalloff) `shiftR` 16 :: Unsigned 16
  in resize $ min 255 score

-- | Check if frequencies are harmonically compatible
frequencyMatch :: Unsigned 16 -> Unsigned 16 -> Bool
frequencyMatch f1 f2
  | f1 == 0 || f2 == 0 = False
  | otherwise =
      let larger = max f1 f2
          smaller = max 1 (min f1 f2)
          ratio = (larger * 100) `div` smaller
      in ratio == 100 || ratio == 150 || ratio == 200 || ratio == 300 ||
         ratio == 162 || ratio == 262  -- φ and φ² ratios

-- | Check if two chambers can sync
canSync :: ChamberNode -> ChamberNode -> Bool
canSync n1 n2 =
  cnCoherence n1 >= syncThreshold &&
  cnCoherence n2 >= syncThreshold &&
  frequencyMatch (cnBaseFrequency n1) (cnBaseFrequency n2)

-- | Update sync state based on coherence and neighbors
updateSyncState :: Unsigned 8 -> Unsigned 4 -> SyncState
updateSyncState coherence neighborCount
  | coherence < syncThreshold = Isolated
  | neighborCount == 0 = Seeking
  | neighborCount >= 2 = Synchronized
  | otherwise = Syncing

-- | Compute link strength between nodes
computeLinkStrength :: ChamberNode -> ChamberNode -> Unsigned 4 -> Unsigned 8
computeLinkStrength = computeResonanceScore

-- | Attenuate phase delta with phi falloff
attenuatePhase :: Signed 16 -> Unsigned 4 -> Signed 16
attenuatePhase phaseDelta distance =
  let attenuation = case distance of
        0 -> 256
        1 -> 158
        2 -> 98
        3 -> 60
        _ -> 37
  in (phaseDelta * resize attenuation) `shiftR` 8

-- | Apply phase update to node
applyPhaseUpdate :: ChamberNode -> Signed 16 -> ChamberNode
applyPhaseUpdate node phaseDelta =
  let newPhase = resize (cnPhaseAngle node) + phaseDelta :: Signed 17
      normalized = if newPhase < 0 then resize (newPhase + 65536)
                   else if newPhase >= 65536 then resize (newPhase - 65536)
                   else resize newPhase
  in node { cnPhaseAngle = normalized }

-- | Compute global sync score for a set of link strengths
computeGlobalSync :: Vec n (Unsigned 8) -> Unsigned 8
computeGlobalSync strengths =
  let total = fold (+) (map resize strengths :: Vec n (Unsigned 16))
      count = length strengths
  in if count == 0 then 0 else resize (total `div` resize count)

-- | Create phase event for propagation
createPhaseEvent :: Unsigned 8 -> Unsigned 8 -> Signed 16 -> Unsigned 8 -> Unsigned 32 -> Unsigned 4 -> PhaseEvent
createPhaseEvent src tgt phase coh ts hops = PhaseEvent
  { peSourceId = src
  , peTargetId = tgt
  , pePhaseDelta = phase
  , peCoherencePulse = coh
  , peTimestamp = ts
  , peHopsRemaining = hops
  }

-- | Process phase event (attenuate and prepare for next hop)
processPhaseEvent :: PhaseEvent -> Maybe PhaseEvent
processPhaseEvent evt
  | peHopsRemaining evt == 0 = Nothing
  | abs (pePhaseDelta evt) < 256 = Nothing  -- Below threshold
  | otherwise = Just $ evt
      { pePhaseDelta = attenuatePhase (pePhaseDelta evt) 1
      , peCoherencePulse = (peCoherencePulse evt * 158) `shiftR` 8
      , peHopsRemaining = peHopsRemaining evt - 1
      }

-- | Initialize chamber node
initChamberNode :: Unsigned 8 -> Unsigned 16 -> ChamberNode
initChamberNode nodeId freq = ChamberNode
  { cnChamberId = nodeId
  , cnBaseFrequency = freq
  , cnCoherence = 0
  , cnPhaseAngle = 0
  , cnSyncState = Isolated
  , cnHarmonicOrder = 1
  , cnPositionX = 0
  , cnPositionY = 0
  , cnPositionZ = 0
  }

-- | Initialize sync link
initSyncLink :: Unsigned 8 -> Unsigned 8 -> Unsigned 4 -> SyncLink
initSyncLink src tgt dist = SyncLink
  { slSourceId = src
  , slTargetId = tgt
  , slLinkType = ResonantLink
  , slLinkStrength = 0
  , slDistance = dist
  , slPhaseOffset = 0
  , slLastSyncTime = 0
  }

-- | Resonance score pipeline
resonanceScorePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ChamberNode, ChamberNode, Unsigned 4)
  -> Signal dom (Unsigned 8)
resonanceScorePipeline input = (\(n1, n2, d) -> computeResonanceScore n1 n2 d) <$> input

-- | Sync check pipeline
syncCheckPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ChamberNode, ChamberNode)
  -> Signal dom Bool
syncCheckPipeline input = uncurry canSync <$> input

-- | Phase propagation pipeline
phasePropagationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ChamberNode, Signed 16)
  -> Signal dom ChamberNode
phasePropagationPipeline input = uncurry applyPhaseUpdate <$> input
