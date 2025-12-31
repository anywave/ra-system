{-|
Module      : RaEntangleBioMap
Description : Biometric-Fragment Entanglement Graphs
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 45: Dynamic entanglement graphs linking biometric signals,
scalar zones, memory fragments, and chamber nodes.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaEntangleBioMap where

import Clash.Prelude

-- | Phi constant scaled
phi16 :: Unsigned 16
phi16 = 1657

-- | Entanglement threshold (0.72 * 255 = 184)
entangleThreshold :: Unsigned 8
entangleThreshold = 184

-- | Coherence delta threshold (0.05 * 255 = 13)
coherenceDeltaThreshold :: Unsigned 8
coherenceDeltaThreshold = 13

-- | Phi cycle ticks (φ * 1000 = 1618)
phiCycleTicks :: Unsigned 16
phiCycleTicks = 1618

-- | Body zone enumeration (8 zones)
data BodyZone
  = ZoneCrown
  | ZoneThirdEye
  | ZoneThroat
  | ZoneHeart
  | ZoneGut
  | ZoneSacral
  | ZoneRoot
  | ZoneBreath
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Phase code for harmonic matching
data PhaseCode = PhaseCode
  { pcHarmonicL   :: Unsigned 4    -- 0-9
  , pcHarmonicM   :: Signed 8      -- -9 to +9
  , pcAmplitude   :: Unsigned 8    -- 0-255
  , pcPhaseAngle  :: Unsigned 16   -- 0-65535 (0-2π)
  } deriving (Generic, NFDataX)

-- | Ra coordinate
data RaCoordinate = RaCoordinate
  { rcTheta :: Unsigned 5
  , rcPhi   :: Unsigned 3
  , rcH     :: Unsigned 3
  } deriving (Generic, NFDataX, Eq)

-- | Bio node (one per body zone)
data BioNode = BioNode
  { bnZone            :: BodyZone
  , bnCoherence       :: Unsigned 8
  , bnPhaseSignature  :: PhaseCode
  , bnLastUpdateTick  :: Unsigned 16
  } deriving (Generic, NFDataX)

-- | Entangled fragment
data EntangledFragment = EntangledFragment
  { efFragmentId    :: Unsigned 8
  , efAnchorCoord   :: RaCoordinate
  , efResonanceScore :: Unsigned 8
  , efActiveLink    :: Bool
  , efPhaseCode     :: PhaseCode
  } deriving (Generic, NFDataX)

-- | Entanglement link
data EntanglementLink = EntanglementLink
  { elBioZone     :: BodyZone
  , elFragmentId  :: Unsigned 8
  , elStrength    :: Unsigned 8
  , elActive      :: Bool
  , elLastSyncTick :: Unsigned 16
  } deriving (Generic, NFDataX)

-- | Update trigger type
data UpdateTrigger
  = TriggerCoherenceDelta
  | TriggerPhiCycle
  | TriggerFragmentChange
  | TriggerManual
  deriving (Generic, NFDataX, Eq, Show)

-- | Create phase code with bounds checking
createPhaseCode :: Unsigned 4 -> Signed 8 -> Unsigned 8 -> Unsigned 16 -> PhaseCode
createPhaseCode l m amp phase = PhaseCode
  { pcHarmonicL = min 9 l
  , pcHarmonicM = max (-resize (min 9 l)) (min (resize (min 9 l)) m)
  , pcAmplitude = amp
  , pcPhaseAngle = phase
  }

-- | Compute phase code similarity (0-255)
phaseSimilarity :: PhaseCode -> PhaseCode -> Unsigned 8
phaseSimilarity p1 p2 =
  let lDiff = if pcHarmonicL p1 > pcHarmonicL p2
              then pcHarmonicL p1 - pcHarmonicL p2
              else pcHarmonicL p2 - pcHarmonicL p1
      mDiff = abs (pcHarmonicM p1 - pcHarmonicM p2)

      -- L similarity: 255 / (1 + diff)
      lSim = 255 `div` (1 + resize lDiff) :: Unsigned 8
      -- M similarity
      mSim = 255 `div` (1 + resize mDiff) :: Unsigned 8

      -- Phase similarity (circular)
      pDiff = if pcPhaseAngle p1 > pcPhaseAngle p2
              then pcPhaseAngle p1 - pcPhaseAngle p2
              else pcPhaseAngle p2 - pcPhaseAngle p1
      pDiffNorm = min pDiff (65536 - pDiff)
      pSim = 255 - resize (pDiffNorm `shiftR` 8) :: Unsigned 8

      -- Amplitude similarity
      aDiff = if pcAmplitude p1 > pcAmplitude p2
              then pcAmplitude p1 - pcAmplitude p2
              else pcAmplitude p2 - pcAmplitude p1
      aSim = 255 - aDiff

      -- Weighted: 30% L, 20% M, 30% phase, 20% amplitude
      weighted = (resize lSim * 3 + resize mSim * 2 +
                  resize pSim * 3 + resize aSim * 2) `div` 10 :: Unsigned 16

  in resize $ min 255 weighted

-- | Check if phase codes match
phaseMatches :: PhaseCode -> PhaseCode -> Unsigned 4 -> Unsigned 4 -> Bool
phaseMatches p1 p2 lTol mTol =
  let lDiff = if pcHarmonicL p1 > pcHarmonicL p2
              then pcHarmonicL p1 - pcHarmonicL p2
              else pcHarmonicL p2 - pcHarmonicL p1
      mDiff = abs (pcHarmonicM p1 - pcHarmonicM p2)
  in lDiff <= lTol && resize mDiff <= mTol

-- | Create bio node
createBioNode :: BodyZone -> Unsigned 8 -> BioNode
createBioNode zone coh = BioNode
  { bnZone = zone
  , bnCoherence = coh
  , bnPhaseSignature = createPhaseCode (resize $ fromEnum zone `mod` 5) 0 coh 0
  , bnLastUpdateTick = 0
  }

-- | Update bio node coherence
updateBioNode :: BioNode -> Unsigned 8 -> Unsigned 16
              -> (BioNode, Bool)  -- (updated node, should trigger)
updateBioNode node newCoh tick =
  let delta = if newCoh > bnCoherence node
              then newCoh - bnCoherence node
              else bnCoherence node - newCoh
      shouldTrigger = delta > coherenceDeltaThreshold
      newNode = BioNode
        { bnZone = bnZone node
        , bnCoherence = newCoh
        , bnPhaseSignature = bnPhaseSignature node
        , bnLastUpdateTick = tick
        }
  in (newNode, shouldTrigger)

-- | Create entangled fragment
createFragment :: Unsigned 8 -> RaCoordinate -> Unsigned 8 -> EntangledFragment
createFragment fid anchor res =
  let phase = createPhaseCode
                (rcH anchor)
                (resize (rcPhi anchor) - 3)
                res
                (resize (rcTheta anchor) * 2427)  -- Scale to 0-65535
  in EntangledFragment
    { efFragmentId = fid
    , efAnchorCoord = anchor
    , efResonanceScore = res
    , efActiveLink = res >= entangleThreshold
    , efPhaseCode = phase
    }

-- | Compute link strength between bio node and fragment
computeLinkStrength :: BioNode -> EntangledFragment -> Unsigned 8
computeLinkStrength bio frag =
  let phaseSim = phaseSimilarity (bnPhaseSignature bio) (efPhaseCode frag)
      cohProduct = (resize (bnCoherence bio) * resize (efResonanceScore frag)) `shiftR` 8 :: Unsigned 8
      -- 40% phase, 60% coherence product
      weighted = (resize phaseSim * 4 + resize cohProduct * 6) `div` 10 :: Unsigned 16
  in resize $ min 255 weighted

-- | Check if link should be active
shouldLinkActivate :: Unsigned 8 -> Bool
shouldLinkActivate strength = strength >= entangleThreshold

-- | Create entanglement link
createLink :: BioNode -> EntangledFragment -> Unsigned 16 -> EntanglementLink
createLink bio frag tick =
  let strength = computeLinkStrength bio frag
  in EntanglementLink
    { elBioZone = bnZone bio
    , elFragmentId = efFragmentId frag
    , elStrength = strength
    , elActive = shouldLinkActivate strength
    , elLastSyncTick = tick
    }

-- | Check if phi cycle update needed
checkPhiCycleUpdate :: Unsigned 16 -> Unsigned 16 -> Bool
checkPhiCycleUpdate lastTick currentTick =
  currentTick - lastTick >= phiCycleTicks

-- | Compute overall coherence from bio nodes (average)
computeOverallCoherence :: Vec 8 BioNode -> Unsigned 8
computeOverallCoherence nodes =
  let total = foldl (\acc n -> acc + resize (bnCoherence n)) 0 nodes :: Unsigned 16
  in resize (total `div` 8)

-- | Count active links
countActiveLinks :: Vec n EntanglementLink -> Unsigned 8
countActiveLinks links =
  foldl (\acc l -> if elActive l then acc + 1 else acc) 0 links

-- | Bio node update pipeline
bioNodePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BioNode, Unsigned 8, Unsigned 16)
  -> Signal dom (BioNode, Bool)
bioNodePipeline input = (\(n, c, t) -> updateBioNode n c t) <$> input

-- | Link strength pipeline
linkStrengthPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BioNode, EntangledFragment)
  -> Signal dom Unsigned 8
linkStrengthPipeline input = uncurry computeLinkStrength <$> input

-- | Phase similarity pipeline
phaseSimilarityPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (PhaseCode, PhaseCode)
  -> Signal dom (Unsigned 8)
phaseSimilarityPipeline input = uncurry phaseSimilarity <$> input

-- | Phi cycle check pipeline
phiCyclePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 16, Unsigned 16)
  -> Signal dom Bool
phiCyclePipeline input = uncurry checkPhiCycleUpdate <$> input
