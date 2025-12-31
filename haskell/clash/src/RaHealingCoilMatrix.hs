{-|
Module      : RaHealingCoilMatrix
Description : Cellular Healing Coil Matrix
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 70: Constructs coil arrangements for localized cellular healing
based on harmonic field data. Uses α thresholds for coil spacing and
supports mirror coil entanglement with biometric or quantum sync.

References HUBBARD_COIL_GENERATOR and ELECTROCULTURE_PARAMETERS.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaHealingCoilMatrix where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Alpha thresholds (scaled to 255)
alphaWideThreshold :: Unsigned 8
alphaWideThreshold = 77     -- 0.3 * 255

alphaDenseThreshold :: Unsigned 8
alphaDenseThreshold = 179   -- 0.7 * 255

-- | Sync methods
data SyncMethod
  = SyncBiometric    -- HRV latency cycles
  | SyncQuantum      -- Zero-latency entangled
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Phase envelope types
data PhaseEnvelope
  = PhaseConstant
  | PhaseRising
  | PhaseFalling
  | PhaseOscillating
  | PhasePulsed
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Tissue types
data TissueType
  = TissueHeart
  | TissueLiver
  | TissueKidney
  | TissueBrain
  | TissueLungs
  | TissueMuscle
  | TissueBone
  | TissueSkin
  | TissueBlood
  | TissueNerve
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Coil orientation
data CoilOrientation
  = OrientToroidal
  | OrientSpherical
  | OrientPlanar
  | OrientBilateral
  | OrientVortex
  | OrientLinear
  | OrientSpiral
  | OrientAxial
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Biometric state
data BioState = BioState
  { bsHRV        :: Unsigned 8
  , bsCoherence  :: Unsigned 8
  , bsLatencyMs  :: Unsigned 8   -- HRV latency in ms / 4
  } deriving (Generic, NFDataX, Eq, Show)

-- | Tissue template
data TissueTemplate = TissueTemplate
  { ttBaseTurns    :: Unsigned 8
  , ttSpacingFactor :: Unsigned 16  -- Scaled by 1024
  , ttOrientation  :: CoilOrientation
  , ttResonanceFreq :: Unsigned 16  -- Hz * 256
  } deriving (Generic, NFDataX, Eq, Show)

-- | Coil matrix parameters
data CoilMatrix = CoilMatrix
  { cmTissue         :: TissueType
  , cmTurns          :: Unsigned 8
  , cmSpacingDensity :: Unsigned 8   -- 0-255 (0.0-1.0)
  , cmPhaseEnvelope  :: PhaseEnvelope
  , cmGain           :: Unsigned 8
  , cmSyncMethod     :: SyncMethod
  , cmHasEntangled   :: Bool
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute spacing density from alpha (linear interpolation)
computeSpacingFromAlpha :: Unsigned 8 -> Unsigned 8
computeSpacingFromAlpha alpha
  | alpha <= alphaWideThreshold  = 51   -- 0.2 * 255
  | alpha >= alphaDenseThreshold = 255  -- 1.0 * 255
  | otherwise =
      let -- Linear interpolation
          range = alphaDenseThreshold - alphaWideThreshold  -- 102
          offset = alpha - alphaWideThreshold
          -- t = offset / range, spacing = 51 + t * 204
          t = (resize offset * 204) `div` resize range :: Unsigned 16
      in 51 + resize t

-- | Get tissue template
getTissueTemplate :: TissueType -> TissueTemplate
getTissueTemplate tissue = case tissue of
  TissueHeart  -> TissueTemplate 12 phi16 OrientToroidal 2004
  TissueLiver  -> TissueTemplate 8 1229 OrientPlanar 10240
  TissueKidney -> TissueTemplate 8 1024 OrientPlanar 5120
  TissueBrain  -> TissueTemplate 21 phi16 OrientSpherical 2560
  TissueLungs  -> TissueTemplate 13 1448 OrientBilateral 18432
  TissueMuscle -> TissueTemplate 5 1024 OrientLinear 12800
  TissueBone   -> TissueTemplate 3 819 OrientSpiral 6400
  TissueSkin   -> TissueTemplate 8 1126 OrientPlanar 7680
  TissueBlood  -> TissueTemplate 13 phi16 OrientVortex 396800
  TissueNerve  -> TissueTemplate 21 phi16 OrientAxial 25600

-- | Compute turn density from base and spacing
computeTurnDensity :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
computeTurnDensity baseTurns spacingDensity =
  let -- multiplier = 0.5 + (density * 1.0 / 255)
      -- scaled: 128 + density
      multiplier = 128 + resize spacingDensity :: Unsigned 16
      result = (resize baseTurns * multiplier) `shiftR` 8
  in max 1 (resize result)

-- | Select phase envelope
selectPhaseEnvelope :: Unsigned 8 -> TissueType -> PhaseEnvelope
selectPhaseEnvelope coherence tissue
  | coherence > 204 = PhaseConstant
  | tissue == TissueHeart || tissue == TissueBlood = PhasePulsed
  | tissue == TissueBrain = PhaseOscillating
  | tissue == TissueMuscle || tissue == TissueBone = PhaseRising
  | coherence < 102 = PhaseRising
  | otherwise = PhaseOscillating

-- | Compute stabilization gain
computeGain :: BioState -> Unsigned 8 -> Unsigned 8
computeGain bio spacingDensity =
  let -- HRV factor: 0.5 + (hrv * 0.5 / 255) -> 128 + hrv/2
      hrvFactor = 128 + (bsHRV bio `shiftR` 1) :: Unsigned 16

      -- Density factor: 0.8 + (density * 0.4 / 255) -> 205 + density*0.4
      densityFactor = 205 + (resize spacingDensity `shiftR` 2) :: Unsigned 16

      -- Coherence factor: 0.7 + (coherence * 0.6 / 255)
      cohFactor = 179 + (resize (bsCoherence bio) * 153 `shiftR` 8) :: Unsigned 16

      -- Combine (all scaled by 256)
      combined = (hrvFactor * densityFactor * cohFactor) `shiftR` 16

  in min 255 (resize combined)

-- | Select sync method
selectSyncMethod :: BioState -> Bool -> SyncMethod
selectSyncMethod bio hasEntangled
  | not hasEntangled = SyncBiometric
  | bsCoherence bio > 179 && bsLatencyMs bio < 13 = SyncQuantum  -- ~50ms
  | otherwise = SyncBiometric

-- | Generate coil matrix
generateCoilMatrix
  :: Unsigned 8        -- Mean alpha
  -> BioState
  -> TissueType
  -> Bool              -- Create entangled pair?
  -> CoilMatrix
generateCoilMatrix meanAlpha bio tissue createPair =
  let template = getTissueTemplate tissue
      spacingDensity = computeSpacingFromAlpha meanAlpha
      turns = computeTurnDensity (ttBaseTurns template) spacingDensity
      phase = selectPhaseEnvelope (bsCoherence bio) tissue
      gain = computeGain bio spacingDensity
      syncMethod = selectSyncMethod bio createPair
  in CoilMatrix tissue turns spacingDensity phase gain syncMethod createPair

-- | Compute latency offset for biometric sync (scaled ms)
computeLatencyOffset :: CoilMatrix -> BioState -> Unsigned 8
computeLatencyOffset matrix bio
  | cmSyncMethod matrix == SyncQuantum = 0
  | otherwise = bsLatencyMs bio  -- Direct pass-through

-- | Validate coil matrix
validateCoilMatrix :: CoilMatrix -> Bool
validateCoilMatrix matrix =
  cmTurns matrix > 0 &&
  cmGain matrix > 0 &&
  cmSpacingDensity matrix >= 26  -- Min 0.1

-- | Coil matrix pipeline input
data CoilInput = CoilInput
  { ciMeanAlpha    :: Unsigned 8
  , ciBioState     :: BioState
  , ciTissue       :: TissueType
  , ciCreatePair   :: Bool
  } deriving (Generic, NFDataX)

-- | Coil matrix pipeline
coilMatrixPipeline
  :: HiddenClockResetEnable dom
  => Signal dom CoilInput
  -> Signal dom CoilMatrix
coilMatrixPipeline input =
  let mkMatrix inp = generateCoilMatrix
        (ciMeanAlpha inp)
        (ciBioState inp)
        (ciTissue inp)
        (ciCreatePair inp)
  in mkMatrix <$> input

-- | Spacing computation pipeline
spacingPipeline
  :: HiddenClockResetEnable dom
  => Signal dom Unsigned 8
  -> Signal dom Unsigned 8
spacingPipeline = fmap computeSpacingFromAlpha

-- | Gain computation pipeline
gainPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BioState, Unsigned 8)
  -> Signal dom Unsigned 8
gainPipeline = fmap (uncurry computeGain)
