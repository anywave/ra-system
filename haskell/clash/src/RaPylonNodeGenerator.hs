{-|
Module      : RaPylonNodeGenerator
Description : Scalar Pylon Node Generators
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 74: Implements configurable scalar pylon nodes with shape-based
amplification, frequency resonance linking, and teleport anchor generation.

Amplification factors: Golod→φ, Tesla→2.0, Hexagonal→√2, Spherical→1.0
Teleport anchor requires BOTH α>0.9 AND HRV coherence lock.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaPylonNodeGenerator where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Sqrt(2) scaled (1.414 * 1024)
sqrt2_16 :: Unsigned 16
sqrt2_16 = 1448

-- | Amplification thresholds
alphaTeleportThreshold :: Unsigned 8
alphaTeleportThreshold = 230   -- 0.9 * 255

hrvCoherenceThreshold :: Unsigned 8
hrvCoherenceThreshold = 204    -- 0.8 * 255

-- | Frequency resonance tolerance (5%)
resonanceTolerance :: Unsigned 8
resonanceTolerance = 13        -- 0.05 * 255

-- | Pylon shape types
data PylonShape
  = ShapeGolodPyramid    -- φ amplification (1.618)
  | ShapeTeslaCoil       -- 2.0 amplification
  | ShapeHexagonalArray  -- √2 amplification (1.414)
  | ShapeSpherical       -- 1.0 amplification (baseline)
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Pylon status
data PylonStatus
  = StatusOffline
  | StatusStandby
  | StatusActive
  | StatusAmplifying
  | StatusTeleportReady
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Anchor type
data AnchorType
  = AnchorNone
  | AnchorLocal         -- Local resonance anchor
  | AnchorLeyline       -- Leyline-linked anchor
  | AnchorTeleport      -- Full teleport anchor
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | 3D coordinate (8.8 fixed point)
data RaCoordinate = RaCoordinate
  { rcX :: Signed 16
  , rcY :: Signed 16
  , rcZ :: Signed 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Biometric state
data BioState = BioState
  { bsHRV        :: Unsigned 8
  , bsCoherence  :: Unsigned 8
  , bsFocusLevel :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Frequency resonance data
data FrequencyResonance = FrequencyResonance
  { frBaseFreq     :: Unsigned 16   -- Hz * 256
  , frHarmonic     :: Unsigned 4    -- Harmonic number (1-15)
  , frPhase        :: Unsigned 16   -- Phase angle * 1024
  } deriving (Generic, NFDataX, Eq, Show)

-- | Leyline link data
data LeylineLink = LeylineLink
  { llIsLinked    :: Bool
  , llProximity   :: Unsigned 8     -- Distance factor (0=at line, 255=far)
  , llStrength    :: Unsigned 8     -- Link strength
  } deriving (Generic, NFDataX, Eq, Show)

-- | Pylon configuration
data PylonConfig = PylonConfig
  { pcShape        :: PylonShape
  , pcPosition     :: RaCoordinate
  , pcBaseFreq     :: Unsigned 16
  , pcPowerLevel   :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Pylon node
data PylonNode = PylonNode
  { pnConfig       :: PylonConfig
  , pnStatus       :: PylonStatus
  , pnAmplification :: Unsigned 16  -- * 1024
  , pnAnchorType   :: AnchorType
  , pnOutputPower  :: Unsigned 16   -- Amplified output
  } deriving (Generic, NFDataX, Eq, Show)

-- | Get amplification factor for shape (scaled by 1024)
getAmplificationFactor :: PylonShape -> Unsigned 16
getAmplificationFactor shape = case shape of
  ShapeGolodPyramid   -> phi16      -- 1.618 * 1024 = 1657
  ShapeTeslaCoil      -> 2048       -- 2.0 * 1024
  ShapeHexagonalArray -> sqrt2_16   -- 1.414 * 1024 = 1448
  ShapeSpherical      -> 1024       -- 1.0 * 1024

-- | Compute amplified output
computeAmplifiedOutput :: Unsigned 8 -> Unsigned 16 -> Unsigned 16
computeAmplifiedOutput powerLevel ampFactor =
  let -- output = power * factor / 1024
      product = resize powerLevel * resize ampFactor :: Unsigned 32
      result = product `shiftR` 10
  in resize (min 65535 result)

-- | Check frequency resonance
checkFrequencyResonance
  :: Unsigned 16  -- Pylon frequency
  -> Unsigned 16  -- Target frequency
  -> Bool
checkFrequencyResonance pylonFreq targetFreq =
  let diff = if pylonFreq > targetFreq
             then pylonFreq - targetFreq
             else targetFreq - pylonFreq
      -- 5% tolerance
      tolerance = (targetFreq `shiftR` 4) + (targetFreq `shiftR` 6)  -- ~5%
  in diff <= tolerance

-- | Check leyline proximity for linking
checkLeylineProximity :: Unsigned 8 -> Bool
checkLeylineProximity proximity = proximity < 128  -- Within close range

-- | Check teleport anchor requirements (BOTH conditions required)
checkTeleportAnchor :: Unsigned 8 -> BioState -> Bool
checkTeleportAnchor alpha bio =
  alpha >= alphaTeleportThreshold &&
  bsHRV bio >= hrvCoherenceThreshold

-- | Compute anchor type based on conditions
computeAnchorType
  :: Unsigned 8    -- Alpha
  -> BioState      -- Biometric state
  -> LeylineLink   -- Leyline link
  -> AnchorType
computeAnchorType alpha bio leyline
  | checkTeleportAnchor alpha bio = AnchorTeleport
  | llIsLinked leyline && llStrength leyline > 128 = AnchorLeyline
  | alpha > 128 = AnchorLocal
  | otherwise = AnchorNone

-- | Compute pylon status
computePylonStatus
  :: Unsigned 8    -- Power level
  -> Unsigned 8    -- Alpha
  -> BioState      -- Biometric state
  -> PylonStatus
computePylonStatus power alpha bio
  | power == 0 = StatusOffline
  | checkTeleportAnchor alpha bio = StatusTeleportReady
  | alpha > 179 = StatusAmplifying  -- > 0.7
  | alpha > 77 = StatusActive       -- > 0.3
  | otherwise = StatusStandby

-- | Generate pylon node
generatePylonNode
  :: PylonConfig
  -> Unsigned 8    -- Alpha
  -> BioState
  -> LeylineLink
  -> PylonNode
generatePylonNode config alpha bio leyline =
  let shape = pcShape config
      ampFactor = getAmplificationFactor shape
      outputPower = computeAmplifiedOutput (pcPowerLevel config) ampFactor
      status = computePylonStatus (pcPowerLevel config) alpha bio
      anchor = computeAnchorType alpha bio leyline
  in PylonNode config status ampFactor anchor outputPower

-- | Check if pylon can link to target frequency
canLinkToFrequency :: PylonNode -> Unsigned 16 -> Bool
canLinkToFrequency pylon targetFreq =
  pnStatus pylon /= StatusOffline &&
  checkFrequencyResonance (pcBaseFreq (pnConfig pylon)) targetFreq

-- | Compute grid expansion factor
computeGridExpansionFactor :: Unsigned 8 -> Unsigned 8
computeGridExpansionFactor nodeCount =
  -- More nodes = higher expansion potential (logarithmic)
  if nodeCount < 10 then 26       -- 0.1
  else if nodeCount < 50 then 51  -- 0.2
  else if nodeCount < 100 then 77 -- 0.3
  else 128                        -- 0.5

-- | Compute teleport stability
computeTeleportStability :: PylonNode -> BioState -> Unsigned 8
computeTeleportStability pylon bio
  | pnAnchorType pylon /= AnchorTeleport = 0
  | otherwise =
      let -- Stability = (HRV + coherence) / 2 * output_factor
          hrvCoh = (resize (bsHRV bio) + resize (bsCoherence bio)) `shiftR` 1 :: Unsigned 16
          outputFactor = resize (pnOutputPower pylon `shiftR` 8) :: Unsigned 16
          stability = (hrvCoh * outputFactor) `shiftR` 8
      in min 255 (resize stability)

-- | Validate pylon configuration
validatePylonConfig :: PylonConfig -> Bool
validatePylonConfig config =
  pcBaseFreq config > 0 &&
  pcPowerLevel config > 0

-- | Pylon generator state
data PylonGenState = PylonGenState
  { pgsActiveNodes :: Unsigned 8
  , pgsLastAnchor  :: AnchorType
  } deriving (Generic, NFDataX)

-- | Initial pylon state
initialPylonState :: PylonGenState
initialPylonState = PylonGenState 0 AnchorNone

-- | Pylon generator input
data PylonGenInput = PylonGenInput
  { pgiConfig   :: PylonConfig
  , pgiAlpha    :: Unsigned 8
  , pgiBio      :: BioState
  , pgiLeyline  :: LeylineLink
  } deriving (Generic, NFDataX)

-- | Pylon generator output
data PylonGenOutput = PylonGenOutput
  { pgoNode           :: PylonNode
  , pgoTeleportStable :: Unsigned 8
  , pgoIsValid        :: Bool
  } deriving (Generic, NFDataX)

-- | Pylon generator pipeline
pylonGeneratorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom PylonGenInput
  -> Signal dom PylonGenOutput
pylonGeneratorPipeline input = mealy pylonMealy initialPylonState input
  where
    pylonMealy state inp =
      let -- Generate node
          node = generatePylonNode
            (pgiConfig inp)
            (pgiAlpha inp)
            (pgiBio inp)
            (pgiLeyline inp)

          -- Compute teleport stability
          teleportStable = computeTeleportStability node (pgiBio inp)

          -- Validate
          isValid = validatePylonConfig (pgiConfig inp)

          -- Update state
          newActive = if isValid && pnStatus node /= StatusOffline
                      then min 255 (pgsActiveNodes state + 1)
                      else pgsActiveNodes state

          newState = PylonGenState newActive (pnAnchorType node)

          output = PylonGenOutput node teleportStable isValid

      in (newState, output)

-- | Amplification factor pipeline
amplificationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom PylonShape
  -> Signal dom Unsigned 16
amplificationPipeline = fmap getAmplificationFactor

-- | Anchor type pipeline
anchorTypePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, BioState, LeylineLink)
  -> Signal dom AnchorType
anchorTypePipeline = fmap (\(a, b, l) -> computeAnchorType a b l)

-- | Teleport anchor check pipeline
teleportAnchorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, BioState)
  -> Signal dom Bool
teleportAnchorPipeline = fmap (uncurry checkTeleportAnchor)

-- | Frequency resonance pipeline
frequencyResonancePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 16, Unsigned 16)
  -> Signal dom Bool
frequencyResonancePipeline = fmap (uncurry checkFrequencyResonance)
