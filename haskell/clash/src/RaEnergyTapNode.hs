{-|
Module      : RaEnergyTapNode
Description : Radiant Energy Tap Node Locator
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 66: Identifies passive tap points in scalar fields where
energy may be extracted via harmonic convergence without closed-loop circuits.

Uses convergence thresholds, chamber geometry modulation, and avatar sync.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaEnergyTapNode where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Convergence thresholds (scaled to 255)
alphaConvergenceMin :: Unsigned 8
alphaConvergenceMin = 224  -- 0.88 * 255

vectorInwardnessMin :: Unsigned 8
vectorInwardnessMin = 179  -- 0.7 * 255

-- | Toroidal special threshold
toroidalAlphaMin :: Unsigned 8
toroidalAlphaMin = 209     -- 0.82 * 255

-- | Dodecahedral flux amplification (1.2 * 256)
dodecahedralFluxAmp :: Unsigned 16
dodecahedralFluxAmp = 307

-- | Avatar sync thresholds
frequencyTolerance :: Unsigned 8
frequencyTolerance = 13    -- 0.05 * 255

hrvCoherenceMin :: Unsigned 8
hrvCoherenceMin = 153      -- 0.6 * 255

-- | Chamber geometry types
data ChamberForm
  = ChamberToroidal
  | ChamberDodecahedral
  | ChamberSpherical
  | ChamberCustom
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | 3D coordinate
data RaCoordinate = RaCoordinate
  { rcX :: Signed 16
  , rcY :: Signed 16
  , rcZ :: Signed 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Gradient vector
data GradientVector = GradientVector
  { gvDx :: Signed 16
  , gvDy :: Signed 16
  , gvDz :: Signed 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Appendage resonance state
data AppendageResonance = AppendageResonance
  { arFrequency    :: Unsigned 16   -- Hz * 256
  , arAmplitude    :: Unsigned 8
  , arHRVCoherence :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Scalar field point
data ScalarFieldPoint = ScalarFieldPoint
  { sfpCoordinate :: RaCoordinate
  , sfpAlpha      :: Unsigned 8     -- Coherence 0-255
  , sfpGradient   :: GradientVector
  } deriving (Generic, NFDataX, Eq, Show)

-- | Avatar with appendage
data Avatar = Avatar
  { avPosition  :: RaCoordinate
  , avAppendage :: AppendageResonance
  } deriving (Generic, NFDataX, Eq, Show)

-- | Tap node
data TapNode = TapNode
  { tnCoordinate   :: RaCoordinate
  , tnScalarAlpha  :: Unsigned 8
  , tnFluxVector   :: GradientVector
  , tnIsActive     :: Bool
  , tnGeometryLink :: ChamberForm
  , tnHasAvatarSync :: Bool
  , tnInwardness   :: Unsigned 8    -- 0-255
  } deriving (Generic, NFDataX, Eq, Show)

-- | Locator input
data LocatorInput = LocatorInput
  { liPoint   :: ScalarFieldPoint
  , liCenter  :: RaCoordinate
  , liChamber :: ChamberForm
  , liAvatar  :: Avatar
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute vector magnitude (approximation)
vectorMagnitude :: GradientVector -> Unsigned 16
vectorMagnitude gv =
  let dx2 = resize (gvDx gv * gvDx gv) :: Unsigned 32
      dy2 = resize (gvDy gv * gvDy gv) :: Unsigned 32
      dz2 = resize (gvDz gv * gvDz gv) :: Unsigned 32
      sum_sq = dx2 + dy2 + dz2
      -- Approximate sqrt using bitshift
  in resize (sum_sq `shiftR` 8)

-- | Compute inward radial vector (from point to center)
computeRadialVector :: RaCoordinate -> RaCoordinate -> GradientVector
computeRadialVector point center = GradientVector
  (rcX center - rcX point)
  (rcY center - rcY point)
  (rcZ center - rcZ point)

-- | Compute dot product (normalized to 0-255)
computeDotProduct :: GradientVector -> GradientVector -> Unsigned 8
computeDotProduct g1 g2 =
  let dot = resize (gvDx g1) * resize (gvDx g2) +
            resize (gvDy g1) * resize (gvDy g2) +
            resize (gvDz g1) * resize (gvDz g2) :: Signed 32
      -- Normalize assuming unit vectors
      normalized = if dot > 0 then resize (dot `shiftR` 8) else 0
  in min 255 normalized

-- | Check convergence criteria
checkConvergence :: Unsigned 8 -> Unsigned 8 -> ChamberForm -> Bool
checkConvergence alpha inwardness chamber =
  let alphaThreshold = case chamber of
        ChamberToroidal -> toroidalAlphaMin
        _               -> alphaConvergenceMin
  in alpha >= alphaThreshold && inwardness >= vectorInwardnessMin

-- | Compute flux magnitude with chamber amplification
computeFluxMagnitude :: GradientVector -> ChamberForm -> Unsigned 16
computeFluxMagnitude gradient chamber =
  let baseMag = vectorMagnitude gradient
  in case chamber of
       ChamberDodecahedral -> (baseMag * dodecahedralFluxAmp) `shiftR` 8
       _                   -> baseMag

-- | Estimate tap node frequency (simplified)
estimateTapFrequency :: Unsigned 8 -> GradientVector -> Unsigned 16
estimateTapFrequency alpha gradient =
  let -- Base freq = 7.83 Hz * 256 = 2004
      baseFreq = 2004 :: Unsigned 16
      -- Scale by (1 + alpha/255) * (1 + mag*0.1)
      alphaFactor = 256 + resize alpha :: Unsigned 16
      mag = vectorMagnitude gradient
      magFactor = 256 + (mag `shiftR` 4) :: Unsigned 16
      freq = (baseFreq * alphaFactor * magFactor) `shiftR` 16
  in resize freq

-- | Check avatar sync
checkAvatarSync :: Unsigned 16 -> AppendageResonance -> Bool
checkAvatarSync tapFreq appendage =
  let avatarFreq = arFrequency appendage
      -- Check ±5% tolerance
      diff = if tapFreq > avatarFreq
             then tapFreq - avatarFreq
             else avatarFreq - tapFreq
      tolerance = (tapFreq * resize frequencyTolerance) `shiftR` 8
      freqMatch = diff <= tolerance

      hrvMatch = arHRVCoherence appendage >= hrvCoherenceMin

  in freqMatch && hrvMatch

-- | Create tap node from input
createTapNode :: LocatorInput -> Maybe TapNode
createTapNode input =
  let point = liPoint input
      center = liCenter input
      chamber = liChamber input
      avatar = liAvatar input

      -- Compute radial and inwardness
      radial = computeRadialVector (sfpCoordinate point) center
      gradient = sfpGradient point
      inwardness = computeDotProduct gradient radial

      -- Check convergence
      converges = checkConvergence (sfpAlpha point) inwardness chamber

  in if not converges
     then Nothing
     else
       let -- Compute flux
           fluxMag = computeFluxMagnitude gradient chamber
           fluxVector = GradientVector
             ((gvDx gradient * resize fluxMag) `shiftR` 8)
             ((gvDy gradient * resize fluxMag) `shiftR` 8)
             ((gvDz gradient * resize fluxMag) `shiftR` 8)

           -- Check avatar sync
           tapFreq = estimateTapFrequency (sfpAlpha point) gradient
           hasSync = checkAvatarSync tapFreq (avAppendage avatar)
           isActive = hasSync

       in Just $ TapNode
            (sfpCoordinate point)
            (sfpAlpha point)
            fluxVector
            isActive
            chamber
            hasSync
            inwardness

-- | Tap node locator state
data LocatorState = LocatorState
  { lsNodeCount :: Unsigned 4
  } deriving (Generic, NFDataX)

-- | Initial locator state
initialLocatorState :: LocatorState
initialLocatorState = LocatorState 0

-- | Tap node locator pipeline
tapNodeLocatorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom LocatorInput
  -> Signal dom (Maybe TapNode)
tapNodeLocatorPipeline input = createTapNode <$> input

-- | Convergence check pipeline
convergenceCheckPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8, ChamberForm)
  -> Signal dom Bool
convergenceCheckPipeline input =
  (\(alpha, inward, chamber) -> checkConvergence alpha inward chamber) <$> input
