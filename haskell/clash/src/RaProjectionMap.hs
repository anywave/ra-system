{-|
Module      : RaProjectionMap
Description : Scalar Field Hypergrid Projection
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 43: Projects scalar field into 810-point hypergrid (27θ × 6φ × 5h)
with gradient flow computation and inversion zone detection.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaProjectionMap where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Grid dimensions
type ThetaSlices = 27
type PhiSegments = 6
type HarmonicDepths = 5
type RaSpaceSize = 810  -- 27 * 6 * 5

-- | Inversion thresholds (scaled to 8-bit)
inversionAlphaThreshold :: Unsigned 8
inversionAlphaThreshold = 77  -- 0.3 * 255

divergenceThreshold :: Unsigned 8
divergenceThreshold = 128  -- 0.5 * 255

-- | Ra coordinate in hypergrid
data RaCoordinate = RaCoordinate
  { rcTheta :: Unsigned 5    -- 0-26 (5 bits)
  , rcPhi   :: Unsigned 3    -- 0-5 (3 bits)
  , rcH     :: Unsigned 3    -- 0-4 (3 bits)
  } deriving (Generic, NFDataX, Eq, Show)

-- | Scalar value at coordinate
data ScalarValue = ScalarValue
  { svPotential :: Unsigned 8   -- 0-255
  , svFlux      :: Unsigned 8
  , svPhase     :: Unsigned 16  -- 0-65535 (0-2π)
  , svCoherence :: Unsigned 8   -- 0-255
  } deriving (Generic, NFDataX)

-- | Gradient vector (signed for direction)
data GradientVector = GradientVector
  { gvDTheta :: Signed 16
  , gvDPhi   :: Signed 16
  , gvDH     :: Signed 16
  } deriving (Generic, NFDataX)

-- | Inversion type
data InversionType
  = GradientFlip
  | LowAlpha
  | Divergence
  | Combined
  deriving (Generic, NFDataX, Eq, Show)

-- | Inversion zone detection result
data InversionZone = InversionZone
  { izCoord     :: RaCoordinate
  , izType      :: InversionType
  , izAlpha     :: Unsigned 8
  , izMagnitude :: Unsigned 16
  } deriving (Generic, NFDataX)

-- | Precomputed harmonic lookup tables (ROM)
-- Theta harmonics: sin(2π * t / 27) scaled to signed 8-bit
thetaHarmonicROM :: Vec ThetaSlices (Signed 8)
thetaHarmonicROM = $(listToVecTH
  [0, 29, 56, 79, 98, 111, 118, 118, 111, 98, 79, 56, 29, 0,
   -29, -56, -79, -98, -111, -118, -118, -111, -98, -79, -56, -29, 0])

-- | Phi harmonics: cos(π * p / 6) scaled to signed 8-bit
phiHarmonicROM :: Vec PhiSegments (Signed 8)
phiHarmonicROM = $(listToVecTH [127, 110, 64, 0, -64, -110])

-- | Depth decay: φ^(-h) * 256
depthDecayROM :: Vec HarmonicDepths (Unsigned 8)
depthDecayROM = $(listToVecTH [255, 158, 98, 60, 37])

-- | Get theta harmonic modulation
getThetaHarmonic :: Unsigned 5 -> Signed 8
getThetaHarmonic t = thetaHarmonicROM !! resize (t `mod` 27)

-- | Get phi harmonic modulation
getPhiHarmonic :: Unsigned 3 -> Signed 8
getPhiHarmonic p = phiHarmonicROM !! resize (p `mod` 6)

-- | Get depth decay factor
getDepthDecay :: Unsigned 3 -> Unsigned 8
getDepthDecay h = depthDecayROM !! resize (min h 4)

-- | Wrap theta coordinate (circular)
wrapTheta :: Signed 8 -> Unsigned 5
wrapTheta t
  | t < 0     = resize ((t + 27) `mod` 27)
  | t >= 27   = resize (t `mod` 27)
  | otherwise = resize t

-- | Wrap phi coordinate (circular)
wrapPhi :: Signed 4 -> Unsigned 3
wrapPhi p
  | p < 0    = resize ((p + 6) `mod` 6)
  | p >= 6   = resize (p `mod` 6)
  | otherwise = resize p

-- | Clamp h coordinate (bounded)
clampH :: Signed 4 -> Unsigned 3
clampH h
  | h < 0    = 0
  | h > 4    = 4
  | otherwise = resize h

-- | Compute coordinate to flat index (for memory addressing)
coordToIndex :: RaCoordinate -> Unsigned 10
coordToIndex (RaCoordinate t p h) =
  resize t * 30 + resize p * 5 + resize h

-- | Compute emergence alpha from scalar value at coordinate
-- Alpha = coherence * potential * harmonic_mod * depth_decay / scale
computeEmergenceAlpha :: ScalarValue -> RaCoordinate -> Unsigned 8
computeEmergenceAlpha sv coord =
  let thetaMod = getThetaHarmonic (rcTheta coord)
      phiMod = getPhiHarmonic (rcPhi coord)
      depthFactor = getDepthDecay (rcH coord)

      -- Normalize harmonics to 0-255 range
      thetaNorm = resize ((128 + resize thetaMod) :: Unsigned 9) :: Unsigned 8
      phiNorm = resize ((128 + resize phiMod) :: Unsigned 9) :: Unsigned 8

      -- Base alpha from coherence * potential
      base = (resize (svCoherence sv) * resize (svPotential sv)) `shiftR` 8 :: Unsigned 16

      -- Apply modulations (divide by 256 at each step to maintain range)
      modulated1 = (base * resize thetaNorm) `shiftR` 8
      modulated2 = (modulated1 * resize phiNorm) `shiftR` 8
      final = (modulated2 * resize depthFactor) `shiftR` 8

  in resize $ min 255 final

-- | Compute gradient at coordinate using central differences
-- Returns (dTheta, dPhi, dH) as signed values
computeGradient :: (RaCoordinate -> Unsigned 8) -> RaCoordinate -> GradientVector
computeGradient alphaLookup coord =
  let t = rcTheta coord
      p = rcPhi coord
      h = rcH coord

      -- Alpha at current position
      alpha0 = alphaLookup coord

      -- Theta neighbors (circular)
      alphaThetaPlus = alphaLookup $ RaCoordinate (wrapTheta (resize t + 1)) p h
      alphaThetaMinus = alphaLookup $ RaCoordinate (wrapTheta (resize t - 1)) p h
      dTheta = (resize alphaThetaPlus - resize alphaThetaMinus) `div` 2

      -- Phi neighbors (circular)
      alphaPhiPlus = alphaLookup $ RaCoordinate t (wrapPhi (resize p + 1)) h
      alphaPhiMinus = alphaLookup $ RaCoordinate t (wrapPhi (resize p - 1)) h
      dPhi = (resize alphaPhiPlus - resize alphaPhiMinus) `div` 2

      -- H neighbors (bounded)
      alphaHPlus = if h < 4 then alphaLookup $ RaCoordinate t p (h + 1) else alpha0
      alphaHMinus = if h > 0 then alphaLookup $ RaCoordinate t p (h - 1) else alpha0
      dH = if h == 0 then resize alphaHPlus - resize alpha0
           else if h == 4 then resize alpha0 - resize alphaHMinus
           else (resize alphaHPlus - resize alphaHMinus) `div` 2

  in GradientVector dTheta dPhi dH

-- | Compute gradient magnitude
gradientMagnitude :: GradientVector -> Unsigned 16
gradientMagnitude (GradientVector dt dp dh) =
  let dt2 = resize (abs dt) * resize (abs dt) :: Unsigned 32
      dp2 = resize (abs dp) * resize (abs dp) :: Unsigned 32
      dh2 = resize (abs dh) * resize (abs dh) :: Unsigned 32
      sumSquares = dt2 + dp2 + dh2
  in resize $ min 65535 sumSquares  -- Simplified (no sqrt in hardware)

-- | Check for sign flip between gradients
isSignFlip :: GradientVector -> GradientVector -> Bool
isSignFlip g1 g2 =
  (gvDTheta g1 * gvDTheta g2 < 0) ||
  (gvDPhi g1 * gvDPhi g2 < 0) ||
  (gvDH g1 * gvDH g2 < 0)

-- | Detect inversion at coordinate
detectInversion :: Unsigned 8 -> GradientVector -> Bool -> Maybe InversionType
detectInversion alpha grad hasFlip
  | hasFlip && alpha < inversionAlphaThreshold = Just Combined
  | hasFlip = Just GradientFlip
  | alpha < inversionAlphaThreshold && gradientMagnitude grad > resize divergenceThreshold =
      Just Divergence
  | alpha < inversionAlphaThreshold = Just LowAlpha
  | otherwise = Nothing

-- | Top-level alpha projection pipeline
alphaProjectionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ScalarValue, RaCoordinate)
  -> Signal dom (Unsigned 8)
alphaProjectionPipeline input = uncurry computeEmergenceAlpha <$> input

-- | Gradient computation pipeline
gradientPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (RaCoordinate -> Unsigned 8, RaCoordinate)
  -> Signal dom GradientVector
gradientPipeline input = uncurry computeGradient <$> input

-- | Inversion detection pipeline
inversionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, GradientVector, Bool)
  -> Signal dom (Maybe InversionType)
inversionPipeline input = (\(a, g, f) -> detectInversion a g f) <$> input
