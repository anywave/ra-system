{-|
Module      : RaPropulsionVectorConduction
Description : Intentional Vector Motion from Scalar Coherence
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 51: Enables intentional vector motion derived from scalar coherence
resonance - ideal for digital twin locomotion, interface navigation, or
assistive overlays.

Based on Hubbard coil-induced phase vectoring, Tesla directed pulses,
and Reich orgone charge movement.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaPropulsionVectorConduction where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Minimum coherence for impulse generation (0.1 * 255)
coherenceMin :: Unsigned 8
coherenceMin = 26

-- | Stability threshold for impulse filtering (0.05 * 255)
stabilityThreshold :: Unsigned 8
stabilityThreshold = 13

-- | Direction epsilon (0.01 * 255)
directionEpsilon :: Unsigned 8
directionEpsilon = 3

-- | Omega format (harmonic signature)
data OmegaFormat = OmegaFormat
  { ofOmegaL     :: Unsigned 4    -- Spherical harmonic l (0-9)
  , ofOmegaM     :: Signed 8      -- Spherical harmonic m (-l to +l)
  , ofPhaseAngle :: Unsigned 16   -- Phase angle (0-65535 = 0-2π)
  , ofAmplitude  :: Unsigned 8    -- Field intensity (0-255)
  } deriving (Generic, NFDataX)

-- | Ra coordinate (spherical position)
data RaCoordinate = RaCoordinate
  { rcTheta :: Unsigned 9         -- 0-511 maps to 0-π
  , rcPhi   :: Unsigned 9         -- 0-511 maps to 0-2π
  , rcH     :: Unsigned 8         -- Height/shell (0-255)
  } deriving (Generic, NFDataX)

-- | Vector impulse (output)
data VectorImpulse = VectorImpulse
  { viDirectionX   :: Signed 16   -- x component (-127 to 127 scaled)
  , viDirectionY   :: Signed 16   -- y component
  , viDirectionZ   :: Signed 16   -- z component
  , viMagnitude    :: Unsigned 8  -- 0-255
  , viHarmonicL    :: Unsigned 4  -- Anchor L
  , viHarmonicM    :: Signed 8    -- Anchor M
  , viValid        :: Bool        -- Valid impulse flag
  } deriving (Generic, NFDataX)

-- | Conduction field parameters
data ConductionField = ConductionField
  { cfCoherenceBias :: Unsigned 8   -- Intention strength (0-255)
  , cfPhaseOffset   :: Unsigned 16  -- φ^n time harmonic offset
  } deriving (Generic, NFDataX)

-- | Scalar gradient (3 components)
data ScalarGradient = ScalarGradient
  { sgDTheta :: Signed 16
  , sgDPhi   :: Signed 16
  , sgDH     :: Signed 16
  } deriving (Generic, NFDataX)

-- | Sine lookup table (quarter wave, 64 entries, scaled to 127)
sineLUT :: Vec 64 (Signed 8)
sineLUT = $(listToVecTH
  [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
   48, 51, 54, 57, 59, 62, 65, 67, 70, 73, 75, 78, 80, 82, 85, 87,
   89, 91, 93, 95, 97, 99, 101, 102, 104, 105, 107, 108, 109, 110, 112, 113,
   114, 115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 120, 121, 121, 121, 121])

-- | Cosine lookup table (quarter wave, 64 entries, scaled to 127)
cosineLUT :: Vec 64 (Signed 8)
cosineLUT = $(listToVecTH
  [127, 126, 126, 126, 125, 125, 124, 123, 122, 121, 120, 119, 118, 116, 115, 113,
   112, 110, 108, 107, 105, 103, 101, 99, 97, 94, 92, 90, 87, 85, 82, 80,
   77, 75, 72, 69, 67, 64, 61, 58, 55, 52, 49, 46, 43, 40, 37, 34,
   31, 28, 25, 22, 19, 16, 12, 9, 6, 3, 0, -3, -6, -9, -12, -15])

-- | Get sine value from angle (0-65535 = 0-2π)
getSine :: Unsigned 16 -> Signed 8
getSine angle =
  let idx = resize (angle `shiftR` 10) :: Unsigned 6
  in sineLUT !! idx

-- | Get cosine value from angle (0-65535 = 0-2π)
getCosine :: Unsigned 16 -> Signed 8
getCosine angle =
  let idx = resize (angle `shiftR` 10) :: Unsigned 6
  in cosineLUT !! idx

-- | Compute gradient magnitude
computeGradientMagnitude :: ScalarGradient -> Unsigned 16
computeGradientMagnitude grad =
  let dtAbs = if sgDTheta grad < 0 then resize (-sgDTheta grad) else resize (sgDTheta grad)
      dpAbs = if sgDPhi grad < 0 then resize (-sgDPhi grad) else resize (sgDPhi grad)
      dhAbs = if sgDH grad < 0 then resize (-sgDH grad) else resize (sgDH grad)
  in dtAbs + dpAbs + dhAbs  -- Manhattan distance for efficiency

-- | Convert spherical gradient to Cartesian direction
sphericalToCartesian
  :: RaCoordinate
  -> ScalarGradient
  -> (Signed 16, Signed 16, Signed 16)
sphericalToCartesian coord grad =
  let -- Get trig values for theta and phi
      theta16 = resize (rcTheta coord) `shiftL` 7  -- Scale to 0-65535
      phi16' = resize (rcPhi coord) `shiftL` 7

      sinTheta = getSine theta16
      cosTheta = getCosine theta16
      sinPhi = getSine phi16'
      cosPhi = getCosine phi16'

      -- Weighted direction from gradient
      -- x = dθ * sin(θ) * cos(φ) + dφ * (-sin(φ)) + dh * cos(θ) * cos(φ)
      -- Simplified: just weight by trig components
      x = ((resize (sgDTheta grad) * resize sinTheta) `shiftR` 7) * resize cosPhi `shiftR` 7
      y = ((resize (sgDTheta grad) * resize sinTheta) `shiftR` 7) * resize sinPhi `shiftR` 7
      z = (resize (sgDTheta grad) * resize cosTheta) `shiftR` 7

  in (resize x, resize y, resize z)

-- | Compute harmonic match score (0-255)
computeHarmonicMatch :: OmegaFormat -> Unsigned 4 -> Signed 8 -> Unsigned 8
computeHarmonicMatch anchor fieldL fieldM =
  let -- L match (exact = 255, off by 1 = 191, etc.)
      lDiff = if resize (ofOmegaL anchor) > resize fieldL
              then resize (ofOmegaL anchor) - resize fieldL
              else resize fieldL - resize (ofOmegaL anchor) :: Unsigned 8
      lScore = if lDiff > 4 then 0 else 255 - (lDiff * 64)

      -- M match
      mAbs1 = if ofOmegaM anchor < 0 then -(ofOmegaM anchor) else ofOmegaM anchor
      mAbs2 = if fieldM < 0 then -fieldM else fieldM
      mDiff = if mAbs1 > mAbs2
              then resize mAbs1 - resize mAbs2
              else resize mAbs2 - resize mAbs1 :: Unsigned 8
      mScore = if mDiff > 6 then 0 else 255 - (mDiff * 38)

      -- Combined score scaled by amplitude
      combined = (resize lScore * resize mScore) `shiftR` 8 :: Unsigned 16
      scaled = (combined * resize (ofAmplitude anchor)) `shiftR` 8

  in resize $ min 255 scaled

-- | Apply phase rotation to direction
applyPhaseRotation
  :: Unsigned 16                     -- Phase angle
  -> (Signed 16, Signed 16, Signed 16)  -- Input direction
  -> (Signed 16, Signed 16, Signed 16)  -- Rotated direction
applyPhaseRotation phase (x, y, z) =
  let cosP = getCosine phase
      sinP = getSine phase

      -- Rotate around z-axis: x' = x*cos - y*sin, y' = x*sin + y*cos
      x' = (x * resize cosP - y * resize sinP) `shiftR` 7
      y' = (x * resize sinP + y * resize cosP) `shiftR` 7

  in (resize x', resize y', z)

-- | Conduct vector impulse from gradient
conductImpulse
  :: ConductionField
  -> RaCoordinate
  -> ScalarGradient
  -> Unsigned 4         -- Field harmonic L
  -> Signed 8           -- Field harmonic M
  -> VectorImpulse
conductImpulse cf coord grad fieldL fieldM =
  let -- Check minimum coherence
      cohOk = cfCoherenceBias cf >= coherenceMin

      -- Check gradient magnitude
      gradMag = computeGradientMagnitude grad
      gradOk = gradMag > 0

      -- Convert to Cartesian direction
      (x, y, z) = sphericalToCartesian coord grad

      -- Create harmonic anchor
      anchor = OmegaFormat fieldL fieldM (cfPhaseOffset cf) (cfCoherenceBias cf)

      -- Apply phase rotation
      (xRot, yRot, zRot) = applyPhaseRotation (cfPhaseOffset cf) (x, y, z)

      -- Compute harmonic match
      harmonicScore = computeHarmonicMatch anchor fieldL fieldM

      -- Compute magnitude: coherence * harmonicMatch * gradientStrength
      gradNorm = min 255 (resize gradMag)
      magProduct = resize (cfCoherenceBias cf) * resize harmonicScore `shiftR` 8 :: Unsigned 16
      magnitude = (magProduct * resize gradNorm) `shiftR` 8 :: Unsigned 16

      -- Check if valid
      isValid = cohOk && gradOk && magnitude > resize directionEpsilon

  in VectorImpulse
       xRot yRot zRot
       (resize $ min 255 magnitude)
       fieldL fieldM
       isValid

-- | Stabilize impulse (filter jitter)
stabilizeImpulse :: VectorImpulse -> VectorImpulse
stabilizeImpulse impulse =
  let magOk = viMagnitude impulse >= stabilityThreshold
      dirMag = abs (viDirectionX impulse) + abs (viDirectionY impulse) + abs (viDirectionZ impulse)
      dirOk = dirMag > resize directionEpsilon
      isValid = viValid impulse && magOk && dirOk
  in impulse { viValid = isValid }

-- | Exponential smoothing state
data SmoothingState = SmoothingState
  { ssPrevX   :: Signed 16
  , ssPrevY   :: Signed 16
  , ssPrevZ   :: Signed 16
  , ssPrevMag :: Unsigned 8
  , ssHasPrev :: Bool
  } deriving (Generic, NFDataX)

-- | Initial smoothing state
initialSmoothingState :: SmoothingState
initialSmoothingState = SmoothingState 0 0 0 0 False

-- | Apply exponential smoothing (alpha = 0.3 ≈ 77/256)
applySmoothing :: SmoothingState -> VectorImpulse -> (SmoothingState, VectorImpulse)
applySmoothing state impulse
  | not (ssHasPrev state) =
      let newState = SmoothingState
            (viDirectionX impulse) (viDirectionY impulse) (viDirectionZ impulse)
            (viMagnitude impulse) True
      in (newState, impulse)
  | otherwise =
      let alpha = 77 :: Unsigned 8  -- 0.3 * 256
          invAlpha = 256 - resize alpha :: Unsigned 16

          -- Smooth direction: new = alpha * current + (1-alpha) * previous
          smoothX = (resize alpha * resize (viDirectionX impulse) +
                     invAlpha * resize (ssPrevX state)) `shiftR` 8
          smoothY = (resize alpha * resize (viDirectionY impulse) +
                     invAlpha * resize (ssPrevY state)) `shiftR` 8
          smoothZ = (resize alpha * resize (viDirectionZ impulse) +
                     invAlpha * resize (ssPrevZ state)) `shiftR` 8
          smoothMag = (resize alpha * resize (viMagnitude impulse) +
                       invAlpha * resize (ssPrevMag state)) `shiftR` 8

          smoothed = impulse
            { viDirectionX = resize smoothX
            , viDirectionY = resize smoothY
            , viDirectionZ = resize smoothZ
            , viMagnitude = resize smoothMag
            }

          newState = SmoothingState
            (viDirectionX smoothed) (viDirectionY smoothed) (viDirectionZ smoothed)
            (viMagnitude smoothed) True

      in (newState, smoothed)

-- | Conduction pipeline
conductionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ConductionField, RaCoordinate, ScalarGradient, Unsigned 4, Signed 8)
  -> Signal dom VectorImpulse
conductionPipeline input =
  let impulse = (\(cf, coord, grad, l, m) -> conductImpulse cf coord grad l m) <$> input
  in stabilizeImpulse <$> impulse

-- | Smoothed conduction pipeline (stateful)
smoothedConductionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ConductionField, RaCoordinate, ScalarGradient, Unsigned 4, Signed 8)
  -> Signal dom VectorImpulse
smoothedConductionPipeline input =
  let rawImpulse = (\(cf, coord, grad, l, m) -> conductImpulse cf coord grad l m) <$> input
      stable = stabilizeImpulse <$> rawImpulse
      (_, smoothed) = unbundle $ mealy smoothingMealy initialSmoothingState stable
  in smoothed
  where
    smoothingMealy state imp = applySmoothing state imp

-- | Screen mapping: impulse to cursor delta
impulseToCursorDelta :: VectorImpulse -> Unsigned 8 -> (Signed 16, Signed 16)
impulseToCursorDelta impulse scale =
  let scaleFactor = resize scale :: Signed 16
      dx = (viDirectionX impulse * resize (viMagnitude impulse) * scaleFactor) `shiftR` 8
      dy = (viDirectionY impulse * resize (viMagnitude impulse) * scaleFactor) `shiftR` 8
  in (resize dx, resize dy)

-- | Z component for scroll behavior
impulseToScroll :: VectorImpulse -> Unsigned 8 -> Signed 16
impulseToScroll impulse scale =
  let scaleFactor = resize scale :: Signed 16
  in (viDirectionZ impulse * resize (viMagnitude impulse) * scaleFactor) `shiftR` 8
