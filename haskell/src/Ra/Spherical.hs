{-|
Module      : Ra.Spherical
Description : θ/φ/r coordinate functions
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Coordinate transforms for the Ra System dimensional mapping:

* θ (theta): Semantic sector ← 27 Repitans
* φ (phi): Access sensitivity ← 6 RACs
* h (harmonic): Coherence depth ← 5 Omega formats
* r (radius): Emergence intensity ← Ankh-normalized scalar
-}
module Ra.Spherical
    ( -- * Coordinate Type
      RaCoordinate(..)
    , mkCoordinate
      -- * Theta (Semantic Sector)
    , thetaFromRepitan
    , repitanFromTheta
    , thetaDistance
      -- * Phi (Access Sensitivity)
    , phiFromRac
    , racFromPhi
      -- * Harmonic (Coherence Depth)
    , harmonicFromOmega
    , omegaFromHarmonic
      -- * Radius (Emergence Intensity)
    , normalizeRadius
    , denormalizeRadius
      -- * Coordinate Operations
    , coordinateDistance
    , isCoordinateValid
    ) where

import Ra.Constants (Ankh(..), ankh)
import Ra.Repitans (Repitan, repitan, repitanIndex, repitanValue, thetaToRepitan)
import Ra.Rac (RacLevel(..), racByLevel)
import Ra.Omega (OmegaFormat(..))

-- | A complete Ra coordinate in 4-dimensional space
data RaCoordinate = RaCoordinate
    { coordTheta    :: !Repitan       -- ^ Semantic sector (1-27)
    , coordPhi      :: !RacLevel      -- ^ Access sensitivity (RAC1-RAC6)
    , coordHarmonic :: !OmegaFormat   -- ^ Coherence depth (Red-Blue)
    , coordRadius   :: !Double        -- ^ Ankh-normalized intensity [0,1]
    } deriving (Show, Eq)

-- | Smart constructor for RaCoordinate
mkCoordinate :: Repitan -> RacLevel -> OmegaFormat -> Double -> Maybe RaCoordinate
mkCoordinate theta phi harmonic radius
    | radius >= 0 && radius <= 1 = Just $ RaCoordinate theta phi harmonic radius
    | otherwise = Nothing

-- | Convert Repitan to theta angle in degrees (0-360)
thetaFromRepitan :: Repitan -> Double
thetaFromRepitan r = repitanValue r * 360.0

-- | Convert theta angle (degrees) to Repitan
repitanFromTheta :: Double -> Repitan
repitanFromTheta = thetaToRepitan

-- | Calculate angular distance between two theta values
-- Accounts for circular wraparound (max distance is 13.5 Repitan bands)
thetaDistance :: Repitan -> Repitan -> Int
thetaDistance r1 r2 =
    let d = abs (repitanIndex r1 - repitanIndex r2)
    in min d (27 - d)

-- | Convert RacLevel to phi value (0-255 encoded)
phiFromRac :: RacLevel -> Int
phiFromRac RAC1 = 0    -- Least restrictive
phiFromRac RAC2 = 43
phiFromRac RAC3 = 85
phiFromRac RAC4 = 128
phiFromRac RAC5 = 170
phiFromRac RAC6 = 255  -- Most restrictive

-- | Convert phi value (0-255) to RacLevel
racFromPhi :: Int -> RacLevel
racFromPhi phi
    | phi < 22  = RAC1
    | phi < 64  = RAC2
    | phi < 107 = RAC3
    | phi < 149 = RAC4
    | phi < 213 = RAC5
    | otherwise = RAC6

-- | Convert OmegaFormat to harmonic index (0-4)
harmonicFromOmega :: OmegaFormat -> Int
harmonicFromOmega Red        = 0
harmonicFromOmega OmegaMajor = 1
harmonicFromOmega Green      = 2
harmonicFromOmega OmegaMinor = 3
harmonicFromOmega Blue       = 4

-- | Convert harmonic index (0-4) to OmegaFormat
omegaFromHarmonic :: Int -> Maybe OmegaFormat
omegaFromHarmonic 0 = Just Red
omegaFromHarmonic 1 = Just OmegaMajor
omegaFromHarmonic 2 = Just Green
omegaFromHarmonic 3 = Just OmegaMinor
omegaFromHarmonic 4 = Just Blue
omegaFromHarmonic _ = Nothing

-- | Normalize a raw radius value to [0, 1] using Ankh
-- r_normalized = r_raw / Ankh
normalizeRadius :: Double -> Double
normalizeRadius r = min 1.0 $ max 0.0 $ r / unAnkh ankh

-- | Denormalize a radius value from [0, 1] to raw scale
-- r_raw = r_normalized × Ankh
denormalizeRadius :: Double -> Double
denormalizeRadius r = r * unAnkh ankh

-- | Calculate weighted distance between two coordinates
-- Returns value in [0, 1] where 0 = identical, 1 = maximally different
coordinateDistance :: RaCoordinate -> RaCoordinate -> Double
coordinateDistance c1 c2 =
    let thetaDist = fromIntegral (thetaDistance (coordTheta c1) (coordTheta c2)) / 13.5
        phiDist = fromIntegral (abs (phiFromRac (coordPhi c1) - phiFromRac (coordPhi c2))) / 255.0
        hDist = fromIntegral (abs (harmonicFromOmega (coordHarmonic c1) - harmonicFromOmega (coordHarmonic c2))) / 4.0
        rDist = abs (coordRadius c1 - coordRadius c2)
        -- Weighted average (from spec: w_θ=0.3, w_φ=0.4, w_h=0.2, w_r=0.1)
    in 0.3 * thetaDist + 0.4 * phiDist + 0.2 * hDist + 0.1 * rDist

-- | Check if a coordinate is valid
isCoordinateValid :: RaCoordinate -> Bool
isCoordinateValid c = coordRadius c >= 0 && coordRadius c <= 1
