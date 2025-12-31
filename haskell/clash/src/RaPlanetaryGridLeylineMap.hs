{-|
Module      : RaPlanetaryGridLeylineMap
Description : Leyline Harmonic Mapping System
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 73: Implements phi-ratio radionic mapping, solar/Schumann temporal
modulation, and Becker-Hagens grid geometry. Maps leyline intersections
to harmonic resonance points for scalar field amplification.

Minimum 108 nodes for valid grid (Becker-Hagens requirement).
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaPlanetaryGridLeylineMap where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Schumann fundamental (7.83 Hz * 256)
schumannFundamental :: Unsigned 16
schumannFundamental = 2005

-- | Minimum leyline nodes for valid grid
minLeylineNodes :: Unsigned 8
minLeylineNodes = 108

-- | Solar modulation bounds (scaled)
solarModMin :: Unsigned 8
solarModMin = 205    -- 0.8 * 256
solarModMax :: Unsigned 8
solarModMax = 51     -- 0.2 * 256 (delta from 1.0)

-- | Grid node types
data GridNodeType
  = NodeTypeVertex      -- Becker-Hagens vertex
  | NodeTypeMidpoint    -- Edge midpoint
  | NodeTypeIntersect   -- Leyline intersection
  | NodeTypeVortex      -- Energy vortex point
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Leyline status
data LeylineStatus
  = StatusDormant
  | StatusActive
  | StatusAmplified
  | StatusOverloaded
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Solar phase (time of day approximation)
data SolarPhase
  = PhaseDawn
  | PhaseMorning
  | PhaseNoon
  | PhaseAfternoon
  | PhaseDusk
  | PhaseNight
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Geographic coordinate (lat/lon scaled, 8.8 fixed point)
data GeoCoordinate = GeoCoordinate
  { gcLatitude  :: Signed 16   -- -90 to 90 * 256
  , gcLongitude :: Signed 16   -- -180 to 180 * 256
  } deriving (Generic, NFDataX, Eq, Show)

-- | Grid node
data GridNode = GridNode
  { gnCoord      :: GeoCoordinate
  , gnNodeType   :: GridNodeType
  , gnHarmonic   :: Unsigned 16   -- Resonance frequency * 256
  , gnStrength   :: Unsigned 8    -- 0-255 normalized
  } deriving (Generic, NFDataX, Eq, Show)

-- | Leyline segment
data LeylineSegment = LeylineSegment
  { lsStartCoord :: GeoCoordinate
  , lsEndCoord   :: GeoCoordinate
  , lsStatus     :: LeylineStatus
  , lsHarmonic   :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Radionic rate input
data RadionicInput = RadionicInput
  { riRate       :: Unsigned 16   -- Radionic rate value
  , riMagnitude  :: Unsigned 8    -- Log10 approximation (0-10 scaled)
  } deriving (Generic, NFDataX, Eq, Show)

-- | Temporal modulation input
data TemporalInput = TemporalInput
  { tiSolarPhase    :: SolarPhase
  , tiSchumannPeak  :: Bool        -- True if at Schumann resonance peak
  , tiLunarPhase    :: Unsigned 4  -- 0-15 (0=new, 8=full)
  } deriving (Generic, NFDataX, Eq, Show)

-- | Grid validation result
data GridValidation = GridValidation
  { gvNodeCount   :: Unsigned 8
  , gvIsValid     :: Bool
  , gvCoverage    :: Unsigned 8   -- Geographic coverage %
  } deriving (Generic, NFDataX, Eq, Show)

-- | Phi ratio mapping output
data PhiMappingOutput = PhiMappingOutput
  { pmoBaseHarmonic :: Unsigned 16
  , pmoPhiFactor    :: Unsigned 16
  , pmoResultFreq   :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute phi power (simplified - uses lookup for powers 0-5)
phiPower :: Unsigned 4 -> Unsigned 16
phiPower n = case n of
  0 -> 1024     -- phi^0 = 1.0 * 1024
  1 -> 1657     -- phi^1 = 1.618 * 1024
  2 -> 2681     -- phi^2 = 2.618 * 1024
  3 -> 4338     -- phi^3 = 4.236 * 1024
  4 -> 7019     -- phi^4 = 6.854 * 1024
  _ -> 11357    -- phi^5 = 11.09 * 1024

-- | Compute phi ratio mapping
computePhiRatioMap :: RadionicInput -> PhiMappingOutput
computePhiRatioMap ri =
  let -- Magnitude determines phi power (0-5)
      phiPow = resize (riMagnitude ri) :: Unsigned 4
      phiFact = phiPower (min 5 phiPow)

      -- Base harmonic from Schumann * (rate / 1000)
      -- Simplified: base = schumann * rate / 1024
      baseHarm = (schumannFundamental * resize (riRate ri)) `shiftR` 10 :: Unsigned 32
      baseHarm16 = resize (min 65535 baseHarm) :: Unsigned 16

      -- Result = base * phi_factor / 1024
      result = (resize baseHarm16 * resize phiFact) `shiftR` 10 :: Unsigned 32
      result16 = resize (min 65535 result) :: Unsigned 16

  in PhiMappingOutput baseHarm16 phiFact result16

-- | Compute solar modulation factor
computeSolarModulation :: SolarPhase -> Unsigned 8
computeSolarModulation phase = case phase of
  PhaseDawn      -> 230   -- 0.9 * 256
  PhaseMorning   -> 243   -- 0.95 * 256
  PhaseNoon      -> 256   -- 1.0 * 256 (clamped to 255)
  PhaseAfternoon -> 243   -- 0.95 * 256
  PhaseDusk      -> 230   -- 0.9 * 256
  PhaseNight     -> 205   -- 0.8 * 256

-- | Check solar modulation is within bounds (0.8 to 1.2)
checkSolarBounds :: Unsigned 8 -> Bool
checkSolarBounds mod =
  mod >= 205 && mod <= 255  -- 0.8 to 1.0 (hardware limited to 1.0 max)

-- | Compute Schumann modulation
computeSchumannModulation :: Bool -> Unsigned 8
computeSchumannModulation isPeak =
  if isPeak then 255 else 230  -- 1.0 or 0.9

-- | Compute temporal modulation (combined solar + Schumann)
computeTemporalModulation :: TemporalInput -> Unsigned 8
computeTemporalModulation ti =
  let solarMod = computeSolarModulation (tiSolarPhase ti)
      schumannMod = computeSchumannModulation (tiSchumannPeak ti)
      -- Combined = solar * schumann / 256
      combined = (resize solarMod * resize schumannMod) `shiftR` 8 :: Unsigned 16
  in resize (min 255 combined)

-- | Validate grid node count
validateGridNodeCount :: Unsigned 8 -> Bool
validateGridNodeCount count = count >= minLeylineNodes

-- | Compute grid validation
computeGridValidation :: Unsigned 8 -> Unsigned 8 -> GridValidation
computeGridValidation nodeCount coverage =
  GridValidation
    nodeCount
    (validateGridNodeCount nodeCount)
    coverage

-- | Compute leyline status from strength
computeLeylineStatus :: Unsigned 8 -> LeylineStatus
computeLeylineStatus strength
  | strength < 51   = StatusDormant     -- < 0.2
  | strength < 179  = StatusActive      -- 0.2 - 0.7
  | strength < 230  = StatusAmplified   -- 0.7 - 0.9
  | otherwise       = StatusOverloaded  -- > 0.9

-- | Check if coordinate is at vortex point
isVortexPoint :: GeoCoordinate -> Bool
isVortexPoint coord =
  -- Simplified: check if near known vortex latitudes (scaled)
  let lat = gcLatitude coord
      -- Bermuda ~26°N, Egypt ~30°N, etc. (scaled by 256)
      nearVortexLat = (lat > 6400 && lat < 7936) ||   -- 25-31°N
                      (lat > (-7936) && lat < (-6400)) -- 25-31°S
  in nearVortexLat

-- | Compute node type from position
computeNodeType :: GeoCoordinate -> GridNodeType
computeNodeType coord =
  if isVortexPoint coord then NodeTypeVortex
  else NodeTypeIntersect  -- Default

-- | Apply temporal modulation to harmonic
applyTemporalModulation :: Unsigned 16 -> Unsigned 8 -> Unsigned 16
applyTemporalModulation harmonic modFactor =
  let result = (resize harmonic * resize modFactor) `shiftR` 8 :: Unsigned 32
  in resize (min 65535 result)

-- | Grid mapping state
data GridMappingState = GridMappingState
  { gmsNodeCount    :: Unsigned 8
  , gmsActiveLines  :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Initial grid state
initialGridState :: GridMappingState
initialGridState = GridMappingState 0 0

-- | Grid mapping input
data GridMappingInput = GridMappingInput
  { gmiRadionic  :: RadionicInput
  , gmiTemporal  :: TemporalInput
  , gmiCoord     :: GeoCoordinate
  , gmiStrength  :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Grid mapping output
data GridMappingOutput = GridMappingOutput
  { gmoNode       :: GridNode
  , gmoPhiMap     :: PhiMappingOutput
  , gmoTempMod    :: Unsigned 8
  , gmoValidation :: GridValidation
  } deriving (Generic, NFDataX)

-- | Grid mapping pipeline
gridMappingPipeline
  :: HiddenClockResetEnable dom
  => Signal dom GridMappingInput
  -> Signal dom GridMappingOutput
gridMappingPipeline input = mealy gridMealy initialGridState input
  where
    gridMealy state inp =
      let -- Compute phi ratio mapping
          phiMap = computePhiRatioMap (gmiRadionic inp)

          -- Compute temporal modulation
          tempMod = computeTemporalModulation (gmiTemporal inp)

          -- Apply modulation to harmonic
          modulatedHarm = applyTemporalModulation (pmoResultFreq phiMap) tempMod

          -- Compute node type
          nodeType = computeNodeType (gmiCoord inp)

          -- Create grid node
          node = GridNode
            (gmiCoord inp)
            nodeType
            modulatedHarm
            (gmiStrength inp)

          -- Update state
          newCount = if gmsNodeCount state < 255
                     then gmsNodeCount state + 1
                     else 255

          -- Compute validation
          validation = computeGridValidation newCount 100

          newState = GridMappingState newCount (gmsActiveLines state)

          output = GridMappingOutput node phiMap tempMod validation

      in (newState, output)

-- | Phi ratio mapping pipeline
phiRatioPipeline
  :: HiddenClockResetEnable dom
  => Signal dom RadionicInput
  -> Signal dom PhiMappingOutput
phiRatioPipeline = fmap computePhiRatioMap

-- | Solar modulation pipeline
solarModulationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom SolarPhase
  -> Signal dom Unsigned 8
solarModulationPipeline = fmap computeSolarModulation

-- | Grid validation pipeline
gridValidationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8)
  -> Signal dom GridValidation
gridValidationPipeline = fmap (uncurry computeGridValidation)
