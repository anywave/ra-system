{-|
Module      : Ra.Projection.Map
Description : Dynamic scalar field grid projection
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Projects the scalar field's emergence gradients across the full 810-state
Ra coordinate space. Tracks live potential evolution and visualizes
emergent "hot zones."

== Coordinate Space

The Ra field is composed of:

* 27 Repitan θ slices
* 6 φ segments (azimuth)
* 5 harmonic depths (h-levels)

Total: 27 × 6 × 5 = 810 discrete coordinate points

== Functionality

* Grid projection: Map field state to alpha values per coordinate
* Gradient overlay: Compute vector flow between points
* Inversion detection: Identify shadow emergence pockets
-}
module Ra.Projection.Map
  ( -- * Core Types
    RaCoordinate(..)
  , EmergenceAlpha
  , GradientVector(..)
  , ScalarField(..)

    -- * Grid Projection
  , projectScalarField
  , projectPoint
  , fullGridProjection

    -- * Gradient Flow
  , computeGradientFlow
  , gradientAt
  , flowDirection

    -- * Inversion Zones
  , InversionState(..)
  , inversionZones
  , isInverted
  , shadowIntensity

    -- * Coordinate Generation
  , allCoordinates
  , repitanRange
  , phiSegments
  , harmonicDepths

    -- * Field Operations
  , fieldAlphaAt
  , fieldPotentialAt
  , fieldCoherenceAt

    -- * Visualization Support
  , alphaToColor
  , gradientToArrow
  , ProjectionView(..)
  , renderView

    -- * Update Mechanics
  , UpdateConfig(..)
  , defaultUpdateConfig
  , debounceUpdate
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Ra coordinate in discrete field space
data RaCoordinate = RaCoordinate
  { rcRepitan   :: !Int           -- ^ θ slice (1-27)
  , rcPhi       :: !Int           -- ^ φ segment (1-6)
  , rcHarmonic  :: !Int           -- ^ h-level (1-5)
  } deriving (Eq, Ord, Show)

-- | Emergence alpha value [0.0, 1.0]
type EmergenceAlpha = Double

-- | Gradient vector in field space
data GradientVector = GradientVector
  { gvDr       :: !Double         -- ^ Radial component (θ direction)
  , gvDphi     :: !Double         -- ^ Azimuthal component (φ direction)
  , gvDh       :: !Double         -- ^ Harmonic component (h direction)
  , gvMagnitude :: !Double        -- ^ Vector magnitude
  } deriving (Eq, Show)

-- | Scalar field representation
data ScalarField = ScalarField
  { sfPotential  :: !(Map RaCoordinate Double)   -- ^ Potential at each point
  , sfCoherence  :: !(Map RaCoordinate Double)   -- ^ Coherence at each point
  , sfPhase      :: !(Map RaCoordinate Double)   -- ^ Phase at each point
  , sfInversion  :: !(Map RaCoordinate InversionState)  -- ^ Inversion states
  , sfTimestamp  :: !Double                      -- ^ Last update time
  } deriving (Eq, Show)

-- | Inversion state
data InversionState
  = Normal              -- ^ Normal polarity
  | Inverted            -- ^ Inverted (shadow)
  | Transitioning       -- ^ Between states
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Grid Projection
-- =============================================================================

-- | Project scalar field to alpha map for all 810 coordinates
projectScalarField :: ScalarField -> Map RaCoordinate EmergenceAlpha
projectScalarField sf =
  Map.fromList [(coord, projectPoint sf coord) | coord <- allCoordinates]

-- | Project single coordinate to alpha
projectPoint :: ScalarField -> RaCoordinate -> EmergenceAlpha
projectPoint sf coord =
  let potential = fieldPotentialAt sf coord
      coherence = fieldCoherenceAt sf coord
      phase = Map.findWithDefault 0.0 coord (sfPhase sf)
      inversion = Map.findWithDefault Normal coord (sfInversion sf)

      -- Base alpha from potential and coherence
      baseAlpha = (potential + coherence) / 2.0

      -- Phase modulation
      phaseModulation = 1.0 + 0.1 * sin phase

      -- Inversion adjustment
      inversionFactor = case inversion of
        Normal -> 1.0
        Inverted -> -0.5
        Transitioning -> 0.5

      -- Window modulation based on harmonic depth
      windowMod = phi ** (fromIntegral (rcHarmonic coord - 1) * (-0.1))

      raw = baseAlpha * phaseModulation * inversionFactor * windowMod
  in clamp01 raw

-- | Full grid projection with metadata
fullGridProjection :: ScalarField -> [(RaCoordinate, EmergenceAlpha, GradientVector)]
fullGridProjection sf =
  let alphaMap = projectScalarField sf
      gradMap = computeGradientFlow sf
  in [ (coord, Map.findWithDefault 0.0 coord alphaMap,
               Map.findWithDefault zeroGradient coord gradMap)
     | coord <- allCoordinates
     ]
  where
    zeroGradient = GradientVector 0 0 0 0

-- =============================================================================
-- Gradient Flow
-- =============================================================================

-- | Compute gradient flow across entire field
computeGradientFlow :: ScalarField -> Map RaCoordinate GradientVector
computeGradientFlow sf =
  Map.fromList [(coord, gradientAt sf coord) | coord <- allCoordinates]

-- | Compute gradient at single point
gradientAt :: ScalarField -> RaCoordinate -> GradientVector
gradientAt sf coord =
  let -- Get alpha at neighboring points
      drPlus = projectPoint sf (shiftRepitan coord 1)
      drMinus = projectPoint sf (shiftRepitan coord (-1))
      dphiPlus = projectPoint sf (shiftPhi coord 1)
      dphiMinus = projectPoint sf (shiftPhi coord (-1))
      dhPlus = projectPoint sf (shiftHarmonic coord 1)
      dhMinus = projectPoint sf (shiftHarmonic coord (-1))

      -- Central differences
      dr = (drPlus - drMinus) / 2.0
      dphi' = (dphiPlus - dphiMinus) / 2.0
      dh = (dhPlus - dhMinus) / 2.0

      -- Magnitude
      mag = sqrt (dr*dr + dphi'*dphi' + dh*dh)
  in GradientVector
      { gvDr = dr
      , gvDphi = dphi'
      , gvDh = dh
      , gvMagnitude = mag
      }

-- | Get flow direction (normalized gradient)
flowDirection :: GradientVector -> (Double, Double, Double)
flowDirection gv
  | gvMagnitude gv < 0.001 = (0, 0, 0)  -- Avoid division by zero
  | otherwise =
      let m = gvMagnitude gv
      in (gvDr gv / m, gvDphi gv / m, gvDh gv / m)

-- Shift coordinate in repitan direction
shiftRepitan :: RaCoordinate -> Int -> RaCoordinate
shiftRepitan c d = c { rcRepitan = wrap 1 27 (rcRepitan c + d) }

-- Shift coordinate in phi direction
shiftPhi :: RaCoordinate -> Int -> RaCoordinate
shiftPhi c d = c { rcPhi = wrap 1 6 (rcPhi c + d) }

-- Shift coordinate in harmonic direction
shiftHarmonic :: RaCoordinate -> Int -> RaCoordinate
shiftHarmonic c d = c { rcHarmonic = clampRange 1 5 (rcHarmonic c + d) }

-- Wrap value in range
wrap :: Int -> Int -> Int -> Int
wrap lo hi val
  | val < lo = hi - (lo - val - 1)
  | val > hi = lo + (val - hi - 1)
  | otherwise = val

-- Clamp to range
clampRange :: Int -> Int -> Int -> Int
clampRange lo hi val = max lo (min hi val)

-- =============================================================================
-- Inversion Zones
-- =============================================================================

-- | Get all coordinates with inverted state
inversionZones :: ScalarField -> [RaCoordinate]
inversionZones sf =
  [ coord | coord <- allCoordinates
          , isInverted sf coord
  ]

-- | Check if coordinate is inverted
isInverted :: ScalarField -> RaCoordinate -> Bool
isInverted sf coord =
  case Map.lookup coord (sfInversion sf) of
    Just Inverted -> True
    _ -> False

-- | Get shadow intensity at coordinate
shadowIntensity :: ScalarField -> RaCoordinate -> Double
shadowIntensity sf coord =
  let inv = Map.findWithDefault Normal coord (sfInversion sf)
      alpha = projectPoint sf coord
  in case inv of
       Inverted -> abs alpha
       Transitioning -> abs alpha * 0.5
       Normal -> 0.0

-- =============================================================================
-- Coordinate Generation
-- =============================================================================

-- | Generate all 810 coordinates
allCoordinates :: [RaCoordinate]
allCoordinates =
  [ RaCoordinate r p h
  | r <- repitanRange
  , p <- phiSegments
  , h <- harmonicDepths
  ]

-- | Repitan range (1-27)
repitanRange :: [Int]
repitanRange = [1..27]

-- | Phi segments (1-6)
phiSegments :: [Int]
phiSegments = [1..6]

-- | Harmonic depths (1-5)
harmonicDepths :: [Int]
harmonicDepths = [1..5]

-- =============================================================================
-- Field Operations
-- =============================================================================

-- | Get alpha at coordinate
fieldAlphaAt :: ScalarField -> RaCoordinate -> EmergenceAlpha
fieldAlphaAt = projectPoint

-- | Get potential at coordinate
fieldPotentialAt :: ScalarField -> RaCoordinate -> Double
fieldPotentialAt sf coord =
  Map.findWithDefault 0.0 coord (sfPotential sf)

-- | Get coherence at coordinate
fieldCoherenceAt :: ScalarField -> RaCoordinate -> Double
fieldCoherenceAt sf coord =
  Map.findWithDefault 0.0 coord (sfCoherence sf)

-- =============================================================================
-- Visualization Support
-- =============================================================================

-- | Convert alpha to RGB color
alphaToColor :: EmergenceAlpha -> (Int, Int, Int)
alphaToColor alpha
  | alpha < 0.1 = (0, 0, 0)                          -- Black (near zero)
  | alpha < 0.3 = (0, 0, colorScale alpha 0.1 0.3)   -- Blue (low)
  | alpha < 0.6 = (colorScale alpha 0.3 0.6, colorScale alpha 0.3 0.6, 255)  -- Cyan-blue
  | alpha < 0.85 = (255, colorScale alpha 0.6 0.85, 0)  -- Orange-gold
  | otherwise = (255, 215, 0)                        -- Gold (high)
  where
    colorScale :: Double -> Double -> Double -> Int
    colorScale a lo hi = round (255 * (a - lo) / (hi - lo))

-- | Convert gradient to arrow representation
gradientToArrow :: GradientVector -> (Double, Double, String)
gradientToArrow gv =
  let (dr, dphi', _) = flowDirection gv
      angle = atan2 dphi' dr
      length' = min 1.0 (gvMagnitude gv * 10)
      arrow = if length' < 0.1 then "·"
              else if angle < 0.79 then "→"
              else if angle < 1.57 then "↗"
              else if angle < 2.36 then "↑"
              else if angle < 3.14 then "↖"
              else "←"
  in (angle, length', arrow)

-- | Projection view for visualization
data ProjectionView = ProjectionView
  { pvSlice       :: !Int              -- ^ Which slice to view
  , pvSliceAxis   :: !SliceAxis        -- ^ Axis for slice
  , pvThreshold   :: !Double           -- ^ Alpha threshold for display
  , pvShowGradient :: !Bool            -- ^ Show gradient arrows
  , pvShowInversion :: !Bool           -- ^ Highlight inversion zones
  } deriving (Eq, Show)

-- | Slice axis
data SliceAxis = RepitanAxis | PhiAxis | HarmonicAxis
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Render view to string representation
renderView :: ScalarField -> ProjectionView -> String
renderView sf view =
  let coords = filterBySlice view allCoordinates
      projected = [(c, projectPoint sf c) | c <- coords]
      filtered = [(c, a) | (c, a) <- projected, a >= pvThreshold view]
  in unlines
       [ showCoord c ++ ": α=" ++ showAlpha a ++ colorCode a
       | (c, a) <- filtered
       ]
  where
    showCoord c = "(" ++ show (rcRepitan c) ++ "," ++
                  show (rcPhi c) ++ "," ++ show (rcHarmonic c) ++ ")"
    showAlpha a = take 5 (show a)
    colorCode a
      | a >= 0.85 = " ★"
      | a >= 0.618 = " ◆"
      | a >= 0.318 = " ●"
      | otherwise = " ○"

-- Filter coordinates by slice
filterBySlice :: ProjectionView -> [RaCoordinate] -> [RaCoordinate]
filterBySlice view = filter matchSlice
  where
    s = pvSlice view
    matchSlice c = case pvSliceAxis view of
      RepitanAxis -> rcRepitan c == s
      PhiAxis -> rcPhi c == s
      HarmonicAxis -> rcHarmonic c == s

-- =============================================================================
-- Update Mechanics
-- =============================================================================

-- | Update configuration
data UpdateConfig = UpdateConfig
  { ucDebounceMs    :: !Int           -- ^ Debounce interval (ms)
  , ucAutoUpdate    :: !Bool          -- ^ Auto-update on field change
  , ucThreshold     :: !Double        -- ^ Minimum change to trigger update
  } deriving (Eq, Show)

-- | Default update configuration
defaultUpdateConfig :: UpdateConfig
defaultUpdateConfig = UpdateConfig
  { ucDebounceMs = 1000   -- 1 Hz max update rate
  , ucAutoUpdate = True
  , ucThreshold = 0.01
  }

-- | Check if update should be debounced
debounceUpdate :: UpdateConfig -> Double -> Double -> Bool
debounceUpdate config lastTime currentTime =
  (currentTime - lastTime) * 1000 >= fromIntegral (ucDebounceMs config)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
