{-|
Module      : Ra.PlanetaryGrid
Description : Planetary leyline mapping and energy grid systems
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements planetary energy grid mapping including leylines, vortex points,
and sacred geometry overlays. Provides tools for locating and analyzing
Earth's energetic infrastructure.

== Planetary Grid Theory

=== Leyline Networks

* Ancient energy pathways across Earth's surface
* Intersection points create power nodes
* Alignment with sacred sites and monuments
* Seasonal and celestial activation patterns

=== Grid Geometries

1. Icosahedral grid (Becker-Hagens)
2. Hartmann grid (magnetic)
3. Curry grid (diagonal)
4. Schumann resonance zones
-}
module Ra.PlanetaryGrid
  ( -- * Core Types
    PlanetaryGrid(..)
  , Leyline(..)
  , GridNode(..)
  , VortexPoint(..)

    -- * Grid Construction
  , createGrid
  , defaultPlanetaryGrid
  , addLeyline
  , addNode

    -- * Leyline Operations
  , findLeylines
  , leylineStrength
  , leylineIntersections
  , alignmentAngle

    -- * Node Analysis
  , NodeType(..)
  , nodeEnergy
  , nodeActivation
  , nearestNode

    -- * Vortex Points
  , VortexPolarity(..)
  , createVortex
  , vortexStrength
  , vortexRange

    -- * Geographic Functions
  , GeoCoord(..)
  , distance
  , bearing
  , pointsAlongLeyline

    -- * Grid Queries
  , gridEnergyAt
  , activeLeylines
  , dominantFrequency

    -- * Sacred Geometry
  , GridGeometry(..)
  , overlayGeometry
  , geometryNodes
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete planetary grid system
data PlanetaryGrid = PlanetaryGrid
  { pgLeylines    :: ![Leyline]        -- ^ All mapped leylines
  , pgNodes       :: ![GridNode]       -- ^ Major grid nodes
  , pgVortices    :: ![VortexPoint]    -- ^ Vortex points
  , pgGeometry    :: !GridGeometry     -- ^ Active geometry overlay
  , pgBaseFreq    :: !Double           -- ^ Base frequency (Hz)
  , pgActivation  :: !Double           -- ^ Global activation level [0, 1]
  } deriving (Eq, Show)

-- | Single leyline definition
data Leyline = Leyline
  { llId          :: !String           -- ^ Leyline identifier
  , llName        :: !String           -- ^ Human name
  , llStart       :: !GeoCoord         -- ^ Start coordinate
  , llEnd         :: !GeoCoord         -- ^ End coordinate
  , llStrength    :: !Double           -- ^ Energy strength [0, 1]
  , llWidth       :: !Double           -- ^ Width in km
  , llFrequency   :: !Double           -- ^ Dominant frequency (Hz)
  , llActive      :: !Bool             -- ^ Currently active
  } deriving (Eq, Show)

-- | Grid node (intersection point)
data GridNode = GridNode
  { gnId          :: !String           -- ^ Node identifier
  , gnName        :: !String           -- ^ Location name
  , gnCoord       :: !GeoCoord         -- ^ Geographic coordinates
  , gnType        :: !NodeType         -- ^ Node classification
  , gnEnergy      :: !Double           -- ^ Energy level [0, 1]
  , gnRadius      :: !Double           -- ^ Influence radius (km)
  , gnConnections :: ![String]         -- ^ Connected leyline IDs
  } deriving (Eq, Show)

-- | Vortex point definition
data VortexPoint = VortexPoint
  { vpId          :: !String           -- ^ Vortex identifier
  , vpCoord       :: !GeoCoord         -- ^ Center coordinate
  , vpPolarity    :: !VortexPolarity   -- ^ Energy polarity
  , vpStrength    :: !Double           -- ^ Vortex strength
  , vpRadius      :: !Double           -- ^ Effect radius (km)
  , vpRotation    :: !Double           -- ^ Rotation rate (deg/hour)
  } deriving (Eq, Show)

-- | Geographic coordinate
data GeoCoord = GeoCoord
  { gcLat         :: !Double           -- ^ Latitude (-90 to 90)
  , gcLon         :: !Double           -- ^ Longitude (-180 to 180)
  , gcAlt         :: !Double           -- ^ Altitude (m)
  } deriving (Eq, Show)

-- | Grid node types
data NodeType
  = NodePrimary      -- ^ Major intersection (5+ leylines)
  | NodeSecondary    -- ^ Secondary intersection (3-4 leylines)
  | NodeTertiary     -- ^ Minor intersection (2 leylines)
  | NodeSacredSite   -- ^ Known sacred site
  | NodeNatural      -- ^ Natural formation
  | NodeConstructed  -- ^ Human-made structure
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Vortex polarity
data VortexPolarity
  = PolarityInflow   -- ^ Energy flows inward
  | PolarityOutflow  -- ^ Energy flows outward
  | PolarityNeutral  -- ^ Balanced flow
  | PolarityOscillating  -- ^ Alternating polarity
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Grid geometry types
data GridGeometry
  = GeometryIcosahedral    -- ^ 20-face icosahedron
  | GeometryDodecahedral   -- ^ 12-face dodecahedron
  | GeometryHartmann       -- ^ Hartmann magnetic grid
  | GeometryCurry          -- ^ Curry diagonal grid
  | GeometryBeckerHagens   -- ^ Combined UVG grid
  | GeometryCustom         -- ^ User-defined
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Grid Construction
-- =============================================================================

-- | Create empty planetary grid
createGrid :: GridGeometry -> PlanetaryGrid
createGrid geometry = PlanetaryGrid
  { pgLeylines = []
  , pgNodes = []
  , pgVortices = []
  , pgGeometry = geometry
  , pgBaseFreq = 7.83  -- Schumann resonance
  , pgActivation = 0.5
  }

-- | Default planetary grid with major features
defaultPlanetaryGrid :: PlanetaryGrid
defaultPlanetaryGrid = PlanetaryGrid
  { pgLeylines = defaultLeylines
  , pgNodes = defaultNodes
  , pgVortices = defaultVortices
  , pgGeometry = GeometryBeckerHagens
  , pgBaseFreq = 7.83
  , pgActivation = phiInverse
  }

-- | Add leyline to grid
addLeyline :: PlanetaryGrid -> Leyline -> PlanetaryGrid
addLeyline grid leyline =
  grid { pgLeylines = leyline : pgLeylines grid }

-- | Add node to grid
addNode :: PlanetaryGrid -> GridNode -> PlanetaryGrid
addNode grid node =
  grid { pgNodes = node : pgNodes grid }

-- =============================================================================
-- Leyline Operations
-- =============================================================================

-- | Find leylines near a coordinate
findLeylines :: PlanetaryGrid -> GeoCoord -> Double -> [Leyline]
findLeylines grid coord maxDist =
  filter (leylineNear coord maxDist) (pgLeylines grid)

-- | Calculate leyline strength at point
leylineStrength :: Leyline -> GeoCoord -> Double
leylineStrength leyline coord =
  let d = distanceToLeyline leyline coord
      falloff = max 0 (1 - d / (llWidth leyline * 2))
  in llStrength leyline * falloff * (if llActive leyline then 1 else 0.3)

-- | Find all leyline intersections
leylineIntersections :: PlanetaryGrid -> [(GeoCoord, [Leyline])]
leylineIntersections grid =
  let lls = pgLeylines grid
      pairs = [(l1, l2) | l1 <- lls, l2 <- lls, llId l1 < llId l2]
      intersects = [(intersectionPoint l1 l2, [l1, l2]) | (l1, l2) <- pairs
                   , let ip = intersectionPoint l1 l2
                   , isValidIntersection ip]
  in intersects

-- | Calculate alignment angle between leylines
alignmentAngle :: Leyline -> Leyline -> Double
alignmentAngle l1 l2 =
  let b1 = bearing (llStart l1) (llEnd l1)
      b2 = bearing (llStart l2) (llEnd l2)
      diff = abs (b1 - b2)
  in if diff > 180 then 360 - diff else diff

-- =============================================================================
-- Node Analysis
-- =============================================================================

-- | Get energy level at node
nodeEnergy :: GridNode -> Double
nodeEnergy node =
  let typeFactor = case gnType node of
        NodePrimary -> phi
        NodeSecondary -> 1.0
        NodeTertiary -> phiInverse
        NodeSacredSite -> phi * phi
        NodeNatural -> 1.0
        NodeConstructed -> phiInverse
      connectionFactor = 1 + fromIntegral (length (gnConnections node)) * 0.1
  in gnEnergy node * typeFactor * connectionFactor

-- | Check if node is activated
nodeActivation :: PlanetaryGrid -> GridNode -> Double
nodeActivation grid node =
  let connectedLeylines = filter ((`elem` gnConnections node) . llId) (pgLeylines grid)
      activeCount = length (filter llActive connectedLeylines)
      totalCount = length connectedLeylines
      activationRatio = if totalCount > 0
                        then fromIntegral activeCount / fromIntegral totalCount
                        else 0
  in gnEnergy node * activationRatio * pgActivation grid

-- | Find nearest node to coordinate
nearestNode :: PlanetaryGrid -> GeoCoord -> Maybe (GridNode, Double)
nearestNode grid coord =
  case pgNodes grid of
    [] -> Nothing
    nodes -> Just $ minimumBy' (\(_, d1) (_, d2) -> compare d1 d2)
             [(n, distance coord (gnCoord n)) | n <- nodes]

-- =============================================================================
-- Vortex Points
-- =============================================================================

-- | Create vortex point
createVortex :: String -> GeoCoord -> VortexPolarity -> Double -> VortexPoint
createVortex vId coord polarity strength = VortexPoint
  { vpId = vId
  , vpCoord = coord
  , vpPolarity = polarity
  , vpStrength = strength
  , vpRadius = strength * 100  -- km
  , vpRotation = case polarity of
      PolarityInflow -> 15
      PolarityOutflow -> -15
      PolarityNeutral -> 0
      PolarityOscillating -> 30
  }

-- | Calculate vortex strength at distance
vortexStrength :: VortexPoint -> Double -> Double
vortexStrength vortex dist =
  let normalized = dist / vpRadius vortex
      falloff = max 0 (1 - normalized)
  in vpStrength vortex * falloff * falloff

-- | Get effective vortex range
vortexRange :: VortexPoint -> Double
vortexRange vortex = vpRadius vortex * phi

-- =============================================================================
-- Geographic Functions
-- =============================================================================

-- | Calculate distance between coordinates (km)
distance :: GeoCoord -> GeoCoord -> Double
distance c1 c2 =
  let lat1 = gcLat c1 * pi / 180
      lat2 = gcLat c2 * pi / 180
      dlat = (gcLat c2 - gcLat c1) * pi / 180
      dlon = (gcLon c2 - gcLon c1) * pi / 180
      a = sin (dlat/2)^(2::Int) + cos lat1 * cos lat2 * sin (dlon/2)^(2::Int)
      c = 2 * atan2 (sqrt a) (sqrt (1-a))
  in 6371 * c  -- Earth radius in km

-- | Calculate bearing between coordinates (degrees)
bearing :: GeoCoord -> GeoCoord -> Double
bearing c1 c2 =
  let lat1 = gcLat c1 * pi / 180
      lat2 = gcLat c2 * pi / 180
      dlon = (gcLon c2 - gcLon c1) * pi / 180
      x = sin dlon * cos lat2
      y = cos lat1 * sin lat2 - sin lat1 * cos lat2 * cos dlon
  in atan2 x y * 180 / pi

-- | Get points along leyline
pointsAlongLeyline :: Leyline -> Int -> [GeoCoord]
pointsAlongLeyline leyline n =
  [ interpolateCoord (llStart leyline) (llEnd leyline) (fromIntegral i / fromIntegral (n-1))
  | i <- [0..n-1]
  ]

-- =============================================================================
-- Grid Queries
-- =============================================================================

-- | Get total grid energy at coordinate
gridEnergyAt :: PlanetaryGrid -> GeoCoord -> Double
gridEnergyAt grid coord =
  let leylineEnergy = sum [leylineStrength ll coord | ll <- pgLeylines grid]
      nodeEnergy' = sum [gnEnergy n * max 0 (1 - distance coord (gnCoord n) / gnRadius n)
                        | n <- pgNodes grid]
      vortexEnergy = sum [vortexStrength v (distance coord (vpCoord v)) | v <- pgVortices grid]
  in (leylineEnergy + nodeEnergy' + vortexEnergy) * pgActivation grid

-- | Get currently active leylines
activeLeylines :: PlanetaryGrid -> [Leyline]
activeLeylines = filter llActive . pgLeylines

-- | Get dominant frequency at location
dominantFrequency :: PlanetaryGrid -> GeoCoord -> Double
dominantFrequency grid coord =
  let nearbyLeylines = findLeylines grid coord 100
      freqs = map llFrequency nearbyLeylines
  in if null freqs then pgBaseFreq grid else sum freqs / fromIntegral (length freqs)

-- =============================================================================
-- Sacred Geometry
-- =============================================================================

-- | Overlay geometry pattern on grid
overlayGeometry :: PlanetaryGrid -> GridGeometry -> PlanetaryGrid
overlayGeometry grid geometry =
  let newNodes = geometryNodes geometry
  in grid { pgGeometry = geometry, pgNodes = pgNodes grid ++ newNodes }

-- | Generate nodes for geometry type
geometryNodes :: GridGeometry -> [GridNode]
geometryNodes GeometryIcosahedral = icosahedralNodes
geometryNodes GeometryDodecahedral = dodecahedralNodes
geometryNodes GeometryBeckerHagens = beckerHagensNodes
geometryNodes _ = []

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Check if leyline is near coordinate
leylineNear :: GeoCoord -> Double -> Leyline -> Bool
leylineNear coord maxDist leyline =
  distanceToLeyline leyline coord <= maxDist

-- | Distance from point to leyline
distanceToLeyline :: Leyline -> GeoCoord -> Double
distanceToLeyline leyline coord =
  let closest = closestPointOnLeyline leyline coord
  in distance coord closest

-- | Find closest point on leyline to coordinate
closestPointOnLeyline :: Leyline -> GeoCoord -> GeoCoord
closestPointOnLeyline leyline coord =
  let start = llStart leyline
      end' = llEnd leyline
      -- Project coord onto line segment
      t = projectOntoSegment start end' coord
  in interpolateCoord start end' (max 0 (min 1 t))

-- | Project point onto line segment (returns parameter t)
projectOntoSegment :: GeoCoord -> GeoCoord -> GeoCoord -> Double
projectOntoSegment start end' point =
  let dx = gcLon end' - gcLon start
      dy = gcLat end' - gcLat start
      lenSq = dx*dx + dy*dy
  in if lenSq < 0.0001 then 0
     else let px = gcLon point - gcLon start
              py = gcLat point - gcLat start
          in (px*dx + py*dy) / lenSq

-- | Interpolate between coordinates
interpolateCoord :: GeoCoord -> GeoCoord -> Double -> GeoCoord
interpolateCoord c1 c2 t = GeoCoord
  { gcLat = gcLat c1 + t * (gcLat c2 - gcLat c1)
  , gcLon = gcLon c1 + t * (gcLon c2 - gcLon c1)
  , gcAlt = gcAlt c1 + t * (gcAlt c2 - gcAlt c1)
  }

-- | Calculate intersection point of two leylines
intersectionPoint :: Leyline -> Leyline -> GeoCoord
intersectionPoint l1 l2 =
  -- Simplified intersection (assumes flat Earth for nearby points)
  let (x1, y1) = (gcLon (llStart l1), gcLat (llStart l1))
      (x2, y2) = (gcLon (llEnd l1), gcLat (llEnd l1))
      (x3, y3) = (gcLon (llStart l2), gcLat (llStart l2))
      (x4, y4) = (gcLon (llEnd l2), gcLat (llEnd l2))
      denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
  in if abs denom < 0.0001
     then GeoCoord 0 0 0  -- Parallel lines
     else let t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
          in GeoCoord (y1 + t*(y2-y1)) (x1 + t*(x2-x1)) 0

-- | Check if intersection is valid
isValidIntersection :: GeoCoord -> Bool
isValidIntersection coord =
  gcLat coord >= -90 && gcLat coord <= 90 &&
  gcLon coord >= -180 && gcLon coord <= 180

-- | Minimum with custom comparison
minimumBy' :: (a -> a -> Ordering) -> [a] -> a
minimumBy' _ [x] = x
minimumBy' cmp (x:xs) = foldr (\a b -> if cmp a b == LT then a else b) x xs
minimumBy' _ [] = error "minimumBy': empty list"

-- =============================================================================
-- Default Data
-- =============================================================================

-- | Default major leylines
defaultLeylines :: [Leyline]
defaultLeylines =
  [ Leyline "michael" "St. Michael's Line" (GeoCoord 51.18 1.83 0) (GeoCoord 50.06 (-5.71) 0) 0.9 5 7.83 True
  , Leyline "mary" "Mary Line" (GeoCoord 51.18 1.83 0) (GeoCoord 50.06 (-5.71) 0) 0.85 4 8.5 True
  , Leyline "apollo" "Apollo Line" (GeoCoord 37.97 23.72 0) (GeoCoord 51.50 (-0.12) 0) 0.8 6 7.83 True
  ]

-- | Default major nodes
defaultNodes :: [GridNode]
defaultNodes =
  [ GridNode "stonehenge" "Stonehenge" (GeoCoord 51.18 (-1.83) 100) NodeSacredSite 0.95 50 ["michael", "mary"]
  , GridNode "glastonbury" "Glastonbury Tor" (GeoCoord 51.14 (-2.70) 150) NodeSacredSite 0.9 30 ["michael", "mary"]
  , GridNode "avebury" "Avebury" (GeoCoord 51.43 (-1.85) 160) NodeSacredSite 0.85 40 ["michael"]
  ]

-- | Default vortex points
defaultVortices :: [VortexPoint]
defaultVortices =
  [ VortexPoint "sedona" (GeoCoord 34.87 (-111.76) 1300) PolarityOutflow 0.9 80 15
  , VortexPoint "uluru" (GeoCoord (-25.34) 131.03 863) PolarityInflow 0.85 100 (-10)
  ]

-- | Icosahedral grid nodes (12 vertices)
icosahedralNodes :: [GridNode]
icosahedralNodes =
  [ GridNode ("ico_" ++ show i) ("Icosahedral " ++ show i) (GeoCoord lat lon 0)
      NodePrimary 0.8 200 []
  | (i, (lat, lon)) <- zip [(1::Int)..] icosahedralCoords
  ]

-- | Icosahedral vertex coordinates
icosahedralCoords :: [(Double, Double)]
icosahedralCoords =
  [ (90, 0), (-90, 0)  -- Poles
  , (26.57, 0), (26.57, 72), (26.57, 144), (26.57, -144), (26.57, -72)
  , (-26.57, 36), (-26.57, 108), (-26.57, 180), (-26.57, -108), (-26.57, -36)
  ]

-- | Dodecahedral grid nodes (20 vertices)
dodecahedralNodes :: [GridNode]
dodecahedralNodes =
  [ GridNode ("dod_" ++ show i) ("Dodecahedral " ++ show i) (GeoCoord lat lon 0)
      NodeSecondary 0.7 150 []
  | (i, (lat, lon)) <- zip [(1::Int)..] dodecahedralCoords
  ]

-- | Dodecahedral vertex coordinates (simplified)
dodecahedralCoords :: [(Double, Double)]
dodecahedralCoords =
  [ (52.62, i * 72) | i <- [0..4] ] ++
  [ (10.81, 36 + i * 72) | i <- [0..4] ] ++
  [ (-10.81, i * 72) | i <- [0..4] ] ++
  [ (-52.62, 36 + i * 72) | i <- [0..4] ]

-- | Becker-Hagens UVG nodes (62 points)
beckerHagensNodes :: [GridNode]
beckerHagensNodes = icosahedralNodes ++ take 10 dodecahedralNodes
