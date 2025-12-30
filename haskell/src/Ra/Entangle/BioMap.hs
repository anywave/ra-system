{-|
Module      : Ra.Entangle.BioMap
Description : Bio-fragment entanglement topology mapper
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Constructs and updates a real-time entanglement graph between:

* User's biometric coherence nodes
* Scalar field concentrations (Ra coordinates)
* Fragment anchors from Ra.Memory
* Chamber field structures from Ra.Chamber

Visualizes and traverses emergent memory linkages, resonance corridors,
and identity gradients.

== Entanglement Theory

Fragments have positions (Ra coordinates), resonance thresholds, and
harmonic fingerprints. Users emit biometric coherence pulses (heart,
breath, neural signatures). Entanglement links form when coherence
matches fragment resonance. Topology evolves based on live field
readings and memory emergence.
-}
module Ra.Entangle.BioMap
  ( -- * Core Types
    EntangledFragment(..)
  , BioNode(..)
  , BodyZone(..)
  , PhaseCode(..)

    -- * Entanglement Graph
  , EntangleGraph
  , emptyGraph
  , graphSize
  , graphLinks

    -- * Link Operations
  , linkFragments
  , unlinkFragment
  , findLinks
  , isLinked

    -- * Fragment Anchors
  , FragmentAnchor(..)
  , FragmentID
  , anchorResonance

    -- * Biometric Input
  , BiometricPulse(..)
  , pulseToBioNodes
  , nodeCoherence

    -- * Entanglement Logic
  , EntangleResult(..)
  , attemptEntangle
  , entangleScore
  , phaseAlignment

    -- * Graph Traversal
  , traverseGraph
  , findPath
  , resonanceCorridor

    -- * Visualization
  , graphToNetwork
  , NetworkNode(..)
  , NetworkEdge(..)

    -- * Real-Time Updates
  , UpdateResult(..)
  , updateGraph
  , pruneStale
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

import Ra.Constants.Extended
  ( phi, phiInverse )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Fragment ID
type FragmentID = String

-- | Entangled fragment state
data EntangledFragment = EntangledFragment
  { efFragmentId     :: !FragmentID
  , efAnchorCoord    :: !RaCoordinate
  , efResonanceScore :: !Double        -- ^ Resonance strength [0, 1]
  , efActiveLink     :: !Bool          -- ^ Currently linked
  , efLastUpdate     :: !Double        -- ^ Timestamp
  } deriving (Eq, Show)

-- | Biometric node in body
data BioNode = BioNode
  { bnUserRegion     :: !BodyZone
  , bnCoherenceValue :: !Double        -- ^ Coherence [0, 1]
  , bnPhaseSignature :: !PhaseCode
  } deriving (Eq, Ord, Show)

-- | Body zone for biometric mapping
data BodyZone
  = Heart           -- ^ Heart center
  | Brain           -- ^ Neural regions
  | SolarPlexus     -- ^ Solar plexus
  | Throat          -- ^ Throat center
  | Root            -- ^ Root/base
  | Crown           -- ^ Crown center
  | ThirdEye        -- ^ Third eye
  | Hands           -- ^ Hand chakras
  | Feet            -- ^ Feet chakras
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Phase code for signature matching
data PhaseCode = PhaseCode
  { pcPhase       :: !Double          -- ^ Phase angle [0, 2*pi]
  , pcFrequency   :: !Double          -- ^ Base frequency (Hz)
  , pcHarmonic    :: !(Int, Int)      -- ^ (l, m) harmonic
  } deriving (Eq, Ord, Show)

-- | Ra coordinate (simplified)
data RaCoordinate = RaCoordinate
  { rcRepitan   :: !Int
  , rcPhi       :: !Int
  , rcHarmonic  :: !Int
  } deriving (Eq, Ord, Show)

-- =============================================================================
-- Entanglement Graph
-- =============================================================================

-- | Entanglement graph: BioNode -> [EntangledFragment]
type EntangleGraph = Map BioNode [EntangledFragment]

-- | Empty graph
emptyGraph :: EntangleGraph
emptyGraph = Map.empty

-- | Get graph size (number of bio nodes)
graphSize :: EntangleGraph -> Int
graphSize = Map.size

-- | Get total link count
graphLinks :: EntangleGraph -> Int
graphLinks = sum . map (length . filter efActiveLink) . Map.elems

-- =============================================================================
-- Link Operations
-- =============================================================================

-- | Link fragments to bio nodes
linkFragments :: [BioNode] -> [FragmentAnchor] -> ScalarField -> EntangleGraph
linkFragments bioNodes anchors field =
  Map.fromList
    [ (node, findMatchingFragments node anchors field)
    | node <- bioNodes
    ]

-- | Find matching fragments for a bio node
findMatchingFragments :: BioNode -> [FragmentAnchor] -> ScalarField -> [EntangledFragment]
findMatchingFragments node anchors field =
  [ EntangledFragment
      { efFragmentId = faId anchor
      , efAnchorCoord = faCoordinate anchor
      , efResonanceScore = score
      , efActiveLink = score >= 0.3
      , efLastUpdate = 0.0
      }
  | anchor <- anchors
  , let score = computeEntangleScore node anchor field
  , score > 0.1  -- Minimum threshold
  ]

-- | Compute entanglement score
computeEntangleScore :: BioNode -> FragmentAnchor -> ScalarField -> Double
computeEntangleScore node anchor field =
  let -- Coherence check
      coherence = bnCoherenceValue node
      floorPassed = coherence >= phiInverse  -- ~0.618

      -- Phase alignment check (within ±0.15)
      nodePhase = pcPhase (bnPhaseSignature node)
      anchorPhase = faPhase anchor
      phaseDelta = abs (nodePhase - anchorPhase)
      phaseAligned = phaseDelta <= 0.15 || phaseDelta >= (2 * pi - 0.15)

      -- Potential flux check
      fluxAtCoord = fieldFluxAt field (faCoordinate anchor)
      fluxPassed = fluxAtCoord > 0.2

      -- Harmonic distance
      nodeHarm = pcHarmonic (bnPhaseSignature node)
      anchorHarm = faHarmonic anchor
      harmDist = harmonicDistance nodeHarm anchorHarm
      harmScore = 1.0 / (1.0 + fromIntegral harmDist)

  in if floorPassed && phaseAligned && fluxPassed
     then coherence * harmScore * phi * 0.5
     else 0.0

-- | Unlink a specific fragment
unlinkFragment :: FragmentID -> EntangleGraph -> EntangleGraph
unlinkFragment fid = Map.map (map unlinkIt)
  where
    unlinkIt ef
      | efFragmentId ef == fid = ef { efActiveLink = False }
      | otherwise = ef

-- | Find links for a bio node
findLinks :: BioNode -> EntangleGraph -> [EntangledFragment]
findLinks node graph = Map.findWithDefault [] node graph

-- | Check if fragment is linked to any node
isLinked :: FragmentID -> EntangleGraph -> Bool
isLinked fid graph =
  any (any (\ef -> efFragmentId ef == fid && efActiveLink ef)) (Map.elems graph)

-- =============================================================================
-- Fragment Anchors
-- =============================================================================

-- | Fragment anchor in field
data FragmentAnchor = FragmentAnchor
  { faId          :: !FragmentID
  , faCoordinate  :: !RaCoordinate
  , faResonance   :: !Double          -- ^ Resonance threshold
  , faPhase       :: !Double          -- ^ Phase signature
  , faHarmonic    :: !(Int, Int)      -- ^ Harmonic fingerprint
  } deriving (Eq, Show)

-- | Get anchor resonance
anchorResonance :: FragmentAnchor -> Double
anchorResonance = faResonance

-- =============================================================================
-- Biometric Input
-- =============================================================================

-- | Biometric pulse from user
data BiometricPulse = BiometricPulse
  { bpHeartCoherence  :: !Double
  , bpBreathPhase     :: !Double
  , bpNeuralState     :: !String
  , bpTimestamp       :: !Double
  } deriving (Eq, Show)

-- | Convert pulse to bio nodes
pulseToBioNodes :: BiometricPulse -> [BioNode]
pulseToBioNodes pulse =
  [ BioNode Heart (bpHeartCoherence pulse)
      (PhaseCode (bpBreathPhase pulse) 7.83 (0, 0))
  , BioNode Brain (neuralCoherence (bpNeuralState pulse))
      (PhaseCode 0.0 (neuralFreq (bpNeuralState pulse)) (1, 0))
  , BioNode SolarPlexus (bpHeartCoherence pulse * 0.8)
      (PhaseCode (bpBreathPhase pulse + pi/4) 10.0 (2, 0))
  ]
  where
    neuralCoherence s = case s of
      "alpha" -> 0.8
      "theta" -> 0.9
      "beta" -> 0.5
      "gamma" -> 0.7
      "delta" -> 0.95
      _ -> 0.5
    neuralFreq s = case s of
      "alpha" -> 10.0
      "theta" -> 6.0
      "beta" -> 20.0
      "gamma" -> 40.0
      "delta" -> 2.0
      _ -> 10.0

-- | Get node coherence
nodeCoherence :: BioNode -> Double
nodeCoherence = bnCoherenceValue

-- =============================================================================
-- Entanglement Logic
-- =============================================================================

-- | Entanglement attempt result
data EntangleResult = EntangleResult
  { erSuccess       :: !Bool
  , erScore         :: !Double
  , erPhaseMatch    :: !Double
  , erMessage       :: !String
  } deriving (Eq, Show)

-- | Attempt to entangle bio node with fragment
attemptEntangle :: BioNode -> FragmentAnchor -> ScalarField -> EntangleResult
attemptEntangle node anchor field =
  let score = computeEntangleScore node anchor field
      success = score >= 0.3

      nodePhase = pcPhase (bnPhaseSignature node)
      anchorPhase = faPhase anchor
      phaseDelta = abs (nodePhase - anchorPhase)
      phaseMatch = 1.0 - min 1.0 (phaseDelta / pi)

      message = if success
                then "Entanglement established"
                else if bnCoherenceValue node < phiInverse
                then "Insufficient coherence"
                else if phaseMatch < 0.5
                then "Phase mismatch"
                else "Flux threshold not met"
  in EntangleResult
      { erSuccess = success
      , erScore = score
      , erPhaseMatch = phaseMatch
      , erMessage = message
      }

-- | Get entanglement score between node and anchor
entangleScore :: BioNode -> FragmentAnchor -> Double
entangleScore node anchor =
  let coherence = bnCoherenceValue node
      resonance = faResonance anchor
  in sqrt (coherence * resonance)

-- | Check phase alignment
phaseAlignment :: BioNode -> FragmentAnchor -> Double
phaseAlignment node anchor =
  let nodePhase = pcPhase (bnPhaseSignature node)
      anchorPhase = faPhase anchor
      delta = abs (nodePhase - anchorPhase)
  in 1.0 - min 1.0 (delta / pi)

-- =============================================================================
-- Graph Traversal
-- =============================================================================

-- | Traverse graph from starting node
traverseGraph :: BioNode -> EntangleGraph -> [(BioNode, EntangledFragment)]
traverseGraph start graph =
  let fragments = findLinks start graph
  in [(start, ef) | ef <- fragments, efActiveLink ef]

-- | Find path between two fragments through bio nodes
findPath :: FragmentID -> FragmentID -> EntangleGraph -> Maybe [BioNode]
findPath from to graph =
  -- Simple DFS to find path
  let allPaths = [ (node, ef)
                 | (node, efs) <- Map.toList graph
                 , ef <- efs
                 , efActiveLink ef
                 ]
      fromNodes = [node | (node, ef) <- allPaths, efFragmentId ef == from]
      toNodes = [node | (node, ef) <- allPaths, efFragmentId ef == to]
  in case (fromNodes, toNodes) of
       (f:_, t:_) -> Just [f, t]  -- Simplified: direct path
       _ -> Nothing

-- | Find resonance corridor (high-coherence path)
resonanceCorridor :: EntangleGraph -> [(BioNode, Double)]
resonanceCorridor graph =
  [ (node, avgResonance)
  | (node, efs) <- Map.toList graph
  , let activeEfs = filter efActiveLink efs
  , not (null activeEfs)
  , let avgResonance = sum (map efResonanceScore activeEfs) /
                       fromIntegral (length activeEfs)
  , avgResonance >= 0.5
  ]

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Network node for visualization
data NetworkNode = NetworkNode
  { nnId      :: !String
  , nnType    :: !String            -- "bio" or "fragment"
  , nnValue   :: !Double            -- Coherence or resonance
  , nnColor   :: !(Int, Int, Int)   -- RGB color
  } deriving (Eq, Show)

-- | Network edge for visualization
data NetworkEdge = NetworkEdge
  { neFrom      :: !String
  , neTo        :: !String
  , neWeight    :: !Double
  , neActive    :: !Bool
  } deriving (Eq, Show)

-- | Convert graph to network representation
graphToNetwork :: EntangleGraph -> ([NetworkNode], [NetworkEdge])
graphToNetwork graph =
  let -- Bio nodes
      bioNodes = [ NetworkNode
                    { nnId = show zone
                    , nnType = "bio"
                    , nnValue = bnCoherenceValue node
                    , nnColor = zoneColor zone
                    }
                 | (node, _) <- Map.toList graph
                 , let zone = bnUserRegion node
                 ]

      -- Fragment nodes
      fragNodes = [ NetworkNode
                    { nnId = efFragmentId ef
                    , nnType = "fragment"
                    , nnValue = efResonanceScore ef
                    , nnColor = (100, 200, 255)
                    }
                  | efs <- Map.elems graph
                  , ef <- efs
                  ]

      -- Edges
      edges = [ NetworkEdge
                { neFrom = show (bnUserRegion node)
                , neTo = efFragmentId ef
                , neWeight = efResonanceScore ef
                , neActive = efActiveLink ef
                }
              | (node, efs) <- Map.toList graph
              , ef <- efs
              ]
  in (bioNodes ++ fragNodes, edges)

-- Zone to color
zoneColor :: BodyZone -> (Int, Int, Int)
zoneColor zone = case zone of
  Heart -> (255, 100, 100)
  Brain -> (200, 100, 255)
  SolarPlexus -> (255, 200, 50)
  Throat -> (100, 200, 255)
  Root -> (200, 50, 50)
  Crown -> (255, 255, 255)
  ThirdEye -> (100, 50, 200)
  Hands -> (200, 255, 200)
  Feet -> (150, 100, 50)

-- =============================================================================
-- Real-Time Updates
-- =============================================================================

-- | Update result
data UpdateResult = UpdateResult
  { urNewGraph      :: !EntangleGraph
  , urLinksAdded    :: !Int
  , urLinksRemoved  :: !Int
  , urMessage       :: !String
  } deriving (Eq, Show)

-- | Update graph with new biometric data
updateGraph :: BiometricPulse -> [FragmentAnchor] -> ScalarField -> EntangleGraph -> UpdateResult
updateGraph pulse anchors field oldGraph =
  let -- Generate new bio nodes from pulse
      bioNodes = pulseToBioNodes pulse

      -- Create new graph
      newGraph = linkFragments bioNodes anchors field

      -- Count changes
      oldLinks = graphLinks oldGraph
      newLinks = graphLinks newGraph
      added = max 0 (newLinks - oldLinks)
      removed = max 0 (oldLinks - newLinks)

      message = "Updated: +" ++ show added ++ "/-" ++ show removed ++ " links"
  in UpdateResult
      { urNewGraph = newGraph
      , urLinksAdded = added
      , urLinksRemoved = removed
      , urMessage = message
      }

-- | Prune stale links (older than threshold)
pruneStale :: Double -> Double -> EntangleGraph -> EntangleGraph
pruneStale currentTime threshold = Map.map (filter isRecent)
  where
    isRecent ef = currentTime - efLastUpdate ef < threshold

-- =============================================================================
-- Scalar Field (simplified)
-- =============================================================================

-- | Scalar field for entanglement calculations
data ScalarField = ScalarField
  { sfFlux      :: !(Map RaCoordinate Double)
  , sfPotential :: !(Map RaCoordinate Double)
  } deriving (Eq, Show)

-- | Get flux at coordinate
fieldFluxAt :: ScalarField -> RaCoordinate -> Double
fieldFluxAt field coord = Map.findWithDefault 0.3 coord (sfFlux field)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Harmonic distance between two (l, m) pairs
harmonicDistance :: (Int, Int) -> (Int, Int) -> Int
harmonicDistance (l1, m1) (l2, m2) =
  abs (l1 - l2) + abs (m1 - m2)
