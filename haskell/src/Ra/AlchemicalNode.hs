{-|
Module      : Ra.AlchemicalNode
Description : ORMES-state stabilization nodes for emergence enhancement
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Embeds ORMES-inspired scalar nodes into resonance chambers to stabilize
or enhance emergence, particularly for sensitive or shadow fragments.

== ORMES Theory

=== Orbital Rearranged Monoatomic Elements

ORMES (also known as m-state elements) exhibit:

* High-spin, low-energy states
* Superconducting properties at room temperature
* Enhanced biofield coupling
* Memory and coherence preservation

=== Mineral Phase States

Nodes are configured with mineral-phase encoding:

* Gold (Au) → Spiritual clarity, crown activation
* Platinum (Pt) → Neural coherence, mental focus
* Iridium (Ir) → Third eye, intuition enhancement
* Rhodium (Rh) → Heart coherence, emotional balance
* Copper (Cu) → Grounding, root stability

=== Emergence Enhancement

Nodes enhance emergence by:

* Providing coherence anchors
* Stabilizing sensitive fragments
* Enabling shadow fragment access
* Creating resonance lattices
-}
module Ra.AlchemicalNode
  ( -- * Core Types
    AlchemicalNode(..)
  , MineralPhase(..)
  , NodeState(..)
  , mkNode

    -- * Node Configuration
  , NodeConfig(..)
  , configFromMineral
  , configForUser

    -- * Emergence Influence
  , NodeInfluence(..)
  , computeInfluence
  , influenceScore
  , proximityEffect

    -- * Chamber Integration
  , NodePlacement(..)
  , placeNode
  , removeNode
  , nodeGrid

    -- * Stabilization
  , StabilizationField(..)
  , stabilize
  , stabilizationStrength
  , coherenceAnchor

    -- * Shadow Enhancement
  , ShadowNode(..)
  , enhanceShadow
  , shadowAccess
  , safetyGate

    -- * Visualization
  , NodeVisual(..)
  , visualize
  , mineralColor
  , phaseGlow
  ) where

import Ra.Constants.Extended
  ( phi, coherenceFloorPOR )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Alchemical node for emergence enhancement
data AlchemicalNode = AlchemicalNode
  { anId            :: !String          -- ^ Node identifier
  , anMineral       :: !MineralPhase    -- ^ Mineral type
  , anPosition      :: !(Double, Double, Double)  -- ^ 3D position
  , anStrength      :: !Double          -- ^ Node strength [0,1]
  , anRadius        :: !Double          -- ^ Influence radius
  , anState         :: !NodeState       -- ^ Current state
  , anCoherence     :: !Double          -- ^ Local coherence [0,1]
  } deriving (Eq, Show)

-- | Mineral phase state
data MineralPhase
  = Gold        -- ^ Spiritual clarity, crown
  | Platinum    -- ^ Neural coherence, mental
  | Iridium     -- ^ Third eye, intuition
  | Rhodium     -- ^ Heart coherence, emotional
  | Copper      -- ^ Grounding, root
  | Silver      -- ^ Lunar, emotional flow
  | Palladium   -- ^ Bridge, transition states
  | Ruthenium   -- ^ DNA activation, cellular
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Node operational state
data NodeState
  = Dormant     -- ^ Not active
  | Activating  -- ^ Powering up
  | Active      -- ^ Fully operational
  | Resonating  -- ^ In resonance with user
  | Depleted    -- ^ Needs recharge
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create a new alchemical node
mkNode :: String -> MineralPhase -> (Double, Double, Double) -> AlchemicalNode
mkNode nodeId mineral pos = AlchemicalNode
  { anId = nodeId
  , anMineral = mineral
  , anPosition = pos
  , anStrength = mineralBaseStrength mineral
  , anRadius = mineralRadius mineral
  , anState = Dormant
  , anCoherence = 0.5
  }

-- Base strength by mineral
mineralBaseStrength :: MineralPhase -> Double
mineralBaseStrength m = case m of
  Gold     -> 0.9
  Platinum -> 0.85
  Iridium  -> 0.8
  Rhodium  -> 0.75
  Copper   -> 0.6
  Silver   -> 0.7
  Palladium -> 0.65
  Ruthenium -> 0.7

-- Influence radius by mineral
mineralRadius :: MineralPhase -> Double
mineralRadius m = case m of
  Gold     -> 2.0
  Platinum -> 1.5
  Iridium  -> 1.8
  Rhodium  -> 1.6
  Copper   -> 1.2
  Silver   -> 1.4
  Palladium -> 1.3
  Ruthenium -> 1.0

-- =============================================================================
-- Node Configuration
-- =============================================================================

-- | Node configuration
data NodeConfig = NodeConfig
  { ncMinerals      :: ![MineralPhase]  -- ^ Active minerals
  , ncGridSize      :: !Int             -- ^ Grid dimensions
  , ncBaseStrength  :: !Double          -- ^ Base strength multiplier
  , ncShadowEnabled :: !Bool            -- ^ Allow shadow enhancement
  , ncUserCoherence :: !Double          -- ^ User coherence factor
  } deriving (Eq, Show)

-- | Create config for specific mineral focus
configFromMineral :: MineralPhase -> NodeConfig
configFromMineral primary = NodeConfig
  { ncMinerals = [primary]
  , ncGridSize = 3
  , ncBaseStrength = 1.0
  , ncShadowEnabled = primary == Iridium || primary == Silver
  , ncUserCoherence = 0.5
  }

-- | Create config optimized for user profile
configForUser :: Double -> [String] -> NodeConfig
configForUser coherence semantics = NodeConfig
  { ncMinerals = selectMinerals semantics
  , ncGridSize = if coherence > 0.7 then 5 else 3
  , ncBaseStrength = 0.8 + coherence * 0.4
  , ncShadowEnabled = "shadow" `elem` semantics
  , ncUserCoherence = coherence
  }

-- Select minerals based on semantic profile
selectMinerals :: [String] -> [MineralPhase]
selectMinerals semantics =
  let mapping = [ ("spiritual", Gold)
                , ("mental", Platinum)
                , ("intuition", Iridium)
                , ("heart", Rhodium)
                , ("grounding", Copper)
                , ("emotional", Silver)
                , ("transition", Palladium)
                , ("healing", Ruthenium)
                ]
      matches = [m | (key, m) <- mapping, key `elem` semantics]
  in if null matches then [Copper] else matches  -- Default to grounding

-- =============================================================================
-- Emergence Influence
-- =============================================================================

-- | Influence on emergence
data NodeInfluence = NodeInfluence
  { niScoreBoost    :: !Double      -- ^ Emergence score modifier
  , niCoherenceBoost :: !Double     -- ^ Coherence boost
  , niStabilization :: !Double      -- ^ Stabilization factor
  , niPhaseShift    :: !Double      -- ^ Phase modulation
  , niMineral       :: !MineralPhase
  } deriving (Eq, Show)

-- | Compute influence from node at distance
computeInfluence :: AlchemicalNode -> Double -> Double -> NodeInfluence
computeInfluence node distance userCoherence =
  let -- Distance falloff
      falloff = if distance < anRadius node
                then 1.0 - (distance / anRadius node)
                else 0.0

      -- State factor
      stateFactor = case anState node of
        Dormant    -> 0.0
        Activating -> 0.3
        Active     -> 1.0
        Resonating -> 1.2
        Depleted   -> 0.1

      -- Combined strength
      strength = anStrength node * falloff * stateFactor

      -- Score boost from mineral type and coherence resonance
      scoreBoost = strength * mineralScoreBoost (anMineral node) * (0.5 + userCoherence * 0.5)

      -- Coherence boost
      coherenceBoost = strength * phi * 0.1

      -- Stabilization from mineral
      stabilization = strength * mineralStabilization (anMineral node)

      -- Phase shift
      phaseShift = strength * mineralPhaseShift (anMineral node)
  in NodeInfluence
      { niScoreBoost = scoreBoost
      , niCoherenceBoost = coherenceBoost
      , niStabilization = stabilization
      , niPhaseShift = phaseShift
      , niMineral = anMineral node
      }

-- | Get total influence score
influenceScore :: NodeInfluence -> Double
influenceScore ni = niScoreBoost ni + niCoherenceBoost ni

-- | Calculate proximity effect
proximityEffect :: [AlchemicalNode] -> (Double, Double, Double) -> Double
proximityEffect nodes pos =
  let influences = map (\n -> computeInfluence n (distance3D pos (anPosition n)) 0.5) nodes
  in sum (map influenceScore influences)

-- Mineral-specific modifiers
mineralScoreBoost :: MineralPhase -> Double
mineralScoreBoost m = case m of
  Gold     -> 0.15
  Platinum -> 0.12
  Iridium  -> 0.10
  Rhodium  -> 0.08
  Copper   -> 0.05
  Silver   -> 0.07
  Palladium -> 0.06
  Ruthenium -> 0.09

mineralStabilization :: MineralPhase -> Double
mineralStabilization m = case m of
  Gold     -> 0.8
  Platinum -> 0.7
  Iridium  -> 0.5
  Rhodium  -> 0.9
  Copper   -> 1.0
  Silver   -> 0.6
  Palladium -> 0.7
  Ruthenium -> 0.6

mineralPhaseShift :: MineralPhase -> Double
mineralPhaseShift m = case m of
  Gold     -> pi / 8
  Platinum -> pi / 12
  Iridium  -> pi / 6
  Rhodium  -> pi / 10
  Copper   -> 0.0
  Silver   -> pi / 4
  Palladium -> pi / 16
  Ruthenium -> pi / 20

-- =============================================================================
-- Chamber Integration
-- =============================================================================

-- | Node placement in chamber
data NodePlacement = NodePlacement
  { npNodes      :: ![AlchemicalNode]
  , npChamberId  :: !String
  , npGridType   :: !String           -- ^ "cubic", "tetrahedral", "spherical"
  , npTotalStrength :: !Double
  } deriving (Eq, Show)

-- | Place node in chamber
placeNode :: AlchemicalNode -> NodePlacement -> NodePlacement
placeNode node placement =
  let newNodes = node : npNodes placement
      newStrength = npTotalStrength placement + anStrength node
  in placement
      { npNodes = newNodes
      , npTotalStrength = newStrength
      }

-- | Remove node from chamber
removeNode :: String -> NodePlacement -> NodePlacement
removeNode nodeId placement =
  let newNodes = filter (\n -> anId n /= nodeId) (npNodes placement)
      newStrength = sum (map anStrength newNodes)
  in placement
      { npNodes = newNodes
      , npTotalStrength = newStrength
      }

-- | Create a grid of nodes
nodeGrid :: NodeConfig -> String -> NodePlacement
nodeGrid config chamberId =
  let size = ncGridSize config
      minerals = ncMinerals config
      spacing = 1.0 / fromIntegral (max 1 (size - 1))

      -- Generate grid positions
      positions = [(fromIntegral x * spacing, fromIntegral y * spacing, fromIntegral z * spacing)
                  | x <- [0..size-1], y <- [0..size-1], z <- [0..size-1]]

      -- Create nodes
      nodes = zipWith3 createGridNode [0..] (cycle minerals) positions
      createGridNode :: Int -> MineralPhase -> (Double, Double, Double) -> AlchemicalNode
      createGridNode idx mineral pos =
        (mkNode (chamberId ++ "_node_" ++ show idx) mineral pos)
          { anStrength = mineralBaseStrength mineral * ncBaseStrength config }
  in NodePlacement
      { npNodes = nodes
      , npChamberId = chamberId
      , npGridType = "cubic"
      , npTotalStrength = sum (map anStrength nodes)
      }

-- =============================================================================
-- Stabilization
-- =============================================================================

-- | Stabilization field from nodes
data StabilizationField = StabilizationField
  { sfStrength    :: !Double      -- ^ Total stabilization
  , sfCoherence   :: !Double      -- ^ Field coherence
  , sfAnchors     :: ![String]    -- ^ Active anchor node IDs
  , sfCoverage    :: !Double      -- ^ Spatial coverage [0,1]
  } deriving (Eq, Show)

-- | Create stabilization from node placement
stabilize :: NodePlacement -> Double -> StabilizationField
stabilize placement userCoherence =
  let activeNodes = filter (\n -> anState n == Active || anState n == Resonating) (npNodes placement)

      -- Total stabilization
      totalStab = sum (map (\n -> anStrength n * mineralStabilization (anMineral n)) activeNodes)

      -- Field coherence from nodes
      nodeCoherence = if null activeNodes then 0 else sum (map anCoherence activeNodes) / fromIntegral (length activeNodes)
      fieldCoherence = (nodeCoherence + userCoherence) / 2

      -- Anchors
      anchors = map anId activeNodes

      -- Coverage estimate
      coverage = min 1.0 (fromIntegral (length activeNodes) * 0.2)
  in StabilizationField
      { sfStrength = min 1.0 totalStab
      , sfCoherence = fieldCoherence
      , sfAnchors = anchors
      , sfCoverage = coverage
      }

-- | Get stabilization strength
stabilizationStrength :: StabilizationField -> Double
stabilizationStrength = sfStrength

-- | Create coherence anchor at position
coherenceAnchor :: [AlchemicalNode] -> (Double, Double, Double) -> Double
coherenceAnchor nodes pos =
  let -- Find closest node
      distances = map (\n -> (distance3D pos (anPosition n), n)) nodes
      sorted = sortByFirst distances
      closest = snd (head sorted)

      -- Anchor strength from closest
      anchorStrength = anStrength closest * anCoherence closest
  in if null nodes then 0 else anchorStrength

-- =============================================================================
-- Shadow Enhancement
-- =============================================================================

-- | Shadow-enhanced node
data ShadowNode = ShadowNode
  { snBase        :: !AlchemicalNode
  , snShadowAccess :: !Double     -- ^ Shadow access level [0,1]
  , snSafetyGate  :: !Bool        -- ^ Safety gate enabled
  , snThreshold   :: !Double      -- ^ Access threshold
  } deriving (Eq, Show)

-- | Enhance node for shadow work
enhanceShadow :: AlchemicalNode -> Double -> ShadowNode
enhanceShadow node userCoherence =
  let -- Shadow minerals have natural access
      baseAccess = case anMineral node of
        Iridium -> 0.6
        Silver  -> 0.4
        Gold    -> 0.3
        _       -> 0.1

      -- Boost from coherence
      accessLevel = baseAccess + userCoherence * 0.3

      -- Safety gate for low coherence
      safetyEnabled = userCoherence < coherenceFloorPOR

      -- Threshold from mineral
      threshold = 0.4 - mineralStabilization (anMineral node) * 0.1
  in ShadowNode
      { snBase = node
      , snShadowAccess = clamp01 accessLevel
      , snSafetyGate = safetyEnabled
      , snThreshold = threshold
      }

-- | Get shadow access level
shadowAccess :: ShadowNode -> Double
shadowAccess = snShadowAccess

-- | Check safety gate
safetyGate :: ShadowNode -> Bool
safetyGate = snSafetyGate

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Visual representation of node
data NodeVisual = NodeVisual
  { nvColor     :: !(Int, Int, Int)   -- ^ RGB color
  , nvGlow      :: !Double            -- ^ Glow intensity [0,1]
  , nvPulseRate :: !Double            -- ^ Pulse frequency (Hz)
  , nvSize      :: !Double            -- ^ Visual size
  , nvSymbol    :: !String            -- ^ Alchemical symbol
  } deriving (Eq, Show)

-- | Create visual for node
visualize :: AlchemicalNode -> NodeVisual
visualize node =
  let color = mineralColor (anMineral node)
      glow = phaseGlow node
      pulseRate = anStrength node * 2.0  -- 0-2 Hz
      size = anRadius node * 0.5
      symbol = mineralSymbol (anMineral node)
  in NodeVisual
      { nvColor = color
      , nvGlow = glow
      , nvPulseRate = pulseRate
      , nvSize = size
      , nvSymbol = symbol
      }

-- | Get mineral color
mineralColor :: MineralPhase -> (Int, Int, Int)
mineralColor m = case m of
  Gold     -> (255, 215, 0)     -- Gold
  Platinum -> (229, 228, 226)   -- Platinum silver
  Iridium  -> (148, 0, 211)     -- Violet
  Rhodium  -> (255, 182, 193)   -- Pink
  Copper   -> (184, 115, 51)    -- Copper brown
  Silver   -> (192, 192, 192)   -- Silver
  Palladium -> (189, 183, 170)  -- Pale gold
  Ruthenium -> (64, 224, 208)   -- Turquoise

-- | Calculate glow intensity
phaseGlow :: AlchemicalNode -> Double
phaseGlow node =
  let stateFactor = case anState node of
        Dormant    -> 0.0
        Activating -> 0.3
        Active     -> 0.6
        Resonating -> 1.0
        Depleted   -> 0.1
  in stateFactor * anCoherence node

-- Mineral alchemical symbols
mineralSymbol :: MineralPhase -> String
mineralSymbol m = case m of
  Gold     -> "☉"  -- Sun
  Platinum -> "♇"  -- Pluto (modern)
  Iridium  -> "☿"  -- Mercury (eye connection)
  Rhodium  -> "♡"  -- Heart
  Copper   -> "♀"  -- Venus
  Silver   -> "☽"  -- Moon
  Palladium -> "⚹"  -- Sextile
  Ruthenium -> "⚕"  -- Caduceus

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | 3D distance
distance3D :: (Double, Double, Double) -> (Double, Double, Double) -> Double
distance3D (x1, y1, z1) (x2, y2, z2) =
  sqrt ((x2-x1)^(2::Int) + (y2-y1)^(2::Int) + (z2-z1)^(2::Int))

-- | Sort by first element
sortByFirst :: Ord a => [(a, b)] -> [(a, b)]
sortByFirst = foldr insertSorted []
  where
    insertSorted x [] = [x]
    insertSorted x (y:ys)
      | fst x <= fst y = x : y : ys
      | otherwise = y : insertSorted x ys

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
