{-|
Module      : Ra.Attractors
Description : Ra attractors and emergence modulation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Metaphysical attractors that can induce, modulate, or gate fragment
emergence via coherence manipulation. Inspired by the "Shinigami love
apples" metaphor from Death Note.

== Attractor Concept

Attractors are symbolic or energetic tokens that increase or alter
the emergence potential of certain fragments. They work by:

* Modifying potential and/or flux
* Adjusting inversion or temporal phase
* Optional gating override

== Phase Components

Attractors have flavors corresponding to phase components:

* Emotional: Heart-centered, relational
* Sensory: Body-based, perceptual
* Archetypal: Universal pattern, symbolic

== Target Filtering

Attractors can target:

* Specific fragment IDs
* Shadow-only fragments
* Guardian-gated fragments
* Fragments matching harmonic patterns
-}
module Ra.Attractors
  ( -- * Attractor Type
    Attractor(..)
  , PhaseComponent(..)
  , mkAttractor
  , attractorStrength

    -- * Target Filtering
  , TargetFilter(..)
  , matchesFilter
  , shadowOnly
  , guardianGated

    -- * Attractor Application
  , AttractorEffect(..)
  , applyAttractor
  , modifyPotential
  , modifyFlux
  , modifyInversion

    -- * Emergence Modification
  , EmergenceCondition(..)
  , modifyEmergence
  , coaxEmergence
  , enticeShadow

    -- * Multiple Attractors
  , AttractorField(..)
  , combineAttractors
  , constructiveSync
  , destructiveCancel

    -- * Example Attractors
  , shinigamiApple
  , healingCrystal
  , groundingStone
  , clarityMirror
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse, coherenceFloorPOR )

-- =============================================================================
-- Attractor Type
-- =============================================================================

-- | Attractor definition
data Attractor = Attractor
  { atSymbol        :: !String         -- ^ Symbol name (e.g., "apple")
  , atFlavor        :: !PhaseComponent -- ^ Phase component type
  , atEnticement    :: !Double         -- ^ Enticement level [0,1]
  , atTargetFilter  :: !TargetFilter   -- ^ What fragments it affects
  , atPotentialMod  :: !Double         -- ^ Potential modifier
  , atFluxMod       :: !Double         -- ^ Flux modifier
  , atInversionBias :: !Double         -- ^ Inversion influence [-1,1]
  } deriving (Eq, Show)

-- | Phase component flavor
data PhaseComponent
  = Emotional    -- ^ Heart-centered, relational
  | Sensory      -- ^ Body-based, perceptual
  | Archetypal   -- ^ Universal pattern, symbolic
  | Cognitive    -- ^ Mental, analytical
  | Spiritual    -- ^ Transcendent, unifying
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create attractor from parameters
mkAttractor :: String -> PhaseComponent -> Double -> Attractor
mkAttractor symbol flavor enticement = Attractor
  { atSymbol = symbol
  , atFlavor = flavor
  , atEnticement = clamp01 enticement
  , atTargetFilter = AllFragments
  , atPotentialMod = 1.0 + enticement * phi * 0.1
  , atFluxMod = 1.0 + enticement * 0.2
  , atInversionBias = 0.0
  }

-- | Get effective attractor strength
attractorStrength :: Attractor -> Double
attractorStrength a = atEnticement a * atPotentialMod a

-- =============================================================================
-- Target Filtering
-- =============================================================================

-- | Filter for attractor targets
data TargetFilter
  = AllFragments              -- ^ Affects all fragments
  | SpecificIds ![String]     -- ^ Only specific fragment IDs
  | ShadowOnly                -- ^ Only shadow fragments
  | GuardianGated             -- ^ Only guardian-gated fragments
  | HarmonicMatch !(Int, Int) -- ^ Fragments matching harmonic
  deriving (Eq, Show)

-- | Check if fragment matches filter
matchesFilter :: TargetFilter -> String -> Bool -> Bool -> (Int, Int) -> Bool
matchesFilter filt fragId isShadow isGuarded harmonic =
  case filt of
    AllFragments -> True
    SpecificIds ids -> fragId `elem` ids
    ShadowOnly -> isShadow
    GuardianGated -> isGuarded
    HarmonicMatch (l, m) -> harmonic == (l, m)

-- | Create shadow-only filter
shadowOnly :: TargetFilter
shadowOnly = ShadowOnly

-- | Create guardian-gated filter
guardianGated :: TargetFilter
guardianGated = GuardianGated

-- =============================================================================
-- Attractor Application
-- =============================================================================

-- | Effect of applying an attractor
data AttractorEffect = AttractorEffect
  { aePotentialDelta :: !Double  -- ^ Change in potential
  , aeFluxDelta      :: !Double  -- ^ Change in flux
  , aeInversionDelta :: !Double  -- ^ Change in inversion tendency
  , aePhaseDelta     :: !Double  -- ^ Change in temporal phase
  , aeGatingOverride :: !Bool    -- ^ Override gating?
  } deriving (Eq, Show)

-- | Apply attractor to get effect
applyAttractor :: Attractor -> Double -> AttractorEffect
applyAttractor a coherence =
  let enticement = atEnticement a
      resonance = coherence * enticement

      potentialDelta = (atPotentialMod a - 1.0) * resonance
      fluxDelta = (atFluxMod a - 1.0) * resonance
      inversionDelta = atInversionBias a * resonance
      phaseDelta = enticement * phi * 0.1

      -- Override gating if enticement is high enough
      override = enticement > 0.8 && coherence > 0.3
  in AttractorEffect
      { aePotentialDelta = potentialDelta
      , aeFluxDelta = fluxDelta
      , aeInversionDelta = inversionDelta
      , aePhaseDelta = phaseDelta
      , aeGatingOverride = override
      }

-- | Modify potential with attractor
modifyPotential :: Attractor -> Double -> Double
modifyPotential a potential =
  let effect = applyAttractor a 1.0
  in potential + aePotentialDelta effect

-- | Modify flux with attractor
modifyFlux :: Attractor -> Double -> Double
modifyFlux a flux =
  let effect = applyAttractor a 1.0
  in flux * (1.0 + aeFluxDelta effect)

-- | Modify inversion state with attractor
modifyInversion :: Attractor -> Bool -> Double -> Bool
modifyInversion a currentInversion coherence =
  let effect = applyAttractor a coherence
      delta = aeInversionDelta effect
  in if delta > 0.5
     then True   -- Push toward inversion
     else if delta < -0.5
     then False  -- Push away from inversion
     else currentInversion

-- =============================================================================
-- Emergence Modification
-- =============================================================================

-- | Simplified emergence condition
data EmergenceCondition = EmergenceCondition
  { ecPotential   :: !Double  -- ^ Field potential
  , ecFlux        :: !Double  -- ^ Flux coherence
  , ecInversion   :: !Bool    -- ^ Inverted state
  , ecPhase       :: !Double  -- ^ Temporal phase
  , ecCoherence   :: !Double  -- ^ User coherence
  } deriving (Eq, Show)

-- | Modify emergence condition with attractor
modifyEmergence :: Attractor -> EmergenceCondition -> EmergenceCondition
modifyEmergence a ec =
  let effect = applyAttractor a (ecCoherence ec)
  in ec
      { ecPotential = ecPotential ec + aePotentialDelta effect
      , ecFlux = ecFlux ec * (1.0 + aeFluxDelta effect)
      , ecInversion = modifyInversion a (ecInversion ec) (ecCoherence ec)
      , ecPhase = ecPhase ec + aePhaseDelta effect
      }

-- | Coax a fragment into emergence that wouldn't normally emerge
coaxEmergence :: Attractor -> EmergenceCondition -> Maybe EmergenceCondition
coaxEmergence a ec =
  let modified = modifyEmergence a ec
      threshold = coherenceFloorPOR
  in if ecPotential modified > threshold || aeGatingOverride (applyAttractor a (ecCoherence ec))
     then Just modified
     else Nothing

-- | Entice shadow fragment to emerge
enticeShadow :: Attractor -> EmergenceCondition -> EmergenceCondition
enticeShadow a ec =
  let shadowBiased = a { atInversionBias = 0.5, atTargetFilter = ShadowOnly }
  in modifyEmergence shadowBiased ec

-- =============================================================================
-- Multiple Attractors
-- =============================================================================

-- | Field of multiple attractors
data AttractorField = AttractorField
  { afAttractors :: ![Attractor]
  , afMode       :: !CombineMode
  , afStrength   :: !Double
  } deriving (Eq, Show)

-- | How to combine multiple attractors
data CombineMode
  = Additive      -- ^ Sum effects
  | Multiplicative -- ^ Multiply effects
  | Dominant       -- ^ Strongest wins
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Combine multiple attractors
combineAttractors :: [Attractor] -> CombineMode -> AttractorField
combineAttractors attractors mode =
  let strength = case mode of
        Additive -> sum (map attractorStrength attractors)
        Multiplicative -> product (map attractorStrength attractors)
        Dominant -> maximum (0 : map attractorStrength attractors)
  in AttractorField
      { afAttractors = attractors
      , afMode = mode
      , afStrength = min 2.0 strength  -- Cap at 2x
      }

-- | Calculate constructive synchronization
constructiveSync :: [Attractor] -> Double
constructiveSync attractors =
  let flavors = map atFlavor attractors
      sameFlavor = length (filter (== head flavors) flavors)
  in fromIntegral sameFlavor / fromIntegral (max 1 (length attractors))

-- | Calculate destructive cancellation
destructiveCancel :: [Attractor] -> Double
destructiveCancel attractors =
  let biases = map atInversionBias attractors
      avgBias = sum biases / fromIntegral (max 1 (length biases))
      variance = sum (map (\b -> (b - avgBias) ** 2) biases) / fromIntegral (max 1 (length biases))
  in sqrt variance  -- High variance = cancellation

-- =============================================================================
-- Example Attractors
-- =============================================================================

-- | The Shinigami Apple - enables access where none should exist
shinigamiApple :: Attractor
shinigamiApple = Attractor
  { atSymbol = "shinigami.apple"
  , atFlavor = Archetypal
  , atEnticement = phiInverse  -- 0.618 - golden enticement
  , atTargetFilter = ShadowOnly
  , atPotentialMod = phi        -- Golden boost
  , atFluxMod = 1.5
  , atInversionBias = 0.21      -- Shadow access threshold
  }

-- | Healing crystal - enhances therapeutic emergence
healingCrystal :: Attractor
healingCrystal = Attractor
  { atSymbol = "healing.crystal"
  , atFlavor = Sensory
  , atEnticement = 0.7
  , atTargetFilter = AllFragments
  , atPotentialMod = 1.3
  , atFluxMod = 1.4
  , atInversionBias = -0.3  -- Reduces inversion
  }

-- | Grounding stone - stabilizes field
groundingStone :: Attractor
groundingStone = Attractor
  { atSymbol = "grounding.stone"
  , atFlavor = Sensory
  , atEnticement = 0.5
  , atTargetFilter = AllFragments
  , atPotentialMod = 0.9   -- Slightly reduces potential
  , atFluxMod = 1.2        -- Increases coherence
  , atInversionBias = -0.5 -- Strongly resists inversion
  }

-- | Clarity mirror - enhances mental fragments
clarityMirror :: Attractor
clarityMirror = Attractor
  { atSymbol = "clarity.mirror"
  , atFlavor = Cognitive
  , atEnticement = 0.6
  , atTargetFilter = HarmonicMatch (2, 2)  -- Mental harmonics
  , atPotentialMod = 1.2
  , atFluxMod = 1.3
  , atInversionBias = 0.0
  }

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
