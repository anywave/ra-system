{-|
Module      : Ra.DomainExtensions
Description : Domain extensions for liminal operations
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Handles mathematically undefined or liminal operations (e.g., division
by zero, coherence = 0) by reframing with symbolic attractors and
safe domain logic.

== The Division by Zero Problem

In standard mathematics: 1/0 → Undefined

In Ra scalar terms: "What is the output of no coherence?" → Null emergence

== The Apple Solution

If you introduce an attractor (like an apple to a Shinigami), you modify
the system state to allow relation where there previously was none.

* The apple induces desire, changing the energetic condition
* This increases potential or coherence, making emergence possible again
* You're not dividing by zero anymore - you're resonating across it

== Domain Extension Types

* NullDivision: Division by zero attempted
* CoherenceAbyss: Coherence = 0, no relation
* AttractorResonance: External attractor induces emergence
* ShadowSurface: Inversion surface exists, may echo
-}
module Ra.DomainExtensions
  ( -- * Extension Types
    DomainExtension(..)
  , isDefined
  , isLiminal

    -- * Attractor Type
  , Attractor(..)
  , mkAttractor
  , shinigamiApple

    -- * Domain Results
  , DomainResult(..)
  , isEmergent
  , resultValue

    -- * Safe Operations
  , safeDivide
  , safeCoherence
  , safeEmergence

    -- * Zero Coherence Handling
  , handleZeroCoherence
  , reframeDivisionByZero
  , induceRelation

    -- * Attractor Application
  , applyAttractorToNull
  , resonanceAcrossZero
  , epsilonSubstitution

    -- * Integration with Ra.Emergence
  , EmergenceHook(..)
  , hookDomainExtension
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse, coherenceFloorPOR )

-- =============================================================================
-- Extension Types
-- =============================================================================

-- | Types of domain extensions
data DomainExtension
  = NullDivision        -- ^ Division by zero attempted
  | CoherenceAbyss      -- ^ Coherence = 0, no relation
  | AttractorResonance  -- ^ External attractor induces emergence
  | ShadowSurface       -- ^ Inversion surface exists, may echo
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Check if extension represents defined state
isDefined :: DomainExtension -> Bool
isDefined AttractorResonance = True
isDefined ShadowSurface = True
isDefined _ = False

-- | Check if extension represents liminal state
isLiminal :: DomainExtension -> Bool
isLiminal = not . isDefined

-- =============================================================================
-- Attractor Type (for Domain Extensions)
-- =============================================================================

-- | Attractor for domain extension (simplified from Ra.Attractors)
data Attractor = Attractor
  { attractorId   :: !String
  , potentialLift :: !Double      -- ^ Influence on emergence score
  , coherenceBias :: !(Maybe Double) -- ^ Optional coherence override
  } deriving (Eq, Show)

-- | Create attractor from parameters
mkAttractor :: String -> Double -> Maybe Double -> Attractor
mkAttractor aid lift bias = Attractor
  { attractorId = aid
  , potentialLift = lift
  , coherenceBias = bias
  }

-- | The Shinigami Apple - metaphoric attractor from Death Note
shinigamiApple :: Attractor
shinigamiApple = Attractor
  { attractorId = "shinigami.apple"
  , potentialLift = phiInverse  -- 0.618 - golden ratio potential
  , coherenceBias = Just 0.21   -- Shadow access threshold
  }

-- =============================================================================
-- Domain Results
-- =============================================================================

-- | Outcome of domain-extended operation
data DomainResult
  = Suppressed                  -- ^ No emergence, operation blocked
  | Reframed !Double            -- ^ New valid value (e.g., 1/ε)
  | EmergentViaAttractor !Attractor -- ^ Emergence enabled by attractor
  deriving (Eq, Show)

-- | Check if result allows emergence
isEmergent :: DomainResult -> Bool
isEmergent (EmergentViaAttractor _) = True
isEmergent (Reframed v) = v > 0
isEmergent Suppressed = False

-- | Extract value from result
resultValue :: DomainResult -> Double
resultValue Suppressed = 0.0
resultValue (Reframed v) = v
resultValue (EmergentViaAttractor a) = potentialLift a

-- =============================================================================
-- Safe Operations
-- =============================================================================

-- | Safe division with domain awareness
safeDivide :: Double -> Double -> DomainResult
safeDivide _ 0 = Suppressed
safeDivide x y = Reframed (x / y)

-- | Safe coherence handling
safeCoherence :: Double -> DomainResult
safeCoherence c
  | c <= 0 = Suppressed
  | c < coherenceFloorPOR = Reframed (c * phi)  -- Boost sub-threshold
  | otherwise = Reframed c

-- | Safe emergence evaluation
safeEmergence :: Double -> Double -> Maybe Attractor -> DomainResult
safeEmergence potential coherence mAttractor
  | coherence <= 0 = handleZeroCoherence mAttractor
  | potential <= 0 = Suppressed
  | otherwise = Reframed (potential * coherence)

-- =============================================================================
-- Zero Coherence Handling
-- =============================================================================

-- | Handle emergence when coherence = 0
handleZeroCoherence :: Maybe Attractor -> DomainResult
handleZeroCoherence Nothing = Suppressed
handleZeroCoherence (Just a) = EmergentViaAttractor a

-- | Reframe division by zero via attractor field
reframeDivisionByZero :: Double -> Attractor -> DomainResult
reframeDivisionByZero x attractor =
  let epsilon = 1e-12
      reframed = x / epsilon
      -- Attractor modulates the reframed value
      modulated = reframed * potentialLift attractor
  in if isNaN modulated || isInfinite modulated
     then EmergentViaAttractor attractor  -- Fall back to attractor emergence
     else Reframed (min 1e10 modulated)   -- Cap at reasonable value

-- | Induce relation where none exists
induceRelation :: Attractor -> Double -> DomainResult
induceRelation attractor baseValue =
  let lift = potentialLift attractor
      biased = baseValue + lift
  in case coherenceBias attractor of
      Nothing -> Reframed biased
      Just bias -> if biased > bias
                   then EmergentViaAttractor attractor
                   else Reframed biased

-- =============================================================================
-- Attractor Application
-- =============================================================================

-- | Apply attractor to null/zero state
applyAttractorToNull :: Attractor -> DomainResult
applyAttractorToNull = EmergentViaAttractor

-- | Resonate across zero (metaphoric bridge)
resonanceAcrossZero :: Double -> Attractor -> DomainResult
resonanceAcrossZero value attractor
  | value == 0 = EmergentViaAttractor attractor
  | value < 0 = Reframed (abs value * potentialLift attractor)  -- Invert and lift
  | otherwise = Reframed (value * (1 + potentialLift attractor))

-- | Epsilon substitution for near-zero values
epsilonSubstitution :: Double -> Double
epsilonSubstitution x
  | abs x < 1e-10 = sign x * 1e-10
  | otherwise = x
  where
    sign v = if v >= 0 then 1 else -1

-- =============================================================================
-- Integration with Ra.Emergence
-- =============================================================================

-- | Hook for integrating with emergence system
data EmergenceHook = EmergenceHook
  { ehExtension  :: !DomainExtension
  , ehResult     :: !DomainResult
  , ehOriginal   :: !Double        -- ^ Original value attempted
  , ehAttractor  :: !(Maybe Attractor)
  } deriving (Eq, Show)

-- | Hook domain extension into emergence evaluation
hookDomainExtension :: Double -> Double -> Maybe Attractor -> EmergenceHook
hookDomainExtension potential coherence mAttractor =
  let extension
        | coherence <= 0 = CoherenceAbyss
        | potential <= 0 = NullDivision
        | otherwise = case mAttractor of
            Just _ -> AttractorResonance
            Nothing -> ShadowSurface  -- Default liminal state

      result = safeEmergence potential coherence mAttractor
  in EmergenceHook
      { ehExtension = extension
      , ehResult = result
      , ehOriginal = potential * coherence
      , ehAttractor = mAttractor
      }
