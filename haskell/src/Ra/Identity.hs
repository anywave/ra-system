{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DerivingStrategies #-}

{-|
Module      : Ra.Identity
Description : Ra Scalar Identity null emergence boundary
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Formalizes the Ra Scalar Identity null emergence boundary:

@
  Phi^n . H_{l,m}(theta, phi) - Delta(ankh) = 0
@

When the golden-scaled field harmonic equals the ankh-derived imbalance,
emergence is suppressed (NullEmergence). This represents the boundary
condition where potential and flux coherence cancel out.

Key insight: The identity boundary is where field conditions exactly
balance, producing no net emergence - a stable null point in the
scalar field topology.
-}
module Ra.Identity
  ( -- * Types
    Potential(..)
  , FluxCoherence(..)
  , HarmonicWeight(..)
  , AnkhDelta(..)
  , WindowDepth(..)
  , NullEpsilon(..)
  , IdentityResult(..)

    -- * Constants
  , invAnkh
  , coherenceFloor
  , defaultEpsilon

    -- * Operators
  , (⊣⊢)

    -- * Core Functions
  , phiTerm
  , ankhBalance
  , evaluateIdentity
  , noRelation
  , checkEmergencePermitted

    -- * Predicates
  , isNullEmergence
  , isActiveEmergence
  , emergenceValue
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)
import Ra.Constants (Ankh(..), ankh, GreenPhi(..), greenPhi)

-- ---------------------------------------------------------------------
-- Types
-- ---------------------------------------------------------------------

-- | Scalar potential at a coordinate, bounded [0, 1]
newtype Potential = Potential { unPotential :: Double }
  deriving stock (Show, Eq, Ord, Generic)
  deriving newtype (Num, NFData)

-- | Flux coherence (field stability), bounded [0, 1]
newtype FluxCoherence = FluxCoherence { unFluxCoherence :: Double }
  deriving stock (Show, Eq, Ord, Generic)
  deriving newtype (Num, NFData)

-- | Harmonic weight for spherical harmonic contribution
newtype HarmonicWeight = HarmonicWeight { unHarmonicWeight :: Double }
  deriving stock (Show, Eq, Ord, Generic)
  deriving newtype (Num, NFData)

-- | Ankh-derived imbalance delta
newtype AnkhDelta = AnkhDelta { unAnkhDelta :: Double }
  deriving stock (Show, Eq, Ord, Generic)
  deriving newtype (Num, NFData)

-- | Window depth index for phi scaling (0 = base level)
newtype WindowDepth = WindowDepth { unWindowDepth :: Int }
  deriving stock (Show, Eq, Ord, Generic)
  deriving newtype (Num, Enum, NFData)

-- | Null emergence epsilon threshold
newtype NullEpsilon = NullEpsilon { unNullEpsilon :: Double }
  deriving stock (Show, Eq, Ord, Generic)
  deriving newtype (Num, NFData)

-- | Result of identity evaluation
data IdentityResult
  = NullEmergence           -- ^ Field balanced, no emergence
  | ActiveEmergence !Double -- ^ Emergence active with intensity
  deriving (Show, Eq, Generic)

instance NFData IdentityResult

-- ---------------------------------------------------------------------
-- Constants
-- ---------------------------------------------------------------------

-- | Inverse of Ankh constant: 1 / 5.08938
invAnkh :: Double
invAnkh = 1.0 / unAnkh ankh

-- | Coherence floor: phi / ankh (minimum coherence for any relation)
-- Below this threshold, no emergence relation exists
coherenceFloor :: Double
coherenceFloor = unGreenPhi greenPhi / unAnkh ankh

-- | Default null epsilon threshold
defaultEpsilon :: NullEpsilon
defaultEpsilon = NullEpsilon 1e-9

-- | Phi value extracted for internal use
phi :: Double
phi = unGreenPhi greenPhi

-- ---------------------------------------------------------------------
-- Operators
-- ---------------------------------------------------------------------

-- | Cancellation operator: computes the difference (imbalance)
-- When a ⊣⊢ b approaches zero, we're at the null emergence boundary
infixl 6 ⊣⊢
(⊣⊢) :: Double -> Double -> Double
a ⊣⊢ b = a - b

-- ---------------------------------------------------------------------
-- Core Functions
-- ---------------------------------------------------------------------

-- | Phi term at given window depth: phi^n
-- phi^0 = 1.0, phi^1 = 1.62, phi^2 = 2.6244, etc.
phiTerm :: WindowDepth -> Double
phiTerm (WindowDepth n)
  | n < 0     = 1.0 / (phi ^ abs n)  -- Negative depth = inverse scaling
  | otherwise = phi ^ n

-- | Compute ankh-derived balance from field parameters
-- Combines potential, flux coherence, and harmonic weight
-- scaled by the inverse ankh constant
ankhBalance :: Potential -> FluxCoherence -> HarmonicWeight -> AnkhDelta
ankhBalance (Potential pot) (FluxCoherence flux) (HarmonicWeight harm) =
  AnkhDelta $ invAnkh * (pot * flux * harm)

-- | Evaluate the scalar identity equation
-- Returns NullEmergence when: |phi^n * harmonic - delta| < epsilon
-- Otherwise returns ActiveEmergence with the residual magnitude
evaluateIdentity
  :: WindowDepth      -- ^ Window depth for phi scaling
  -> Double           -- ^ Harmonic value H_{l,m}
  -> AnkhDelta        -- ^ Ankh-derived delta
  -> NullEpsilon      -- ^ Epsilon threshold
  -> IdentityResult
evaluateIdentity depth harmonic (AnkhDelta delta) (NullEpsilon eps) =
  let phiScaled = phiTerm depth * harmonic
      residual = phiScaled ⊣⊢ delta
  in if abs residual < eps
       then NullEmergence
       else ActiveEmergence (abs residual)

-- | Check if coherence is below the floor (no relation possible)
-- When True, the observer has insufficient coherence for any
-- field interaction - they exist outside the emergence domain
noRelation :: Double -> Bool
noRelation coh = coh < coherenceFloor

-- | Combined check: first tests relational substrate, then identity balance
-- Returns NullEmergence if:
--   1. Coherence is below floor (no relational substrate), OR
--   2. Identity equation is satisfied (field balanced)
checkEmergencePermitted
  :: Double           -- ^ Coherence value
  -> WindowDepth      -- ^ Window depth for phi scaling
  -> Double           -- ^ Harmonic value H_{l,m}
  -> Potential        -- ^ Scalar potential
  -> FluxCoherence    -- ^ Flux coherence
  -> HarmonicWeight   -- ^ Harmonic weight
  -> NullEpsilon      -- ^ Epsilon threshold
  -> IdentityResult
checkEmergencePermitted coh depth harmonic pot flux hw eps
  | noRelation coh = NullEmergence
  | otherwise      = evaluateIdentity depth harmonic (ankhBalance pot flux hw) eps

-- ---------------------------------------------------------------------
-- Predicates
-- ---------------------------------------------------------------------

-- | Check if result is null emergence
isNullEmergence :: IdentityResult -> Bool
isNullEmergence NullEmergence = True
isNullEmergence _             = False

-- | Check if result is active emergence
isActiveEmergence :: IdentityResult -> Bool
isActiveEmergence (ActiveEmergence _) = True
isActiveEmergence _                   = False

-- | Extract emergence value (0 for null, residual for active)
emergenceValue :: IdentityResult -> Double
emergenceValue NullEmergence        = 0.0
emergenceValue (ActiveEmergence v)  = v
