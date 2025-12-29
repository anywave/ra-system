{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DerivingStrategies #-}

{-|
Module      : Ra.Emergence
Description : Field to content emergence algorithm
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Reference implementation for field → content emergence algorithm.
Transforms scalar field + coordinate into fragment emergence.

Key insight: Content doesn't "exist" until field conditions are met.
Partial alignment → partial content (preview, blur, summary).
-}
module Ra.Emergence
  ( -- * Configuration
    EmergenceParams(..)
  , defaultParams

    -- * Types
  , ContentForm(..)
  , PartialForm(..)
  , EmergenceResult(..)
  , EmergentContent(..)
  , EmergenceScore(..)
  , Potential(..)
  , FluxCoherence(..)

    -- * Core Functions
  , computeEmergenceScore
  , temporalModifier
  , selectContentForm
  , evaluateEmergence
  , classifyEmergence

    -- * Predicates
  , isNoRelation
  , isDark
  , isPartial
  , isFull

    -- * Invariants
  , scoreInvariant
  , alphaInvariant
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)
import Ra.Scalar (Inversion(..))
import Ra.Temporal (AlignmentMultiplier(..), WindowPhase(..))
import Ra.Identity (coherenceFloor, noRelation)

-- | Emergence algorithm parameters
data EmergenceParams = EmergenceParams
  { epPotentialWeight  :: !Double   -- ^ Weight for potential (default 0.7)
  , epFluxWeight       :: !Double   -- ^ Weight for flux coherence (default 0.3)
  , epFullThreshold    :: !Double   -- ^ Score for full emergence
  , epPartialThreshold :: !Double   -- ^ Score for partial emergence
  , epShadowThreshold  :: !Double   -- ^ Score for shadow emergence
  } deriving (Show, Eq, Generic)

-- | Default parameters derived from Ra constants
defaultParams :: EmergenceParams
defaultParams = EmergenceParams
  { epPotentialWeight  = 0.7
  , epFluxWeight       = 0.3
  , epFullThreshold    = 0.85    -- ~RAC1 threshold
  , epPartialThreshold = 0.40    -- ~RAC3 threshold
  , epShadowThreshold  = 0.25    -- ~RAC5 threshold
  }

-- | Content manifestation forms
data ContentForm
  = FormFull      -- ^ Complete, unredacted
  | FormPreview   -- ^ First N characters/frames
  | FormSummary   -- ^ AI-generated summary
  | FormBlur      -- ^ Degraded/obscured version
  | FormPointer   -- ^ Just the ID, no content
  | FormShadow    -- ^ Inverted/complement form
  | FormNone      -- ^ No content
  deriving (Show, Eq, Ord, Generic)

instance NFData ContentForm

-- | Partial content forms for degraded emergence
data PartialForm
  = Preview     -- ^ First N characters/frames
  | Summary     -- ^ AI-generated summary
  | Blur        -- ^ Degraded/obscured version
  | Pointer     -- ^ Just the ID, no content
  deriving (Show, Eq, Ord, Generic)

instance NFData PartialForm

-- | High-level emergence classification result
-- Order of evaluation: NoRelation → Blocked → Dark → Partial → Full
data EmergenceResult
  = NoRelation              -- ^ No relational substrate (coh < coherenceFloor)
  | Dark                    -- ^ Field exists but unreachable (coh < 0.40)
  | Partial !PartialForm !Double  -- ^ Partial alignment with form and alpha
  | Full                    -- ^ Full alignment (coh >= 0.85)
  | Shadow                  -- ^ Inverted state
  | Blocked                 -- ^ Consent denied (future)
  deriving (Show, Eq, Generic)

instance NFData EmergenceResult

-- | Emergence score ∈ [0, 1]
newtype EmergenceScore = EmergenceScore { unScore :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Scalar potential at coordinate ∈ [0, 1]
newtype Potential = Potential { unPotential :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Flux coherence (field stability) ∈ [0, 1]
newtype FluxCoherence = FluxCoherence { unFlux :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Emerged content with metadata
data EmergentContent = EmergentContent
  { ecFragmentId :: !String
  , ecForm       :: !ContentForm
  , ecAlpha      :: !Double       -- ^ Emergence intensity [0, 1]
  , ecMetadata   :: ![(String, String)]
  } deriving (Show, Eq, Generic)

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 = max 0.0 . min 1.0

-- | Temporal modifier based on window phase
-- Peaks at phase=0.5 (window center), range [0.8, 1.0]
-- Uses linear shape per Architect (not sine)
temporalModifier :: WindowPhase -> Double
temporalModifier (WindowPhase phase) =
  0.8 + 0.2 * (1.0 - abs (phase - 0.5) * 2.0)

-- | Compute raw emergence score
computeEmergenceScore
  :: EmergenceParams
  -> Potential
  -> FluxCoherence
  -> WindowPhase
  -> AlignmentMultiplier
  -> EmergenceScore
computeEmergenceScore params (Potential pot) (FluxCoherence flux) phase (AlignmentMultiplier mult) =
  EmergenceScore $ clamp01 $ rawScore * modifier * mult
  where
    rawScore = epPotentialWeight params * pot + epFluxWeight params * flux
    modifier = temporalModifier phase

-- | Select content form based on emergence score
-- Uses discrete bands per Architect
selectContentForm :: EmergenceParams -> EmergenceScore -> Inversion -> ContentForm
selectContentForm params (EmergenceScore score) inversion
  | inversion == Inverted =
      if score >= epShadowThreshold params
        then FormShadow
        else FormNone
  | score >= epFullThreshold params = FormFull
  | score >= 0.75 = FormPreview
  | score >= 0.55 = FormSummary
  | score >= 0.35 = FormBlur
  | score >= epPartialThreshold params = FormPointer
  | otherwise = FormNone

-- | Compute alpha (emergence intensity) based on form
computeAlpha :: EmergenceParams -> EmergenceScore -> ContentForm -> Double
computeAlpha _ _ FormFull = 1.0
computeAlpha _ _ FormNone = 0.0
computeAlpha params (EmergenceScore score) _ =
  clamp01 $ (score - epPartialThreshold params) /
            (epFullThreshold params - epPartialThreshold params)

-- | Main emergence evaluation
evaluateEmergence
  :: EmergenceParams
  -> String            -- ^ Fragment ID
  -> Potential
  -> FluxCoherence
  -> WindowPhase
  -> AlignmentMultiplier
  -> Inversion
  -> EmergentContent
evaluateEmergence params fragId pot flux phase mult inv =
  EmergentContent
    { ecFragmentId = fragId
    , ecForm = form
    , ecAlpha = clamp01 alpha
    , ecMetadata =
        [ ("emergence_score", show (unScore score))
        , ("inversion", show inv)
        ]
    }
  where
    score = computeEmergenceScore params pot flux phase mult
    form = selectContentForm params score inv
    alpha = computeAlpha params score form

-- | Classify emergence based on coherence value
-- Evaluation order:
--   1. NoRelation if coherence < coherenceFloor (~0.318)
--   2. Dark if coherence < 0.40
--   3. Partial with form based on coherence band
--   4. Full if coherence >= 0.85
classifyEmergence :: Double -> Inversion -> EmergenceResult
classifyEmergence coh inv
  | noRelation coh = NoRelation
  | inv == Inverted = Shadow
  | coh >= 0.85 = Full
  | coh >= 0.75 = Partial Preview alpha
  | coh >= 0.55 = Partial Summary alpha
  | coh >= 0.40 = Partial Blur alpha
  | coh >= coherenceFloor = Dark  -- Between floor and 0.40
  | otherwise = NoRelation        -- Should not reach here
  where
    -- Alpha interpolates within partial range [0.40, 0.85]
    alpha = (coh - 0.40) / (0.85 - 0.40)

-- ---------------------------------------------------------------------
-- Predicates
-- ---------------------------------------------------------------------

-- | Check if result is NoRelation
isNoRelation :: EmergenceResult -> Bool
isNoRelation NoRelation = True
isNoRelation _          = False

-- | Check if result is Dark
isDark :: EmergenceResult -> Bool
isDark Dark = True
isDark _    = False

-- | Check if result is Partial
isPartial :: EmergenceResult -> Bool
isPartial (Partial _ _) = True
isPartial _             = False

-- | Check if result is Full
isFull :: EmergenceResult -> Bool
isFull Full = True
isFull _    = False

-- ---------------------------------------------------------------------
-- Invariants
-- ---------------------------------------------------------------------

-- | Invariant: emergence score in [0, 1]
scoreInvariant :: EmergenceScore -> Bool
scoreInvariant (EmergenceScore s) = s >= 0 && s <= 1

-- | Invariant: alpha in [0, 1]
alphaInvariant :: EmergentContent -> Bool
alphaInvariant ec = ecAlpha ec >= 0 && ecAlpha ec <= 1
