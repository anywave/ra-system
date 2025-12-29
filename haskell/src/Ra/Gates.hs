{-|
Module      : Ra.Gates
Description : AccessResult type + gating logic from spec Section 4
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Access gating determines whether a fragment/signal can emerge based on
coherence and consent levels.

From Section 4 of ra_integration_spec.md:

@
AccessLevel(user_coherence, fragment_rac) → {FullAccess, PartialAccess(α), Blocked}

threshold(R_f) = RAC(R_f) / RAC₁
C_floor = φ_green / Ankh ≈ 0.3183
C_ceiling = 1.0

If C_u ≥ threshold(R_f):
    return FullAccess
Else If C_u ≥ C_floor:
    α = (C_u - C_floor) / (threshold(R_f) - C_floor)
    return PartialAccess(α)
Else:
    return Blocked
@
-}
module Ra.Gates
    ( -- * Access Result Type
      AccessResult(..)
    , isFullAccess
    , isPartialAccess
    , isBlocked
    , accessAlpha
      -- * Coherence Bounds
    , coherenceFloor
    , coherenceCeiling
      -- * Gating Functions
    , accessLevel
    , racThreshold
    , canAccess
    , effectiveCoherence
      -- * Partial Emergence
    , partialEmergence
      -- * Resonance
    , resonanceScore
    , ResonanceWeights(..)
    , defaultWeights
    ) where

import Ra.Constants (Ankh(..), ankh, GreenPhi(..), greenPhi)
import Ra.Rac (RacLevel(..), racValue, RacValue(..), racValueNormalized)
import Ra.Repitans (Repitan, repitanValue, nextRepitan)

-- | Result of access gating check
data AccessResult
    = FullAccess                -- ^ Complete emergence allowed
    | PartialAccess !Double     -- ^ Partial emergence with intensity α ∈ (0, 1)
    | Blocked                   -- ^ No emergence allowed
    deriving (Show, Eq)

-- | Check if result is FullAccess
isFullAccess :: AccessResult -> Bool
isFullAccess FullAccess = True
isFullAccess _ = False

-- | Check if result is PartialAccess
isPartialAccess :: AccessResult -> Bool
isPartialAccess (PartialAccess _) = True
isPartialAccess _ = False

-- | Check if result is Blocked
isBlocked :: AccessResult -> Bool
isBlocked Blocked = True
isBlocked _ = False

-- | Extract alpha value from PartialAccess, 0 for Blocked, 1 for FullAccess
accessAlpha :: AccessResult -> Double
accessAlpha FullAccess = 1.0
accessAlpha (PartialAccess a) = a
accessAlpha Blocked = 0.0

-- | Coherence floor: φ_green / Ankh = 1.62 / 5.08938 ≈ 0.3183
coherenceFloor :: Double
coherenceFloor = unGreenPhi greenPhi / unAnkh ankh

-- | Coherence ceiling: 1.0
coherenceCeiling :: Double
coherenceCeiling = 1.0

-- | Get threshold for a RAC level (normalized to RAC1)
racThreshold :: RacLevel -> Double
racThreshold = racValueNormalized

-- | Core gating function from spec Section 4.1
-- Determines access level based on user coherence and fragment RAC requirement
accessLevel :: Double -> RacLevel -> AccessResult
accessLevel userCoherence fragmentRac
    | userCoherence >= threshold = FullAccess
    | userCoherence >= coherenceFloor =
        let alpha = (userCoherence - coherenceFloor) / (threshold - coherenceFloor)
        in PartialAccess (max 0 $ min 1 alpha)
    | otherwise = Blocked
  where
    threshold = racThreshold fragmentRac

-- | Simple check if access is allowed (not Blocked)
canAccess :: Double -> RacLevel -> Bool
canAccess coherence rac = not $ isBlocked $ accessLevel coherence rac

-- | Calculate effective coherence given access result
-- Maps FullAccess → 1.0, PartialAccess(α) → α, Blocked → 0.0
effectiveCoherence :: AccessResult -> Double
effectiveCoherence = accessAlpha

-- | Calculate partial emergence within a Repitan band
-- From spec Section 4.4
partialEmergence :: Repitan -> Double -> Double
partialEmergence currentBand alpha =
    let bandLow = repitanValue currentBand
        bandHigh = repitanValue (nextRepitan currentBand)
    in bandLow + alpha * (bandHigh - bandLow)

-- | Weights for resonance score calculation
data ResonanceWeights = ResonanceWeights
    { weightTheta    :: !Double  -- ^ θ alignment weight
    , weightPhi      :: !Double  -- ^ φ access weight
    , weightHarmonic :: !Double  -- ^ h harmonic match weight
    , weightRadius   :: !Double  -- ^ r intensity weight
    } deriving (Show, Eq)

-- | Default weights from spec Section 5.3
-- w_θ = 0.3, w_φ = 0.4, w_h = 0.2, w_r = 0.1
defaultWeights :: ResonanceWeights
defaultWeights = ResonanceWeights
    { weightTheta = 0.3
    , weightPhi = 0.4
    , weightHarmonic = 0.2
    , weightRadius = 0.1
    }

-- | Calculate composite resonance score
-- resonance = w_θ × θ_match + w_φ × φ_access + w_h × h_match + w_r × r_intensity
resonanceScore
    :: ResonanceWeights  -- ^ Weights (should sum to 1)
    -> Double            -- ^ θ match score [0, 1]
    -> Double            -- ^ φ access score [0, 1]
    -> Double            -- ^ h harmonic match [0, 1]
    -> Double            -- ^ r intensity [0, 1]
    -> Double            -- ^ Composite score [0, 1]
resonanceScore weights thetaMatch phiAccess harmonicMatch intensity =
    let ResonanceWeights wt wp wh wr = weights
    in wt * thetaMatch + wp * phiAccess + wh * harmonicMatch + wr * intensity
