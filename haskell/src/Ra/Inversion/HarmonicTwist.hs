{-|
Module      : Ra.Inversion.HarmonicTwist
Description : Harmonic inversion and shadow emergence mechanics
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements harmonic inversion mechanics for shadow emergence, phase reversal,
and torsion field twisting. When coherence drops below critical thresholds
or Δ(ankh) inverts, fragments may emerge in shadow form.

== Inversion Theory

=== Phase Inversion

Normal emergence: coherence → content manifestation
Inverted emergence: anti-coherence → shadow manifestation

The twist point occurs at:

* Coherence below φ⁻¹ (~0.618)
* Δ(ankh) sign reversal
* Torsion field polarity flip

=== Shadow Harmonics

Shadow harmonics are the complement of normal harmonics:
H'_{l,m} = (-1)^(l+m) × H_{l,m}

This creates mirror-image emergence patterns.
-}
module Ra.Inversion.HarmonicTwist
  ( -- * Core Types
    TwistState(..)
  , InversionPoint(..)
  , ShadowHarmonic(..)
  , TwistResult(..)

    -- * Inversion Detection
  , detectInversion
  , checkTwistConditions
  , inversionThreshold

    -- * Harmonic Twisting
  , twistHarmonic
  , untwistHarmonic
  , shadowComplement

    -- * Phase Reversal
  , PhaseReversal(..)
  , reversePhase
  , phaseAntinode

    -- * Torsion Manipulation
  , TorsionTwist(..)
  , applyTorsionTwist
  , neutralizeTorsion

    -- * Shadow Emergence
  , ShadowEmergence(..)
  , computeShadowEmergence
  , shadowToNormal

    -- * Ankh Delta Analysis
  , AnkhDelta(..)
  , computeAnkhDelta
  , deltaInversion

    -- * Twist Field
  , TwistField(..)
  , generateTwistField
  , fieldPolarity
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Current twist state
data TwistState
  = TwistNormal       -- ^ Standard emergence
  | TwistInverted     -- ^ Shadow emergence active
  | TwistTransitional -- ^ Between states
  | TwistNull         -- ^ No twist (neutral)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Point where inversion occurs
data InversionPoint = InversionPoint
  { ipCoherence   :: !Double          -- ^ Coherence at inversion
  , ipAnkhDelta   :: !Double          -- ^ Δ(ankh) at inversion
  , ipTorsion     :: !TorsionPolarity -- ^ Torsion state
  , ipHarmonic    :: !(Int, Int)      -- ^ (l, m) at inversion
  , ipTimestamp   :: !Int             -- ^ φ^n tick
  } deriving (Eq, Show)

-- | Shadow harmonic representation
data ShadowHarmonic = ShadowHarmonic
  { shL         :: !Int       -- ^ Harmonic degree
  , shM         :: !Int       -- ^ Harmonic order
  , shSign      :: !Int       -- ^ (-1)^(l+m) factor
  , shAmplitude :: !Double    -- ^ Shadow amplitude
  , shPhase     :: !Double    -- ^ Phase offset
  } deriving (Eq, Show)

-- | Result of twist operation
data TwistResult
  = TwistSuccess !TwistState !ShadowHarmonic
  | TwistFailed !String
  | TwistPartial !Double !ShadowHarmonic  -- ^ Partial twist with factor
  deriving (Eq, Show)

-- | Torsion polarity
data TorsionPolarity
  = PolarityPositive   -- ^ Right-hand rotation
  | PolarityNegative   -- ^ Left-hand rotation
  | PolarityNeutral    -- ^ No rotation
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Inversion Detection
-- =============================================================================

-- | Detect if inversion conditions are met
detectInversion :: Double -> Double -> TorsionPolarity -> Maybe InversionPoint
detectInversion coherence ankhDelta torsion
  | coherence < inversionThreshold && ankhDelta < 0 =
      Just InversionPoint
        { ipCoherence = coherence
        , ipAnkhDelta = ankhDelta
        , ipTorsion = torsion
        , ipHarmonic = (0, 0)  -- Will be set by caller
        , ipTimestamp = 0
        }
  | torsion == PolarityNegative && coherence < phiInverse =
      Just InversionPoint
        { ipCoherence = coherence
        , ipAnkhDelta = ankhDelta
        , ipTorsion = torsion
        , ipHarmonic = (0, 0)
        , ipTimestamp = 0
        }
  | otherwise = Nothing

-- | Check multiple twist conditions
checkTwistConditions :: Double       -- ^ Coherence
                     -> Double       -- ^ Ankh delta
                     -> TorsionPolarity
                     -> (Int, Int)   -- ^ Harmonic (l, m)
                     -> [TwistCondition]
checkTwistConditions coh delta torsion (l, m) =
  let conditions = []
      cond1 = if coh < inversionThreshold
              then [CoherenceBelow inversionThreshold]
              else []
      cond2 = if delta < 0
              then [AnkhInverted delta]
              else []
      cond3 = if torsion == PolarityNegative
              then [TorsionReversed]
              else []
      cond4 = if odd (l + m)
              then [OddHarmonic l m]
              else []
  in conditions ++ cond1 ++ cond2 ++ cond3 ++ cond4

-- | Twist condition types
data TwistCondition
  = CoherenceBelow !Double
  | AnkhInverted !Double
  | TorsionReversed
  | OddHarmonic !Int !Int
  | PhaseDiscontinuity !Double
  deriving (Eq, Show)

-- | Coherence threshold for inversion
inversionThreshold :: Double
inversionThreshold = phiInverse * phiInverse  -- ~0.382

-- =============================================================================
-- Harmonic Twisting
-- =============================================================================

-- | Twist a harmonic to its shadow form
twistHarmonic :: Int -> Int -> Double -> ShadowHarmonic
twistHarmonic l m amplitude =
  let sign = if even (l + m) then 1 else (-1)
      shadowAmp = amplitude * abs (fromIntegral sign)
      phaseShift = if sign < 0 then pi else 0
  in ShadowHarmonic
    { shL = l
    , shM = m
    , shSign = sign
    , shAmplitude = shadowAmp
    , shPhase = phaseShift
    }

-- | Untwist shadow harmonic back to normal
untwistHarmonic :: ShadowHarmonic -> (Int, Int, Double)
untwistHarmonic sh =
  let amp = shAmplitude sh * fromIntegral (shSign sh)
  in (shL sh, shM sh, amp)

-- | Compute shadow complement of harmonic
shadowComplement :: Int -> Int -> (Int, Int, Int)
shadowComplement l m =
  let sign = (-1) ^ (l + m)
  in (l, -m, sign)  -- Complement has negated m

-- =============================================================================
-- Phase Reversal
-- =============================================================================

-- | Phase reversal specification
data PhaseReversal = PhaseReversal
  { prOriginalPhase :: !Double   -- ^ Original phase [0, 2π]
  , prReversedPhase :: !Double   -- ^ Reversed phase
  , prNodeCount     :: !Int      -- ^ Number of nodes crossed
  , prAmplitudeLoss :: !Double   -- ^ Energy loss during reversal
  } deriving (Eq, Show)

-- | Reverse phase by π
reversePhase :: Double -> PhaseReversal
reversePhase original =
  let reversed = if original + pi > 2 * pi
                 then original + pi - 2 * pi
                 else original + pi
      nodes = 1  -- Crossing one node
      loss = 1 - phiInverse  -- ~0.382 loss
  in PhaseReversal original reversed nodes loss

-- | Compute phase antinode location
phaseAntinode :: Double -> Double -> Double
phaseAntinode phase1 phase2 =
  let avg = (phase1 + phase2) / 2
      antinode = if avg + pi / 2 > 2 * pi
                 then avg + pi / 2 - 2 * pi
                 else avg + pi / 2
  in antinode

-- =============================================================================
-- Torsion Manipulation
-- =============================================================================

-- | Torsion twist operation
data TorsionTwist = TorsionTwist
  { ttOriginal   :: !TorsionPolarity
  , ttResulting  :: !TorsionPolarity
  , ttMagnitude  :: !Double          -- ^ Twist strength [0, 1]
  , ttEnergy     :: !Double          -- ^ Energy required
  } deriving (Eq, Show)

-- | Apply torsion twist to polarity
applyTorsionTwist :: TorsionPolarity -> Double -> TorsionTwist
applyTorsionTwist original magnitude =
  let resulting = case original of
        PolarityPositive -> if magnitude > 0.5 then PolarityNegative else PolarityPositive
        PolarityNegative -> if magnitude > 0.5 then PolarityPositive else PolarityNegative
        PolarityNeutral -> if magnitude > 0.7 then PolarityPositive else PolarityNeutral
      energy = magnitude * phi  -- Scaled by golden ratio
  in TorsionTwist original resulting magnitude energy

-- | Neutralize torsion
neutralizeTorsion :: TorsionPolarity -> Double -> TorsionPolarity
neutralizeTorsion polarity strength
  | strength > 0.9 = PolarityNeutral
  | strength > 0.5 = case polarity of
      PolarityPositive -> PolarityNeutral
      PolarityNegative -> PolarityNeutral
      PolarityNeutral -> PolarityNeutral
  | otherwise = polarity

-- =============================================================================
-- Shadow Emergence
-- =============================================================================

-- | Shadow emergence result
data ShadowEmergence = ShadowEmergence
  { seHarmonic     :: !ShadowHarmonic
  , seCoherence    :: !Double           -- ^ Shadow coherence
  , sePhaseOffset  :: !Double           -- ^ Phase from normal
  , seIntensity    :: !Double           -- ^ Emergence intensity
  , seInversionPt  :: !(Maybe InversionPoint)
  } deriving (Eq, Show)

-- | Compute shadow emergence from conditions
computeShadowEmergence :: Double       -- ^ Coherence
                       -> Double       -- ^ Ankh delta
                       -> (Int, Int)   -- ^ Harmonic
                       -> Double       -- ^ Amplitude
                       -> ShadowEmergence
computeShadowEmergence coh delta (l, m) amp =
  let shadow = twistHarmonic l m amp
      shadowCoh = 1.0 - coh  -- Inverted coherence
      phaseOff = if delta < 0 then pi else pi / 2
      intensity = shadowCoh * abs delta
      invPt = detectInversion coh delta PolarityNegative
  in ShadowEmergence
    { seHarmonic = shadow
    , seCoherence = shadowCoh
    , sePhaseOffset = phaseOff
    , seIntensity = min 1.0 intensity
    , seInversionPt = invPt
    }

-- | Convert shadow emergence to normal form
shadowToNormal :: ShadowEmergence -> (Double, Double, (Int, Int))
shadowToNormal se =
  let (l, m, amp) = untwistHarmonic (seHarmonic se)
      normalCoh = 1.0 - seCoherence se
  in (normalCoh, amp, (l, m))

-- =============================================================================
-- Ankh Delta Analysis
-- =============================================================================

-- | Ankh delta measurement
data AnkhDelta = AnkhDelta
  { adValue      :: !Double    -- ^ Delta value (can be negative)
  , adRate       :: !Double    -- ^ Rate of change
  , adPolarity   :: !DeltaPolarity
  , adStability  :: !Double    -- ^ Stability index [0, 1]
  } deriving (Eq, Show)

-- | Delta polarity
data DeltaPolarity
  = DeltaPositive   -- ^ Symmetry increasing
  | DeltaNegative   -- ^ Symmetry decreasing
  | DeltaZero       -- ^ At balance point
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Compute ankh delta from measurements
computeAnkhDelta :: Double -> Double -> AnkhDelta
computeAnkhDelta current previous =
  let delta = current - previous
      rate = abs delta
      polarity
        | delta > 0.001 = DeltaPositive
        | delta < -0.001 = DeltaNegative
        | otherwise = DeltaZero
      stability = 1.0 - min 1.0 (rate * 10)
  in AnkhDelta delta rate polarity stability

-- | Check if delta indicates inversion
deltaInversion :: AnkhDelta -> Bool
deltaInversion ad =
  adPolarity ad == DeltaNegative && adValue ad < -phiInverse

-- =============================================================================
-- Twist Field
-- =============================================================================

-- | Twist field representation
data TwistField = TwistField
  { tfCenter    :: !(Double, Double, Double)  -- ^ Field center
  , tfRadius    :: !Double                    -- ^ Effect radius
  , tfStrength  :: !Double                    -- ^ Twist strength
  , tfPolarity  :: !TorsionPolarity           -- ^ Field polarity
  , tfGradient  :: ![(Double, Double)]        -- ^ Radial gradient
  } deriving (Eq, Show)

-- | Generate twist field at location
generateTwistField :: (Double, Double, Double) -> Double -> Double -> TwistField
generateTwistField center radius strength =
  let polarity = if strength > 0 then PolarityPositive else PolarityNegative
      gradient = [(r, strength * exp (-r / radius))
                 | r <- [0, radius/10 .. radius]]
  in TwistField
    { tfCenter = center
    , tfRadius = radius
    , tfStrength = abs strength
    , tfPolarity = polarity
    , tfGradient = gradient
    }

-- | Get field polarity at distance from center
fieldPolarity :: TwistField -> Double -> TorsionPolarity
fieldPolarity tf distance
  | distance > tfRadius tf = PolarityNeutral
  | distance < tfRadius tf * 0.1 = tfPolarity tf
  | otherwise =
      let relDist = distance / tfRadius tf
          threshold = phiInverse
      in if relDist < threshold
         then tfPolarity tf
         else PolarityNeutral
