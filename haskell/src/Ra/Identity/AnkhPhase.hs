{-|
Module      : Ra.Identity.AnkhPhase
Description : Ankh-phase consent reversal threshold mechanics
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements ankh-phase consent reversal thresholds using delta(ankh) values
to simulate fragment inversion or shadow emergence when coherence thresholds
invert during biometric phase drops.

== Ankh-Phase Theory

=== Consent Reversal

When delta(ankh) crosses critical thresholds:

* Positive delta: Consent strengthening, coherence building
* Negative delta: Consent weakening, potential shadow emergence
* Inversion point: Complete phase flip, fragment may invert

=== Shadow Emergence Triggers

1. Coherence drop below phi^-2 (~0.382)
2. Sustained negative delta(ankh)
3. Biometric phase desynchronization
4. Torsion field polarity reversal
-}
module Ra.Identity.AnkhPhase
  ( -- * Core Types
    AnkhPhaseState(..)
  , PhasePolarity(..)
  , ConsentLevel(..)
  , InversionEvent(..)

    -- * Phase Tracking
  , initAnkhPhase
  , updateAnkhPhase
  , currentDelta

    -- * Threshold Detection
  , inversionThreshold
  , checkInversion
  , shadowEmergenceRisk

    -- * Consent Dynamics
  , consentStrength
  , consentDecay
  , restoreConsent

    -- * Fragment Effects
  , FragmentPhaseEffect(..)
  , applyPhaseEffect
  , invertFragment

    -- * Biometric Integration
  , biometricPhaseSync
  , phaseDesyncDetect
  , recoverSync
  ) where

import Ra.Constants.Extended (phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Ankh phase state tracking
data AnkhPhaseState = AnkhPhaseState
  { apsCurrentAnkh   :: !Double          -- ^ Current ankh value
  , apsPreviousAnkh  :: !Double          -- ^ Previous ankh value
  , apsDelta         :: !Double          -- ^ Delta(ankh) = current - previous
  , apsPolarity      :: !PhasePolarity   -- ^ Current phase polarity
  , apsConsent       :: !ConsentLevel    -- ^ Consent level
  , apsInversionCount :: !Int            -- ^ Number of inversions
  , apsTicks         :: !Int             -- ^ Ticks since last inversion
  } deriving (Eq, Show)

-- | Phase polarity states
data PhasePolarity
  = PolarityPositive   -- ^ Normal, constructive phase
  | PolarityNeutral    -- ^ Balanced, transition state
  | PolarityNegative   -- ^ Inverted, shadow-prone phase
  | PolarityInverting  -- ^ Actively inverting
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Consent level enumeration
data ConsentLevel
  = ConsentFull        -- ^ Full consent, stable
  | ConsentDiminished  -- ^ Reduced consent
  | ConsentSuspended   -- ^ Temporarily suspended
  | ConsentEmergency   -- ^ Emergency override active
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Inversion event record
data InversionEvent = InversionEvent
  { ieTimestamp    :: !Int              -- ^ When inversion occurred
  , iePriorAnkh    :: !Double           -- ^ Ankh before inversion
  , iePostAnkh     :: !Double           -- ^ Ankh after inversion
  , ieTrigger      :: !InversionTrigger -- ^ What triggered it
  , ieSeverity     :: !Double           -- ^ Severity 0-1
  } deriving (Eq, Show)

-- | Inversion trigger types
data InversionTrigger
  = TriggerCoherenceDrop    -- ^ Coherence fell below threshold
  | TriggerDeltaSustained   -- ^ Sustained negative delta
  | TriggerBiometricDesync  -- ^ Biometric phase desync
  | TriggerTorsionFlip      -- ^ Torsion field flipped
  | TriggerExternal         -- ^ External event
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Phase Tracking
-- =============================================================================

-- | Initialize ankh phase state
initAnkhPhase :: Double -> AnkhPhaseState
initAnkhPhase initialAnkh = AnkhPhaseState
  { apsCurrentAnkh = initialAnkh
  , apsPreviousAnkh = initialAnkh
  , apsDelta = 0
  , apsPolarity = PolarityPositive
  , apsConsent = ConsentFull
  , apsInversionCount = 0
  , apsTicks = 0
  }

-- | Update ankh phase with new reading
updateAnkhPhase :: AnkhPhaseState -> Double -> AnkhPhaseState
updateAnkhPhase state newAnkh =
  let delta = newAnkh - apsCurrentAnkh state
      newPolarity = determinePolarity delta (apsPolarity state)
      newConsent = updateConsent delta (apsConsent state)
      inverted = checkInversion state
      invCount = if inverted then apsInversionCount state + 1 else apsInversionCount state
      ticks = if inverted then 0 else apsTicks state + 1
  in state
    { apsCurrentAnkh = newAnkh
    , apsPreviousAnkh = apsCurrentAnkh state
    , apsDelta = delta
    , apsPolarity = newPolarity
    , apsConsent = newConsent
    , apsInversionCount = invCount
    , apsTicks = ticks
    }

-- | Get current delta value
currentDelta :: AnkhPhaseState -> Double
currentDelta = apsDelta

-- =============================================================================
-- Threshold Detection
-- =============================================================================

-- | Critical inversion threshold (phi^-2)
inversionThreshold :: Double
inversionThreshold = phiInverse * phiInverse  -- ~0.382

-- | Check if inversion is occurring
checkInversion :: AnkhPhaseState -> Bool
checkInversion state =
  apsCurrentAnkh state < inversionThreshold &&
  apsDelta state < -0.05 &&
  apsPolarity state == PolarityNegative

-- | Calculate shadow emergence risk [0, 1]
shadowEmergenceRisk :: AnkhPhaseState -> Double
shadowEmergenceRisk state =
  let ankhFactor = 1 - min 1 (apsCurrentAnkh state / phiInverse)
      deltaFactor = if apsDelta state < 0 then abs (apsDelta state) * 2 else 0
      polarityFactor = case apsPolarity state of
        PolarityPositive -> 0
        PolarityNeutral -> 0.2
        PolarityNegative -> 0.5
        PolarityInverting -> 0.8
      consentFactor = case apsConsent state of
        ConsentFull -> 0
        ConsentDiminished -> 0.2
        ConsentSuspended -> 0.5
        ConsentEmergency -> 0.3  -- Emergency has protections
  in min 1.0 (ankhFactor * 0.4 + deltaFactor * 0.3 + polarityFactor * 0.2 + consentFactor * 0.1)

-- =============================================================================
-- Consent Dynamics
-- =============================================================================

-- | Calculate consent strength [0, 1]
consentStrength :: AnkhPhaseState -> Double
consentStrength state =
  let baseStrength = case apsConsent state of
        ConsentFull -> 1.0
        ConsentDiminished -> 0.7
        ConsentSuspended -> 0.3
        ConsentEmergency -> 0.5
      ankhModifier = min 1 (apsCurrentAnkh state / phiInverse)
  in baseStrength * ankhModifier

-- | Apply consent decay over time
consentDecay :: AnkhPhaseState -> Int -> AnkhPhaseState
consentDecay state ticks =
  let decayRate = 0.01 * fromIntegral ticks
      newAnkh = max 0 (apsCurrentAnkh state - decayRate)
  in updateAnkhPhase state newAnkh

-- | Attempt to restore consent
restoreConsent :: AnkhPhaseState -> Double -> AnkhPhaseState
restoreConsent state boost =
  let newAnkh = min 1.0 (apsCurrentAnkh state + boost)
      newConsent = if newAnkh > phiInverse
                   then ConsentFull
                   else if newAnkh > inversionThreshold
                        then ConsentDiminished
                        else apsConsent state
  in (updateAnkhPhase state newAnkh) { apsConsent = newConsent }

-- =============================================================================
-- Fragment Effects
-- =============================================================================

-- | Phase effect on fragments
data FragmentPhaseEffect = FragmentPhaseEffect
  { fpeAmplification :: !Double   -- ^ Amplitude modifier
  , fpePhaseShift    :: !Double   -- ^ Phase shift amount
  , fpeInverted      :: !Bool     -- ^ Is fragment inverted
  , fpeShadowBleed   :: !Double   -- ^ Shadow content bleed-through
  } deriving (Eq, Show)

-- | Apply phase effect to fragment parameters
applyPhaseEffect :: AnkhPhaseState -> FragmentPhaseEffect
applyPhaseEffect state =
  let risk = shadowEmergenceRisk state
      inverted = apsPolarity state == PolarityInverting || risk > 0.7
      amplification = if inverted then -phiInverse else phiInverse + apsCurrentAnkh state
      phaseShift = apsDelta state * pi
      shadowBleed = risk * (1 - apsCurrentAnkh state)
  in FragmentPhaseEffect
    { fpeAmplification = amplification
    , fpePhaseShift = phaseShift
    , fpeInverted = inverted
    , fpeShadowBleed = shadowBleed
    }

-- | Force fragment inversion
invertFragment :: AnkhPhaseState -> (AnkhPhaseState, InversionEvent)
invertFragment state =
  let newAnkh = 1 - apsCurrentAnkh state
      event = InversionEvent
        { ieTimestamp = apsTicks state
        , iePriorAnkh = apsCurrentAnkh state
        , iePostAnkh = newAnkh
        , ieTrigger = TriggerExternal
        , ieSeverity = abs (newAnkh - apsCurrentAnkh state)
        }
      newState = state
        { apsCurrentAnkh = newAnkh
        , apsPreviousAnkh = apsCurrentAnkh state
        , apsDelta = newAnkh - apsCurrentAnkh state
        , apsPolarity = PolarityInverting
        , apsInversionCount = apsInversionCount state + 1
        , apsTicks = 0
        }
  in (newState, event)

-- =============================================================================
-- Biometric Integration
-- =============================================================================

-- | Biometric phase synchronization state
data BiometricSync = BiometricSync
  { bsHRVPhase      :: !Double    -- ^ HRV phase alignment
  , bsRespiratoryPhase :: !Double -- ^ Respiratory phase
  , bsCoherencePhase :: !Double   -- ^ Overall coherence phase
  , bsSyncQuality   :: !Double    -- ^ Sync quality [0, 1]
  } deriving (Eq, Show)

-- | Synchronize with biometric phase
biometricPhaseSync :: AnkhPhaseState -> Double -> Double -> Double -> (AnkhPhaseState, BiometricSync)
biometricPhaseSync state hrv resp coherence =
  let avgPhase = (hrv + resp + coherence) / 3
      syncQuality = 1 - abs (apsCurrentAnkh state - avgPhase)
      sync = BiometricSync hrv resp coherence syncQuality
      ankhAdjustment = (avgPhase - apsCurrentAnkh state) * 0.1
      newState = updateAnkhPhase state (apsCurrentAnkh state + ankhAdjustment)
  in (newState, sync)

-- | Detect phase desynchronization
phaseDesyncDetect :: AnkhPhaseState -> BiometricSync -> Maybe InversionTrigger
phaseDesyncDetect state sync
  | bsSyncQuality sync < 0.3 = Just TriggerBiometricDesync
  | apsDelta state < -0.1 && bsSyncQuality sync < 0.5 = Just TriggerCoherenceDrop
  | otherwise = Nothing

-- | Attempt to recover synchronization
recoverSync :: AnkhPhaseState -> BiometricSync -> AnkhPhaseState
recoverSync state sync =
  let targetAnkh = bsCoherencePhase sync
      stepSize = 0.05
      direction = if targetAnkh > apsCurrentAnkh state then stepSize else -stepSize
      newAnkh = apsCurrentAnkh state + direction
  in updateAnkhPhase state (max 0 (min 1 newAnkh))

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Determine polarity from delta
determinePolarity :: Double -> PhasePolarity -> PhasePolarity
determinePolarity delta currentPolarity
  | delta > 0.05 = PolarityPositive
  | delta < -0.05 = case currentPolarity of
      PolarityNegative -> PolarityInverting
      _ -> PolarityNegative
  | otherwise = PolarityNeutral

-- | Update consent based on delta
updateConsent :: Double -> ConsentLevel -> ConsentLevel
updateConsent delta current
  | delta > 0.1 = case current of
      ConsentDiminished -> ConsentFull
      ConsentSuspended -> ConsentDiminished
      _ -> current
  | delta < -0.1 = case current of
      ConsentFull -> ConsentDiminished
      ConsentDiminished -> ConsentSuspended
      _ -> current
  | otherwise = current
