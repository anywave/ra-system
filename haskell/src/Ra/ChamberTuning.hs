{-|
Module      : Ra.ChamberTuning
Description : Biophysical resonance tuning for coherence chambers
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements real-time biophysical tuning of coherence chambers via closed-loop
biometric feedback, allowing dynamic adjustment of chamber parameters to
optimize resonance with the user's physiological state.

== Tuning Principles

=== Resonance Optimization

The chamber's field configuration can be tuned in real-time by:

* Adjusting harmonic gate thresholds based on HRV coherence
* Modulating orgone charge rate from GSR fluctuations
* Shifting resonance targets to match breathing rhythm
* Applying phi-window corrections from EEG alpha phase

=== Biophysical Coupling

Biometric → Chamber mapping:

* HRV coherence → Gate threshold adjustment (±20%)
* GSR (arousal) → Orgone charge/discharge rate
* Respiration → Resonance target frequency modulation
* EEG alpha → Phase lock window alignment
* Heart rate → Base field intensity scaling

=== Feedback Modes

* Entrainment: Chamber guides user toward target state
* Responsive: Chamber follows user state changes
* Adaptive: Chamber learns optimal parameters over time
* Manual: User directly controls tuning parameters

== Safety Constraints

* Maximum tuning rate limited to prevent sudden changes
* Coherence collapse triggers automatic chamber reset
* Shadow fragment handling restricted during rapid tuning
-}
module Ra.ChamberTuning
  ( -- * Tuning State
    TuningState(..)
  , TuningMode(..)
  , mkTuningState
  , tuningCoherence
  , tuningDrift

    -- * Biometric Coupling
  , BiometricCoupling(..)
  , CouplingStrength(..)
  , computeCoupling
  , couplingToModifier

    -- * Parameter Adjustment
  , TuningAdjustment(..)
  , AdjustmentTarget(..)
  , computeAdjustment
  , applyAdjustment
  , constrainAdjustment

    -- * Real-Time Tuning Loop
  , TuningLoop(..)
  , LoopPhase(..)
  , initTuningLoop
  , stepTuningLoop
  , loopConverged

    -- * Resonance Optimization
  , OptimizationGoal(..)
  , ResonanceScore(..)
  , evaluateResonance
  , optimizeResonance
  , gradientStep

    -- * Chamber Integration
  , TunedChamber(..)
  , tuneChamber
  , applyTuningState
  , resetTuning

    -- * Phi-Window Alignment
  , PhiAlignment(..)
  , alignToPhiWindow
  , phiCorrectionFactor
  , optimalPhiPhase

    -- * Adaptation Learning
  , AdaptiveProfile(..)
  , TuningHistory(..)
  , learnFromSession
  , suggestTuning
  , profileSimilarity
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse, coherenceFloorPOR )

-- =============================================================================
-- Tuning State
-- =============================================================================

-- | Current tuning state for a chamber
data TuningState = TuningState
  { tsMode           :: !TuningMode
  , tsCurrentPhase   :: !Double       -- ^ Current phi phase [0, 2*pi]
  , tsGateModifier   :: !Double       -- ^ Gate threshold modifier [0.8, 1.2]
  , tsChargeRate     :: !Double       -- ^ Orgone charge rate modifier [0.5, 2.0]
  , tsResonanceShift :: !Double       -- ^ Resonance frequency shift (Hz)
  , tsIntensityScale :: !Double       -- ^ Field intensity scale [0.5, 1.5]
  , tsLastUpdate     :: !Double       -- ^ Time since last update (s)
  , tsConverged      :: !Bool         -- ^ Has tuning converged?
  } deriving (Eq, Show)

-- | Tuning mode
data TuningMode
  = Entrainment     -- ^ Chamber guides user toward target
  | Responsive      -- ^ Chamber follows user state
  | Adaptive        -- ^ Chamber learns optimal parameters
  | Manual          -- ^ User directly controls tuning
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create initial tuning state
mkTuningState :: TuningMode -> TuningState
mkTuningState mode = TuningState
  { tsMode = mode
  , tsCurrentPhase = 0.0
  , tsGateModifier = 1.0
  , tsChargeRate = 1.0
  , tsResonanceShift = 0.0
  , tsIntensityScale = 1.0
  , tsLastUpdate = 0.0
  , tsConverged = False
  }

-- | Get current tuning coherence
tuningCoherence :: TuningState -> Double
tuningCoherence ts =
  let phaseCoherence = cos (tsCurrentPhase ts) * 0.5 + 0.5
      gateBalance = 1.0 - abs (tsGateModifier ts - 1.0) * 2
      chargeBalance = 1.0 - abs (tsChargeRate ts - 1.0) * 0.5
  in (phaseCoherence + gateBalance + chargeBalance) / 3.0

-- | Calculate drift from baseline
tuningDrift :: TuningState -> Double
tuningDrift ts =
  let gateDrift = abs (tsGateModifier ts - 1.0)
      chargeDrift = abs (tsChargeRate ts - 1.0) / 2.0
      resonanceDrift = abs (tsResonanceShift ts) / 10.0
      intensityDrift = abs (tsIntensityScale ts - 1.0)
  in (gateDrift + chargeDrift + resonanceDrift + intensityDrift) / 4.0

-- =============================================================================
-- Biometric Coupling
-- =============================================================================

-- | Biometric coupling configuration
data BiometricCoupling = BiometricCoupling
  { bcHRVWeight      :: !Double       -- ^ Weight for HRV influence
  , bcGSRWeight      :: !Double       -- ^ Weight for GSR influence
  , bcBreathWeight   :: !Double       -- ^ Weight for respiration
  , bcEEGWeight      :: !Double       -- ^ Weight for EEG alpha
  , bcHeartWeight    :: !Double       -- ^ Weight for heart rate
  , bcStrength       :: !CouplingStrength
  } deriving (Eq, Show)

-- | Coupling strength level
data CouplingStrength
  = Loose           -- ^ Weak coupling (damped response)
  | Moderate        -- ^ Balanced coupling
  | Tight           -- ^ Strong coupling (quick response)
  | Locked          -- ^ Maximum coupling (1:1 mapping)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Compute coupling values from biometric input
computeCoupling :: (Double, Double, Double, Double, Double) -> BiometricCoupling -> (Double, Double, Double, Double, Double)
computeCoupling (hrv, gsr, breath, eeg, heart) coupling =
  let strengthFactor = case bcStrength coupling of
        Loose    -> 0.25
        Moderate -> 0.5
        Tight    -> 0.75
        Locked   -> 1.0

      hrvMod = (hrv - 0.5) * bcHRVWeight coupling * strengthFactor
      gsrMod = (gsr - 0.5) * bcGSRWeight coupling * strengthFactor
      breathMod = (breath - 0.5) * bcBreathWeight coupling * strengthFactor
      eegMod = (eeg - 0.5) * bcEEGWeight coupling * strengthFactor
      heartMod = (heart - 60.0) / 60.0 * bcHeartWeight coupling * strengthFactor
  in (hrvMod, gsrMod, breathMod, eegMod, heartMod)

-- | Convert coupling to tuning modifier
couplingToModifier :: BiometricCoupling -> Double -> Double
couplingToModifier coupling baseValue =
  let totalWeight = bcHRVWeight coupling + bcGSRWeight coupling +
                    bcBreathWeight coupling + bcEEGWeight coupling +
                    bcHeartWeight coupling
      normalizedWeight = if totalWeight > 0 then totalWeight / 5.0 else 1.0
  in baseValue * normalizedWeight

-- =============================================================================
-- Parameter Adjustment
-- =============================================================================

-- | A tuning adjustment
data TuningAdjustment = TuningAdjustment
  { taTarget        :: !AdjustmentTarget
  , taDelta         :: !Double         -- ^ Change amount
  , taReason        :: !String         -- ^ Why this adjustment
  , taPriority      :: !Int            -- ^ Priority (lower = higher)
  } deriving (Eq, Show)

-- | Target for adjustment
data AdjustmentTarget
  = GateThreshold     -- ^ Harmonic gate threshold
  | ChargeRate        -- ^ Orgone charge rate
  | ResonanceFreq     -- ^ Resonance target frequency
  | FieldIntensity    -- ^ Field intensity
  | PhaseOffset       -- ^ Phi phase offset
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Compute adjustment from biometric delta
computeAdjustment :: AdjustmentTarget -> Double -> Double -> TuningAdjustment
computeAdjustment target current desired =
  let delta = desired - current
      clampedDelta = clamp delta (-0.2) 0.2  -- Max 20% change
      reason = case target of
        GateThreshold  -> "HRV coherence shift"
        ChargeRate     -> "GSR arousal change"
        ResonanceFreq  -> "Breath rhythm drift"
        FieldIntensity -> "Heart rate variation"
        PhaseOffset    -> "EEG alpha phase"
  in TuningAdjustment
      { taTarget = target
      , taDelta = clampedDelta
      , taReason = reason
      , taPriority = fromEnum target
      }

-- | Apply adjustment to tuning state
applyAdjustment :: TuningAdjustment -> TuningState -> TuningState
applyAdjustment adj ts = case taTarget adj of
  GateThreshold ->
    ts { tsGateModifier = clamp (tsGateModifier ts + taDelta adj) 0.8 1.2 }
  ChargeRate ->
    ts { tsChargeRate = clamp (tsChargeRate ts + taDelta adj) 0.5 2.0 }
  ResonanceFreq ->
    ts { tsResonanceShift = clamp (tsResonanceShift ts + taDelta adj) (-10.0) 10.0 }
  FieldIntensity ->
    ts { tsIntensityScale = clamp (tsIntensityScale ts + taDelta adj) 0.5 1.5 }
  PhaseOffset ->
    ts { tsCurrentPhase = wrapPhase (tsCurrentPhase ts + taDelta adj * pi) }

-- | Constrain adjustment to safe limits
constrainAdjustment :: TuningAdjustment -> Double -> TuningAdjustment
constrainAdjustment adj maxRate =
  adj { taDelta = clamp (taDelta adj) (-maxRate) maxRate }

-- =============================================================================
-- Real-Time Tuning Loop
-- =============================================================================

-- | Tuning loop state
data TuningLoop = TuningLoop
  { tlPhase           :: !LoopPhase
  , tlIterations      :: !Int
  , tlLastScore       :: !Double      -- ^ Last resonance score
  , tlBestScore       :: !Double      -- ^ Best score achieved
  , tlBestState       :: !TuningState -- ^ State at best score
  , tlLearningRate    :: !Double      -- ^ Current learning rate
  , tlStableCount     :: !Int         -- ^ Consecutive stable iterations
  } deriving (Eq, Show)

-- | Loop phase
data LoopPhase
  = Initializing    -- ^ Setting up initial state
  | Exploring       -- ^ Searching parameter space
  | Refining        -- ^ Fine-tuning around optimum
  | Maintaining     -- ^ Holding stable state
  | Resetting       -- ^ Returning to baseline
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Initialize tuning loop
initTuningLoop :: TuningState -> TuningLoop
initTuningLoop ts = TuningLoop
  { tlPhase = Initializing
  , tlIterations = 0
  , tlLastScore = 0.0
  , tlBestScore = 0.0
  , tlBestState = ts
  , tlLearningRate = 0.1
  , tlStableCount = 0
  }

-- | Step the tuning loop
stepTuningLoop :: ResonanceScore -> TuningState -> TuningLoop -> (TuningLoop, [TuningAdjustment])
stepTuningLoop score ts loop =
  let currentScore = rsOverall score
      improvement = currentScore - tlLastScore loop

      -- Update best if improved
      (newBest, newBestState) = if currentScore > tlBestScore loop
                                 then (currentScore, ts)
                                 else (tlBestScore loop, tlBestState loop)

      -- Update stable count
      newStable = if abs improvement < 0.01
                  then tlStableCount loop + 1
                  else 0

      -- Determine phase transition
      newPhase = case tlPhase loop of
        Initializing | tlIterations loop > 5 -> Exploring
        Exploring | newStable > 10 -> Refining
        Refining | newStable > 20 -> Maintaining
        Maintaining | abs improvement > 0.1 -> Exploring  -- Destabilized
        phase -> phase

      -- Adjust learning rate
      newRate = case newPhase of
        Initializing -> 0.1
        Exploring -> 0.05
        Refining -> 0.02
        Maintaining -> 0.01
        Resetting -> 0.1

      -- Generate adjustments based on phase
      adjustments = case newPhase of
        Initializing -> []
        Exploring -> exploreAdjustments score newRate
        Refining -> refineAdjustments score newRate
        Maintaining -> maintainAdjustments score
        Resetting -> resetAdjustments ts

      newLoop = loop
        { tlPhase = newPhase
        , tlIterations = tlIterations loop + 1
        , tlLastScore = currentScore
        , tlBestScore = newBest
        , tlBestState = newBestState
        , tlLearningRate = newRate
        , tlStableCount = newStable
        }
  in (newLoop, adjustments)

-- | Check if loop has converged
loopConverged :: TuningLoop -> Bool
loopConverged loop = tlPhase loop == Maintaining && tlStableCount loop > 30

-- Generate exploration adjustments
exploreAdjustments :: ResonanceScore -> Double -> [TuningAdjustment]
exploreAdjustments score rate =
  [ TuningAdjustment GateThreshold (rate * (0.5 - rsGateAlignment score)) "exploration" 1
  , TuningAdjustment ChargeRate (rate * (rsOrgoneResonance score - 0.5)) "exploration" 2
  , TuningAdjustment FieldIntensity (rate * (rsPhaseCoherence score - 0.5)) "exploration" 3
  ]

-- Generate refinement adjustments
refineAdjustments :: ResonanceScore -> Double -> [TuningAdjustment]
refineAdjustments score rate =
  [ TuningAdjustment GateThreshold (rate * 0.5 * (0.5 - rsGateAlignment score)) "refinement" 1
  , TuningAdjustment ChargeRate (rate * 0.5 * (rsOrgoneResonance score - 0.5)) "refinement" 2
  ]

-- Generate maintenance adjustments (small corrections only)
maintainAdjustments :: ResonanceScore -> [TuningAdjustment]
maintainAdjustments score =
  if rsOverall score < coherenceFloorPOR
  then [TuningAdjustment GateThreshold 0.01 "maintenance" 1]
  else []

-- Generate reset adjustments
resetAdjustments :: TuningState -> [TuningAdjustment]
resetAdjustments ts =
  [ TuningAdjustment GateThreshold (1.0 - tsGateModifier ts) "reset" 1
  , TuningAdjustment ChargeRate (1.0 - tsChargeRate ts) "reset" 2
  , TuningAdjustment ResonanceFreq (-tsResonanceShift ts) "reset" 3
  , TuningAdjustment FieldIntensity (1.0 - tsIntensityScale ts) "reset" 4
  ]

-- =============================================================================
-- Resonance Optimization
-- =============================================================================

-- | Optimization goal
data OptimizationGoal
  = MaxCoherence      -- ^ Maximize overall coherence
  | TargetState Double Double Double  -- ^ Target specific (gate, charge, intensity)
  | MinDrift          -- ^ Minimize drift from baseline
  | BalancedResonance -- ^ Balance all parameters
  deriving (Eq, Show)

-- | Resonance quality score
data ResonanceScore = ResonanceScore
  { rsOverall         :: !Double      -- ^ Overall score [0,1]
  , rsGateAlignment   :: !Double      -- ^ Gate alignment quality
  , rsOrgoneResonance :: !Double      -- ^ Orgone resonance quality
  , rsPhaseCoherence  :: !Double      -- ^ Phase lock coherence
  , rsTargetMatch     :: !Double      -- ^ Resonance target match
  } deriving (Eq, Show)

-- | Evaluate resonance for current state
evaluateResonance :: TuningState -> (Double, Double, Double, Double) -> ResonanceScore
evaluateResonance ts (hrv, gsr, breath, eeg) =
  let -- Gate alignment: how well gates match HRV
      gateAlignment = 1.0 - abs (tsGateModifier ts - hrv)

      -- Orgone resonance: charge rate matches GSR
      orgoneResonance = 1.0 - abs (tsChargeRate ts - 1.0 - gsr * 0.5)

      -- Phase coherence: EEG alpha phase lock
      phaseCoherence = cos (tsCurrentPhase ts - eeg * 2 * pi) * 0.5 + 0.5

      -- Target match: resonance shift matches breath
      targetMatch = 1.0 - abs (tsResonanceShift ts - (breath - 0.5) * 5.0) / 10.0

      -- Overall weighted
      overall = gateAlignment * 0.3 + orgoneResonance * 0.25 +
                phaseCoherence * 0.25 + targetMatch * 0.2
  in ResonanceScore
      { rsOverall = clamp01 overall
      , rsGateAlignment = clamp01 gateAlignment
      , rsOrgoneResonance = clamp01 orgoneResonance
      , rsPhaseCoherence = clamp01 phaseCoherence
      , rsTargetMatch = clamp01 targetMatch
      }

-- | Optimize resonance toward goal
optimizeResonance :: OptimizationGoal -> TuningState -> ResonanceScore -> TuningState
optimizeResonance goal ts score = case goal of
  MaxCoherence ->
    applyGradient ts (gradientStep score 0.05)

  TargetState gateTarget chargeTarget intensityTarget ->
    ts { tsGateModifier = tsGateModifier ts + 0.02 * (gateTarget - tsGateModifier ts)
       , tsChargeRate = tsChargeRate ts + 0.02 * (chargeTarget - tsChargeRate ts)
       , tsIntensityScale = tsIntensityScale ts + 0.02 * (intensityTarget - tsIntensityScale ts)
       }

  MinDrift ->
    ts { tsGateModifier = tsGateModifier ts * 0.99 + 1.0 * 0.01
       , tsChargeRate = tsChargeRate ts * 0.99 + 1.0 * 0.01
       , tsResonanceShift = tsResonanceShift ts * 0.99
       , tsIntensityScale = tsIntensityScale ts * 0.99 + 1.0 * 0.01
       }

  BalancedResonance ->
    let avgDrift = tuningDrift ts
        correction = avgDrift * 0.1
    in ts { tsGateModifier = clamp (tsGateModifier ts - sign (tsGateModifier ts - 1.0) * correction) 0.8 1.2
          , tsChargeRate = clamp (tsChargeRate ts - sign (tsChargeRate ts - 1.0) * correction) 0.5 2.0
          }

-- | Compute gradient step
gradientStep :: ResonanceScore -> Double -> (Double, Double, Double, Double)
gradientStep score rate =
  let gateGrad = (rsGateAlignment score - 0.5) * rate
      chargeGrad = (rsOrgoneResonance score - 0.5) * rate
      intensityGrad = (rsPhaseCoherence score - 0.5) * rate
      phaseGrad = (rsTargetMatch score - 0.5) * rate
  in (gateGrad, chargeGrad, intensityGrad, phaseGrad)

-- Apply gradient to tuning state
applyGradient :: TuningState -> (Double, Double, Double, Double) -> TuningState
applyGradient ts (gateGrad, chargeGrad, intensityGrad, phaseGrad) =
  ts { tsGateModifier = clamp (tsGateModifier ts + gateGrad) 0.8 1.2
     , tsChargeRate = clamp (tsChargeRate ts + chargeGrad) 0.5 2.0
     , tsIntensityScale = clamp (tsIntensityScale ts + intensityGrad) 0.5 1.5
     , tsCurrentPhase = wrapPhase (tsCurrentPhase ts + phaseGrad)
     }

-- =============================================================================
-- Chamber Integration
-- =============================================================================

-- | A chamber with active tuning
data TunedChamber = TunedChamber
  { tcChamberId     :: !String
  , tcTuningState   :: !TuningState
  , tcTuningLoop    :: !TuningLoop
  , tcCoupling      :: !BiometricCoupling
  , tcLastBiometrics :: !(Double, Double, Double, Double)  -- ^ (hrv, gsr, breath, eeg)
  , tcActive        :: !Bool
  } deriving (Eq, Show)

-- | Tune a chamber based on biometric input
tuneChamber :: TunedChamber -> (Double, Double, Double, Double) -> TunedChamber
tuneChamber tc biometrics =
  let -- Evaluate current resonance
      score = evaluateResonance (tcTuningState tc) biometrics

      -- Step the tuning loop
      (newLoop, adjustments) = stepTuningLoop score (tcTuningState tc) (tcTuningLoop tc)

      -- Apply adjustments
      newState = foldr applyAdjustment (tcTuningState tc) adjustments

      -- Mark as converged if loop converged
      finalState = newState { tsConverged = loopConverged newLoop }
  in tc
      { tcTuningState = finalState
      , tcTuningLoop = newLoop
      , tcLastBiometrics = biometrics
      }

-- | Apply tuning state to chamber parameters (returns modifier values)
applyTuningState :: TuningState -> (Double, Double, Double, Double)
applyTuningState ts =
  ( tsGateModifier ts
  , tsChargeRate ts
  , tsResonanceShift ts
  , tsIntensityScale ts
  )

-- | Reset tuning to baseline
resetTuning :: TunedChamber -> TunedChamber
resetTuning tc = tc
  { tcTuningState = mkTuningState (tsMode (tcTuningState tc))
  , tcTuningLoop = initTuningLoop (mkTuningState (tsMode (tcTuningState tc)))
  }

-- =============================================================================
-- Phi-Window Alignment
-- =============================================================================

-- | Phi alignment state
data PhiAlignment = PhiAlignment
  { paPhase           :: !Double      -- ^ Current phase [0, 2*pi]
  , paTargetPhase     :: !Double      -- ^ Target phi phase
  , paAlignmentError  :: !Double      -- ^ Current error
  , paLocked          :: !Bool        -- ^ Phase locked?
  } deriving (Eq, Show)

-- | Align tuning to phi window
alignToPhiWindow :: TuningState -> Int -> PhiAlignment
alignToPhiWindow ts windowIndex =
  let targetPhase = fromIntegral windowIndex * (2 * pi / phi)
      phaseError = abs (tsCurrentPhase ts - targetPhase)
      wrappedError = min phaseError (2 * pi - phaseError)
      locked = wrappedError < 0.1
  in PhiAlignment
      { paPhase = tsCurrentPhase ts
      , paTargetPhase = targetPhase
      , paAlignmentError = wrappedError
      , paLocked = locked
      }

-- | Compute phi correction factor
phiCorrectionFactor :: PhiAlignment -> Double
phiCorrectionFactor pa =
  let errorNorm = paAlignmentError pa / (2 * pi)
  in 1.0 - errorNorm * phi  -- Correction increases with error

-- | Find optimal phi phase for resonance
optimalPhiPhase :: ResonanceScore -> Double
optimalPhiPhase score =
  let basePhase = rsOverall score * 2 * pi
      phiOffset = rsPhaseCoherence score * phiInverse
  in wrapPhase (basePhase + phiOffset)

-- =============================================================================
-- Adaptation Learning
-- =============================================================================

-- | Adaptive profile learned from sessions
data AdaptiveProfile = AdaptiveProfile
  { apUserId          :: !String
  , apOptimalGate     :: !Double      -- ^ Learned optimal gate modifier
  , apOptimalCharge   :: !Double      -- ^ Learned optimal charge rate
  , apOptimalIntensity :: !Double     -- ^ Learned optimal intensity
  , apSessionCount    :: !Int         -- ^ Sessions used for learning
  , apConfidence      :: !Double      -- ^ Learning confidence [0,1]
  } deriving (Eq, Show)

-- | Session tuning history
data TuningHistory = TuningHistory
  { thStates          :: ![TuningState]   -- ^ State snapshots
  , thScores          :: ![Double]        -- ^ Corresponding scores
  , thDuration        :: !Double          -- ^ Session duration (s)
  , thConverged       :: !Bool            -- ^ Did tuning converge?
  } deriving (Eq, Show)

-- | Learn from completed session
learnFromSession :: AdaptiveProfile -> TuningHistory -> AdaptiveProfile
learnFromSession profile history =
  if not (thConverged history) || length (thStates history) < 10
  then profile  -- Not enough data
  else
    let -- Find state at best score
        bestIdx = findBestIdx (thScores history)
        bestState = thStates history !! bestIdx

        -- Blend with existing profile
        alpha = 0.1  -- Learning rate
        newGate = apOptimalGate profile * (1 - alpha) + tsGateModifier bestState * alpha
        newCharge = apOptimalCharge profile * (1 - alpha) + tsChargeRate bestState * alpha
        newIntensity = apOptimalIntensity profile * (1 - alpha) + tsIntensityScale bestState * alpha

        -- Update confidence
        newConfidence = min 1.0 (apConfidence profile + 0.05)
    in profile
        { apOptimalGate = newGate
        , apOptimalCharge = newCharge
        , apOptimalIntensity = newIntensity
        , apSessionCount = apSessionCount profile + 1
        , apConfidence = newConfidence
        }

-- | Suggest initial tuning from profile
suggestTuning :: AdaptiveProfile -> TuningState
suggestTuning profile =
  if apConfidence profile < 0.3
  then mkTuningState Adaptive  -- Not enough confidence
  else TuningState
      { tsMode = Adaptive
      , tsCurrentPhase = 0.0
      , tsGateModifier = apOptimalGate profile
      , tsChargeRate = apOptimalCharge profile
      , tsResonanceShift = 0.0
      , tsIntensityScale = apOptimalIntensity profile
      , tsLastUpdate = 0.0
      , tsConverged = False
      }

-- | Calculate profile similarity for cross-user suggestions
profileSimilarity :: AdaptiveProfile -> AdaptiveProfile -> Double
profileSimilarity p1 p2 =
  let gateSim = 1.0 - abs (apOptimalGate p1 - apOptimalGate p2)
      chargeSim = 1.0 - abs (apOptimalCharge p1 - apOptimalCharge p2) / 1.5
      intensitySim = 1.0 - abs (apOptimalIntensity p1 - apOptimalIntensity p2)
  in (gateSim + chargeSim + intensitySim) / 3.0

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)

-- | Clamp value to range
clamp :: Double -> Double -> Double -> Double
clamp x lo hi = max lo (min hi x)

-- | Wrap phase to [0, 2*pi]
wrapPhase :: Double -> Double
wrapPhase p
  | p < 0 = wrapPhase (p + 2 * pi)
  | p >= 2 * pi = wrapPhase (p - 2 * pi)
  | otherwise = p

-- | Sign function
sign :: Double -> Double
sign x
  | x > 0 = 1
  | x < 0 = -1
  | otherwise = 0

-- | Find index of maximum value
findBestIdx :: [Double] -> Int
findBestIdx [] = 0
findBestIdx xs = snd $ maximum $ zip xs [0..]
