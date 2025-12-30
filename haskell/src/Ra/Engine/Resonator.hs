{-|
Module      : Ra.Engine.Resonator
Description : Self-regulating resonance engine using Tesla coil and Joe Cell principles
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Feedback-driven resonance engine that continuously modulates user state using
Tesla coil field behavior and Joe Cell charge separation dynamics. This serves
as a self-regulating biometric tuning system.

== Engine Architecture

=== Tesla-Joe Hybrid Structure

* Core: Bifilar Tesla-style coil pulsing at tuned biometrics
* Containment: Joe-style scalar chamber with concentric cylinders
* Feedback: Biometric loop triggers harmonic dampening/amplification

=== Field Behavior

* Energy builds in concentric coils until threshold
* Releases via harmonic gate when coherence = resonance lock
* Longitudinal spikes align user field with chamber field
-}
module Ra.Engine.Resonator
  ( -- * Core Types
    ResonanceEngine(..)
  , TeslaCore(..)
  , JoeChamber(..)
  , CylinderLayer(..)

    -- * Engine Construction
  , createEngine
  , configureTeslaCore
  , configureJoeChamber
  , addCylinderLayer

    -- * Resonance Control
  , ResonanceState(..)
  , ResonanceAdjustment(..)
  , modulateResonance
  , currentResonance

    -- * Biometric Input
  , BiometricSnapshot(..)
  , processSnapshot
  , snapshotToField

    -- * Field Operations
  , ChamberField(..)
  , fieldCharge
  , fieldRelease
  , harmonicGate

    -- * Feedback System
  , FeedbackLoop(..)
  , initFeedback
  , feedbackCycle
  , selfRegulate

    -- * Safety Systems
  , SafetyState(..)
  , checkSafety
  , emergencyDamp
  , resetEngine
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete resonance engine
data ResonanceEngine = ResonanceEngine
  { reTeslaCore    :: !TeslaCore           -- ^ Bifilar Tesla coil core
  , reJoeChamber   :: !JoeChamber          -- ^ Joe Cell containment
  , reResonance    :: !ResonanceState      -- ^ Current resonance state
  , reFeedback     :: !FeedbackLoop        -- ^ Feedback system
  , reSafety       :: !SafetyState         -- ^ Safety status
  , reActive       :: !Bool                -- ^ Engine active
  , reCycles       :: !Int                 -- ^ Completed cycles
  } deriving (Eq, Show)

-- | Tesla-style bifilar coil core
data TeslaCore = TeslaCore
  { tcCoilType     :: !CoilType            -- ^ Coil configuration
  , tcWindings     :: !Int                 -- ^ Number of windings
  , tcFrequency    :: !Double              -- ^ Operating frequency (Hz)
  , tcPhase        :: !Double              -- ^ Phase offset [0, 2pi]
  , tcAmplitude    :: !Double              -- ^ Current amplitude [0, 1]
  , tcCharge       :: !Double              -- ^ Stored charge [0, 1]
  } deriving (Eq, Show)

-- | Coil configuration types
data CoilType
  = CoilBifilar        -- ^ Bifilar wound (inverse phase)
  | CoilCaduceus       -- ^ Caduceus/serpent wound
  | CoilToroidal       -- ^ Toroidal geometry
  | CoilPancake        -- ^ Flat spiral
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Joe Cell-style containment chamber
data JoeChamber = JoeChamber
  { jcCylinders    :: ![CylinderLayer]     -- ^ Concentric cylinders
  , jcWaterLevel   :: !Double              -- ^ Water fill level [0, 1]
  , jcChargeState  :: !ChargeState         -- ^ Current charge state
  , jcScalarField  :: !Double              -- ^ Scalar field strength [0, 1]
  , jcAlignment    :: !Double              -- ^ Scalar alignment [0, 1]
  } deriving (Eq, Show)

-- | Single cylinder layer in Joe Cell
data CylinderLayer = CylinderLayer
  { clRadius       :: !Double              -- ^ Cylinder radius (normalized)
  , clMaterial     :: !CylinderMaterial    -- ^ Material type
  , clCharge       :: !Double              -- ^ Layer charge [0, 1]
  , clDelay        :: !Double              -- ^ Field delay (ms)
  } deriving (Eq, Show)

-- | Cylinder material types
data CylinderMaterial
  = MaterialSteel      -- ^ Stainless steel
  | MaterialAluminum   -- ^ Aluminum
  | MaterialCopper     -- ^ Copper
  | MaterialGlass      -- ^ Glass insulator
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Charge state
data ChargeState
  = ChargeNeutral      -- ^ No significant charge
  | ChargeBuilding     -- ^ Charge accumulating
  | ChargeFull         -- ^ Maximum charge
  | ChargeDischarging  -- ^ Releasing charge
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Engine Construction
-- =============================================================================

-- | Create default resonance engine
createEngine :: ResonanceEngine
createEngine = ResonanceEngine
  { reTeslaCore = defaultTeslaCore
  , reJoeChamber = defaultJoeChamber
  , reResonance = defaultResonance
  , reFeedback = defaultFeedback
  , reSafety = SafetyOk
  , reActive = False
  , reCycles = 0
  }

-- | Configure Tesla core parameters
configureTeslaCore :: ResonanceEngine -> CoilType -> Double -> ResonanceEngine
configureTeslaCore engine coilType freq =
  let core = (reTeslaCore engine)
        { tcCoilType = coilType
        , tcFrequency = freq
        }
  in engine { reTeslaCore = core }

-- | Configure Joe chamber
configureJoeChamber :: ResonanceEngine -> Double -> ResonanceEngine
configureJoeChamber engine waterLevel =
  let chamber = (reJoeChamber engine) { jcWaterLevel = clamp01 waterLevel }
  in engine { reJoeChamber = chamber }

-- | Add cylinder layer to Joe chamber
addCylinderLayer :: ResonanceEngine -> Double -> CylinderMaterial -> ResonanceEngine
addCylinderLayer engine radius material =
  let chamber = reJoeChamber engine
      newLayer = CylinderLayer radius material 0 (radius * 10)
      newChamber = chamber { jcCylinders = newLayer : jcCylinders chamber }
  in engine { reJoeChamber = newChamber }

-- =============================================================================
-- Resonance Control
-- =============================================================================

-- | Current resonance state
data ResonanceState = ResonanceState
  { rsScalarShift  :: !Double              -- ^ Scalar field shift [-1, 1]
  , rsAlignment    :: !Double              -- ^ Chamber alignment [0, 1]
  , rsHarmonic     :: !Int                 -- ^ Active harmonic
  , rsLocked       :: !Bool                -- ^ Resonance locked
  , rsStability    :: !Double              -- ^ Stability factor [0, 1]
  } deriving (Eq, Show)

-- | Resonance adjustment output
data ResonanceAdjustment = ResonanceAdjustment
  { raShiftDelta   :: !Double              -- ^ Change in scalar shift
  , raNewAlignment :: !Double              -- ^ New alignment value
  , raHarmonicChange :: !Int               -- ^ Harmonic shift
  , raEnergy       :: !Double              -- ^ Energy transferred
  } deriving (Eq, Show)

-- | Main modulation function
modulateResonance :: BiometricSnapshot -> ChamberField -> ResonanceAdjustment
modulateResonance snapshot field =
  let -- Calculate coherence match
      coherenceMatch = bsCoherence snapshot * cfStrength field

      -- Determine shift based on HRV and respiration
      hrvFactor = bsHRV snapshot / 100
      respFactor = bsRespiration snapshot / 20

      -- Calculate adjustment
      shiftDelta = (coherenceMatch - 0.5) * phi * hrvFactor
      newAlignment = clamp01 (cfAlignment field + respFactor * 0.1)

      -- Harmonic stepping
      harmonicShift = if coherenceMatch > phiInverse then 1 else 0

      -- Energy based on phase coherence
      energy = bsCoherencePhase snapshot * cfStrength field

  in ResonanceAdjustment
    { raShiftDelta = shiftDelta
    , raNewAlignment = newAlignment
    , raHarmonicChange = harmonicShift
    , raEnergy = energy
    }

-- | Get current resonance state
currentResonance :: ResonanceEngine -> ResonanceState
currentResonance = reResonance

-- =============================================================================
-- Biometric Input
-- =============================================================================

-- | Biometric snapshot input
data BiometricSnapshot = BiometricSnapshot
  { bsHRV          :: !Double              -- ^ Heart rate variability (ms)
  , bsRespiration  :: !Double              -- ^ Breaths per minute
  , bsCoherence    :: !Double              -- ^ Overall coherence [0, 1]
  , bsCoherencePhase :: !Double            -- ^ Phase coherence [0, 1]
  , bsTimestamp    :: !Int                 -- ^ Capture timestamp
  } deriving (Eq, Show)

-- | Process biometric snapshot through engine
processSnapshot :: ResonanceEngine -> BiometricSnapshot -> (ResonanceEngine, ResonanceAdjustment)
processSnapshot engine snapshot =
  let field = snapshotToField engine snapshot
      adjustment = modulateResonance snapshot field

      -- Update resonance state
      currentRes = reResonance engine
      newResonance = currentRes
        { rsScalarShift = clamp11 (rsScalarShift currentRes + raShiftDelta adjustment)
        , rsAlignment = raNewAlignment adjustment
        , rsHarmonic = rsHarmonic currentRes + raHarmonicChange adjustment
        , rsLocked = raNewAlignment adjustment > phiInverse
        }

      -- Update chamber
      chamber = reJoeChamber engine
      newChamber = chamber
        { jcScalarField = clamp01 (jcScalarField chamber + raEnergy adjustment * 0.1)
        , jcAlignment = raNewAlignment adjustment
        }

  in (engine { reResonance = newResonance, reJoeChamber = newChamber, reCycles = reCycles engine + 1 }, adjustment)

-- | Convert snapshot to chamber field
snapshotToField :: ResonanceEngine -> BiometricSnapshot -> ChamberField
snapshotToField engine snapshot =
  let chamber = reJoeChamber engine
      core = reTeslaCore engine
  in ChamberField
    { cfStrength = jcScalarField chamber * tcAmplitude core
    , cfAlignment = jcAlignment chamber
    , cfFrequency = tcFrequency core
    , cfPhase = tcPhase core
    , cfCoherence = bsCoherence snapshot
    }

-- =============================================================================
-- Field Operations
-- =============================================================================

-- | Chamber field state
data ChamberField = ChamberField
  { cfStrength     :: !Double              -- ^ Field strength [0, 1]
  , cfAlignment    :: !Double              -- ^ Field alignment [0, 1]
  , cfFrequency    :: !Double              -- ^ Dominant frequency (Hz)
  , cfPhase        :: !Double              -- ^ Phase [0, 2pi]
  , cfCoherence    :: !Double              -- ^ Field coherence [0, 1]
  } deriving (Eq, Show)

-- | Charge the chamber field
fieldCharge :: ResonanceEngine -> Double -> ResonanceEngine
fieldCharge engine amount =
  let core = reTeslaCore engine
      newCharge = clamp01 (tcCharge core + amount)
      newCore = core { tcCharge = newCharge }

      chamber = reJoeChamber engine
      newChamber = chamber
        { jcChargeState = if newCharge > 0.8 then ChargeFull
                          else if newCharge > 0.3 then ChargeBuilding
                          else ChargeNeutral
        }
  in engine { reTeslaCore = newCore, reJoeChamber = newChamber }

-- | Release field energy
fieldRelease :: ResonanceEngine -> (ResonanceEngine, Double)
fieldRelease engine =
  let core = reTeslaCore engine
      released = tcCharge core * tcAmplitude core
      newCore = core { tcCharge = tcCharge core * 0.2 }  -- Keep some residual

      chamber = reJoeChamber engine
      newChamber = chamber { jcChargeState = ChargeDischarging }

  in (engine { reTeslaCore = newCore, reJoeChamber = newChamber }, released)

-- | Harmonic gate control
harmonicGate :: ResonanceEngine -> Double -> Bool
harmonicGate engine coherence =
  let res = reResonance engine
      ankh = phi / (1 + abs (rsScalarShift res))
      tolerance = ankh * 0.1
  in abs (coherence - ankh) < tolerance

-- =============================================================================
-- Feedback System
-- =============================================================================

-- | Feedback loop state
data FeedbackLoop = FeedbackLoop
  { flGain         :: !Double              -- ^ Loop gain [0, 2]
  , flDamping      :: !Double              -- ^ Damping factor [0, 1]
  , flHistory      :: ![Double]            -- ^ Recent coherence values
  , flTarget       :: !Double              -- ^ Target coherence
  , flConverged    :: !Bool                -- ^ Convergence achieved
  } deriving (Eq, Show)

-- | Initialize feedback loop
initFeedback :: Double -> FeedbackLoop
initFeedback target = FeedbackLoop
  { flGain = 1.0
  , flDamping = 0.3
  , flHistory = []
  , flTarget = target
  , flConverged = False
  }

-- | Execute one feedback cycle
feedbackCycle :: ResonanceEngine -> BiometricSnapshot -> ResonanceEngine
feedbackCycle engine snapshot =
  let (newEngine, _adjustment) = processSnapshot engine snapshot
      fb = reFeedback newEngine

      -- Update history
      newHistory = bsCoherence snapshot : take 9 (flHistory fb)

      -- Check convergence (3 consecutive values near target)
      converged = checkConvergence newHistory (flTarget fb)

      -- Adjust gain based on error
      err = abs (bsCoherence snapshot - flTarget fb)
      newGain = if err > 0.2 then min 2.0 (flGain fb * 1.1)
                else max 0.5 (flGain fb * 0.95)

      newFeedback = fb
        { flHistory = newHistory
        , flGain = newGain
        , flConverged = converged
        }

  in newEngine { reFeedback = newFeedback }

-- | Self-regulate to maintain coherence
selfRegulate :: ResonanceEngine -> ResonanceEngine
selfRegulate engine =
  let fb = reFeedback engine
      res = reResonance engine

      -- Apply damping if unstable
      damped = if rsStability res < 0.3
               then engine { reFeedback = fb { flDamping = min 0.8 (flDamping fb + 0.1) } }
               else engine

      -- Release if fully charged and aligned
      released = if jcChargeState (reJoeChamber damped) == ChargeFull && rsAlignment res > phiInverse
                 then fst (fieldRelease damped)
                 else damped

  in released

-- =============================================================================
-- Safety Systems
-- =============================================================================

-- | Safety state
data SafetyState
  = SafetyOk          -- ^ Normal operation
  | SafetyWarning     -- ^ Elevated but manageable
  | SafetyCritical    -- ^ Requires intervention
  | SafetyShutdown    -- ^ Emergency shutdown active
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Check safety status
checkSafety :: ResonanceEngine -> SafetyState
checkSafety engine =
  let core = reTeslaCore engine
      res = reResonance engine
      fb = reFeedback engine

      -- Check for runaway conditions
      chargeHigh = tcCharge core > 0.95
      unstable = rsStability res < 0.2
      gainHigh = flGain fb > 1.8

  in if chargeHigh && unstable then SafetyCritical
     else if chargeHigh || gainHigh then SafetyWarning
     else SafetyOk

-- | Emergency damping
emergencyDamp :: ResonanceEngine -> ResonanceEngine
emergencyDamp engine =
  let core = (reTeslaCore engine) { tcCharge = 0.1, tcAmplitude = 0.2 }
      fb = (reFeedback engine) { flGain = 0.5, flDamping = 0.8 }
  in engine
    { reTeslaCore = core
    , reFeedback = fb
    , reSafety = SafetyShutdown
    , reActive = False
    }

-- | Reset engine to default state
resetEngine :: ResonanceEngine -> ResonanceEngine
resetEngine engine = engine
  { reTeslaCore = defaultTeslaCore
  , reJoeChamber = defaultJoeChamber
  , reResonance = defaultResonance
  , reFeedback = defaultFeedback
  , reSafety = SafetyOk
  , reActive = False
  , reCycles = 0
  }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Default Tesla core
defaultTeslaCore :: TeslaCore
defaultTeslaCore = TeslaCore
  { tcCoilType = CoilBifilar
  , tcWindings = 100
  , tcFrequency = 7.83  -- Schumann
  , tcPhase = 0
  , tcAmplitude = 0.5
  , tcCharge = 0
  }

-- | Default Joe chamber with 4 cylinders
defaultJoeChamber :: JoeChamber
defaultJoeChamber = JoeChamber
  { jcCylinders =
    [ CylinderLayer 0.05 MaterialSteel 0 5
    , CylinderLayer 0.10 MaterialSteel 0 10
    , CylinderLayer 0.15 MaterialSteel 0 15
    , CylinderLayer 0.20 MaterialSteel 0 20
    ]
  , jcWaterLevel = 0.8
  , jcChargeState = ChargeNeutral
  , jcScalarField = 0.5
  , jcAlignment = 0.5
  }

-- | Default resonance state
defaultResonance :: ResonanceState
defaultResonance = ResonanceState
  { rsScalarShift = 0
  , rsAlignment = 0.5
  , rsHarmonic = 1
  , rsLocked = False
  , rsStability = 0.8
  }

-- | Default feedback loop
defaultFeedback :: FeedbackLoop
defaultFeedback = FeedbackLoop
  { flGain = 1.0
  , flDamping = 0.3
  , flHistory = []
  , flTarget = phiInverse
  , flConverged = False
  }

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 = max 0 . min 1

-- | Clamp to [-1, 1]
clamp11 :: Double -> Double
clamp11 = max (-1) . min 1

-- | Check convergence
checkConvergence :: [Double] -> Double -> Bool
checkConvergence history target =
  length history >= 3 &&
  all (\v -> abs (v - target) < 0.1) (take 3 history)
