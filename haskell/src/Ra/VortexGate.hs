{-|
Module      : Ra.VortexGate
Description : Scalar vortex gate as consent-based portal
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements a dynamic portal system that activates only when a user meets
specific coherence thresholds AND consent states. Styled as a torsion vortex
with scalar field dynamics.

== Vortex Gate Theory

=== Torsion Portal Mechanics

The gate operates via:

* Rotating scalar vortex field
* Coherence-locked activation threshold
* Consent state verification
* Harmonic key matching

=== Activation Requirements

Gates require:

* Minimum coherence level
* Specific consent state (FULL, ACTIVE, etc.)
* Optional harmonic key matching
* Stable field conditions

=== Portal States

* Dormant: No activity
* Charging: Building coherence
* Aligned: Ready for activation
* Open: Passage available
* Closing: Winding down
-}
module Ra.VortexGate
  ( -- * Core Types
    VortexGate(..)
  , GateState(..)
  , GateConfig(..)
  , mkVortexGate

    -- * Activation
  , ActivationResult(..)
  , attemptActivation
  , activationProgress
  , forceClose

    -- * Coherence Requirements
  , CoherenceReq(..)
  , checkCoherence
  , coherenceDeficit

    -- * Consent Requirements
  , ConsentReq(..)
  , ConsentState(..)
  , checkConsent
  , requiredConsent

    -- * Harmonic Keys
  , HarmonicKey(..)
  , keyMatch
  , generateKey

    -- * Vortex Dynamics
  , VortexField(..)
  , vortexStrength
  , vortexPhase
  , rotationSpeed

    -- * Passage
  , PassageResult(..)
  , attemptPassage
  , passageAllowed

    -- * Visualization
  , GateVisual(..)
  , visualize
  , vortexColors
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Vortex gate configuration
data VortexGate = VortexGate
  { vgId          :: !String
  , vgState       :: !GateState
  , vgConfig      :: !GateConfig
  , vgVortex      :: !VortexField
  , vgChargeLevel :: !Double        -- ^ Current charge [0,1]
  , vgLastAccess  :: !Double        -- ^ Last access timestamp
  } deriving (Eq, Show)

-- | Gate operational state
data GateState
  = Dormant     -- ^ No activity
  | Charging    -- ^ Building coherence
  | Aligned     -- ^ Ready for activation
  | Open        -- ^ Passage available
  | Closing     -- ^ Winding down
  | Locked      -- ^ Explicitly locked
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Gate configuration
data GateConfig = GateConfig
  { gcCoherenceReq  :: !CoherenceReq
  , gcConsentReq    :: !ConsentReq
  , gcHarmonicKey   :: !(Maybe HarmonicKey)
  , gcOpenDuration  :: !Double        -- ^ How long gate stays open (s)
  , gcChargeDuration :: !Double       -- ^ Time to charge (s)
  } deriving (Eq, Show)

-- | Create vortex gate
mkVortexGate :: String -> GateConfig -> VortexGate
mkVortexGate gateId config = VortexGate
  { vgId = gateId
  , vgState = Dormant
  , vgConfig = config
  , vgVortex = mkVortexField (crThreshold (gcCoherenceReq config))
  , vgChargeLevel = 0.0
  , vgLastAccess = 0.0
  }

-- =============================================================================
-- Activation
-- =============================================================================

-- | Activation result
data ActivationResult = ActivationResult
  { arSuccess     :: !Bool
  , arNewState    :: !GateState
  , arMessage     :: !String
  , arProgress    :: !Double        -- ^ Progress toward activation [0,1]
  } deriving (Eq, Show)

-- | Attempt gate activation
attemptActivation :: Double -> ConsentState -> Maybe (Int, Int) -> VortexGate -> (VortexGate, ActivationResult)
attemptActivation coherence consent harmonic gate =
  let config = vgConfig gate

      -- Check coherence
      cohResult = checkCoherence coherence (gcCoherenceReq config)

      -- Check consent
      conResult = checkConsent consent (gcConsentReq config)

      -- Check harmonic key (if required)
      keyResult = case (gcHarmonicKey config, harmonic) of
        (Nothing, _) -> True
        (Just key, Just h) -> keyMatch key h
        (Just _, Nothing) -> False

      allPassed = cohResult && conResult && keyResult

      -- Determine new state
      currentState = vgState gate
      newState = case currentState of
        Dormant | allPassed -> Charging
        Charging | allPassed && vgChargeLevel gate >= 1.0 -> Aligned
        Charging | not allPassed -> Dormant
        Aligned | allPassed -> Open
        Open -> Open
        Closing -> if allPassed then Open else Closing
        Locked -> Locked
        _ -> currentState

      -- Update charge
      newCharge = if allPassed && currentState `elem` [Dormant, Charging]
                  then min 1.0 (vgChargeLevel gate + 0.1)
                  else if not allPassed
                  then max 0.0 (vgChargeLevel gate - 0.05)
                  else vgChargeLevel gate

      -- Progress
      progress = case newState of
        Dormant -> 0.0
        Charging -> newCharge
        Aligned -> 1.0
        Open -> 1.0
        Closing -> 0.5
        Locked -> 0.0

      -- Message
      message = case newState of
        Dormant -> "Gate dormant"
        Charging -> "Charging: " ++ show (round (newCharge * 100) :: Int) ++ "%"
        Aligned -> "Gate aligned - ready"
        Open -> "Gate OPEN"
        Closing -> "Gate closing"
        Locked -> "Gate locked"

      newGate = gate
        { vgState = newState
        , vgChargeLevel = newCharge
        }
  in (newGate, ActivationResult
      { arSuccess = newState == Open
      , arNewState = newState
      , arMessage = message
      , arProgress = progress
      })

-- | Get activation progress
activationProgress :: VortexGate -> Double
activationProgress gate = case vgState gate of
  Open -> 1.0
  Aligned -> 0.9
  Charging -> vgChargeLevel gate * 0.8
  _ -> 0.0

-- | Force close gate
forceClose :: VortexGate -> VortexGate
forceClose gate = gate
  { vgState = Closing
  , vgChargeLevel = max 0.0 (vgChargeLevel gate - 0.3)
  }

-- =============================================================================
-- Coherence Requirements
-- =============================================================================

-- | Coherence requirement
data CoherenceReq = CoherenceReq
  { crThreshold   :: !Double        -- ^ Minimum coherence required
  , crStability   :: !Double        -- ^ Required stability duration (s)
  , crMargin      :: !Double        -- ^ Allowed margin below threshold
  } deriving (Eq, Show)

-- | Check coherence requirement
checkCoherence :: Double -> CoherenceReq -> Bool
checkCoherence coherence req =
  coherence >= crThreshold req - crMargin req

-- | Calculate coherence deficit
coherenceDeficit :: Double -> CoherenceReq -> Double
coherenceDeficit coherence req =
  max 0 (crThreshold req - coherence)

-- =============================================================================
-- Consent Requirements
-- =============================================================================

-- | Consent requirement
data ConsentReq = ConsentReq
  { conrMinLevel   :: !ConsentState
  , conrVerified   :: !Bool          -- ^ Require verified consent
  , conrRecent     :: !Bool          -- ^ Require recent consent update
  } deriving (Eq, Show)

-- | Consent state
data ConsentState
  = NoConsent
  | Suspended
  | Diminished
  | Active
  | Full
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Check consent requirement
checkConsent :: ConsentState -> ConsentReq -> Bool
checkConsent consent req = consent >= conrMinLevel req

-- | Get required consent level
requiredConsent :: ConsentReq -> ConsentState
requiredConsent = conrMinLevel

-- =============================================================================
-- Harmonic Keys
-- =============================================================================

-- | Harmonic key for gate access
data HarmonicKey = HarmonicKey
  { hkL         :: !Int
  , hkM         :: !Int
  , hkTolerance :: !Int           -- ^ Allowed deviation
  } deriving (Eq, Show)

-- | Check if harmonic matches key
keyMatch :: HarmonicKey -> (Int, Int) -> Bool
keyMatch key (l, m) =
  abs (hkL key - l) <= hkTolerance key &&
  abs (hkM key - m) <= hkTolerance key

-- | Generate key from parameters
generateKey :: Int -> Int -> Int -> HarmonicKey
generateKey l m tol = HarmonicKey l m tol

-- =============================================================================
-- Vortex Dynamics
-- =============================================================================

-- | Vortex field state
data VortexField = VortexField
  { vfStrength    :: !Double        -- ^ Field strength [0,1]
  , vfPhase       :: !Double        -- ^ Rotation phase [0, 2*pi]
  , vfSpeed       :: !Double        -- ^ Rotation speed (rad/s)
  , vfRadius      :: !Double        -- ^ Vortex radius
  , vfDepth       :: !Double        -- ^ Vortex depth
  } deriving (Eq, Show)

-- | Create vortex field
mkVortexField :: Double -> VortexField
mkVortexField coherenceReq = VortexField
  { vfStrength = coherenceReq * phi
  , vfPhase = 0.0
  , vfSpeed = 2 * pi * phiInverse  -- One rotation per phi seconds
  , vfRadius = 1.0
  , vfDepth = coherenceReq * 2.0
  }

-- | Get vortex strength
vortexStrength :: VortexField -> Double
vortexStrength = vfStrength

-- | Get vortex phase
vortexPhase :: VortexField -> Double
vortexPhase = vfPhase

-- | Get rotation speed
rotationSpeed :: VortexField -> Double
rotationSpeed = vfSpeed

-- =============================================================================
-- Passage
-- =============================================================================

-- | Passage attempt result
data PassageResult = PassageResult
  { prAllowed     :: !Bool
  , prGate        :: !VortexGate
  , prMessage     :: !String
  , prFragmentId  :: !(Maybe String)
  } deriving (Eq, Show)

-- | Attempt passage through gate
attemptPassage :: String -> Double -> ConsentState -> VortexGate -> PassageResult
attemptPassage userId coherence consent gate =
  let allowed = vgState gate == Open &&
                checkCoherence coherence (gcCoherenceReq (vgConfig gate)) &&
                checkConsent consent (gcConsentReq (vgConfig gate))

      message = if allowed
                then "Passage granted to " ++ userId
                else "Passage denied: " ++ passageReason gate coherence consent

      -- Update gate access time
      newGate = gate { vgLastAccess = 0.0 }  -- Would use current time
  in PassageResult
      { prAllowed = allowed
      , prGate = newGate
      , prMessage = message
      , prFragmentId = if allowed then Just userId else Nothing
      }

-- Get passage denial reason
passageReason :: VortexGate -> Double -> ConsentState -> String
passageReason gate coherence consent
  | vgState gate /= Open = "Gate not open"
  | not (checkCoherence coherence (gcCoherenceReq (vgConfig gate))) = "Insufficient coherence"
  | not (checkConsent consent (gcConsentReq (vgConfig gate))) = "Insufficient consent"
  | otherwise = "Unknown"

-- | Check if passage is allowed
passageAllowed :: PassageResult -> Bool
passageAllowed = prAllowed

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Gate visual representation
data GateVisual = GateVisual
  { gvColor       :: !(Int, Int, Int)   -- ^ RGB color
  , gvGlow        :: !Double            -- ^ Glow intensity [0,1]
  , gvRotation    :: !Double            -- ^ Current rotation angle
  , gvRadius      :: !Double            -- ^ Visual radius
  , gvPattern     :: !String            -- ^ Vortex pattern type
  } deriving (Eq, Show)

-- | Create visual from gate
visualize :: VortexGate -> GateVisual
visualize gate =
  let state = vgState gate
      vortex = vgVortex gate

      color = stateColor state
      glow = case state of
        Open -> 1.0
        Aligned -> 0.8
        Charging -> vgChargeLevel gate * 0.6
        _ -> 0.1

      rotation = vfPhase vortex
      radius = vfRadius vortex * (if state == Open then 1.5 else 1.0)
      pattern = statePattern state
  in GateVisual
      { gvColor = color
      , gvGlow = glow
      , gvRotation = rotation
      , gvRadius = radius
      , gvPattern = pattern
      }

-- State to color
stateColor :: GateState -> (Int, Int, Int)
stateColor state = case state of
  Dormant -> (50, 50, 80)
  Charging -> (100, 100, 200)
  Aligned -> (100, 200, 255)
  Open -> (50, 255, 150)
  Closing -> (200, 150, 100)
  Locked -> (150, 50, 50)

-- State to pattern
statePattern :: GateState -> String
statePattern state = case state of
  Dormant -> "static"
  Charging -> "pulse"
  Aligned -> "spiral"
  Open -> "vortex"
  Closing -> "collapse"
  Locked -> "barrier"

-- | Get vortex colors (gradient)
vortexColors :: VortexGate -> [(Double, (Int, Int, Int))]
vortexColors gate =
  let (r, g, b) = stateColor (vgState gate)
      intensity = vgChargeLevel gate
      scale :: Double -> Int -> Int
      scale s v = round (fromIntegral v * s)
  in [ (0.0, (scale 0.3 r, scale 0.3 g, scale 0.3 b))
     , (0.5, (scale intensity r, scale intensity g, scale intensity b))
     , (1.0, (r, g, b))
     ]
