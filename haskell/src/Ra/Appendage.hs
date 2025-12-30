{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

{-|
Module      : Ra.Appendage
Description : Resonant Appendages - scalar-field-based digital limbs
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Resonant Appendages are scalar-field-based digital limbs that manifest as
extensions of a user's intention and coherence. They serve as interactive
vector arms that allow a user's digital twin to manipulate screen elements
through direct intention.

Key features:
- Consent and coherence gating for activation
- Maps field position (RaCoordinate) to screen-space (ScreenTarget)
- Accessibility-focused: ignores physical limb limitations
- Gesture as invocation model

The vector arm is not metaphor; it is a projected intentional interface.
-}
module Ra.Appendage
  ( -- * Core Types
    ResonantAppendage(..)
  , IntentSignal(..)
  , AppendageAction(..)
  , ScreenTarget(..)
  , AppendageState(..)
  , ActivationResult(..)

    -- * Smart Constructors
  , mkResonantAppendage
  , mkIntentSignal
  , mkScreenTarget

    -- * Activation Logic
  , activateAppendage
  , evaluateAppendageActivation
  , canActivate

    -- * Constants
  , appendageActivationThreshold
  , intentCoherenceWeight
  , biometricCoherenceWeight

    -- * Helpers
  , combineCoherence
  , mapToScreenSpace
  , appendageReady

    -- * Control Quality (Flux Coherence Mapping)
  , ControlQuality(..)
  , fluxToControlQuality
  , effectiveRange
  , controlNoise
  , controlLatency

    -- * Fragment Archetype Appendages
  , FragmentArchetype(..)
  , archetypeAppendages
  , createArchetypeAppendages

    -- * Consent-Based Restrictions
  , ConsentLevel(..)
  , ConsentRestrictions(..)
  , applyConsentRestrictions
  , restrictedActivation

    -- * Emotion-Based Mutations
  , FragmentEmotion(..)
  , AppendageVisual(..)
  , AppendageBehavior(..)
  , AppendageMutation(..)
  , mutateByEmotion
  , applyEmotionToControl
  , defaultVisual
  , defaultBehavior
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)
import Data.Maybe (isJust)

import Ra.Scalar (Coordinate(..), FragmentReluctance(..), HarmonicSignature(..))
import Ra.Rac (RacLevel(..))
import Ra.Gates (accessLevel, isBlocked)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Actions that an appendage can perform
data AppendageAction
  = Grasp      -- ^ Grab/select an element
  | Point      -- ^ Highlight/focus without selection
  | Pull       -- ^ Drag toward user
  | Push       -- ^ Push away from user
  | Rotate     -- ^ Rotate element
  | Pinch      -- ^ Zoom in gesture
  | Expand     -- ^ Zoom out gesture
  | Wave       -- ^ Attention/dismiss gesture
  | Invoke     -- ^ Activate/execute gesture
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | Screen target for appendage interaction
data ScreenTarget = ScreenTarget
  { stX        :: !Double        -- ^ X coordinate (normalized 0-1)
  , stY        :: !Double        -- ^ Y coordinate (normalized 0-1)
  , stElementId :: !(Maybe String) -- ^ Optional UI element ID
  , stDepth    :: !Double        -- ^ Z-depth for layered UIs (0 = front)
  } deriving (Eq, Show, Generic, NFData)

-- | Smart constructor for ScreenTarget, clamping coordinates to [0, 1]
mkScreenTarget :: Double -> Double -> Maybe String -> Double -> ScreenTarget
mkScreenTarget x y elemId depth = ScreenTarget
  { stX = clamp01 x
  , stY = clamp01 y
  , stElementId = elemId
  , stDepth = max 0 depth
  }
  where clamp01 v = max 0.0 (min 1.0 v)

-- | User's focused semantic + biometric coherence
data IntentSignal = IntentSignal
  { isSemanticFocus   :: !Double        -- ^ Focus coherence [0, 1]
  , isBiometricSync   :: !Double        -- ^ Biometric alignment [0, 1]
  , isHarmonicProfile :: !(Maybe HarmonicSignature) -- ^ User's harmonic signature
  , isIntentVector    :: !(Double, Double, Double)  -- ^ 3D intent direction (normalized)
  } deriving (Eq, Show, Generic, NFData)

-- | Smart constructor for IntentSignal, clamping coherence values
mkIntentSignal
  :: Double                       -- ^ Semantic focus [0, 1]
  -> Double                       -- ^ Biometric sync [0, 1]
  -> Maybe HarmonicSignature      -- ^ Harmonic profile
  -> (Double, Double, Double)     -- ^ Intent vector (will be normalized)
  -> IntentSignal
mkIntentSignal focus bio harmonic vec = IntentSignal
  { isSemanticFocus = clamp01 focus
  , isBiometricSync = clamp01 bio
  , isHarmonicProfile = harmonic
  , isIntentVector = normalizeVec vec
  }
  where
    clamp01 v = max 0.0 (min 1.0 v)
    normalizeVec (x, y, z) =
      let mag = sqrt (x*x + y*y + z*z)
      in if mag > 0 then (x/mag, y/mag, z/mag) else (0, 0, 1)

-- | State of an appendage
data AppendageState
  = Dormant           -- ^ Not active, not visible
  | Awakening         -- ^ Transitioning to active
  | Active            -- ^ Fully manifested and responsive
  | Fading            -- ^ Coherence dropping, losing form
  | Blocked           -- ^ Consent denied
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | A resonant appendage - digital limb for intention-based interaction
data ResonantAppendage = ResonantAppendage
  { raId           :: !String              -- ^ Unique identifier
  , raAction       :: !AppendageAction     -- ^ Current/default action type
  , raRange        :: !Double              -- ^ Effective range (normalized 0-1)
  , raCoherence    :: !Double              -- ^ Current coherence level [0, 1]
  , raState        :: !AppendageState      -- ^ Current state
  , raTarget       :: !(Maybe ScreenTarget) -- ^ Current target (if any)
  , raReluctance   :: !FragmentReluctance  -- ^ Internal consent constraints
  , raAccessLevel  :: !RacLevel            -- ^ Required access level
  } deriving (Eq, Show, Generic, NFData)

-- | Smart constructor for ResonantAppendage
mkResonantAppendage
  :: String                 -- ^ Unique ID
  -> AppendageAction        -- ^ Action type
  -> Double                 -- ^ Range
  -> RacLevel               -- ^ Required access level
  -> FragmentReluctance     -- ^ Reluctance constraints
  -> ResonantAppendage
mkResonantAppendage aid action range rac reluc = ResonantAppendage
  { raId = aid
  , raAction = action
  , raRange = max 0.0 (min 1.0 range)
  , raCoherence = 0.0
  , raState = Dormant
  , raTarget = Nothing
  , raReluctance = reluc
  , raAccessLevel = rac
  }

-- | Result of appendage activation attempt
data ActivationResult
  = Activated ResonantAppendage      -- ^ Successfully activated
  | PartialActivation Double ResonantAppendage  -- ^ Partially activated with alpha
  | InsufficientCoherence Double     -- ^ Coherence too low (with current value)
  | AccessDenied RacLevel            -- ^ Access level insufficient
  | ReluctanceBlocked                -- ^ Internal reluctance prevents activation
  | GuardianRejected                 -- ^ Harmonic signature mismatch
  | NoTarget                         -- ^ No valid target specified
  deriving (Eq, Show, Generic)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Minimum coherence required for appendage activation
appendageActivationThreshold :: Double
appendageActivationThreshold = 0.6

-- | Weight for semantic focus in combined coherence
intentCoherenceWeight :: Double
intentCoherenceWeight = 0.6

-- | Weight for biometric sync in combined coherence
biometricCoherenceWeight :: Double
biometricCoherenceWeight = 0.4

-- =============================================================================
-- Core Logic
-- =============================================================================

-- | Combine intent signal components into a single coherence value
combineCoherence :: IntentSignal -> Double
combineCoherence is =
  intentCoherenceWeight * isSemanticFocus is +
  biometricCoherenceWeight * isBiometricSync is

-- | Check if appendage can potentially activate (basic checks)
canActivate :: IntentSignal -> ResonantAppendage -> Bool
canActivate intent app =
  let coh = combineCoherence intent
      floor' = reluctanceFloor (raReluctance app)
  in coh >= appendageActivationThreshold && coh >= floor'

-- | Map field coordinate to screen space
-- Uses theta for X, phi for Y, shell for depth
mapToScreenSpace :: Coordinate -> ScreenTarget
mapToScreenSpace _coord =
  -- Simplified mapping: would use actual RaCoordinate fields in full impl
  mkScreenTarget 0.5 0.5 Nothing 0.0

-- | Check if appendage is ready to respond
appendageReady :: ResonantAppendage -> Bool
appendageReady app = raState app == Active

-- | Evaluate appendage activation with full consent/coherence gating
evaluateAppendageActivation
  :: IntentSignal           -- ^ User's intent signal
  -> ResonantAppendage      -- ^ Appendage to activate
  -> Maybe ScreenTarget     -- ^ Target to interact with
  -> ActivationResult
evaluateAppendageActivation intent app target
  -- 1. Check if target is specified
  | isJust target && isNothing (raTarget app) || isJust target =
      evaluateWithTarget intent app (maybe (raTarget app) Just target)
  | otherwise =
      NoTarget
  where
    isNothing Nothing = True
    isNothing _ = False

evaluateWithTarget :: IntentSignal -> ResonantAppendage -> Maybe ScreenTarget -> ActivationResult
evaluateWithTarget intent app target =
  let coh = combineCoherence intent
      reluc = raReluctance app
      floor' = reluctanceFloor reluc
      guardMatch = matchGuardian (isHarmonicProfile intent) (guardian reluc)
      accessResult = accessLevel coh (raAccessLevel app)
  in
    -- 2. Check reluctance floor
    if coh < floor'
    then ReluctanceBlocked
    -- 3. Check guardian harmonic
    else if not guardMatch
    then GuardianRejected
    -- 4. Check access level
    else if isBlocked accessResult
    then AccessDenied (raAccessLevel app)
    -- 5. Check activation threshold
    else if coh < appendageActivationThreshold
    then InsufficientCoherence coh
    -- 6. Full or partial activation
    else if coh >= 0.9
    then Activated (activatedApp coh target)
    else PartialActivation coh (activatedApp coh target)
  where
    activatedApp c t = app
      { raCoherence = c
      , raState = if c >= 0.9 then Active else Awakening
      , raTarget = t
      }

    matchGuardian :: Maybe HarmonicSignature -> Maybe HarmonicSignature -> Bool
    matchGuardian _ Nothing = True
    matchGuardian Nothing _ = True
    matchGuardian (Just user) (Just guard) =
      hsL user == hsL guard && hsM user == hsM guard

-- | Main activation function
activateAppendage
  :: IntentSignal
  -> ResonantAppendage
  -> ScreenTarget
  -> ActivationResult
activateAppendage intent app target =
  evaluateAppendageActivation intent app (Just target)

-- =============================================================================
-- Control Quality (Flux Coherence Mapping)
-- =============================================================================

-- | Control quality parameters derived from flux coherence
-- Higher coherence = better control (longer range, less noise, lower latency)
data ControlQuality = ControlQuality
  { cqEffectiveRange :: !Double    -- ^ Effective range multiplier [0.1, 1.0]
  , cqNoise          :: !Double    -- ^ Position noise amplitude [0.0, 0.5]
  , cqLatency        :: !Double    -- ^ Response latency in ms [10, 500]
  , cqStability      :: !Double    -- ^ Stability factor [0.0, 1.0]
  } deriving (Eq, Show, Generic, NFData)

-- | Transform flux coherence to control quality parameters
-- Uses golden ratio (phi) for aesthetic decay curves
fluxToControlQuality :: Double -> ControlQuality
fluxToControlQuality flux =
  let coh = max 0.0 (min 1.0 flux)
      phi = 1.618033988749895  -- Golden ratio

      -- Range: 0.1 at coh=0, 1.0 at coh=1 (linear)
      range' = 0.1 + 0.9 * coh

      -- Noise: 0.5 at coh=0, 0.0 at coh=1 (inverse)
      noise' = 0.5 * (1.0 - coh)

      -- Latency: 500ms at coh=0, 10ms at coh=1 (exponential decay)
      latency' = 10.0 + 490.0 * (1.0 - coh ** phi)

      -- Stability: 0.0 at coh=0, 1.0 at coh=1 (sigmoid-like)
      stability' = tanh' (coh * 2.0)

  in ControlQuality
    { cqEffectiveRange = range'
    , cqNoise = noise'
    , cqLatency = latency'
    , cqStability = stability'
    }
  where
    -- Approximate tanh for stability curve
    tanh' x = (exp x - exp (-x)) / (exp x + exp (-x))

-- | Get effective range given flux coherence
effectiveRange :: Double -> Double
effectiveRange = cqEffectiveRange . fluxToControlQuality

-- | Get control noise amplitude given flux coherence
controlNoise :: Double -> Double
controlNoise = cqNoise . fluxToControlQuality

-- | Get control latency in milliseconds given flux coherence
controlLatency :: Double -> Double
controlLatency = cqLatency . fluxToControlQuality

-- =============================================================================
-- Fragment Archetype Appendage Assignment
-- =============================================================================

-- | Fragment archetype determines appendage capabilities
data FragmentArchetype
  = ObserverArchetype      -- ^ Passive observation, minimal appendages
  | ManipulatorArchetype   -- ^ Active manipulation, full appendage suite
  | CommunicatorArchetype  -- ^ Communication focus, pointing/waving
  | GuardianArchetype      -- ^ Protection focus, blocking/shielding
  | CreatorArchetype       -- ^ Creation focus, pulling/rotating
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | Default appendage configuration based on archetype
archetypeAppendages :: FragmentArchetype -> [AppendageAction]
archetypeAppendages archetype = case archetype of
  ObserverArchetype     -> [Point]
  ManipulatorArchetype  -> [Grasp, Pull, Push, Rotate]
  CommunicatorArchetype -> [Point, Wave, Invoke]
  GuardianArchetype     -> [Push, Wave]
  CreatorArchetype      -> [Grasp, Pull, Rotate, Pinch, Expand]

-- | Create appendages for a fragment based on its archetype
-- Returns list of appendages with appropriate access levels
createArchetypeAppendages
  :: String            -- ^ Fragment ID prefix
  -> FragmentArchetype -- ^ Fragment's archetype
  -> RacLevel          -- ^ Required access level
  -> FragmentReluctance -- ^ Reluctance constraints
  -> [ResonantAppendage]
createArchetypeAppendages prefix archetype rac reluc =
  [ mkResonantAppendage
      (prefix ++ "_" ++ show action)
      action
      (rangeForAction action)
      rac
      reluc
  | action <- archetypeAppendages archetype
  ]
  where
    -- Different actions have different default ranges
    rangeForAction Grasp  = 0.3
    rangeForAction Point  = 1.0
    rangeForAction Pull   = 0.4
    rangeForAction Push   = 0.5
    rangeForAction Rotate = 0.2
    rangeForAction Pinch  = 0.15
    rangeForAction Expand = 0.6
    rangeForAction Wave   = 0.8
    rangeForAction Invoke = 0.5

-- =============================================================================
-- Consent-Based Restrictions
-- =============================================================================

-- | Consent level for appendage access control
-- Maps to ACSP (Avatar Consent State Protocol) states
data ConsentLevel
  = FullConsent           -- ^ Full access to all appendage capabilities
  | DiminishedConsent     -- ^ Reduced range and precision
  | SuspendedConsent      -- ^ Minimal interaction only (Point)
  | NoConsent             -- ^ All appendages blocked
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | Restrictions applied based on consent level
data ConsentRestrictions = ConsentRestrictions
  { crAllowedActions   :: ![AppendageAction]   -- ^ Actions permitted at this consent level
  , crRangeMultiplier  :: !Double              -- ^ Range reduction factor [0, 1]
  , crCoherenceBonus   :: !Double              -- ^ Extra coherence required [-1, 0]
  , crLatencyPenalty   :: !Double              -- ^ Additional latency in ms [0, inf)
  } deriving (Eq, Show, Generic, NFData)

-- | Get restrictions for a given consent level
applyConsentRestrictions :: ConsentLevel -> ConsentRestrictions
applyConsentRestrictions level = case level of
  FullConsent -> ConsentRestrictions
    { crAllowedActions = [minBound .. maxBound]  -- All actions
    , crRangeMultiplier = 1.0
    , crCoherenceBonus = 0.0
    , crLatencyPenalty = 0.0
    }
  DiminishedConsent -> ConsentRestrictions
    { crAllowedActions = [Point, Grasp, Pull, Push, Wave]  -- No Rotate, Pinch, Expand, Invoke
    , crRangeMultiplier = 0.7
    , crCoherenceBonus = -0.1  -- Requires 10% more coherence
    , crLatencyPenalty = 50.0  -- 50ms extra latency
    }
  SuspendedConsent -> ConsentRestrictions
    { crAllowedActions = [Point, Wave]  -- Minimal interaction
    , crRangeMultiplier = 0.3
    , crCoherenceBonus = -0.2  -- Requires 20% more coherence
    , crLatencyPenalty = 200.0  -- 200ms extra latency
    }
  NoConsent -> ConsentRestrictions
    { crAllowedActions = []  -- No actions allowed
    , crRangeMultiplier = 0.0
    , crCoherenceBonus = -1.0  -- Effectively impossible
    , crLatencyPenalty = 10000.0  -- Infinite latency
    }

-- | Attempt activation with consent restrictions
restrictedActivation
  :: ConsentLevel         -- ^ Current consent level
  -> IntentSignal         -- ^ User's intent
  -> ResonantAppendage    -- ^ Appendage to activate
  -> ScreenTarget         -- ^ Target location
  -> ActivationResult
restrictedActivation consent intent app target =
  let restrictions = applyConsentRestrictions consent
      action = raAction app
      allowed = action `elem` crAllowedActions restrictions
      adjustedCoherence = combineCoherence intent + crCoherenceBonus restrictions
      adjustedIntent = intent { isSemanticFocus = adjustedCoherence }
      adjustedApp = app { raRange = raRange app * crRangeMultiplier restrictions }
  in
    if not allowed
    then AccessDenied (raAccessLevel app)
    else if consent == NoConsent
    then AccessDenied (raAccessLevel app)
    else activateAppendage adjustedIntent adjustedApp target

-- =============================================================================
-- Emotion-Based Appendage Mutations
-- =============================================================================

-- | Fragment emotional states that affect appendage form
data FragmentEmotion
  = Calm              -- ^ Baseline state, smooth motion
  | Focused           -- ^ Heightened precision, reduced range
  | Anxious           -- ^ Tremor, unstable targeting
  | Joyful            -- ^ Expanded range, fluid motion
  | Grieving          -- ^ Reduced responsiveness, heavy movement
  | Protective        -- ^ Shield-like behavior, defensive posture
  | Curious           -- ^ Extended reach, exploratory motion
  | Withdrawn         -- ^ Contracted form, minimal presence
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | Visual characteristics of appendage based on emotion
data AppendageVisual = AppendageVisual
  { avColor       :: !(Double, Double, Double)  -- ^ RGB color [0-1]
  , avAlpha       :: !Double                     -- ^ Opacity [0-1]
  , avPulseRate   :: !Double                     -- ^ Visual pulse frequency Hz
  , avTrailLength :: !Double                     -- ^ Motion trail length [0-1]
  , avGlowRadius  :: !Double                     -- ^ Glow effect radius [0-1]
  } deriving (Eq, Show, Generic, NFData)

-- | Behavioral modifications based on emotion
data AppendageBehavior = AppendageBehavior
  { abRangeModifier     :: !Double  -- ^ Range multiplier [0.1-2.0]
  , abPrecisionModifier :: !Double  -- ^ Precision multiplier [0.1-2.0]
  , abSpeedModifier     :: !Double  -- ^ Speed multiplier [0.1-2.0]
  , abTremorAmplitude   :: !Double  -- ^ Random position noise [0-0.5]
  , abResponseDelay     :: !Double  -- ^ Additional delay ms [0-500]
  } deriving (Eq, Show, Generic, NFData)

-- | Combined visual and behavioral mutation
data AppendageMutation = AppendageMutation
  { amVisual   :: !AppendageVisual
  , amBehavior :: !AppendageBehavior
  } deriving (Eq, Show, Generic, NFData)

-- | Default visual (calm state)
defaultVisual :: AppendageVisual
defaultVisual = AppendageVisual
  { avColor = (0.4, 0.6, 0.9)      -- Soft blue
  , avAlpha = 0.8
  , avPulseRate = 0.5              -- Slow, steady pulse
  , avTrailLength = 0.3
  , avGlowRadius = 0.1
  }

-- | Default behavior (calm state)
defaultBehavior :: AppendageBehavior
defaultBehavior = AppendageBehavior
  { abRangeModifier = 1.0
  , abPrecisionModifier = 1.0
  , abSpeedModifier = 1.0
  , abTremorAmplitude = 0.0
  , abResponseDelay = 0.0
  }

-- | Transform appendage appearance and behavior based on emotion
mutateByEmotion :: FragmentEmotion -> AppendageMutation
mutateByEmotion emotion = case emotion of
  Calm -> AppendageMutation defaultVisual defaultBehavior

  Focused -> AppendageMutation
    (defaultVisual
      { avColor = (0.2, 0.8, 0.4)   -- Sharp green
      , avAlpha = 0.95
      , avPulseRate = 0.0           -- No pulse, steady
      , avTrailLength = 0.1         -- Minimal trail
      , avGlowRadius = 0.05
      })
    (defaultBehavior
      { abRangeModifier = 0.7       -- Reduced range
      , abPrecisionModifier = 1.5   -- Enhanced precision
      , abSpeedModifier = 1.2
      })

  Anxious -> AppendageMutation
    (defaultVisual
      { avColor = (0.9, 0.7, 0.2)   -- Warning yellow
      , avAlpha = 0.6
      , avPulseRate = 3.0           -- Rapid flickering
      , avTrailLength = 0.5         -- Scattered trail
      , avGlowRadius = 0.2
      })
    (defaultBehavior
      { abRangeModifier = 0.8
      , abPrecisionModifier = 0.5   -- Poor precision
      , abSpeedModifier = 1.5       -- Jerky, fast
      , abTremorAmplitude = 0.15    -- Visible tremor
      , abResponseDelay = 50.0
      })

  Joyful -> AppendageMutation
    (defaultVisual
      { avColor = (1.0, 0.8, 0.3)   -- Warm gold
      , avAlpha = 0.9
      , avPulseRate = 1.0           -- Rhythmic pulse
      , avTrailLength = 0.6         -- Flowing trail
      , avGlowRadius = 0.3          -- Expanded glow
      })
    (defaultBehavior
      { abRangeModifier = 1.3       -- Extended range
      , abPrecisionModifier = 0.9
      , abSpeedModifier = 1.1       -- Slightly faster
      })

  Grieving -> AppendageMutation
    (defaultVisual
      { avColor = (0.3, 0.3, 0.5)   -- Muted blue-grey
      , avAlpha = 0.5
      , avPulseRate = 0.2           -- Very slow
      , avTrailLength = 0.7         -- Long, fading trail
      , avGlowRadius = 0.05
      })
    (defaultBehavior
      { abRangeModifier = 0.6       -- Contracted
      , abPrecisionModifier = 0.7
      , abSpeedModifier = 0.5       -- Heavy, slow
      , abResponseDelay = 200.0     -- Delayed response
      })

  Protective -> AppendageMutation
    (defaultVisual
      { avColor = (0.6, 0.2, 0.8)   -- Purple shield
      , avAlpha = 1.0
      , avPulseRate = 0.8
      , avTrailLength = 0.2
      , avGlowRadius = 0.4          -- Strong barrier glow
      })
    (defaultBehavior
      { abRangeModifier = 0.5       -- Close-in defense
      , abPrecisionModifier = 1.2
      , abSpeedModifier = 1.4       -- Quick reactive
      })

  Curious -> AppendageMutation
    (defaultVisual
      { avColor = (0.3, 0.9, 0.9)   -- Cyan exploratory
      , avAlpha = 0.7
      , avPulseRate = 1.5           -- Active scanning
      , avTrailLength = 0.4
      , avGlowRadius = 0.25
      })
    (defaultBehavior
      { abRangeModifier = 1.5       -- Extended reach
      , abPrecisionModifier = 0.8
      , abSpeedModifier = 0.9       -- Deliberate
      })

  Withdrawn -> AppendageMutation
    (defaultVisual
      { avColor = (0.2, 0.2, 0.3)   -- Dark, faded
      , avAlpha = 0.3
      , avPulseRate = 0.1           -- Almost imperceptible
      , avTrailLength = 0.0         -- No trail
      , avGlowRadius = 0.0
      })
    (defaultBehavior
      { abRangeModifier = 0.2       -- Minimal reach
      , abPrecisionModifier = 0.6
      , abSpeedModifier = 0.4       -- Reluctant
      , abResponseDelay = 300.0
      })

-- | Apply emotion-based mutation to appendage control quality
applyEmotionToControl :: FragmentEmotion -> ControlQuality -> ControlQuality
applyEmotionToControl emotion cq =
  let mutation = mutateByEmotion emotion
      behavior = amBehavior mutation
  in cq
    { cqEffectiveRange = cqEffectiveRange cq * abRangeModifier behavior
    , cqNoise = cqNoise cq + abTremorAmplitude behavior
    , cqLatency = cqLatency cq + abResponseDelay behavior
    , cqStability = cqStability cq * abPrecisionModifier behavior
    }
