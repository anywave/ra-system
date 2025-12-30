{-|
Module      : Ra.Visualizer.Surfaces
Description : Scalar-interactive surface shader system with biometric coherence coupling
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Dynamic surface rendering system where digital surfaces, interfaces, and textures
respond to user scalar state (via Ra coherence model) and environmental torsion fields.

== Surface Applications

=== Responsive UI Elements

* Coherence-modulated material textures
* Consent-aware digital membranes
* Bio-reactive UI halos
* Permission-gated interface surfaces

=== Torsion-Based Effects

* Spin-induced refractive distortion
* Clockwise/counter-clockwise polarization
* Structural transformation overlays
* Scalar field visualization

=== Biometric Coupling

* Real-time coherence level mapping
* Heart rate variability response
* HRV stability indicators
* Consent state visual encoding
-}
module Ra.Visualizer.Surfaces
  ( -- * Core Types
    SurfaceState(..)
  , SpinState(..)
  , ConsentLevel(..)
  , ShaderParams(..)

    -- * Surface Configuration
  , SurfaceConfig(..)
  , defaultSurfaceConfig
  , configureSurface

    -- * Style Resolution
  , resolveSurfaceStyle
  , applyTorsionDistortion
  , calculateHueShift
  , computeTransparency

    -- * Consent Overlays
  , ConsentOverlay(..)
  , generateOverlay
  , showBioDome
  , applyTurbulence

    -- * Environment Surfaces
  , EnvironmentSurface(..)
  , BioHarmonicSignature(..)
  , createSurface
  , checkAlignment
  , surfaceReaction

    -- * Surface Interaction
  , InteractionResult(..)
  , InteractionType(..)
  , onTouch
  , onHover
  , onRelease

    -- * State Management
  , SurfaceStateManager(..)
  , initManager
  , updateSurfaceState
  , resetToNeutral
  , inactivityTimeout
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Shader state model for scalar-interactive surfaces
data SurfaceState = SurfaceState
  { ssCoherenceLevel :: !Double          -- ^ Coherence level [0, 1]
  , ssTorsionSpin    :: !SpinState       -- ^ Clockwise or counter spin
  , ssResonanceFreq  :: !Double          -- ^ Resonance frequency (Hz)
  , ssPermissionGate :: !ConsentLevel    -- ^ Current consent level
  , ssStability      :: !Double          -- ^ Surface stability [0, 1]
  , ssLastUpdate     :: !Int             -- ^ Last update timestamp
  } deriving (Eq, Show)

-- | Torsion spin state
data SpinState
  = SpinClockwise        -- ^ Clockwise rotation (positive torsion)
  | SpinCounter          -- ^ Counter-clockwise (negative torsion)
  | SpinNeutral          -- ^ No spin (balanced)
  | SpinOscillating      -- ^ Alternating spin
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Consent level for permission gating
data ConsentLevel
  = ConsentSuspended     -- ^ Consent withdrawn - show bio-dome
  | ConsentPassive       -- ^ Minimal consent - reduced interactivity
  | ConsentEngaged       -- ^ Full consent - full interactivity
  | ConsentAmplified     -- ^ Enhanced consent - boosted effects
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Shader parameters resolved from surface state
data ShaderParams = ShaderParams
  { spRefractiveIndex    :: !Double      -- ^ Refractive distortion factor
  , spHueShift           :: !Double      -- ^ Hue shift [0, 360]
  , spAlpha              :: !Double      -- ^ Alpha transparency [0, 1]
  , spHardness           :: !Double      -- ^ Surface hardness [0, 1]
  , spEmissive           :: !Double      -- ^ Emissive glow [0, 1]
  , spBloom              :: !Double      -- ^ Bloom radius [0, 2]
  , spTurbulence         :: !Double      -- ^ Turbulence intensity [0, 1]
  , spInteractZone       :: !Double      -- ^ Interaction zone radius
  } deriving (Eq, Show)

-- =============================================================================
-- Surface Configuration
-- =============================================================================

-- | Surface configuration settings
data SurfaceConfig = SurfaceConfig
  { scBaseFrequency      :: !Double      -- ^ Base resonance frequency
  , scSensitivity        :: !Double      -- ^ Biometric sensitivity [0, 1]
  , scReactTime          :: !Int         -- ^ Reaction time (ms)
  , scInactivityTimeout  :: !Int         -- ^ Timeout before reset (ms)
  , scTurbulenceThreshold :: !Double     -- ^ Coherence threshold for turbulence
  , scBloomMultiplier    :: !Double      -- ^ Bloom effect multiplier
  } deriving (Eq, Show)

-- | Default surface configuration
defaultSurfaceConfig :: SurfaceConfig
defaultSurfaceConfig = SurfaceConfig
  { scBaseFrequency = 528.0
  , scSensitivity = 0.7
  , scReactTime = 200
  , scInactivityTimeout = 30000
  , scTurbulenceThreshold = 0.2
  , scBloomMultiplier = phi
  }

-- | Configure surface with custom settings
configureSurface :: SurfaceConfig -> SurfaceState -> SurfaceState
configureSurface config state =
  state { ssResonanceFreq = scBaseFrequency config }

-- =============================================================================
-- Style Resolution
-- =============================================================================

-- | Resolve surface state to shader parameters
resolveSurfaceStyle :: SurfaceState -> ShaderParams
resolveSurfaceStyle state =
  let refract = applyTorsionDistortion (ssTorsionSpin state) (ssCoherenceLevel state)
      hue = calculateHueShift (ssResonanceFreq state)
      (alpha, hardness) = computeTransparency (ssCoherenceLevel state) (ssPermissionGate state)
      emissive = calculateEmissive state
      bloom = calculateBloom state
      turbulence = calculateTurbulence state
      interactZone = calculateInteractZone (ssPermissionGate state)
  in ShaderParams
    { spRefractiveIndex = refract
    , spHueShift = hue
    , spAlpha = alpha
    , spHardness = hardness
    , spEmissive = emissive
    , spBloom = bloom
    , spTurbulence = turbulence
    , spInteractZone = interactZone
    }

-- | Apply torsion-based refractive distortion
applyTorsionDistortion :: SpinState -> Double -> Double
applyTorsionDistortion spinState coherence =
  let baseDistortion = case spinState of
        SpinClockwise -> 0.15
        SpinCounter -> (-0.15)
        SpinNeutral -> 0
        SpinOscillating -> 0.1 * sin (coherence * pi)
      coherenceModulation = coherence * phiInverse
  in 1.0 + baseDistortion * coherenceModulation

-- | Calculate hue shift from resonance frequency
calculateHueShift :: Double -> Double
calculateHueShift freq =
  let normalized = (freq - 396) / (852 - 396)  -- Solfeggio range
      hue = normalized * 360
  in max 0 (min 360 hue)

-- | Compute alpha transparency and hardness from coherence and consent
computeTransparency :: Double -> ConsentLevel -> (Double, Double)
computeTransparency coherence consent =
  let baseAlpha = case consent of
        ConsentSuspended -> 0.3
        ConsentPassive -> 0.6
        ConsentEngaged -> 0.9
        ConsentAmplified -> 1.0
      alpha = baseAlpha * coherence
      hardness = coherence * phi / 2
  in (alpha, min 1.0 hardness)

-- =============================================================================
-- Consent Overlays
-- =============================================================================

-- | Consent overlay visualization
data ConsentOverlay = ConsentOverlay
  { coActive           :: !Bool          -- ^ Overlay active
  , coType             :: !OverlayType   -- ^ Overlay type
  , coOpacity          :: !Double        -- ^ Overlay opacity [0, 1]
  , coRadius           :: !Double        -- ^ Bio-dome radius
  , coTurbulence       :: !Double        -- ^ Turbulence level [0, 1]
  , coColor            :: !(Double, Double, Double)  -- ^ RGB color
  } deriving (Eq, Show)

-- | Overlay type
data OverlayType
  = OverlayBioDome       -- ^ Translucent bio-dome field
  | OverlayTurbulent     -- ^ Turbulent distortion
  | OverlayWarning       -- ^ Warning indicator
  | OverlayNone          -- ^ No overlay
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Generate consent overlay from surface state
generateOverlay :: SurfaceState -> ConsentOverlay
generateOverlay state =
  case ssPermissionGate state of
    ConsentSuspended ->
      showBioDome (ssCoherenceLevel state)
    _ ->
      if ssCoherenceLevel state < 0.2
      then applyTurbulence (ssCoherenceLevel state)
      else ConsentOverlay
        { coActive = False
        , coType = OverlayNone
        , coOpacity = 0
        , coRadius = 0
        , coTurbulence = 0
        , coColor = (0, 0, 0)
        }

-- | Show translucent bio-dome field overlay
showBioDome :: Double -> ConsentOverlay
showBioDome coherence = ConsentOverlay
  { coActive = True
  , coType = OverlayBioDome
  , coOpacity = 0.6 * (1 - coherence)
  , coRadius = 2.0 * phi
  , coTurbulence = 0.1
  , coColor = (0.2, 0.4, 0.8)  -- Blue tint
  }

-- | Apply turbulence overlay for low coherence
applyTurbulence :: Double -> ConsentOverlay
applyTurbulence coherence = ConsentOverlay
  { coActive = True
  , coType = OverlayTurbulent
  , coOpacity = 0.8 * (0.2 - coherence) / 0.2
  , coRadius = 1.5
  , coTurbulence = 1.0 - coherence * 5
  , coColor = (0.8, 0.3, 0.2)  -- Red-orange tint
  }

-- =============================================================================
-- Environment Surfaces
-- =============================================================================

-- | Environment surface with bio-harmonic signature
data EnvironmentSurface = EnvironmentSurface
  { esId               :: !String        -- ^ Surface identifier
  , esSignature        :: !BioHarmonicSignature  -- ^ Base signature
  , esState            :: !SurfaceState  -- ^ Current state
  , esInteractionStack :: ![InteractionType]  -- ^ Active interactions
  , esBloomState       :: !Double        -- ^ Current bloom level
  , esLastTouch        :: !Int           -- ^ Last touch timestamp
  } deriving (Eq, Show)

-- | Bio-harmonic signature for surface alignment
data BioHarmonicSignature = BioHarmonicSignature
  { bhsFrequency       :: !Double        -- ^ Signature frequency (Hz)
  , bhsCoherence       :: !Double        -- ^ Baseline coherence [0, 1]
  , bhsPhase           :: !Double        -- ^ Phase offset [0, 2pi]
  , bhsHarmonics       :: ![Double]      -- ^ Harmonic multipliers
  } deriving (Eq, Show)

-- | Create new environment surface
createSurface :: String -> BioHarmonicSignature -> EnvironmentSurface
createSurface surfaceId sig = EnvironmentSurface
  { esId = surfaceId
  , esSignature = sig
  , esState = defaultSurfaceState
  , esInteractionStack = []
  , esBloomState = 0
  , esLastTouch = 0
  }

-- | Check biometric alignment with surface
checkAlignment :: EnvironmentSurface -> BiometricSnapshot -> Bool
checkAlignment surface snapshot =
  let sigFreq = bhsFrequency (esSignature surface)
      bioFreq = bsFrequency snapshot
      freqDiff = abs (sigFreq - bioFreq) / max 1 sigFreq
      coherenceDiff = abs (bhsCoherence (esSignature surface) - bsCoherence snapshot)
  in freqDiff < 0.1 && coherenceDiff < phiInverse

-- | Calculate surface reaction to biometric input
surfaceReaction :: EnvironmentSurface -> BiometricSnapshot -> SurfaceReaction
surfaceReaction surface snapshot =
  let aligned = checkAlignment surface snapshot
      coherence = bsCoherence snapshot
  in if aligned
     then SurfaceReaction
       { srAligned = True
       , srBloomDelta = phi * 0.2 * coherence
       , srTurbulenceDelta = (-0.1)
       , srEmissiveDelta = 0.15 * coherence
       }
     else SurfaceReaction
       { srAligned = False
       , srBloomDelta = (-0.1)
       , srTurbulenceDelta = 0.3 * (1 - coherence)
       , srEmissiveDelta = (-0.1)
       }

-- =============================================================================
-- Surface Interaction
-- =============================================================================

-- | Result of surface interaction
data InteractionResult = InteractionResult
  { irSuccess          :: !Bool          -- ^ Interaction successful
  , irNewState         :: !SurfaceState  -- ^ Updated surface state
  , irVisualFeedback   :: !ShaderParams  -- ^ Visual feedback params
  , irHapticIntensity  :: !Double        -- ^ Haptic feedback [0, 1]
  } deriving (Eq, Show)

-- | Interaction type
data InteractionType
  = InteractTouch        -- ^ Direct touch
  | InteractHover        -- ^ Hover/proximity
  | InteractGaze         -- ^ Gaze tracking
  | InteractGesture      -- ^ Gesture input
  | InteractRelease      -- ^ Release/disengage
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Handle touch interaction
onTouch :: EnvironmentSurface -> BiometricSnapshot -> Int -> InteractionResult
onTouch surface snapshot timestamp =
  let aligned = checkAlignment surface snapshot
      reaction = surfaceReaction surface snapshot
      newState = (esState surface)
        { ssCoherenceLevel = min 1.0 (ssCoherenceLevel (esState surface) + srBloomDelta reaction * 0.5)
        , ssLastUpdate = timestamp
        }
      params = resolveSurfaceStyle newState
      boostedParams = if aligned
                      then params { spBloom = spBloom params * phi, spEmissive = min 1.0 (spEmissive params + 0.2) }
                      else params { spTurbulence = min 1.0 (spTurbulence params + 0.3) }
  in InteractionResult
    { irSuccess = aligned
    , irNewState = newState
    , irVisualFeedback = boostedParams
    , irHapticIntensity = if aligned then 0.3 else 0.7
    }

-- | Handle hover interaction
onHover :: EnvironmentSurface -> BiometricSnapshot -> Int -> InteractionResult
onHover surface snapshot timestamp =
  let aligned = checkAlignment surface snapshot
      newState = (esState surface)
        { ssCoherenceLevel = ssCoherenceLevel (esState surface) * 0.98 + bsCoherence snapshot * 0.02
        , ssLastUpdate = timestamp
        }
      params = resolveSurfaceStyle newState
      adjustedParams = params
        { spInteractZone = spInteractZone params * (if aligned then phi else phiInverse)
        }
  in InteractionResult
    { irSuccess = True
    , irNewState = newState
    , irVisualFeedback = adjustedParams
    , irHapticIntensity = 0.1
    }

-- | Handle release interaction
onRelease :: EnvironmentSurface -> Int -> InteractionResult
onRelease surface timestamp =
  let decayedState = (esState surface)
        { ssCoherenceLevel = ssCoherenceLevel (esState surface) * phiInverse
        , ssLastUpdate = timestamp
        }
      params = resolveSurfaceStyle decayedState
  in InteractionResult
    { irSuccess = True
    , irNewState = decayedState
    , irVisualFeedback = params
    , irHapticIntensity = 0
    }

-- =============================================================================
-- State Management
-- =============================================================================

-- | Surface state manager for multiple surfaces
data SurfaceStateManager = SurfaceStateManager
  { ssmSurfaces        :: ![EnvironmentSurface]  -- ^ Managed surfaces
  , ssmConfig          :: !SurfaceConfig  -- ^ Global configuration
  , ssmCurrentTime     :: !Int           -- ^ Current timestamp
  , ssmActiveCount     :: !Int           -- ^ Active surface count
  } deriving (Eq, Show)

-- | Initialize state manager
initManager :: SurfaceConfig -> SurfaceStateManager
initManager config = SurfaceStateManager
  { ssmSurfaces = []
  , ssmConfig = config
  , ssmCurrentTime = 0
  , ssmActiveCount = 0
  }

-- | Update surface state with new biometric data
updateSurfaceState :: SurfaceStateManager -> String -> BiometricSnapshot -> Int -> SurfaceStateManager
updateSurfaceState manager surfaceId snapshot timestamp =
  let updatedSurfaces = map (updateIfMatch surfaceId snapshot timestamp) (ssmSurfaces manager)
  in manager
    { ssmSurfaces = updatedSurfaces
    , ssmCurrentTime = timestamp
    }

-- | Reset surface to neutral state
resetToNeutral :: SurfaceState -> SurfaceState
resetToNeutral state = state
  { ssCoherenceLevel = 0.5
  , ssTorsionSpin = SpinNeutral
  , ssPermissionGate = ConsentPassive
  , ssStability = 0.5
  }

-- | Check and apply inactivity timeout
inactivityTimeout :: SurfaceStateManager -> EnvironmentSurface -> Bool
inactivityTimeout manager surface =
  let elapsed = ssmCurrentTime manager - esLastTouch surface
  in elapsed > scInactivityTimeout (ssmConfig manager)

-- =============================================================================
-- Helper Types and Functions
-- =============================================================================

-- | Biometric snapshot for interaction
data BiometricSnapshot = BiometricSnapshot
  { bsCoherence        :: !Double        -- ^ Current coherence [0, 1]
  , bsFrequency        :: !Double        -- ^ Bio-frequency (Hz)
  , bsHRV              :: !Double        -- ^ Heart rate variability
  , bsIntensity        :: !Double        -- ^ Signal intensity [0, 1]
  } deriving (Eq, Show)

-- | Surface reaction data
data SurfaceReaction = SurfaceReaction
  { srAligned          :: !Bool          -- ^ Is aligned
  , srBloomDelta       :: !Double        -- ^ Bloom change
  , srTurbulenceDelta  :: !Double        -- ^ Turbulence change
  , srEmissiveDelta    :: !Double        -- ^ Emissive change
  } deriving (Eq, Show)

-- | Default surface state
defaultSurfaceState :: SurfaceState
defaultSurfaceState = SurfaceState
  { ssCoherenceLevel = 0.5
  , ssTorsionSpin = SpinNeutral
  , ssResonanceFreq = 528.0
  , ssPermissionGate = ConsentPassive
  , ssStability = 0.5
  , ssLastUpdate = 0
  }

-- | Calculate emissive glow from state
calculateEmissive :: SurfaceState -> Double
calculateEmissive state =
  let base = ssCoherenceLevel state * 0.5
      consentBoost = case ssPermissionGate state of
        ConsentAmplified -> 0.3
        ConsentEngaged -> 0.1
        _ -> 0
  in min 1.0 (base + consentBoost)

-- | Calculate bloom radius from state
calculateBloom :: SurfaceState -> Double
calculateBloom state =
  let base = ssCoherenceLevel state * phi
      spinBoost = case ssTorsionSpin state of
        SpinClockwise -> 0.2
        SpinOscillating -> 0.3
        _ -> 0
  in min 2.0 (base + spinBoost)

-- | Calculate turbulence from state
calculateTurbulence :: SurfaceState -> Double
calculateTurbulence state =
  if ssCoherenceLevel state < 0.2
  then (0.2 - ssCoherenceLevel state) * 5
  else 0

-- | Calculate interaction zone radius
calculateInteractZone :: ConsentLevel -> Double
calculateInteractZone consent = case consent of
  ConsentSuspended -> 0.5
  ConsentPassive -> 1.0
  ConsentEngaged -> 1.5
  ConsentAmplified -> 2.0 * phi

-- | Update surface if ID matches
updateIfMatch :: String -> BiometricSnapshot -> Int -> EnvironmentSurface -> EnvironmentSurface
updateIfMatch targetId snapshot timestamp surface =
  if esId surface == targetId
  then surface
    { esState = (esState surface)
        { ssCoherenceLevel = bsCoherence snapshot
        , ssLastUpdate = timestamp
        }
    , esLastTouch = timestamp
    }
  else surface
