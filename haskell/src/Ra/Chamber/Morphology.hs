{-|
Module      : Ra.Chamber.Morphology
Description : Scalar field-driven chamber modulation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Dynamically adapts the chamber's structure, geometry, and response states
based on live scalar field resonance gradients and emergence harmonics.
Chambers evolve with user coherence and field alignment.

== Morphogenic Principles

Chambers are not static constructs—they are morphogenic fields influenced by:

* Scalar potential intensity (from Ra.Scalar)
* Coherence flow (from Ra.Pipeline)
* Emergence alignment (from Ra.Identity)
* Biometric resonance (from Ra.Tuning)

== Chamber Forms

* Egg: Primordial/receptive state
* Dodecahedron: Platonic stability
* Toroid: Flow-oriented buffer
* HelixSpindle: DNA-resonant
* CaduceusAligned: Healing polarity
* HarmonicShell: N-layer resonance
-}
module Ra.Chamber.Morphology
  ( -- * Core Types
    ChamberForm(..)
  , ChamberState(..)
  , MorphEvent(..)

    -- * Morphology Resolution
  , UserResonanceProfile(..)
  , resolveChamberForm
  , formFromField

    -- * Morphing Protocol
  , TransitionType(..)
  , morphTransition
  , transitionDuration

    -- * State Management
  , updateMorphology
  , FieldContext(..)
  , applyMorphEvent

    -- * Form Properties
  , formSymmetry
  , formResonance
  , formCapacity

    -- * Biometric Coupling
  , breatheForm
  , heartPhaseCoupling
  , neuralAlignment

    -- * Visualization Events
  , VisualEffect(..)
  , morphToVisual
  , formGlow
  , formPulse

    -- * Stability Analysis
  , StabilityMetric(..)
  , analyzeStability
  , autoStabilize
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Chamber geometric form
data ChamberForm
  = Egg                     -- ^ Primordial/receptive
  | Dodecahedron            -- ^ Platonic stability (12 faces)
  | Toroid                  -- ^ Flow-oriented buffer
  | HelixSpindle            -- ^ DNA-resonant double helix
  | CaduceusAligned         -- ^ Healing polarity (intertwined)
  | HarmonicShell !Int      -- ^ N-layer harmonic resonance
  deriving (Eq, Show)

-- | Chamber morphology state
data ChamberState = ChamberState
  { csForm          :: !ChamberForm     -- ^ Current geometric form
  , csScale         :: !Double          -- ^ Size scale factor
  , csRotation      :: !Double          -- ^ Rotation phase [0, 2*pi]
  , csCoherence     :: !Double          -- ^ Internal coherence [0, 1]
  , csPotential     :: !Double          -- ^ Stored potential
  , csPhase         :: !MorphPhase      -- ^ Current morph phase
  , csGlow          :: !Double          -- ^ Visual glow intensity [0, 1]
  , csBreathPhase   :: !Double          -- ^ Biometric breath coupling
  } deriving (Eq, Show)

-- | Morph phase
data MorphPhase
  = Stable          -- ^ Form is stable
  | Transitioning   -- ^ Between forms
  | Pulsing         -- ^ Rhythmic pulsation
  | Expanding       -- ^ Growing
  | Contracting     -- ^ Shrinking
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Morphing transition event
data MorphEvent
  = GradualPhaseShift !ChamberForm !Double   -- ^ Smooth transition to form over duration
  | RapidCollapse                            -- ^ Quick collapse to minimal state
  | ResonantExpansion !Double                -- ^ Expand by factor
  | PulseEvent !Double                       -- ^ Single pulse with intensity
  | FormLock                                 -- ^ Lock current form
  | FormRelease                              -- ^ Release form lock
  deriving (Eq, Show)

-- =============================================================================
-- Morphology Resolution
-- =============================================================================

-- | User resonance profile for form resolution
data UserResonanceProfile = UserResonanceProfile
  { urpCoherence    :: !Double          -- ^ User coherence level [0, 1]
  , urpHeartPhase   :: !Double          -- ^ Heart rhythm phase
  , urpBreathPhase  :: !Double          -- ^ Breath rhythm phase
  , urpNeuralState  :: !String          -- ^ Neural state (alpha, theta, etc.)
  , urpIntention    :: !String          -- ^ User intention category
  } deriving (Eq, Show)

-- | Resolve chamber form from field and user profile
resolveChamberForm :: ScalarField -> UserResonanceProfile -> ChamberForm
resolveChamberForm field profile =
  let -- Field metrics
      intensity = sfIntensity field
      flux = sfFluxCoherence field
      harmonicLevel = sfHarmonicDepth field

      -- User metrics
      coherence = urpCoherence profile
      neural = urpNeuralState profile

      -- Combined score
      combinedCoherence = (intensity + coherence) / 2.0

  in -- Form resolution based on thresholds
     if flux > 0.3 then
       Toroid  -- Fallback to toroidal buffering for instability
     else if combinedCoherence >= 0.85 && harmonicLevel >= 4 then
       HarmonicShell harmonicLevel
     else if combinedCoherence >= 0.75 && neural == "theta" then
       CaduceusAligned
     else if combinedCoherence >= 0.65 then
       HelixSpindle
     else if combinedCoherence >= 0.5 then
       Dodecahedron
     else
       Egg

-- | Determine form from field alone
formFromField :: ScalarField -> ChamberForm
formFromField field =
  let intensity = sfIntensity field
  in if intensity >= 0.8 then HarmonicShell 5
     else if intensity >= 0.6 then Dodecahedron
     else if intensity >= 0.4 then Toroid
     else Egg

-- Scalar field representation (simplified for morphology)
data ScalarField = ScalarField
  { sfIntensity     :: !Double        -- ^ Field intensity [0, 1]
  , sfFluxCoherence :: !Double        -- ^ Flux coherence [0, 1]
  , sfHarmonicDepth :: !Int           -- ^ Harmonic depth (1-5)
  , sfPhase         :: !Double        -- ^ Field phase
  } deriving (Eq, Show)

-- =============================================================================
-- Morphing Protocol
-- =============================================================================

-- | Transition type
data TransitionType
  = Instant         -- ^ Immediate change
  | Smooth          -- ^ Gradual interpolation
  | Phased          -- ^ Step-wise transition
  | Resonant        -- ^ Wave-like morphing
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Calculate morph transition between forms
morphTransition :: ChamberForm -> ChamberForm -> TransitionType
morphTransition from to = case (from, to) of
  -- Same form = instant
  _ | from == to -> Instant

  -- Egg transitions are always smooth
  (Egg, _) -> Smooth
  (_, Egg) -> Smooth

  -- Toroid is emergency buffer = instant
  (_, Toroid) -> Instant
  (Toroid, _) -> Phased

  -- Harmonic shells resonate
  (HarmonicShell _, HarmonicShell _) -> Resonant

  -- Caduceus healing requires smooth
  (_, CaduceusAligned) -> Smooth
  (CaduceusAligned, _) -> Smooth

  -- Default
  _ -> Phased

-- | Get transition duration in seconds
transitionDuration :: TransitionType -> Double
transitionDuration tt = case tt of
  Instant -> 0.0
  Smooth -> 0.3 * phi    -- ~0.485s
  Phased -> 0.2
  Resonant -> phi        -- Golden transition time

-- =============================================================================
-- State Management
-- =============================================================================

-- | Field context for morphology updates
data FieldContext = FieldContext
  { fcField         :: !ScalarField
  , fcUserProfile   :: !UserResonanceProfile
  , fcDeltaTime     :: !Double        -- ^ Time since last update
  , fcEventQueue    :: ![MorphEvent]  -- ^ Pending events
  } deriving (Eq, Show)

-- | Update chamber morphology based on context
updateMorphology :: FieldContext -> ChamberState -> (ChamberState, MorphEvent)
updateMorphology ctx state =
  let -- Determine target form
      targetForm = resolveChamberForm (fcField ctx) (fcUserProfile ctx)

      -- Check if form change needed
      currentForm = csForm state
      needsChange = currentForm /= targetForm

      -- Determine event
      event = if needsChange
              then GradualPhaseShift targetForm (transitionDuration (morphTransition currentForm targetForm))
              else if csCoherence state < urpCoherence (fcUserProfile ctx) * 0.9
              then ResonantExpansion 1.05
              else PulseEvent (csCoherence state)

      -- Apply biometric breathing
      breathedState = breatheForm (fcDeltaTime ctx) (urpBreathPhase (fcUserProfile ctx)) state

      -- Update glow based on coherence
      glowLevel = urpCoherence (fcUserProfile ctx) * phi * 0.8

      newState = breathedState
        { csGlow = min 1.0 glowLevel
        , csPhase = if needsChange then Transitioning else Stable
        }
  in (newState, event)

-- | Apply morph event to chamber state
applyMorphEvent :: MorphEvent -> ChamberState -> ChamberState
applyMorphEvent event state = case event of
  GradualPhaseShift form _ ->
    state { csForm = form, csPhase = Transitioning }

  RapidCollapse ->
    state { csForm = Egg, csScale = 0.5, csPhase = Contracting }

  ResonantExpansion factor ->
    state { csScale = csScale state * factor, csPhase = Expanding }

  PulseEvent intensity ->
    state { csGlow = intensity, csPhase = Pulsing }

  FormLock ->
    state { csPhase = Stable }

  FormRelease ->
    state { csPhase = Stable }

-- =============================================================================
-- Form Properties
-- =============================================================================

-- | Get form symmetry order
formSymmetry :: ChamberForm -> Int
formSymmetry form = case form of
  Egg -> 1              -- Continuous
  Dodecahedron -> 12    -- 12-fold
  Toroid -> 1           -- Continuous
  HelixSpindle -> 2     -- Binary
  CaduceusAligned -> 2  -- Binary
  HarmonicShell n -> n  -- N-fold

-- | Get form resonance factor
formResonance :: ChamberForm -> Double
formResonance form = case form of
  Egg -> 0.5
  Dodecahedron -> phi
  Toroid -> 1.0
  HelixSpindle -> phi * phi
  CaduceusAligned -> phi ** 3
  HarmonicShell n -> phi ** fromIntegral n

-- | Get form capacity (how much it can hold)
formCapacity :: ChamberForm -> Double
formCapacity form = case form of
  Egg -> 1.0
  Dodecahedron -> phi * 2
  Toroid -> phi * 3
  HelixSpindle -> phi * 4
  CaduceusAligned -> phi * 5
  HarmonicShell n -> phi ** fromIntegral n * 2

-- =============================================================================
-- Biometric Coupling
-- =============================================================================

-- | Apply breath modulation to form
breatheForm :: Double -> Double -> ChamberState -> ChamberState
breatheForm dt breathPhase state =
  let -- Breath modulates scale
      breathMod = 1.0 + 0.05 * sin breathPhase
      newScale = csScale state * breathMod

      -- Update breath phase tracking
      newBreathPhase = csBreathPhase state + dt * 2 * pi / 4.0  -- 4 second breath cycle
  in state
      { csScale = newScale
      , csBreathPhase = newBreathPhase
      }

-- | Couple heart phase to form glow
heartPhaseCoupling :: Double -> ChamberState -> ChamberState
heartPhaseCoupling heartPhase state =
  let -- Heart rhythm modulates glow
      heartMod = 0.7 + 0.3 * sin heartPhase
      newGlow = csGlow state * heartMod
  in state { csGlow = min 1.0 newGlow }

-- | Align form with neural state
neuralAlignment :: String -> ChamberState -> ChamberState
neuralAlignment neuralState state =
  let -- Neural states affect rotation speed
      rotMod = case neuralState of
        "delta" -> 0.1    -- Very slow
        "theta" -> 0.3    -- Slow
        "alpha" -> 0.5    -- Medium
        "beta" -> 0.8     -- Fast
        "gamma" -> 1.0    -- Very fast
        _ -> 0.5

      newRotation = csRotation state + rotMod * 0.1
  in state { csRotation = newRotation }

-- =============================================================================
-- Visualization Events
-- =============================================================================

-- | Visual effect for rendering
data VisualEffect = VisualEffect
  { veType        :: !EffectType
  , veIntensity   :: !Double        -- ^ Effect intensity [0, 1]
  , veDuration    :: !Double        -- ^ Duration in seconds
  , veColor       :: !(Int, Int, Int)  -- ^ RGB color
  } deriving (Eq, Show)

-- | Effect type
data EffectType
  = GlowEffect
  | PulseEffect
  | RippleEffect
  | TransformEffect
  | ShimmerEffect
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Convert morph event to visual effect
morphToVisual :: MorphEvent -> VisualEffect
morphToVisual event = case event of
  GradualPhaseShift _ dur ->
    VisualEffect TransformEffect 0.8 dur (100, 200, 255)

  RapidCollapse ->
    VisualEffect RippleEffect 1.0 0.2 (255, 100, 50)

  ResonantExpansion _ ->
    VisualEffect GlowEffect 0.9 0.5 (255, 215, 100)

  PulseEvent intensity ->
    VisualEffect PulseEffect intensity 0.3 (150, 255, 200)

  FormLock ->
    VisualEffect ShimmerEffect 0.5 0.1 (200, 200, 255)

  FormRelease ->
    VisualEffect ShimmerEffect 0.3 0.1 (255, 255, 200)

-- | Get glow parameters for form
formGlow :: ChamberForm -> (Double, (Int, Int, Int))
formGlow form = case form of
  Egg -> (0.3, (255, 220, 180))           -- Warm glow
  Dodecahedron -> (0.6, (100, 200, 255))  -- Blue crystal
  Toroid -> (0.5, (200, 255, 200))        -- Green flow
  HelixSpindle -> (0.7, (255, 180, 255))  -- Purple DNA
  CaduceusAligned -> (0.8, (255, 215, 0)) -- Gold healing
  HarmonicShell n -> (0.5 + fromIntegral n * 0.1, (255, 255, 255))  -- White shells

-- | Get pulse parameters for form
formPulse :: ChamberForm -> Double -> (Double, Double)
formPulse form coherence =
  let baseRate = case form of
        Egg -> 0.5
        Dodecahedron -> 1.0
        Toroid -> 1.5
        HelixSpindle -> 2.0
        CaduceusAligned -> phi
        HarmonicShell _ -> phiInverse

      rate = baseRate * (1.0 + coherence * 0.5)
      amplitude = 0.1 + coherence * 0.2
  in (rate, amplitude)

-- =============================================================================
-- Stability Analysis
-- =============================================================================

-- | Stability metrics
data StabilityMetric = StabilityMetric
  { smOverall      :: !Double        -- ^ Overall stability [0, 1]
  , smFormFit      :: !Double        -- ^ How well form fits context
  , smCoherence    :: !Double        -- ^ Coherence stability
  , smFlux         :: !Double        -- ^ Flux level (inverse stability)
  , smRecommendation :: !String      -- ^ Recommended action
  } deriving (Eq, Show)

-- | Analyze chamber stability
analyzeStability :: ChamberState -> FieldContext -> StabilityMetric
analyzeStability state ctx =
  let -- Form fit based on resolved vs current
      resolvedForm = resolveChamberForm (fcField ctx) (fcUserProfile ctx)
      currentForm = csForm state
      formFit = if resolvedForm == currentForm then 1.0 else 0.5

      -- Coherence from user profile
      coherence = urpCoherence (fcUserProfile ctx)

      -- Flux from field
      flux = sfFluxCoherence (fcField ctx)

      -- Overall = weighted average
      overall = (formFit * 0.3 + coherence * 0.4 + (1.0 - flux) * 0.3)

      -- Recommendation
      recommendation
        | overall >= 0.8 = "Maintain current form"
        | flux > 0.5 = "Switch to Toroid buffering"
        | formFit < 0.5 = "Transition to " ++ show resolvedForm
        | coherence < 0.4 = "Stabilize user coherence first"
        | otherwise = "Monitor and adjust"

  in StabilityMetric
      { smOverall = overall
      , smFormFit = formFit
      , smCoherence = coherence
      , smFlux = flux
      , smRecommendation = recommendation
      }

-- | Auto-stabilize chamber if needed
autoStabilize :: ChamberState -> FieldContext -> ChamberState
autoStabilize state ctx =
  let metrics = analyzeStability state ctx
  in if smOverall metrics < 0.4
     then -- Force toroid for stability
       state { csForm = Toroid, csPhase = Stable, csGlow = 0.3 }
     else if smFlux metrics > 0.5
     then -- Reduce scale to contain flux
       state { csScale = csScale state * 0.9, csPhase = Contracting }
     else
       state
