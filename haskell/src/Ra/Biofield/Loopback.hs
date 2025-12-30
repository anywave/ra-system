{-|
Module      : Ra.Biofield.Loopback
Description : Closed-loop biofield feedback system with scalar emergence coupling
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Closed-loop feedback system where scalar emergence activity dynamically
modulates biometric parameters, producing physiological feedback in the
user via resonant coupling. Models the user's biofield as both emitter
and receiver, completing the scalar resonance loop.

== Biofield Loopback Theory

=== Feedback Architecture

* Biometric input (breath, HRV, GSR) feeds coherence calculation
* Coherence drives emergence state classification
* Emergence state modulates feedback channels
* Feedback channels influence biometric state (closed loop)

=== Resonant Reinforcement

* Coherence increase from emergence → improved biometrics → stronger emergence
* Coherence suppression (via inversion/overload) → stagnation → rest initiation
* Shadow content emergence → gradual field tension reduction

=== Reference Sources

* KAALI_BECK_BLOOD_ELECTRIFICATION.md - Bioelectric interaction
* ELECTROMAGNETIC_HEALING_FREQUENCIES.md - Frequency-to-biological mappings
* REICH_ORGONE_ACCUMULATOR.md - Biophysical energy storage
-}
module Ra.Biofield.Loopback
  ( -- * Core Types
    BiofieldState(..)
  , BiometricInput(..)
  , EmergenceGlow(..)
  , AvatarFieldFrame(..)

    -- * Feedback Channels
  , EMChannel(..)
  , ChannelConfig(..)
  , FeedbackOutput(..)

    -- * Emergence States
  , EmergenceResult(..)
  , classifyEmergence
  , classifyGlow

    -- * Coherence Computation
  , computeCoherence
  , computeScalarTension
  , computeAnkhDelta

    -- * Feedback Loop Core
  , LoopState(..)
  , initLoopState
  , updateFeedbackLoop
  , stepLoop

    -- * Modulation Paths
  , ModulationEffect(..)
  , modulateFromEmergence
  , applyHRVModulation
  , applyGSRModulation
  , applyTensionRelease

    -- * Resonant Reinforcement
  , ReinforcementMode(..)
  , computeReinforcement
  , checkStagnation
  , initiateRestPeriod

    -- * Channel Output
  , generateChannelOutput
  , pulseLight
  , modulateAudio
  , triggerHaptic
  , activateMicrocurrent

    -- * Simulation
  , simulateLoop
  , SimulationResult(..)
  , runSimulation
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete biofield state for loopback system
data BiofieldState = BiofieldState
  { bfsCoherencePhase  :: !Double        -- ^ Current coherence phase [0, 2pi]
  , bfsScalarTension   :: !Double        -- ^ Scalar field tension [0, 1]
  , bfsAnkhDelta       :: !Double        -- ^ Ankh deviation [-1, 1]
  , bfsFeedbackChannels :: ![EMChannel]  -- ^ Active feedback channels
  , bfsEmergenceLevel  :: !Double        -- ^ Current emergence level [0, 1]
  , bfsLoopGain        :: !Double        -- ^ Feedback loop gain [0, 2]
  , bfsRestMode        :: !Bool          -- ^ Rest period active
  } deriving (Eq, Show)

-- | Biometric input from sensors
data BiometricInput = BiometricInput
  { biBreathRate       :: !Double        -- ^ Breath rate (Hz), optimal ~6.5
  , biHRV              :: !Double        -- ^ Heart rate variability [0, 1]
  , biGSR              :: !Double        -- ^ Galvanic skin response [0, 1]
  , biFocus            :: !Double        -- ^ Mental focus level [0, 1]
  , biTimestamp        :: !Int           -- ^ Input timestamp
  } deriving (Eq, Show)

-- | Emergence glow levels (Clash-compatible enum)
data EmergenceGlow
  = GlowNone           -- ^ No emergence activity
  | GlowLow            -- ^ Minimal emergence
  | GlowModerate       -- ^ Moderate emergence
  | GlowHigh           -- ^ Strong emergence
  | GlowPeak           -- ^ Peak emergence state
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Avatar field frame for visual feedback
data AvatarFieldFrame = AvatarFieldFrame
  { affGlowState       :: !EmergenceGlow -- ^ Current glow level
  , affCoherence       :: !Double        -- ^ Coherence value [0, 1]
  , affFieldIntensity  :: !Double        -- ^ Field intensity [0, 1]
  , affPhaseAngle      :: !Double        -- ^ Phase angle [0, 2pi]
  , affPulseActive     :: !Bool          -- ^ Pulse event active
  } deriving (Eq, Show)

-- =============================================================================
-- Feedback Channels
-- =============================================================================

-- | Electromagnetic feedback channel types
data EMChannel
  = ChannelLight LightConfig       -- ^ Pulsed LED feedback
  | ChannelAudio AudioConfig       -- ^ Resonance tone modulation
  | ChannelHaptic HapticConfig     -- ^ Vibrotactile entrainment
  | ChannelTactile TactileConfig   -- ^ Microcurrent pads
  deriving (Eq, Show)

-- | Light channel configuration
data LightConfig = LightConfig
  { lcFrequency        :: !Double        -- ^ Pulse frequency (Hz)
  , lcIntensity        :: !Double        -- ^ Light intensity [0, 1]
  , lcColor            :: !(Double, Double, Double)  -- ^ RGB values
  , lcDutyCycle        :: !Double        -- ^ Duty cycle [0, 1]
  } deriving (Eq, Show)

-- | Audio channel configuration
data AudioConfig = AudioConfig
  { acBaseFrequency    :: !Double        -- ^ Base tone frequency (Hz)
  , acModulationDepth  :: !Double        -- ^ Modulation depth [0, 1]
  , acVolume           :: !Double        -- ^ Volume level [0, 1]
  , acBinauralBeat     :: !Double        -- ^ Binaural beat offset (Hz)
  } deriving (Eq, Show)

-- | Haptic channel configuration
data HapticConfig = HapticConfig
  { hcFrequency        :: !Double        -- ^ Vibration frequency (Hz)
  , hcIntensity        :: !Double        -- ^ Haptic intensity [0, 1]
  , hcPattern          :: !HapticPattern -- ^ Vibration pattern
  , hcDuration         :: !Int           -- ^ Pulse duration (ms)
  } deriving (Eq, Show)

-- | Haptic vibration patterns
data HapticPattern
  = PatternContinuous    -- ^ Steady vibration
  | PatternPulse         -- ^ Pulsed vibration
  | PatternRamp          -- ^ Ramping intensity
  | PatternHeartbeat     -- ^ Heartbeat-like pattern
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Tactile electrical configuration
data TactileConfig = TactileConfig
  { tcMicrocurrent     :: !Double        -- ^ Current level (microamps)
  , tcWaveform         :: !Waveform      -- ^ Signal waveform
  , tcFrequency        :: !Double        -- ^ Stimulation frequency (Hz)
  , tcElectrodes       :: !Int           -- ^ Active electrode count
  } deriving (Eq, Show)

-- | Electrical waveform types
data Waveform
  = WaveformSine         -- ^ Sinusoidal
  | WaveformSquare       -- ^ Square wave
  | WaveformTriangle     -- ^ Triangle wave
  | WaveformPulsed       -- ^ Pulsed DC
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Generic channel configuration wrapper
data ChannelConfig = ChannelConfig
  { ccEnabled          :: !Bool          -- ^ Channel enabled
  , ccGain             :: !Double        -- ^ Output gain [0, 2]
  , ccThreshold        :: !Double        -- ^ Activation threshold [0, 1]
  , ccFallback         :: !Bool          -- ^ Use fallback mode
  } deriving (Eq, Show)

-- | Feedback output for all channels
data FeedbackOutput = FeedbackOutput
  { foLightPulse       :: !(Maybe LightConfig)
  , foAudioTone        :: !(Maybe AudioConfig)
  , foHapticVibration  :: !(Maybe HapticConfig)
  , foTactileStim      :: !(Maybe TactileConfig)
  , foTimestamp        :: !Int
  } deriving (Eq, Show)

-- =============================================================================
-- Emergence States
-- =============================================================================

-- | Emergence result classification
data EmergenceResult
  = EmergenceFull        -- ^ Full coherent emergence
  | EmergencePartial     -- ^ Partial emergence
  | EmergenceShadow      -- ^ Shadow content emergence
  | EmergenceNone        -- ^ No emergence
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Classify emergence from coherence and tension
classifyEmergence :: Double -> Double -> EmergenceResult
classifyEmergence coherence tension
  | coherence >= phi * 0.5 && tension < 0.3 = EmergenceFull
  | coherence >= phiInverse && tension < 0.5 = EmergencePartial
  | tension > 0.6 = EmergenceShadow
  | otherwise = EmergenceNone

-- | Classify coherence level into glow state
classifyGlow :: Double -> EmergenceGlow
classifyGlow c
  | c < 0.2   = GlowNone
  | c < 0.4   = GlowLow
  | c < phiInverse = GlowModerate
  | c < phi * 0.5 = GlowHigh
  | otherwise = GlowPeak

-- =============================================================================
-- Coherence Computation
-- =============================================================================

-- | Compute coherence from biometric input
-- Optimal breath rate is ~6.5 Hz (resonant frequency)
computeCoherence :: BiometricInput -> Double
computeCoherence input =
  let breathProximity = 1.0 - abs (6.5 - biBreathRate input) / 6.5
      hrvFactor = biHRV input
      focusFactor = biFocus input
      -- Weighted combination
      rawCoherence = breathProximity * 0.4 + hrvFactor * 0.4 + focusFactor * 0.2
  in max 0 (min 1 (rawCoherence * phi / 2))

-- | Compute scalar tension from biometric state
computeScalarTension :: BiometricInput -> BiofieldState -> Double
computeScalarTension input state =
  let gsrContribution = biGSR input * 0.5
      ankhContribution = abs (bfsAnkhDelta state) * 0.3
      baselineTension = bfsScalarTension state * phiInverse
  in max 0 (min 1 (gsrContribution + ankhContribution + baselineTension))

-- | Compute ankh delta (deviation from optimal coherence)
computeAnkhDelta :: Double -> Double -> Double
computeAnkhDelta currentCoherence targetCoherence =
  let delta = currentCoherence - targetCoherence
  in max (-1) (min 1 delta)

-- =============================================================================
-- Feedback Loop Core
-- =============================================================================

-- | Complete loop state
data LoopState = LoopState
  { lsBiofield         :: !BiofieldState   -- ^ Current biofield state
  , lsFrame            :: !AvatarFieldFrame -- ^ Current avatar frame
  , lsOutput           :: !FeedbackOutput   -- ^ Current output
  , lsIteration        :: !Int              -- ^ Loop iteration count
  , lsStagnationCount  :: !Int              -- ^ Stagnation counter
  , lsConverged        :: !Bool             -- ^ Loop converged
  } deriving (Eq, Show)

-- | Initialize loop state
initLoopState :: LoopState
initLoopState = LoopState
  { lsBiofield = defaultBiofieldState
  , lsFrame = defaultAvatarFrame
  , lsOutput = emptyFeedbackOutput
  , lsIteration = 0
  , lsStagnationCount = 0
  , lsConverged = False
  }

-- | Core feedback loop update (Clash-compatible pure function)
updateFeedbackLoop :: BiometricInput -> AvatarFieldFrame -> (AvatarFieldFrame, Bool)
updateFeedbackLoop input prev =
  let cLevel = computeCoherence input
      gNew = classifyGlow cLevel
      phaseShift = if cLevel > affCoherence prev then 0.1 else (-0.05)
      newPhase = mod' (affPhaseAngle prev + phaseShift) (2 * pi)
      frameOut = prev
        { affGlowState = gNew
        , affCoherence = cLevel
        , affFieldIntensity = cLevel * phi / 2
        , affPhaseAngle = newPhase
        , affPulseActive = gNew /= affGlowState prev
        }
      pulseChanged = gNew /= affGlowState prev
  in (frameOut, pulseChanged)

-- | Single loop step with full state update
stepLoop :: BiometricInput -> LoopState -> LoopState
stepLoop input state =
  let -- Update avatar frame
      (newFrame, _pulseChanged) = updateFeedbackLoop input (lsFrame state)

      -- Compute new coherence and tension
      coherence = computeCoherence input
      tension = computeScalarTension input (lsBiofield state)
      ankhDelta = computeAnkhDelta coherence phiInverse

      -- Classify emergence
      emergence = classifyEmergence coherence tension

      -- Apply modulation effects
      modEffect = modulateFromEmergence emergence (lsBiofield state)

      -- Update biofield state
      newBiofield = (lsBiofield state)
        { bfsCoherencePhase = affPhaseAngle newFrame
        , bfsScalarTension = tension * meNewTension modEffect
        , bfsAnkhDelta = ankhDelta
        , bfsEmergenceLevel = coherence
        , bfsLoopGain = meGainAdjust modEffect
        , bfsRestMode = checkStagnation state
        }

      -- Generate channel output
      newOutput = generateChannelOutput newBiofield newFrame (biTimestamp input)

      -- Check for stagnation
      stagnating = abs (coherence - affCoherence (lsFrame state)) < 0.01
      newStagnation = if stagnating
                      then lsStagnationCount state + 1
                      else 0

  in state
    { lsBiofield = newBiofield
    , lsFrame = newFrame
    , lsOutput = newOutput
    , lsIteration = lsIteration state + 1
    , lsStagnationCount = newStagnation
    , lsConverged = coherence > phi * 0.5 && tension < 0.2
    }

-- =============================================================================
-- Modulation Paths
-- =============================================================================

-- | Modulation effect from emergence state
data ModulationEffect = ModulationEffect
  { meHRVDelta         :: !Double        -- ^ HRV change target
  , meGSRDelta         :: !Double        -- ^ GSR change target
  , meTensionDelta     :: !Double        -- ^ Tension change
  , meGainAdjust       :: !Double        -- ^ Loop gain adjustment
  , meNewTension       :: !Double        -- ^ New tension multiplier
  } deriving (Eq, Show)

-- | Generate modulation effect from emergence result
modulateFromEmergence :: EmergenceResult -> BiofieldState -> ModulationEffect
modulateFromEmergence emergence state = case emergence of
  EmergenceFull -> ModulationEffect
    { meHRVDelta = 0.1 * phi          -- Raise HRV (entrainment)
    , meGSRDelta = (-0.05)            -- Lower GSR (calm)
    , meTensionDelta = (-0.1)         -- Release tension
    , meGainAdjust = min 2.0 (bfsLoopGain state * 1.1)
    , meNewTension = phiInverse
    }
  EmergencePartial -> ModulationEffect
    { meHRVDelta = 0.05
    , meGSRDelta = 0
    , meTensionDelta = 0
    , meGainAdjust = bfsLoopGain state
    , meNewTension = 1.0
    }
  EmergenceShadow -> ModulationEffect
    { meHRVDelta = 0
    , meGSRDelta = 0.1               -- GSR increase (somatic unease)
    , meTensionDelta = (-0.05)       -- Gradual tension release
    , meGainAdjust = max 0.5 (bfsLoopGain state * 0.9)
    , meNewTension = 0.9
    }
  EmergenceNone -> ModulationEffect
    { meHRVDelta = 0
    , meGSRDelta = 0
    , meTensionDelta = 0
    , meGainAdjust = bfsLoopGain state
    , meNewTension = 1.0
    }

-- | Apply HRV modulation based on emergence
applyHRVModulation :: EmergenceResult -> Double -> Double
applyHRVModulation emergence currentHRV = case emergence of
  EmergenceFull -> min 1.0 (currentHRV + 0.1 * phi)
  EmergencePartial -> min 1.0 (currentHRV + 0.05)
  _ -> currentHRV

-- | Apply GSR modulation for unresolved ankh delta
applyGSRModulation :: Double -> Double -> Double
applyGSRModulation ankhDelta currentGSR =
  let change = abs ankhDelta * 0.1
  in if ankhDelta > 0.3
     then min 1.0 (currentGSR + change)
     else max 0 (currentGSR - change * phiInverse)

-- | Apply tension release for shadow emergence
applyTensionRelease :: EmergenceResult -> Double -> Double
applyTensionRelease emergence currentTension = case emergence of
  EmergenceShadow -> max 0 (currentTension - 0.05)
  EmergenceFull -> max 0 (currentTension - 0.1)
  _ -> currentTension

-- =============================================================================
-- Resonant Reinforcement
-- =============================================================================

-- | Reinforcement mode
data ReinforcementMode
  = ReinforcementPositive  -- ^ Coherence building
  | ReinforcementNegative  -- ^ Coherence suppression
  | ReinforcementNeutral   -- ^ Stable state
  | ReinforcementRest      -- ^ Rest period
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Compute reinforcement mode from loop state
computeReinforcement :: LoopState -> ReinforcementMode
computeReinforcement state
  | lsConverged state = ReinforcementPositive
  | bfsRestMode (lsBiofield state) = ReinforcementRest
  | lsStagnationCount state > 10 = ReinforcementNegative
  | otherwise = ReinforcementNeutral

-- | Check if loop is stagnating
checkStagnation :: LoopState -> Bool
checkStagnation state =
  lsStagnationCount state > 15 || bfsScalarTension (lsBiofield state) > 0.8

-- | Initiate rest period (returns modified biofield)
initiateRestPeriod :: BiofieldState -> BiofieldState
initiateRestPeriod state = state
  { bfsRestMode = True
  , bfsLoopGain = 0.5
  , bfsScalarTension = bfsScalarTension state * phiInverse
  }

-- =============================================================================
-- Channel Output
-- =============================================================================

-- | Generate feedback output for all channels
generateChannelOutput :: BiofieldState -> AvatarFieldFrame -> Int -> FeedbackOutput
generateChannelOutput biofield frame timestamp =
  let coherence = affCoherence frame
      glow = affGlowState frame
      tension = bfsScalarTension biofield
  in FeedbackOutput
    { foLightPulse = if coherence > 0.3
                     then Just (pulseLight glow coherence)
                     else Nothing
    , foAudioTone = if coherence > 0.2
                    then Just (modulateAudio coherence tension)
                    else Nothing
    , foHapticVibration = if affPulseActive frame
                          then Just (triggerHaptic glow)
                          else Nothing
    , foTactileStim = if tension > 0.5 && coherence > phiInverse
                      then Just (activateMicrocurrent tension)
                      else Nothing
    , foTimestamp = timestamp
    }

-- | Generate pulsed LED configuration
pulseLight :: EmergenceGlow -> Double -> LightConfig
pulseLight glow coherence = LightConfig
  { lcFrequency = case glow of
      GlowNone -> 1.0
      GlowLow -> 4.0
      GlowModerate -> 7.83  -- Schumann resonance
      GlowHigh -> 10.0      -- Alpha
      GlowPeak -> 40.0      -- Gamma
  , lcIntensity = coherence * phi / 2
  , lcColor = glowToColor glow
  , lcDutyCycle = 0.5
  }

-- | Generate audio tone configuration
modulateAudio :: Double -> Double -> AudioConfig
modulateAudio coherence tension = AudioConfig
  { acBaseFrequency = 432.0 + coherence * 96  -- 432-528 Hz range
  , acModulationDepth = tension * 0.3
  , acVolume = coherence * 0.7
  , acBinauralBeat = if coherence > phiInverse then 7.83 else 4.0
  }

-- | Generate haptic configuration for pulse event
triggerHaptic :: EmergenceGlow -> HapticConfig
triggerHaptic glow = HapticConfig
  { hcFrequency = case glow of
      GlowNone -> 50
      GlowLow -> 100
      GlowModerate -> 150
      GlowHigh -> 200
      GlowPeak -> 250
  , hcIntensity = glowIntensity glow
  , hcPattern = if glow >= GlowHigh then PatternHeartbeat else PatternPulse
  , hcDuration = 100
  }

-- | Generate microcurrent configuration
activateMicrocurrent :: Double -> TactileConfig
activateMicrocurrent tension = TactileConfig
  { tcMicrocurrent = 50 + tension * 150  -- 50-200 microamps
  , tcWaveform = WaveformSine
  , tcFrequency = 7.83  -- Schumann resonance
  , tcElectrodes = 2
  }

-- =============================================================================
-- Simulation
-- =============================================================================

-- | Simulation result
data SimulationResult = SimulationResult
  { srFrames           :: ![AvatarFieldFrame]  -- ^ All frames
  , srFinalState       :: !LoopState           -- ^ Final loop state
  , srConverged        :: !Bool                -- ^ Did converge
  , srIterations       :: !Int                 -- ^ Total iterations
  , srPeakCoherence    :: !Double              -- ^ Maximum coherence reached
  } deriving (Eq, Show)

-- | Simulate feedback loop with input sequence
simulateLoop :: [BiometricInput] -> [AvatarFieldFrame]
simulateLoop inputs = map lsFrame $ scanl (flip stepLoop) initLoopState inputs

-- | Run full simulation with result summary
runSimulation :: [BiometricInput] -> SimulationResult
runSimulation inputs =
  let states = scanl (flip stepLoop) initLoopState inputs
      frames = map lsFrame states
      finalState = if null states then initLoopState else last states
      peakCoh = maximum (map affCoherence frames)
  in SimulationResult
    { srFrames = frames
    , srFinalState = finalState
    , srConverged = lsConverged finalState
    , srIterations = length inputs
    , srPeakCoherence = peakCoh
    }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Default biofield state
defaultBiofieldState :: BiofieldState
defaultBiofieldState = BiofieldState
  { bfsCoherencePhase = 0
  , bfsScalarTension = 0.3
  , bfsAnkhDelta = 0
  , bfsFeedbackChannels = []
  , bfsEmergenceLevel = 0
  , bfsLoopGain = 1.0
  , bfsRestMode = False
  }

-- | Default avatar frame
defaultAvatarFrame :: AvatarFieldFrame
defaultAvatarFrame = AvatarFieldFrame
  { affGlowState = GlowNone
  , affCoherence = 0
  , affFieldIntensity = 0
  , affPhaseAngle = 0
  , affPulseActive = False
  }

-- | Empty feedback output
emptyFeedbackOutput :: FeedbackOutput
emptyFeedbackOutput = FeedbackOutput
  { foLightPulse = Nothing
  , foAudioTone = Nothing
  , foHapticVibration = Nothing
  , foTactileStim = Nothing
  , foTimestamp = 0
  }

-- | Convert glow level to RGB color
glowToColor :: EmergenceGlow -> (Double, Double, Double)
glowToColor glow = case glow of
  GlowNone -> (0.2, 0.2, 0.3)      -- Dim blue-gray
  GlowLow -> (0.3, 0.4, 0.6)       -- Soft blue
  GlowModerate -> (0.4, 0.6, 0.8)  -- Medium blue
  GlowHigh -> (0.6, 0.8, 1.0)      -- Bright blue-white
  GlowPeak -> (1.0, 1.0, 1.0)      -- Pure white

-- | Get intensity from glow level
glowIntensity :: EmergenceGlow -> Double
glowIntensity glow = case glow of
  GlowNone -> 0.1
  GlowLow -> 0.3
  GlowModerate -> 0.5
  GlowHigh -> 0.7
  GlowPeak -> 1.0

-- | Floating point modulo
mod' :: Double -> Double -> Double
mod' x y = x - fromIntegral (floor (x / y) :: Int) * y
