{-|
Module      : Ra.Biofield
Description : Biofield loopback feedback system
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Closed-loop feedback where scalar emergence dynamically modulates
biometric parameters, producing physiological feedback via resonant
coupling. Models the user's biofield as both emitter and receiver.

== Loopback Principle

The biofield loopback completes the scalar resonance loop:

1. Biometrics generate field state
2. Field state influences emergence
3. Emergence feeds back to body
4. Body responds with new biometrics

== Modulation Paths

* FULL emergence -> raises HRV (entrainment)
* Unresolved ankh_delta -> triggers GSR change (somatic unease)
* Shadow emergence -> gradually lowers field tension

== Effect Channels

* Light (pulsed LED feedback)
* Audio (resonance tone modulation)
* Haptic (vibrotactile entrainment)
* Electrical (microcurrent pads)
-}
module Ra.Biofield
  ( -- * Biofield State
    BiofieldState(..)
  , mkBiofieldState
  , updateBiofield
  , biofieldCoherence

    -- * Effect Channels
  , EMChannel(..)
  , ChannelOutput(..)
  , activeChannels
  , channelIntensity

    -- * Modulation Paths
  , ModulationPath(..)
  , EmergenceInfluence(..)
  , computeInfluence
  , applyInfluence

    -- * Resonant Reinforcement
  , ReinforcementLoop(..)
  , LoopState(..)
  , evaluateLoop
  , shouldReinforce
  , shouldSuppress

    -- * Feedback Generation
  , BiofieldFeedback(..)
  , generateFeedback
  , feedbackToChannels

    -- * Simulation
  , SimulatedUser(..)
  , simulateResponse
  , simulationStep
  ) where

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Biofield State
-- =============================================================================

-- | Current biofield state
data BiofieldState = BiofieldState
  { bsCoherencePhase :: !Double      -- ^ Phase coherence [0,1]
  , bsScalarTension  :: !Double      -- ^ Field tension [0,1]
  , bsAnkhDelta      :: !Double      -- ^ Δ(ankh) imbalance [-1,1]
  , bsFeedbackChannels :: ![EMChannel]
  , bsHistory        :: ![Double]    -- ^ Coherence history
  } deriving (Eq, Show)

-- | Create initial biofield state
mkBiofieldState :: Double -> BiofieldState
mkBiofieldState coherence = BiofieldState
  { bsCoherencePhase = clamp01 coherence
  , bsScalarTension = 0.5
  , bsAnkhDelta = 0.0
  , bsFeedbackChannels = [Light, Audio]  -- Default channels
  , bsHistory = [coherence]
  }

-- | Update biofield from emergence result
updateBiofield :: EmergenceInfluence -> BiofieldState -> BiofieldState
updateBiofield influence bs =
  let newCoh = clamp01 (bsCoherencePhase bs + eiCoherenceDelta influence)
      newTension = clamp01 (bsScalarTension bs + eiTensionDelta influence)
      newAnkh = clamp (bsAnkhDelta bs + eiAnkhDelta influence) (-1) 1
  in bs
      { bsCoherencePhase = newCoh
      , bsScalarTension = newTension
      , bsAnkhDelta = newAnkh
      , bsHistory = newCoh : take 100 (bsHistory bs)
      }

-- | Get current biofield coherence
biofieldCoherence :: BiofieldState -> Double
biofieldCoherence = bsCoherencePhase

-- =============================================================================
-- Effect Channels
-- =============================================================================

-- | Electromagnetic feedback channel
data EMChannel
  = Light      -- ^ Pulsed LED feedback
  | Audio      -- ^ Resonance tone modulation
  | Haptic     -- ^ Vibrotactile entrainment
  | Electrical -- ^ Microcurrent pads
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Output for a channel
data ChannelOutput = ChannelOutput
  { coChannel   :: !EMChannel
  , coIntensity :: !Double     -- ^ [0,1]
  , coFrequency :: !Double     -- ^ Hz
  , coPattern   :: !String     -- ^ Pattern description
  } deriving (Eq, Show)

-- | Get active channels from biofield state
activeChannels :: BiofieldState -> [EMChannel]
activeChannels = bsFeedbackChannels

-- | Calculate intensity for channel based on coherence
channelIntensity :: EMChannel -> BiofieldState -> Double
channelIntensity channel bs =
  let baseCoh = bsCoherencePhase bs
      tensionMod = 1.0 - bsScalarTension bs * 0.3
  in case channel of
      Light -> baseCoh * tensionMod
      Audio -> baseCoh * 0.8
      Haptic -> baseCoh * tensionMod * 0.6
      Electrical -> baseCoh * 0.4  -- Conservative for safety

-- =============================================================================
-- Modulation Paths
-- =============================================================================

-- | Modulation path type
data ModulationPath
  = HRVPath      -- ^ Heart rate variability modulation
  | GSRPath      -- ^ Galvanic skin response
  | BreathPath   -- ^ Respiration rate/depth
  | TensionPath  -- ^ Muscle tension release
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Influence from emergence on biofield
data EmergenceInfluence = EmergenceInfluence
  { eiEmergenceType   :: !String     -- ^ FULL, PARTIAL, SHADOW
  , eiCoherenceDelta  :: !Double     -- ^ Change in coherence
  , eiTensionDelta    :: !Double     -- ^ Change in tension
  , eiAnkhDelta       :: !Double     -- ^ Ankh balance shift
  , eiPaths           :: ![ModulationPath]
  } deriving (Eq, Show)

-- | Compute influence from emergence result
computeInfluence :: String -> Double -> EmergenceInfluence
computeInfluence emergenceType alpha =
  case emergenceType of
    "FULL" -> EmergenceInfluence
      { eiEmergenceType = "FULL"
      , eiCoherenceDelta = 0.05 * alpha
      , eiTensionDelta = -0.03 * alpha  -- Reduces tension
      , eiAnkhDelta = 0.02 * alpha
      , eiPaths = [HRVPath, BreathPath]
      }
    "PARTIAL" -> EmergenceInfluence
      { eiEmergenceType = "PARTIAL"
      , eiCoherenceDelta = 0.02 * alpha
      , eiTensionDelta = 0.0
      , eiAnkhDelta = 0.01 * alpha
      , eiPaths = [HRVPath]
      }
    "SHADOW" -> EmergenceInfluence
      { eiEmergenceType = "SHADOW"
      , eiCoherenceDelta = -0.01
      , eiTensionDelta = -0.05  -- Shadow work releases tension
      , eiAnkhDelta = -0.03
      , eiPaths = [GSRPath, TensionPath]
      }
    _ -> EmergenceInfluence
      { eiEmergenceType = "NONE"
      , eiCoherenceDelta = 0.0
      , eiTensionDelta = 0.01  -- Slight tension increase
      , eiAnkhDelta = 0.0
      , eiPaths = []
      }

-- | Apply influence to biometric values
applyInfluence :: EmergenceInfluence -> (Double, Double, Double) -> (Double, Double, Double)
applyInfluence ei (hrv, gsr, breath) =
  let hrvMod = if HRVPath `elem` eiPaths ei
               then hrv + eiCoherenceDelta ei * 0.5
               else hrv
      gsrMod = if GSRPath `elem` eiPaths ei
               then gsr + eiTensionDelta ei * 0.3
               else gsr
      breathMod = if BreathPath `elem` eiPaths ei
                  then breath + eiCoherenceDelta ei * 0.2
                  else breath
  in (clamp01 hrvMod, clamp01 gsrMod, clamp01 breathMod)

-- =============================================================================
-- Resonant Reinforcement
-- =============================================================================

-- | Reinforcement loop state
data ReinforcementLoop = ReinforcementLoop
  { rlState         :: !LoopState
  , rlStrength      :: !Double    -- ^ Loop strength [0,1]
  , rlDirection     :: !Bool      -- ^ True = positive, False = negative
  , rlCycleCount    :: !Int
  } deriving (Eq, Show)

-- | State of the feedback loop
data LoopState
  = Amplifying    -- ^ Coherence increasing
  | Stabilizing   -- ^ Coherence stable
  | Suppressing   -- ^ Coherence decreasing
  | Resting       -- ^ Loop inactive
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Evaluate loop state from biofield
evaluateLoop :: BiofieldState -> ReinforcementLoop
evaluateLoop bs =
  let history = bsHistory bs
      trend = coherenceTrend history
      strength = abs trend * 10  -- Scale trend to strength
      state
        | trend > 0.05 = Amplifying
        | trend > 0.01 = Stabilizing
        | trend < -0.05 = Suppressing
        | otherwise = Resting
  in ReinforcementLoop
      { rlState = state
      , rlStrength = min 1.0 strength
      , rlDirection = trend > 0
      , rlCycleCount = 0
      }

-- | Calculate coherence trend
coherenceTrend :: [Double] -> Double
coherenceTrend [] = 0.0
coherenceTrend [_] = 0.0
coherenceTrend history =
  let recent = take 5 history
      oldest = last recent
      newest = head recent
  in (newest - oldest) / fromIntegral (length recent)

-- | Check if should reinforce current state
shouldReinforce :: ReinforcementLoop -> Bool
shouldReinforce rl = rlState rl == Amplifying && rlDirection rl

-- | Check if should suppress (enter rest)
shouldSuppress :: ReinforcementLoop -> Bool
shouldSuppress rl = rlState rl == Suppressing || rlStrength rl < 0.1

-- =============================================================================
-- Feedback Generation
-- =============================================================================

-- | Complete biofield feedback
data BiofieldFeedback = BiofieldFeedback
  { bfChannelOutputs :: ![ChannelOutput]
  , bfInstruction    :: !String
  , bfLoopState      :: !LoopState
  , bfIntensity      :: !Double
  } deriving (Eq, Show)

-- | Generate feedback from biofield state
generateFeedback :: BiofieldState -> BiofieldFeedback
generateFeedback bs =
  let loop = evaluateLoop bs
      channels = activeChannels bs
      outputs = map (generateChannelOutput bs) channels

      instruction = case rlState loop of
        Amplifying -> "Coherence rising. Maintain current rhythm."
        Stabilizing -> "Field stable. Continue breathing pattern."
        Suppressing -> "Rest period initiated. Relax deeply."
        Resting -> "Awaiting engagement."

      intensity = rlStrength loop * bsCoherencePhase bs
  in BiofieldFeedback
      { bfChannelOutputs = outputs
      , bfInstruction = instruction
      , bfLoopState = rlState loop
      , bfIntensity = intensity
      }

-- | Generate output for specific channel
generateChannelOutput :: BiofieldState -> EMChannel -> ChannelOutput
generateChannelOutput bs channel =
  let intensity = channelIntensity channel bs
      baseFreq = 7.83  -- Schumann base
      freq = case channel of
        Light -> baseFreq * phi
        Audio -> 432 * (bsCoherencePhase bs + 0.5)
        Haptic -> baseFreq
        Electrical -> 0.5  -- Very low for safety

      pattern = case channel of
        Light -> "pulse_sync"
        Audio -> "binaural_theta"
        Haptic -> "heartbeat"
        Electrical -> "micro_stim"
  in ChannelOutput
      { coChannel = channel
      , coIntensity = intensity
      , coFrequency = freq
      , coPattern = pattern
      }

-- | Convert feedback to channel list
feedbackToChannels :: BiofieldFeedback -> [EMChannel]
feedbackToChannels = map coChannel . bfChannelOutputs

-- =============================================================================
-- Simulation
-- =============================================================================

-- | Simulated user for testing
data SimulatedUser = SimulatedUser
  { suHRV       :: !Double
  , suGSR       :: !Double
  , suBreath    :: !Double
  , suBiofield  :: !BiofieldState
  , suTime      :: !Double
  } deriving (Eq, Show)

-- | Simulate user response to feedback
simulateResponse :: BiofieldFeedback -> SimulatedUser -> SimulatedUser
simulateResponse feedback user =
  let intensity = bfIntensity feedback
      loopState = bfLoopState feedback

      -- Simulate biometric changes based on loop state
      (hrvDelta, gsrDelta, breathDelta) = case loopState of
        Amplifying -> (0.02 * intensity, -0.01 * intensity, 0.01 * intensity)
        Stabilizing -> (0.005, 0.0, 0.005)
        Suppressing -> (-0.01, 0.02, -0.01)
        Resting -> (0.0, 0.0, 0.0)

      newHRV = clamp01 (suHRV user + hrvDelta)
      newGSR = clamp01 (suGSR user + gsrDelta)
      newBreath = clamp01 (suBreath user + breathDelta)

      -- Update biofield
      newCoherence = (newHRV + newBreath) / 2
      newBiofield = (suBiofield user) { bsCoherencePhase = newCoherence }
  in user
      { suHRV = newHRV
      , suGSR = newGSR
      , suBreath = newBreath
      , suBiofield = newBiofield
      }

-- | Run one simulation step
simulationStep :: Double -> SimulatedUser -> SimulatedUser
simulationStep dt user =
  let biofield = suBiofield user
      feedback = generateFeedback biofield
      responded = simulateResponse feedback user
  in responded { suTime = suTime user + dt }

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)

-- | Clamp value to range
clamp :: Double -> Double -> Double -> Double
clamp x lo hi = max lo (min hi x)
