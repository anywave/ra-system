{-|
Module      : Ra.Propulsion.BiofeedbackHarness
Description : Bio-coupled propulsion control through coherence feedback
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements biofeedback harness for coupling biological coherence states
to propulsion vector control. The harness translates intention patterns
into directed scalar field manipulation.

== Biofeedback Theory

=== Intention-Vector Coupling

Biological coherence states map to propulsion vectors:

* Heart coherence → Primary thrust vector
* Neural coherence → Navigation precision
* Respiratory rhythm → Thrust modulation
* Skin conductance → Field sensitivity

=== Harness Architecture

1. Sensor Layer: Biometric input processing
2. Translation Layer: Signal to vector conversion
3. Amplification Layer: Coherence-boosted output
4. Feedback Layer: Closed-loop stability
-}
module Ra.Propulsion.BiofeedbackHarness
  ( -- * Core Types
    BiofeedbackHarness(..)
  , HarnessState(..)
  , BioChannel(..)
  , IntentionVector(..)

    -- * Harness Operations
  , initializeHarness
  , updateHarness
  , calibrateHarness

    -- * Channel Management
  , activateChannel
  , deactivateChannel
  , channelCoherence

    -- * Intention Processing
  , processIntention
  , intentionToVector
  , amplifyIntention

    -- * Feedback Loop
  , FeedbackLoop(..)
  , createFeedbackLoop
  , updateFeedback
  , loopStability

    -- * Safety Limits
  , SafetyLimits(..)
  , defaultLimits
  , checkLimits
  , applyLimits

    -- * Harness Metrics
  , HarnessMetrics(..)
  , computeMetrics
  , coherenceEfficiency
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Biofeedback harness configuration
data BiofeedbackHarness = BiofeedbackHarness
  { bhChannels    :: ![BioChannel]          -- ^ Active bio channels
  , bhState       :: !HarnessState          -- ^ Current harness state
  , bhGain        :: !Double                -- ^ Overall gain factor
  , bhCalibration :: !(Double, Double)      -- ^ (offset, scale) calibration
  , bhLimits      :: !SafetyLimits          -- ^ Safety limits
  } deriving (Eq, Show)

-- | Harness operational state
data HarnessState
  = HarnessIdle        -- ^ Not actively processing
  | HarnessActive      -- ^ Processing bio signals
  | HarnessCalibrating -- ^ In calibration mode
  | HarnessSafeMode    -- ^ Limited operation (safety triggered)
  | HarnessError       -- ^ Error state
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Biometric input channel
data BioChannel = BioChannel
  { bcType       :: !ChannelType    -- ^ Channel type
  , bcValue      :: !Double         -- ^ Current value [0, 1]
  , bcBaseline   :: !Double         -- ^ Calibrated baseline
  , bcWeight     :: !Double         -- ^ Channel weight in mix
  , bcActive     :: !Bool           -- ^ Channel active flag
  } deriving (Eq, Show)

-- | Channel type enumeration
data ChannelType
  = ChannelHeart       -- ^ Heart rate variability
  | ChannelNeural      -- ^ EEG coherence
  | ChannelRespiratory -- ^ Breathing pattern
  | ChannelGalvanic    -- ^ Skin conductance
  | ChannelMuscular    -- ^ EMG tension
  | ChannelThermal     -- ^ Skin temperature
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Intention vector from bio processing
data IntentionVector = IntentionVector
  { ivDirection  :: !(Double, Double, Double)  -- ^ Intended direction
  , ivIntensity  :: !Double                    -- ^ Intention strength [0, 1]
  , ivClarity    :: !Double                    -- ^ Intention clarity [0, 1]
  , ivStability  :: !Double                    -- ^ Temporal stability [0, 1]
  } deriving (Eq, Show)

-- =============================================================================
-- Harness Operations
-- =============================================================================

-- | Initialize biofeedback harness
initializeHarness :: [ChannelType] -> BiofeedbackHarness
initializeHarness channelTypes =
  let channels = map createChannel channelTypes
  in BiofeedbackHarness
    { bhChannels = channels
    , bhState = HarnessIdle
    , bhGain = 1.0
    , bhCalibration = (0, 1)
    , bhLimits = defaultLimits
    }
  where
    createChannel ct = BioChannel
      { bcType = ct
      , bcValue = 0
      , bcBaseline = 0.5
      , bcWeight = channelWeight ct
      , bcActive = False
      }
    channelWeight ChannelHeart = phi
    channelWeight ChannelNeural = phi * phiInverse
    channelWeight ChannelRespiratory = phiInverse
    channelWeight ChannelGalvanic = phiInverse * phiInverse
    channelWeight ChannelMuscular = 0.5
    channelWeight ChannelThermal = 0.3

-- | Update harness with new bio readings
updateHarness :: BiofeedbackHarness -> [(ChannelType, Double)] -> BiofeedbackHarness
updateHarness harness readings =
  let updatedChannels = map (updateChannel readings) (bhChannels harness)
      newState = determineState updatedChannels (bhLimits harness)
  in harness { bhChannels = updatedChannels, bhState = newState }
  where
    updateChannel rds ch =
      case lookup (bcType ch) rds of
        Just v  -> ch { bcValue = v }
        Nothing -> ch
    determineState chs limits =
      let coherence = overallCoherence chs
      in if coherence < slMinCoherence limits
         then HarnessSafeMode
         else if coherence > slMinCoherence limits * phi
              then HarnessActive
              else HarnessIdle

-- | Calibrate harness baselines
calibrateHarness :: BiofeedbackHarness -> [Double] -> BiofeedbackHarness
calibrateHarness harness baselines =
  let calibrated = zipWith setBaseline (bhChannels harness) (baselines ++ repeat 0.5)
      avgBaseline = sum baselines / fromIntegral (length baselines)
      scale = if avgBaseline > 0 then 1 / avgBaseline else 1
  in harness
    { bhChannels = calibrated
    , bhState = HarnessCalibrating
    , bhCalibration = (avgBaseline, scale)
    }
  where
    setBaseline ch bl = ch { bcBaseline = bl }

-- =============================================================================
-- Channel Management
-- =============================================================================

-- | Activate a bio channel
activateChannel :: BiofeedbackHarness -> ChannelType -> BiofeedbackHarness
activateChannel harness ct =
  let updated = map activate (bhChannels harness)
  in harness { bhChannels = updated }
  where
    activate ch = if bcType ch == ct then ch { bcActive = True } else ch

-- | Deactivate a bio channel
deactivateChannel :: BiofeedbackHarness -> ChannelType -> BiofeedbackHarness
deactivateChannel harness ct =
  let updated = map deactivate (bhChannels harness)
  in harness { bhChannels = updated }
  where
    deactivate ch = if bcType ch == ct then ch { bcActive = False } else ch

-- | Get coherence for a channel
channelCoherence :: BioChannel -> Double
channelCoherence ch =
  let deviation = abs (bcValue ch - bcBaseline ch)
      normalized = 1 - min 1 (deviation * 2)
  in normalized * bcWeight ch

-- =============================================================================
-- Intention Processing
-- =============================================================================

-- | Process bio channels into intention vector
processIntention :: BiofeedbackHarness -> IntentionVector
processIntention harness =
  let activeChannels = filter bcActive (bhChannels harness)
      coherences = map channelCoherence activeChannels
      avgCoherence = if null coherences then 0 else sum coherences / fromIntegral (length coherences)
      direction = computeDirection activeChannels
      clarity = coherenceToClarity avgCoherence
      stability = computeStability activeChannels
  in IntentionVector
    { ivDirection = direction
    , ivIntensity = avgCoherence * bhGain harness
    , ivClarity = clarity
    , ivStability = stability
    }

-- | Convert intention to propulsion vector
intentionToVector :: IntentionVector -> Double -> (Double, Double, Double, Double)
intentionToVector iv thrust =
  let (dx, dy, dz) = ivDirection iv
      magnitude = thrust * ivIntensity iv * ivClarity iv
  in (dx * magnitude, dy * magnitude, dz * magnitude, ivStability iv)

-- | Amplify intention through coherence boost
amplifyIntention :: IntentionVector -> Double -> IntentionVector
amplifyIntention iv boostFactor =
  let newIntensity = min 1.0 (ivIntensity iv * boostFactor)
      clarityBoost = 1 + (boostFactor - 1) * phiInverse
      newClarity = min 1.0 (ivClarity iv * clarityBoost)
  in iv
    { ivIntensity = newIntensity
    , ivClarity = newClarity
    }

-- =============================================================================
-- Feedback Loop
-- =============================================================================

-- | Feedback loop state
data FeedbackLoop = FeedbackLoop
  { flTarget     :: !Double    -- ^ Target coherence
  , flCurrent    :: !Double    -- ^ Current coherence
  , flError      :: !Double    -- ^ Error signal
  , flIntegral   :: !Double    -- ^ Integral term
  , flDerivative :: !Double    -- ^ Derivative term
  , flOutput     :: !Double    -- ^ Control output
  } deriving (Eq, Show)

-- | Create new feedback loop
createFeedbackLoop :: Double -> FeedbackLoop
createFeedbackLoop target = FeedbackLoop
  { flTarget = target
  , flCurrent = 0
  , flError = target
  , flIntegral = 0
  , flDerivative = 0
  , flOutput = 0
  }

-- | Update feedback loop with new reading
updateFeedback :: FeedbackLoop -> Double -> FeedbackLoop
updateFeedback fl current =
  let newError = flTarget fl - current
      newIntegral = flIntegral fl + newError * 0.1  -- Simple integration
      newDerivative = newError - flError fl
      -- PID-like control output
      kp = phi         -- Proportional gain
      ki = phiInverse  -- Integral gain
      kd = 0.1         -- Derivative gain
      output = kp * newError + ki * newIntegral + kd * newDerivative
  in fl
    { flCurrent = current
    , flError = newError
    , flIntegral = newIntegral
    , flDerivative = newDerivative
    , flOutput = max (-1) (min 1 output)  -- Clamp output
    }

-- | Compute loop stability
loopStability :: FeedbackLoop -> Double
loopStability fl =
  let errorMag = abs (flError fl)
      derivMag = abs (flDerivative fl)
      stability = 1 - min 1 (errorMag + derivMag * 0.5)
  in stability

-- =============================================================================
-- Safety Limits
-- =============================================================================

-- | Safety limit configuration
data SafetyLimits = SafetyLimits
  { slMaxIntensity  :: !Double   -- ^ Maximum intensity [0, 1]
  , slMaxRate       :: !Double   -- ^ Maximum rate of change
  , slMinCoherence  :: !Double   -- ^ Minimum coherence for operation
  , slTimeout       :: !Int      -- ^ Inactivity timeout (ticks)
  , slEmergencyStop :: !Bool     -- ^ Emergency stop flag
  } deriving (Eq, Show)

-- | Default safety limits
defaultLimits :: SafetyLimits
defaultLimits = SafetyLimits
  { slMaxIntensity = phiInverse  -- ~0.618 max intensity
  , slMaxRate = 0.1              -- 10% change per update
  , slMinCoherence = phiInverse * phiInverse  -- ~0.382 minimum
  , slTimeout = 100
  , slEmergencyStop = False
  }

-- | Check if values are within limits
checkLimits :: SafetyLimits -> IntentionVector -> Bool
checkLimits limits iv =
  not (slEmergencyStop limits) &&
  ivIntensity iv <= slMaxIntensity limits &&
  ivClarity iv >= slMinCoherence limits

-- | Apply safety limits to intention
applyLimits :: SafetyLimits -> IntentionVector -> IntentionVector
applyLimits limits iv =
  iv
    { ivIntensity = min (slMaxIntensity limits) (ivIntensity iv)
    , ivClarity = max (slMinCoherence limits) (ivClarity iv)
    }

-- =============================================================================
-- Harness Metrics
-- =============================================================================

-- | Harness performance metrics
data HarnessMetrics = HarnessMetrics
  { hmOverallCoherence :: !Double   -- ^ Combined coherence
  , hmActiveChannels   :: !Int      -- ^ Number of active channels
  , hmEfficiency       :: !Double   -- ^ Processing efficiency
  , hmStability        :: !Double   -- ^ Overall stability
  , hmLatency          :: !Double   -- ^ Processing latency
  } deriving (Eq, Show)

-- | Compute harness metrics
computeMetrics :: BiofeedbackHarness -> HarnessMetrics
computeMetrics harness =
  let channels = bhChannels harness
      activeCount = length $ filter bcActive channels
      coherence = overallCoherence channels
      efficiency = coherenceEfficiency harness
      stability = channelStability channels
  in HarnessMetrics
    { hmOverallCoherence = coherence
    , hmActiveChannels = activeCount
    , hmEfficiency = efficiency
    , hmStability = stability
    , hmLatency = 1 / (1 + coherence * phi)  -- Lower latency with higher coherence
    }

-- | Compute coherence efficiency
coherenceEfficiency :: BiofeedbackHarness -> Double
coherenceEfficiency harness =
  let (offset, scale) = bhCalibration harness
      channels = bhChannels harness
      coherence = overallCoherence channels
      calibrated = (coherence - offset) * scale
  in min 1.0 (max 0 calibrated) * bhGain harness

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Compute overall coherence from channels
overallCoherence :: [BioChannel] -> Double
overallCoherence [] = 0
overallCoherence channels =
  let activeChannels = filter bcActive channels
      weightedSum = sum [channelCoherence ch | ch <- activeChannels]
      totalWeight = sum [bcWeight ch | ch <- activeChannels]
  in if totalWeight > 0 then weightedSum / totalWeight else 0

-- | Compute direction from bio channels
computeDirection :: [BioChannel] -> (Double, Double, Double)
computeDirection channels =
  let heart = findChannel ChannelHeart channels
      neural = findChannel ChannelNeural channels
      resp = findChannel ChannelRespiratory channels
      x = maybe 0 bcValue heart
      y = maybe 0 bcValue neural
      z = maybe 0 bcValue resp
      len = sqrt (x*x + y*y + z*z)
  in if len > 0 then (x/len, y/len, z/len) else (0, 0, 1)

-- | Find channel by type
findChannel :: ChannelType -> [BioChannel] -> Maybe BioChannel
findChannel ct channels =
  case filter (\c -> bcType c == ct) channels of
    (ch:_) -> Just ch
    []     -> Nothing

-- | Convert coherence to clarity
coherenceToClarity :: Double -> Double
coherenceToClarity coh = coh * phi / (1 + coh)

-- | Compute stability from channels
computeStability :: [BioChannel] -> Double
computeStability channels =
  let values = map bcValue channels
      avg = sum values / fromIntegral (length values)
      variance = sum [(v - avg) ^ (2::Int) | v <- values] / fromIntegral (length values)
  in 1 - min 1 (sqrt variance)

-- | Stability from individual channels
channelStability :: [BioChannel] -> Double
channelStability [] = 0
channelStability channels =
  let deviations = [abs (bcValue ch - bcBaseline ch) | ch <- channels]
      avgDeviation = sum deviations / fromIntegral (length deviations)
  in 1 - min 1 avgDeviation
