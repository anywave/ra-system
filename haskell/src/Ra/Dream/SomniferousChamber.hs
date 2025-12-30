{-|
Module      : Ra.Dream.SomniferousChamber
Description : Somniferous chamber for enhanced dream work
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements the somniferous chamber system for enhanced dream induction,
lucid dreaming support, and consciousness exploration during sleep states.
Integrates biometric feedback, scalar field generation, and acoustic
entrainment.

== Chamber Theory

=== Environmental Control

* Temperature regulation for optimal REM
* Light spectrum management
* Acoustic entrainment frequencies
* Electromagnetic field shaping

=== Consciousness Entrainment

1. Brainwave frequency targeting (theta, delta)
2. Binaural beat generation
3. Isochronic tone patterns
4. Scalar wave consciousness interface
-}
module Ra.Dream.SomniferousChamber
  ( -- * Core Types
    SomniferousChamber(..)
  , ChamberMode(..)
  , EntrainmentProfile(..)
  , ChamberEnvironment(..)

    -- * Chamber Control
  , initializeChamber
  , activateChamber
  , deactivateChamber
  , setChamberMode

    -- * Entrainment Settings
  , setEntrainment
  , targetBrainwave
  , binauralFrequency
  , isochronicPattern

    -- * Environment Control
  , setTemperature
  , setLighting
  , setAcoustics
  , setFieldStrength

    -- * Session Management
  , DreamSession(..)
  , startSession
  , endSession
  , sessionProgress

    -- * Biometric Integration
  , BiometricFeedback(..)
  , processBiometrics
  , adaptToSleepStage
  , coherenceFromBiometrics

    -- * Safety Systems
  , SafetyProtocol(..)
  , checkSafety
  , emergencyWake
  , safetyOverride
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete chamber state
data SomniferousChamber = SomniferousChamber
  { scMode         :: !ChamberMode          -- ^ Current operating mode
  , scEnvironment  :: !ChamberEnvironment   -- ^ Environmental settings
  , scEntrainment  :: !EntrainmentProfile   -- ^ Entrainment configuration
  , scSession      :: !(Maybe DreamSession) -- ^ Active session
  , scBiometrics   :: !(Maybe BiometricFeedback)  -- ^ Latest biometrics
  , scSafety       :: !SafetyProtocol       -- ^ Safety settings
  , scActive       :: !Bool                 -- ^ Chamber active
  , scCoherence    :: !Double               -- ^ Overall coherence [0, 1]
  } deriving (Eq, Show)

-- | Chamber operating modes
data ChamberMode
  = ModeStandby        -- ^ Ready but inactive
  | ModeSleepInduction -- ^ Inducing sleep
  | ModeREMEnhancement -- ^ Enhancing REM
  | ModeLucidSupport   -- ^ Supporting lucidity
  | ModeDeepRest       -- ^ Deep sleep restoration
  | ModeWakeTransition -- ^ Gentle wake-up
  | ModeEmergency      -- ^ Emergency/safety mode
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Entrainment profile configuration
data EntrainmentProfile = EntrainmentProfile
  { epTargetFreq    :: !Double      -- ^ Target brainwave frequency (Hz)
  , epBinauralBase  :: !Double      -- ^ Binaural base frequency (Hz)
  , epBinauralBeat  :: !Double      -- ^ Binaural beat frequency (Hz)
  , epIsochronic    :: !Bool        -- ^ Isochronic tones enabled
  , epIsoRate       :: !Double      -- ^ Isochronic pulse rate
  , epIntensity     :: !Double      -- ^ Overall intensity [0, 1]
  , epRampTime      :: !Int         -- ^ Ramp time in seconds
  } deriving (Eq, Show)

-- | Chamber environment settings
data ChamberEnvironment = ChamberEnvironment
  { ceTemperature  :: !Double       -- ^ Temperature (Celsius)
  , ceHumidity     :: !Double       -- ^ Humidity percentage
  , ceLightLevel   :: !Double       -- ^ Light level [0, 1]
  , ceLightColor   :: !LightSpectrum -- ^ Light color
  , ceAcousticLevel :: !Double      -- ^ Acoustic level [0, 1]
  , ceFieldStrength :: !Double      -- ^ EM field strength [0, 1]
  , ceOxygen       :: !Double       -- ^ Oxygen percentage
  } deriving (Eq, Show)

-- | Light spectrum options
data LightSpectrum
  = SpectrumOff          -- ^ Lights off
  | SpectrumWarmRed      -- ^ Warm red (melatonin-friendly)
  | SpectrumAmber        -- ^ Amber
  | SpectrumDimWhite     -- ^ Dim white
  | SpectrumBlue         -- ^ Blue (alertness)
  | SpectrumCustom       -- ^ Custom spectrum
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Chamber Control
-- =============================================================================

-- | Initialize chamber with defaults
initializeChamber :: SomniferousChamber
initializeChamber = SomniferousChamber
  { scMode = ModeStandby
  , scEnvironment = defaultEnvironment
  , scEntrainment = defaultEntrainment
  , scSession = Nothing
  , scBiometrics = Nothing
  , scSafety = defaultSafety
  , scActive = False
  , scCoherence = 0.5
  }

-- | Activate chamber
activateChamber :: SomniferousChamber -> SomniferousChamber
activateChamber chamber =
  if safetyCheck chamber
  then chamber { scActive = True, scMode = ModeSleepInduction }
  else chamber { scMode = ModeEmergency }

-- | Deactivate chamber
deactivateChamber :: SomniferousChamber -> SomniferousChamber
deactivateChamber chamber =
  chamber
    { scActive = False
    , scMode = ModeStandby
    , scSession = Nothing
    }

-- | Set chamber mode
setChamberMode :: SomniferousChamber -> ChamberMode -> SomniferousChamber
setChamberMode chamber mode =
  let env = adjustEnvironmentForMode mode (scEnvironment chamber)
      ent = adjustEntrainmentForMode mode (scEntrainment chamber)
  in chamber { scMode = mode, scEnvironment = env, scEntrainment = ent }

-- =============================================================================
-- Entrainment Settings
-- =============================================================================

-- | Set entrainment profile
setEntrainment :: SomniferousChamber -> EntrainmentProfile -> SomniferousChamber
setEntrainment chamber profile =
  chamber { scEntrainment = profile }

-- | Target specific brainwave frequency
targetBrainwave :: SomniferousChamber -> BrainwaveBand -> SomniferousChamber
targetBrainwave chamber band =
  let freq = bandFrequency band
      profile = (scEntrainment chamber) { epTargetFreq = freq }
  in chamber { scEntrainment = profile }

-- | Set binaural beat frequency
binauralFrequency :: SomniferousChamber -> Double -> Double -> SomniferousChamber
binauralFrequency chamber baseFreq beatFreq =
  let profile = (scEntrainment chamber)
        { epBinauralBase = baseFreq
        , epBinauralBeat = beatFreq
        }
  in chamber { scEntrainment = profile }

-- | Set isochronic pattern
isochronicPattern :: SomniferousChamber -> Double -> Bool -> SomniferousChamber
isochronicPattern chamber rate enabled =
  let profile = (scEntrainment chamber)
        { epIsochronic = enabled
        , epIsoRate = rate
        }
  in chamber { scEntrainment = profile }

-- =============================================================================
-- Environment Control
-- =============================================================================

-- | Set chamber temperature
setTemperature :: SomniferousChamber -> Double -> SomniferousChamber
setTemperature chamber temp =
  let env = (scEnvironment chamber) { ceTemperature = clamp 15 30 temp }
  in chamber { scEnvironment = env }

-- | Set chamber lighting
setLighting :: SomniferousChamber -> Double -> LightSpectrum -> SomniferousChamber
setLighting chamber level spectrum =
  let env = (scEnvironment chamber)
        { ceLightLevel = clamp 0 1 level
        , ceLightColor = spectrum
        }
  in chamber { scEnvironment = env }

-- | Set acoustic settings
setAcoustics :: SomniferousChamber -> Double -> SomniferousChamber
setAcoustics chamber level =
  let env = (scEnvironment chamber) { ceAcousticLevel = clamp 0 1 level }
  in chamber { scEnvironment = env }

-- | Set electromagnetic field strength
setFieldStrength :: SomniferousChamber -> Double -> SomniferousChamber
setFieldStrength chamber strength =
  let env = (scEnvironment chamber) { ceFieldStrength = clamp 0 1 strength }
  in chamber { scEnvironment = env }

-- =============================================================================
-- Session Management
-- =============================================================================

-- | Dream session data
data DreamSession = DreamSession
  { dsSessionId    :: !String          -- ^ Session identifier
  , dsStartTime    :: !Int             -- ^ Start timestamp
  , dsTargetDuration :: !Int           -- ^ Target duration (minutes)
  , dsCurrentPhase :: !SleepPhase      -- ^ Current sleep phase
  , dsLucidPeriods :: !Int             -- ^ Lucid period count
  , dsREMDuration  :: !Int             -- ^ REM duration (minutes)
  , dsCoherence    :: !Double          -- ^ Average coherence
  , dsComplete     :: !Bool            -- ^ Session completed
  } deriving (Eq, Show)

-- | Sleep phases
data SleepPhase
  = PhaseAwake         -- ^ Still awake
  | PhaseN1            -- ^ Light sleep stage 1
  | PhaseN2            -- ^ Light sleep stage 2
  | PhaseN3            -- ^ Deep sleep
  | PhaseREM           -- ^ REM sleep
  | PhaseTransition    -- ^ Between phases
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Start new dream session
startSession :: SomniferousChamber -> Int -> Int -> SomniferousChamber
startSession chamber startTime duration =
  let session = DreamSession
        { dsSessionId = "session_" ++ show startTime
        , dsStartTime = startTime
        , dsTargetDuration = duration
        , dsCurrentPhase = PhaseAwake
        , dsLucidPeriods = 0
        , dsREMDuration = 0
        , dsCoherence = scCoherence chamber
        , dsComplete = False
        }
  in activateChamber (chamber { scSession = Just session })

-- | End current session
endSession :: SomniferousChamber -> SomniferousChamber
endSession chamber =
  let updatedSession = fmap (\s -> s { dsComplete = True }) (scSession chamber)
  in setChamberMode (chamber { scSession = updatedSession }) ModeWakeTransition

-- | Get session progress
sessionProgress :: SomniferousChamber -> Maybe Double
sessionProgress chamber = do
  session <- scSession chamber
  let elapsed = dsREMDuration session + dsLucidPeriods session * 5  -- Simplified
      target = dsTargetDuration session
  return $ if target > 0 then fromIntegral elapsed / fromIntegral target else 0

-- =============================================================================
-- Biometric Integration
-- =============================================================================

-- | Biometric feedback data
data BiometricFeedback = BiometricFeedback
  { bfHeartRate    :: !Double          -- ^ Heart rate (BPM)
  , bfHRV          :: !Double          -- ^ Heart rate variability
  , bfBreathRate   :: !Double          -- ^ Breaths per minute
  , bfEEGDominant  :: !Double          -- ^ Dominant EEG frequency
  , bfSkinTemp     :: !Double          -- ^ Skin temperature
  , bfMovement     :: !Double          -- ^ Movement level [0, 1]
  , bfCoherence    :: !Double          -- ^ Biometric coherence
  } deriving (Eq, Show)

-- | Process biometric feedback
processBiometrics :: SomniferousChamber -> BiometricFeedback -> SomniferousChamber
processBiometrics chamber bio =
  let newCoherence = (scCoherence chamber + bfCoherence bio) / 2
      sleepPhase = detectSleepPhase bio
      adapted = adaptToSleepStage chamber sleepPhase
  in adapted { scBiometrics = Just bio, scCoherence = newCoherence }

-- | Adapt chamber to detected sleep stage
adaptToSleepStage :: SomniferousChamber -> SleepPhase -> SomniferousChamber
adaptToSleepStage chamber phase =
  let newMode = phaseToMode phase
      updatedSession = fmap (updateSessionPhase phase) (scSession chamber)
  in setChamberMode (chamber { scSession = updatedSession }) newMode

-- | Calculate coherence from biometrics
coherenceFromBiometrics :: BiometricFeedback -> Double
coherenceFromBiometrics bio =
  let hrvFactor = min 1.0 (bfHRV bio / 100)
      moveFactor = 1 - bfMovement bio
      breathFactor = min 1.0 (15 / max 1 (abs (bfBreathRate bio - 12)))
  in (hrvFactor + moveFactor + breathFactor) / 3

-- =============================================================================
-- Safety Systems
-- =============================================================================

-- | Safety protocol configuration
data SafetyProtocol = SafetyProtocol
  { spMaxDuration    :: !Int           -- ^ Maximum session duration (minutes)
  , spMinCoherence   :: !Double        -- ^ Minimum coherence threshold
  , spMaxFieldStrength :: !Double      -- ^ Maximum field strength
  , spEmergencyThresholds :: !EmergencyThresholds  -- ^ Emergency triggers
  , spAutoWake       :: !Bool          -- ^ Auto-wake enabled
  , spMonitoringActive :: !Bool        -- ^ Active monitoring
  } deriving (Eq, Show)

-- | Emergency trigger thresholds
data EmergencyThresholds = EmergencyThresholds
  { etMaxHeartRate  :: !Double         -- ^ Max heart rate
  , etMinHeartRate  :: !Double         -- ^ Min heart rate
  , etMaxMovement   :: !Double         -- ^ Max movement (distress)
  , etMinCoherence  :: !Double         -- ^ Min coherence
  } deriving (Eq, Show)

-- | Check safety status
checkSafety :: SomniferousChamber -> SafetyStatus
checkSafety chamber =
  case scBiometrics chamber of
    Nothing -> SafetyUnknown
    Just bio ->
      let thresholds = spEmergencyThresholds (scSafety chamber)
          hrOk = bfHeartRate bio < etMaxHeartRate thresholds &&
                 bfHeartRate bio > etMinHeartRate thresholds
          moveOk = bfMovement bio < etMaxMovement thresholds
          cohOk = scCoherence chamber > etMinCoherence thresholds
      in if hrOk && moveOk && cohOk
         then SafetyOk
         else SafetyAlert (describeIssue bio thresholds)

-- | Safety status
data SafetyStatus
  = SafetyOk
  | SafetyAlert !String
  | SafetyUnknown
  deriving (Eq, Show)

-- | Emergency wake protocol
emergencyWake :: SomniferousChamber -> SomniferousChamber
emergencyWake chamber =
  let env = (scEnvironment chamber)
        { ceLightLevel = 0.7
        , ceLightColor = SpectrumBlue
        , ceAcousticLevel = 0.5
        }
  in chamber
    { scMode = ModeEmergency
    , scEnvironment = env
    , scActive = False
    }

-- | Safety override (requires explicit permission)
safetyOverride :: SomniferousChamber -> String -> SomniferousChamber
safetyOverride chamber _permission =
  -- In real implementation, would validate permission
  chamber { scSafety = (scSafety chamber) { spMonitoringActive = False } }

-- =============================================================================
-- Brainwave Bands
-- =============================================================================

-- | Brainwave frequency bands
data BrainwaveBand
  = BandDelta      -- ^ 0.5-4 Hz (deep sleep)
  | BandTheta      -- ^ 4-8 Hz (meditation, light sleep)
  | BandAlpha      -- ^ 8-12 Hz (relaxation)
  | BandBeta       -- ^ 12-30 Hz (alertness)
  | BandGamma      -- ^ 30-100 Hz (high cognition)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Get center frequency for band
bandFrequency :: BrainwaveBand -> Double
bandFrequency BandDelta = 2.0
bandFrequency BandTheta = 6.0
bandFrequency BandAlpha = 10.0
bandFrequency BandBeta = 20.0
bandFrequency BandGamma = 40.0

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Default environment settings
defaultEnvironment :: ChamberEnvironment
defaultEnvironment = ChamberEnvironment
  { ceTemperature = 18.5
  , ceHumidity = 50
  , ceLightLevel = 0
  , ceLightColor = SpectrumOff
  , ceAcousticLevel = 0.3
  , ceFieldStrength = phiInverse
  , ceOxygen = 21
  }

-- | Default entrainment profile
defaultEntrainment :: EntrainmentProfile
defaultEntrainment = EntrainmentProfile
  { epTargetFreq = 4.0  -- Theta
  , epBinauralBase = 200
  , epBinauralBeat = 4.0
  , epIsochronic = True
  , epIsoRate = 4.0
  , epIntensity = 0.5
  , epRampTime = 300  -- 5 minutes
  }

-- | Default safety protocol
defaultSafety :: SafetyProtocol
defaultSafety = SafetyProtocol
  { spMaxDuration = 480  -- 8 hours
  , spMinCoherence = 0.3
  , spMaxFieldStrength = 0.8
  , spEmergencyThresholds = EmergencyThresholds 120 40 0.9 0.2
  , spAutoWake = True
  , spMonitoringActive = True
  }

-- | Safety check for activation
safetyCheck :: SomniferousChamber -> Bool
safetyCheck chamber =
  let env = scEnvironment chamber
      tempOk = ceTemperature env >= 15 && ceTemperature env <= 30
      fieldOk = ceFieldStrength env <= spMaxFieldStrength (scSafety chamber)
  in tempOk && fieldOk

-- | Adjust environment for mode
adjustEnvironmentForMode :: ChamberMode -> ChamberEnvironment -> ChamberEnvironment
adjustEnvironmentForMode mode env = case mode of
  ModeSleepInduction -> env
    { ceLightLevel = 0.05
    , ceLightColor = SpectrumWarmRed
    , ceTemperature = 18.5
    }
  ModeREMEnhancement -> env
    { ceLightLevel = 0
    , ceFieldStrength = min 1.0 (ceFieldStrength env * phi)
    }
  ModeLucidSupport -> env
    { ceFieldStrength = ceFieldStrength env * phiInverse
    }
  ModeDeepRest -> env
    { ceLightLevel = 0
    , ceAcousticLevel = 0.1
    , ceTemperature = 17.5
    }
  ModeWakeTransition -> env
    { ceLightLevel = 0.3
    , ceLightColor = SpectrumAmber
    }
  ModeEmergency -> env
    { ceLightLevel = 0.8
    , ceLightColor = SpectrumBlue
    }
  ModeStandby -> env

-- | Adjust entrainment for mode
adjustEntrainmentForMode :: ChamberMode -> EntrainmentProfile -> EntrainmentProfile
adjustEntrainmentForMode mode profile = case mode of
  ModeSleepInduction -> profile { epTargetFreq = 6.0, epBinauralBeat = 6.0 }  -- Theta
  ModeREMEnhancement -> profile { epTargetFreq = 4.0, epBinauralBeat = 4.0 }  -- Deep theta
  ModeLucidSupport -> profile { epTargetFreq = 8.0, epBinauralBeat = 8.0 }    -- Alpha-theta border
  ModeDeepRest -> profile { epTargetFreq = 2.0, epBinauralBeat = 2.0 }        -- Delta
  ModeWakeTransition -> profile { epTargetFreq = 10.0, epBinauralBeat = 10.0 } -- Alpha
  _ -> profile

-- | Detect sleep phase from biometrics
detectSleepPhase :: BiometricFeedback -> SleepPhase
detectSleepPhase bio
  | bfMovement bio > 0.5 = PhaseAwake
  | bfEEGDominant bio < 4 = PhaseN3
  | bfEEGDominant bio < 8 && bfMovement bio < 0.1 = PhaseREM
  | bfEEGDominant bio < 8 = PhaseN2
  | bfEEGDominant bio < 12 = PhaseN1
  | otherwise = PhaseAwake

-- | Map sleep phase to chamber mode
phaseToMode :: SleepPhase -> ChamberMode
phaseToMode PhaseAwake = ModeSleepInduction
phaseToMode PhaseN1 = ModeSleepInduction
phaseToMode PhaseN2 = ModeREMEnhancement
phaseToMode PhaseN3 = ModeDeepRest
phaseToMode PhaseREM = ModeLucidSupport
phaseToMode PhaseTransition = ModeStandby

-- | Update session with detected phase
updateSessionPhase :: SleepPhase -> DreamSession -> DreamSession
updateSessionPhase phase session =
  let remInc = if phase == PhaseREM then 1 else 0
  in session
    { dsCurrentPhase = phase
    , dsREMDuration = dsREMDuration session + remInc
    }

-- | Clamp value to range
clamp :: Double -> Double -> Double -> Double
clamp minVal maxVal = max minVal . min maxVal

-- | Describe safety issue
describeIssue :: BiometricFeedback -> EmergencyThresholds -> String
describeIssue bio thresh
  | bfHeartRate bio > etMaxHeartRate thresh = "Heart rate too high"
  | bfHeartRate bio < etMinHeartRate thresh = "Heart rate too low"
  | bfMovement bio > etMaxMovement thresh = "Excessive movement detected"
  | otherwise = "Coherence below threshold"
