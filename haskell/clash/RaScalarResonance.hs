{-|
Module      : RaScalarResonance
Description : Scalar resonance biofeedback loop for healing
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 10 v1.1: Enhanced with raw normalization, 10Hz gating, access gating,
session FSM, safety limits, and explicit output buses.
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaScalarResonance where

import Clash.Prelude
import qualified Prelude as P

-- Types
type Fixed8 = Unsigned 8
type DriftValue = Signed 16
type FreqHz = Unsigned 16
type ChakraIndex = Index 7
type ChakraDrift = Vec 7 DriftValue
type SampleCounter = Unsigned 7
type DORTimer = Unsigned 5

-- Access Gating (Prompt 8)
data AccessResult = AccessOK | AccessDenied | AccessExpired | AccessPending
  deriving (Generic, NFDataX, Show, Eq)

gateCheck :: AccessResult -> Bool
gateCheck AccessOK = True
gateCheck _ = False

-- Session FSM
data SessionState = SessionIdle | Baseline | Alignment | Entrainment | Integration | Complete
  deriving (Generic, NFDataX, Show, Eq)

baselineCycles, alignmentMinCycles, entrainmentMinCycles, integrationCycles :: Unsigned 16
baselineCycles = 300
alignmentMinCycles = 1200
entrainmentMinCycles = 9000
integrationCycles = 3000

-- Raw Biometric Input
data RawBiometric = RawBiometric
  { hrvMs :: Unsigned 8, eegAlphaUv :: Unsigned 8
  , gsrUs :: Unsigned 8, breathCpm :: Unsigned 8
  } deriving (Generic, NFDataX, Show, Eq)

data BiometricInput = BiometricInput
  { hrv :: Fixed8, eegAlpha :: Fixed8
  , skinConductance :: Fixed8, breathVariability :: Fixed8
  } deriving (Generic, NFDataX, Show, Eq)

data CoherenceState = CoherenceState
  { coherenceLevel :: Fixed8, emotionalTension :: Fixed8, chakraDrift :: ChakraDrift
  } deriving (Generic, NFDataX, Show, Eq)

data ScalarCommand = ScalarCommand
  { targetChakra :: ChakraIndex, polarityValue :: DriftValue
  , targetPotential :: DriftValue, entrainLow :: Unsigned 16
  , entrainHigh :: Unsigned 16, dorClearing :: Bool
  } deriving (Generic, NFDataX, Show, Eq)

data AdaptMode = Reinforce | Adjust | PeakPulse | Stabilize | EmergencyStab
  deriving (Generic, NFDataX, Show, Eq)

-- Output Buses
data AudioOutput = AudioOutput
  { audioPrimary :: FreqHz, audioSecondary :: FreqHz, audioCarrier :: FreqHz
  } deriving (Generic, NFDataX, Show, Eq)

data RGB = RGB { red :: Unsigned 8, green :: Unsigned 8, blue :: Unsigned 8 }
  deriving (Generic, NFDataX, Show, Eq)

data Pattern = PatternBreath | PatternWave | PatternPulse | PatternStatic
  deriving (Generic, NFDataX, Show, Eq)

data VisualOutput = VisualOutput { rgbColor :: RGB, pattern :: Pattern, intensity :: Fixed8 }
  deriving (Generic, NFDataX, Show, Eq)

data PulseConfig = PulseConfig
  { pulseFreq :: Unsigned 8, pulseDuty :: Unsigned 8, pulseIntensity :: Fixed8 }
  deriving (Generic, NFDataX, Show, Eq)

data TactileOutput = TactileOutput { tactilePulse :: PulseConfig, tactileActive :: Bool }
  deriving (Generic, NFDataX, Show, Eq)

data HarmonicOutput = HarmonicOutput
  { audioOut :: AudioOutput, visualOut :: VisualOutput
  , tactileOut :: TactileOutput, phiSync :: Bool }
  deriving (Generic, NFDataX, Show, Eq)

data FeedbackState = FeedbackState
  { prevCoherence :: Fixed8, coherenceDelta :: Signed 16
  , adaptationMode :: AdaptMode, cycleCount :: Unsigned 16, dorTimer :: DORTimer }
  deriving (Generic, NFDataX, Show, Eq)

data SystemState = SystemState
  { sessionPhase :: SessionState, feedbackState :: FeedbackState
  , phaseCycles :: Unsigned 16, sampleCounter :: SampleCounter, accessValid :: Bool }
  deriving (Generic, NFDataX, Show, Eq)

data ResonanceOutput = ResonanceOutput
  { scalarCmd :: ScalarCommand, harmonicOut :: HarmonicOutput
  , currentPhase :: SessionState, feedback :: FeedbackState
  , validOutput :: Bool, safetyAlert :: Bool }
  deriving (Generic, NFDataX, Show, Eq)

-- Constants
chakraFrequencies :: Vec 7 FreqHz
chakraFrequencies = 396 :> 417 :> 528 :> 639 :> 741 :> 852 :> 963 :> Nil

harmonicSupport :: Vec 7 FreqHz
harmonicSupport = 264 :> 278 :> 352 :> 426 :> 494 :> 568 :> 642 :> Nil

chakraColors :: Vec 7 RGB
chakraColors = RGB 255 0 0 :> RGB 255 127 0 :> RGB 255 255 0 :> RGB 0 255 0 :>
               RGB 0 127 255 :> RGB 75 0 130 :> RGB 148 0 211 :> Nil

coherenceFloor, coherenceThreshold :: Fixed8
coherenceFloor = 26
coherenceThreshold = 5

maxDORDuration :: DORTimer
maxDORDuration = 30

maxPolarityCap :: DriftValue
maxPolarityCap = 64

peakThreshold :: Signed 16
peakThreshold = 31

thetaCarrier :: FreqHz
thetaCarrier = 78

-- Normalization
normalizeBiometrics :: RawBiometric -> BiometricInput
normalizeBiometrics raw =
  let hrvNorm = resize (satSub SatBound (hrvMs raw) 20) :: Unsigned 16
      hrvScaled = satMul SatBound (resize hrvNorm) 2
      eegNorm = resize (eegAlphaUv raw) :: Unsigned 16
      eegScaled = satMul SatBound eegNorm 2 + (eegNorm `shiftR` 1)
      breathNorm = resize (satSub SatBound (breathCpm raw) 4) :: Unsigned 16
      breathScaled = satMul SatBound (resize breathNorm) 16
  in BiometricInput (resize (min 255 hrvScaled)) (resize (min 255 eegScaled))
                    (gsrUs raw) (resize (min 255 breathScaled))

-- Sample Gating (10 Hz)
sampleGate :: SampleCounter -> Bool
sampleGate counter = counter == 0

nextSampleCounter :: SampleCounter -> SampleCounter
nextSampleCounter c = if c >= 99 then 0 else c + 1

-- Coherence
calculateCoherence :: BiometricInput -> Fixed8
calculateCoherence bio =
  let hrvVal = resize (hrv bio) :: Unsigned 16
      eegVal = resize (eegAlpha bio) :: Unsigned 16
      breathVal = resize (breathVariability bio) :: Unsigned 16
  in resize ((hrvVal * 40 + eegVal * 35 + breathVal * 25) `div` 100)

calculateTension :: BiometricInput -> Fixed8
calculateTension bio =
  let scVal = resize (skinConductance bio) :: Unsigned 16
      hrvInv = 255 - resize (hrv bio) :: Unsigned 16
  in resize ((scVal * 60 + hrvInv * 40) `div` 100)

extractChakraDrift :: BiometricInput -> Fixed8 -> ChakraDrift
extractChakraDrift bio tension =
  let alpha = resize (eegAlpha bio) :: Signed 16
      sc = resize (skinConductance bio) :: Signed 16
      tensionS = resize tension :: Signed 16
      baseDrift = alpha - 128
  in (baseDrift + (sc `shiftR` 4)) :> (baseDrift - (sc `shiftR` 5)) :>
     (baseDrift + (tensionS `shiftR` 3)) :> (negate (tensionS `shiftR` 2)) :>
     (baseDrift `shiftR` 1) :> (baseDrift `shiftR` 2) :> (baseDrift `shiftR` 3) :> Nil

processBiometrics :: BiometricInput -> CoherenceState
processBiometrics bio = CoherenceState (calculateCoherence bio) (calculateTension bio)
                                       (extractChakraDrift bio (calculateTension bio))

-- Session FSM
sessionStep :: SessionState -> Unsigned 16 -> Fixed8 -> SessionState
sessionStep phase cycles coherence = case phase of
  SessionIdle -> SessionIdle
  Baseline | cycles >= baselineCycles -> Alignment
           | otherwise -> Baseline
  Alignment | cycles >= alignmentMinCycles && coherence >= 128 -> Entrainment
            | otherwise -> Alignment
  Entrainment | cycles >= entrainmentMinCycles && coherence >= 179 -> Integration
              | otherwise -> Entrainment
  Integration | cycles >= integrationCycles -> Complete
              | otherwise -> Integration
  Complete -> Complete

-- Scalar Alignment
findMaxDrift :: ChakraDrift -> (ChakraIndex, DriftValue)
findMaxDrift drift = foldr maxMag (0, drift !! 0) (zip indicesI drift)
  where maxMag (i, v) (maxI, maxV) = if abs v > abs maxV then (i, v) else (maxI, maxV)

saturatePolarity :: DriftValue -> DriftValue
saturatePolarity p | p > maxPolarityCap = maxPolarityCap
                   | p < negate maxPolarityCap = negate maxPolarityCap
                   | otherwise = p

scalarAlign :: CoherenceState -> DORTimer -> ScalarCommand
scalarAlign state dorTime =
  let (idx, maxDrift) = findMaxDrift (chakraDrift state)
      polar = saturatePolarity (negate maxDrift)
      tension = resize (emotionalTension state) :: Unsigned 16
      needsDOR = emotionalTension state > 179 && dorTime < maxDORDuration
  in ScalarCommand idx polar polar (3000 + tension * 2) (5500 + tension * 3) needsDOR

-- Harmonic Generation
generateAudio :: ChakraIndex -> AudioOutput
generateAudio idx = AudioOutput (chakraFrequencies !! idx) (harmonicSupport !! idx) thetaCarrier

generateVisual :: ChakraIndex -> AdaptMode -> Fixed8 -> VisualOutput
generateVisual idx mode coh = VisualOutput (chakraColors !! idx)
  (case mode of
     PeakPulse -> PatternPulse
     Reinforce -> PatternWave
     Stabilize -> PatternBreath
     EmergencyStab -> PatternStatic
     _ -> PatternBreath) coh

generateTactile :: Fixed8 -> AdaptMode -> TactileOutput
generateTactile tension mode =
  let intens = satSub SatBound 255 tension
      freq = case mode of
               PeakPulse -> 40
               Stabilize -> 10
               _ -> 20
  in TactileOutput (PulseConfig freq 128 intens) (intens > 30)

generateHarmonics :: ChakraIndex -> CoherenceState -> AdaptMode -> HarmonicOutput
generateHarmonics idx state mode = HarmonicOutput (generateAudio idx)
  (generateVisual idx mode (coherenceLevel state))
  (generateTactile (emotionalTension state) mode) True

-- Feedback
updateFeedback :: Fixed8 -> FeedbackState -> Bool -> FeedbackState
updateFeedback newCoh prev dorActive =
  let delta = resize newCoh - resize (prevCoherence prev) :: Signed 16
      mode = if newCoh < coherenceFloor then EmergencyStab
             else if delta > resize peakThreshold then PeakPulse
             else if delta > resize coherenceThreshold then Reinforce
             else if delta < negate (resize coherenceThreshold) then Adjust
             else Stabilize
      newDor = if dorActive && dorTimer prev < maxDORDuration then dorTimer prev + 1
               else if not dorActive then 0 else dorTimer prev
  in FeedbackState newCoh delta mode (satAdd SatBound (cycleCount prev) 1) newDor

-- Main Processing
initSystemState :: SystemState
initSystemState = SystemState SessionIdle (FeedbackState 128 0 Stabilize 0 0) 0 0 False

processResonance :: RawBiometric -> AccessResult -> SystemState -> (SystemState, ResonanceOutput)
processResonance rawBio access state =
  let shouldProcess = sampleGate (sampleCounter state)
      nextCounter = nextSampleCounter (sampleCounter state)
      hasAccess = gateCheck access || accessValid state
      newPhase = if hasAccess && sessionPhase state == SessionIdle then Baseline else sessionPhase state
      (newState, output) = if shouldProcess && hasAccess
        then processActive rawBio state { sessionPhase = newPhase, accessValid = hasAccess }
        else (state { sampleCounter = nextCounter }, makeIdleOutput state)
  in (newState { sampleCounter = nextCounter }, output)

processActive :: RawBiometric -> SystemState -> (SystemState, ResonanceOutput)
processActive rawBio state =
  let bio = normalizeBiometrics rawBio
      cohState = processBiometrics bio
      nextPhase = sessionStep (sessionPhase state) (phaseCycles state) (coherenceLevel cohState)
      resetCycles = nextPhase /= sessionPhase state
      cmd = scalarAlign cohState (dorTimer (feedbackState state))
      newFb = updateFeedback (coherenceLevel cohState) (feedbackState state) (dorClearing cmd)
      harmonics = generateHarmonics (targetChakra cmd) cohState (adaptationMode newFb)
      safety = adaptationMode newFb == EmergencyStab || dorTimer newFb >= maxDORDuration
      newCycles = if resetCycles then 0 else phaseCycles state + 1
  in (state { sessionPhase = nextPhase, feedbackState = newFb, phaseCycles = newCycles },
      ResonanceOutput cmd harmonics nextPhase newFb True safety)

makeIdleOutput :: SystemState -> ResonanceOutput
makeIdleOutput state = ResonanceOutput (ScalarCommand 0 0 0 3000 5500 False)
  (HarmonicOutput (AudioOutput 528 352 78) (VisualOutput (RGB 0 128 0) PatternStatic 64)
                  (TactileOutput (PulseConfig 10 64 32) False) False)
  (sessionPhase state) (feedbackState state) False False

-- Signal Processing
resonanceProcessor :: HiddenClockResetEnable dom => Signal dom (RawBiometric, AccessResult) -> Signal dom ResonanceOutput
resonanceProcessor = mealy (\st (raw, acc) -> processResonance raw acc st) initSystemState

{-# ANN scalarResonanceTop (Synthesize { t_name = "scalar_resonance_unit"
  , t_inputs = [PortName "clk", PortName "rst", PortName "en"
              , PortProduct "bio_raw" [PortName "hrv_ms", PortName "eeg_uv", PortName "gsr_us", PortName "breath_cpm"]
              , PortName "access_result"]
  , t_output = PortProduct "resonance_out" [PortName "scalar_cmd"
              , PortProduct "audio" [PortName "primary_hz", PortName "secondary_hz", PortName "carrier_hz"]
              , PortProduct "visual" [PortName "rgb", PortName "pattern", PortName "intensity"]
              , PortProduct "tactile" [PortName "pulse_config", PortName "active"]
              , PortName "session_phase", PortName "feedback", PortName "valid", PortName "safety_alert"]}) #-}
scalarResonanceTop :: Clock System -> Reset System -> Enable System
                   -> Signal System RawBiometric -> Signal System AccessResult -> Signal System ResonanceOutput
scalarResonanceTop clk rst en rawBio access =
  exposeClockResetEnable resonanceProcessor clk rst en (bundle (rawBio, access))

testRawInputs :: Vec 4 RawBiometric
testRawInputs = RawBiometric 85 45 128 12 :> RawBiometric 130 80 60 14 :>
                RawBiometric 40 30 220 8 :> RawBiometric 100 55 150 10 :> Nil

testBench :: Signal System Bool
testBench = done where
  clk = tbSystemClockGen (not <$> done)
  rst = systemResetGen
  out = scalarResonanceTop clk rst enableGen (stimuliGenerator clk rst testRawInputs) (pure AccessOK)
  done = register clk rst enableGen False (validOutput <$> out)

chakraName :: ChakraIndex -> String
chakraName 0 = "Root"
chakraName 1 = "Sacral"
chakraName 2 = "Solar"
chakraName 3 = "Heart"
chakraName 4 = "Throat"
chakraName 5 = "Third Eye"
chakraName 6 = "Crown"
chakraName _ = "Unknown"

phaseName :: SessionState -> String
phaseName SessionIdle = "IDLE"
phaseName Baseline = "BASELINE"
phaseName Alignment = "ALIGNMENT"
phaseName Entrainment = "ENTRAINMENT"
phaseName Integration = "INTEGRATION"
phaseName Complete = "COMPLETE"
