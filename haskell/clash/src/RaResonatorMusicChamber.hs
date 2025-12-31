{-|
Module      : RaResonatorMusicChamber
Description : Biometric-Driven Music Chamber Sound Field
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 47: A biometric-driven music/sound chamber that maps coherence
to rhythm/BPM, uses φ^(1+coherence) loop timing, applies avatar tones,
and implements inversion state (waveform polarity + stereo flip).
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaResonatorMusicChamber where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | BPM constants
bpmBase :: Unsigned 8
bpmBase = 60

bpmRange :: Unsigned 8
bpmRange = 80

-- | Biometric input ranges (for normalization)
heartRateMin :: Unsigned 8
heartRateMin = 50

heartRateMax :: Unsigned 8
heartRateMax = 140

hrvMax :: Unsigned 8
hrvMax = 100

respirationMin :: Unsigned 8
respirationMin = 6

respirationMax :: Unsigned 8
respirationMax = 18

-- | Harmonic scale types
data HarmonicScale
  = PhiMajor
  | Root10Minor
  | CustomScale
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Sound spatial profiles
data SoundSpatialProfile
  = StereoSwirl
  | VortexSpiral
  | CenterPulse
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Inversion state
data InversionState
  = InvNormal
  | InvInverted
  deriving (Generic, NFDataX, Eq, Show)

-- | Avatar timbre tones
data AvatarTone
  = Metallic
  | Crystalline
  | Hollow
  | Warm
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Resonance modulation type
data ResonanceModType
  = CoherencePulse
  | FluxDrift
  | InversionShift
  | AvatarOverlay
  deriving (Generic, NFDataX, Eq, Show)

-- | Biometric input (fixed-point scaled)
data BiometricInput = BiometricInput
  { biHeartRate       :: Unsigned 8   -- 50-140 BPM
  , biHrv             :: Unsigned 8   -- 0-100 ms
  , biRespiration     :: Unsigned 8   -- 6-18 rpm
  , biSkinConductance :: Unsigned 8   -- 0-255 (0-1 scaled)
  , biCoherence       :: Unsigned 8   -- 0-255 (0-1 scaled)
  } deriving (Generic, NFDataX)

-- | Temporal envelope
data TemporalEnvelope = TemporalEnvelope
  { teBpm          :: Unsigned 8      -- Beats per minute
  , teLoopDuration :: Unsigned 16     -- Loop duration in ticks
  } deriving (Generic, NFDataX)

-- | Avatar profile
data AvatarProfile = AvatarProfile
  { apTone   :: AvatarTone
  , apInvert :: Bool
  } deriving (Generic, NFDataX)

-- | Chamber configuration
data ChamberConfig = ChamberConfig
  { ccScale   :: HarmonicScale
  , ccSpatial :: SoundSpatialProfile
  } deriving (Generic, NFDataX)

-- | Chamber sound field output
data ChamberSoundField = ChamberSoundField
  { csfScale           :: HarmonicScale
  , csfEnvelope        :: TemporalEnvelope
  , csfSpatial         :: SoundSpatialProfile
  , csfAvatarTone      :: AvatarTone
  , csfCoherencePulse  :: Unsigned 8
  , csfFluxDrift       :: Unsigned 8
  , csfWaveformInverted :: Bool
  , csfStereoInverted  :: Bool
  } deriving (Generic, NFDataX)

-- | Diagnostic frame
data DiagnosticFrame = DiagnosticFrame
  { dfBpmOut          :: Unsigned 8
  , dfLoopOut         :: Unsigned 16
  , dfAvatarTone      :: AvatarTone
  , dfInversionActive :: Bool
  } deriving (Generic, NFDataX)

-- | Phi power lookup table for loop timing
-- φ^(1+n/8) for n=0..8 (0..1 coherence range)
-- Scaled to 16-bit ticks (multiply by ~100 for sub-second resolution)
phiLoopROM :: Vec 9 (Unsigned 16)
phiLoopROM = $(listToVecTH
  [ 162   -- φ^1.00 * 100 = 161.8
  , 171   -- φ^1.125 * 100
  , 180   -- φ^1.25 * 100
  , 190   -- φ^1.375 * 100
  , 200   -- φ^1.50 * 100
  , 211   -- φ^1.625 * 100
  , 223   -- φ^1.75 * 100
  , 235   -- φ^1.875 * 100
  , 262   -- φ^2.00 * 100 = 261.8
  ])

-- | Compute BPM from coherence
-- BPM = 60 + coherence * 80 / 255
computeBpm :: Unsigned 8 -> Unsigned 8
computeBpm coherence =
  let scaled = (resize coherence * resize bpmRange) `shiftR` 8 :: Unsigned 16
  in bpmBase + resize scaled

-- | Compute phi loop duration from coherence
-- Uses ROM lookup for φ^(1 + coherence/255)
computePhiLoop :: Unsigned 8 -> Unsigned 16
computePhiLoop coherence =
  let idx = coherence `shiftR` 5  -- Map 0-255 to 0-7
  in phiLoopROM !! resize (min 8 idx)

-- | Normalize HRV to 0-255 range
normalizeHrv :: Unsigned 8 -> Unsigned 8
normalizeHrv hrv = if hrv > hrvMax then 255 else (hrv * 255) `div` hrvMax

-- | Normalize respiration to 0-255 range
normalizeRespiration :: Unsigned 8 -> Unsigned 8
normalizeRespiration resp
  | resp < respirationMin = 0
  | resp > respirationMax = 255
  | otherwise = ((resp - respirationMin) * 255) `div` (respirationMax - respirationMin)

-- | Clamp biometric input to valid ranges
clampBiometricInput :: BiometricInput -> BiometricInput
clampBiometricInput bio = BiometricInput
  { biHeartRate = max heartRateMin (min heartRateMax (biHeartRate bio))
  , biHrv = min hrvMax (biHrv bio)
  , biRespiration = max respirationMin (min respirationMax (biRespiration bio))
  , biSkinConductance = biSkinConductance bio
  , biCoherence = biCoherence bio
  }

-- | Apply waveform inversion
applyWaveformInversion :: Signed 16 -> Bool -> Signed 16
applyWaveformInversion sample inverted = if inverted then -sample else sample

-- | Apply stereo inversion (swap channels)
applyStereoInversion :: (Signed 16, Signed 16) -> Bool -> (Signed 16, Signed 16)
applyStereoInversion (left, right) inverted =
  if inverted then (right, left) else (left, right)

-- | Generate chamber sound field
generateChamberSound
  :: BiometricInput
  -> AvatarProfile
  -> ChamberConfig
  -> ChamberSoundField
generateChamberSound bio avatar config =
  let clamped = clampBiometricInput bio
      bpm = computeBpm (biCoherence clamped)
      loopDur = computePhiLoop (biCoherence clamped)
      envelope = TemporalEnvelope bpm loopDur
      flux = normalizeHrv (biHrv clamped)
  in ChamberSoundField
    { csfScale = ccScale config
    , csfEnvelope = envelope
    , csfSpatial = ccSpatial config
    , csfAvatarTone = apTone avatar
    , csfCoherencePulse = biCoherence clamped
    , csfFluxDrift = flux
    , csfWaveformInverted = apInvert avatar
    , csfStereoInverted = apInvert avatar
    }

-- | Extract tempo from sound field
extractTempo :: ChamberSoundField -> Unsigned 8
extractTempo = teBpm . csfEnvelope

-- | Extract loop duration from sound field
extractLoopDuration :: ChamberSoundField -> Unsigned 16
extractLoopDuration = teLoopDuration . csfEnvelope

-- | Check if waveform is inverted
isWaveformInverted :: ChamberSoundField -> Bool
isWaveformInverted = csfWaveformInverted

-- | Check if stereo is inverted
isStereoInverted :: ChamberSoundField -> Bool
isStereoInverted = csfStereoInverted

-- | Get avatar tone from sound field
getAvatarTone :: ChamberSoundField -> AvatarTone
getAvatarTone = csfAvatarTone

-- | Simulate diagnostics
simulateDiagnostics
  :: BiometricInput
  -> AvatarProfile
  -> ChamberConfig
  -> DiagnosticFrame
simulateDiagnostics bio avatar config =
  let sf = generateChamberSound bio avatar config
  in DiagnosticFrame
    { dfBpmOut = extractTempo sf
    , dfLoopOut = extractLoopDuration sf
    , dfAvatarTone = getAvatarTone sf
    , dfInversionActive = apInvert avatar
    }

-- | BPM computation pipeline
bpmPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom (Unsigned 8)
bpmPipeline = fmap computeBpm

-- | Phi loop pipeline
phiLoopPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom (Unsigned 16)
phiLoopPipeline = fmap computePhiLoop

-- | Sound field generation pipeline
soundFieldPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BiometricInput, AvatarProfile, ChamberConfig)
  -> Signal dom ChamberSoundField
soundFieldPipeline input = (\(b, a, c) -> generateChamberSound b a c) <$> input

-- | Waveform inversion pipeline
waveformInversionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Signed 16, Bool)
  -> Signal dom (Signed 16)
waveformInversionPipeline input = uncurry applyWaveformInversion <$> input

-- | Stereo inversion pipeline
stereoInversionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom ((Signed 16, Signed 16), Bool)
  -> Signal dom (Signed 16, Signed 16)
stereoInversionPipeline input = uncurry applyStereoInversion <$> input

-- | Diagnostic pipeline
diagnosticPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BiometricInput, AvatarProfile, ChamberConfig)
  -> Signal dom DiagnosticFrame
diagnosticPipeline input = (\(b, a, c) -> simulateDiagnostics b a c) <$> input
