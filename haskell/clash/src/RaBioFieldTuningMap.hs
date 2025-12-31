{-|
Module      : RaBioFieldTuningMap
Description : Per-User Healing Frequency Tuning Maps
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 69: Dynamically constructs per-user tuning maps based on healing
frequency profiles and biometric input. References RIFE, Lakhovsky, and
Kaali-Beck therapeutic frequency sources.

Supports organ-specific frequencies, waveform selection, and chamber modulation.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaBioFieldTuningMap where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Schumann fundamental (7.83Hz * 256 = 2004)
schumannFreq :: Unsigned 16
schumannFreq = 2004

-- | Organ indices
data OrganType
  = OrganHeart
  | OrganLiver
  | OrganKidney
  | OrganBrain
  | OrganLungs
  | OrganBlood
  | OrganImmune
  | OrganGeneral
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Waveform types
data Waveform
  = WaveformSine      -- Gentle regulation
  | WaveformSquare    -- RIFE-pulsed delivery
  | WaveformTriangle  -- Ramping field tests
  | WaveformSawtooth  -- Asymmetric charge
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Phase lock modes
data PhaseLock
  = PhaseFree
  | PhaseSchumann
  | PhaseCardiac
  | PhaseRespiratory
  | PhaseCustom
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Chamber geometry
data ChamberForm
  = ChamberToroidal
  | ChamberDodecahedral
  | ChamberSpherical
  | ChamberPyramidal
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Chamber modulation mode
data ChamberMode
  = ModeClosedLoop   -- Modulates frequency
  | ModeReadOnly     -- Reflects phase only
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Biometric state
data BioState = BioState
  { bsHRV          :: Unsigned 8   -- 0-255 scaled
  , bsPulse        :: Unsigned 8   -- BPM
  , bsCoherence    :: Unsigned 8   -- 0-255 scaled
  , bsStressIndex  :: Unsigned 8   -- 0-255 scaled
  } deriving (Generic, NFDataX, Eq, Show)

-- | Single tuning wave
data TuningWave = TuningWave
  { twOrgan       :: OrganType
  , twFrequency   :: Unsigned 16   -- Hz * 256
  , twWaveform    :: Waveform
  , twPhaseLock   :: PhaseLock
  , twAmplitude   :: Unsigned 8    -- 0-255
  } deriving (Generic, NFDataX, Eq, Show)

-- | Get base frequency for organ (Hz * 256)
getOrganBaseFreq :: OrganType -> Unsigned 16
getOrganBaseFreq organ = case organ of
  OrganHeart   -> schumannFreq      -- 7.83 Hz
  OrganLiver   -> 10240             -- 40 Hz
  OrganKidney  -> 5120              -- 20 Hz
  OrganBrain   -> 2560              -- 10 Hz
  OrganLungs   -> 18432             -- 72 Hz
  OrganBlood   -> 396800            -- 1550 Hz (Kaali-Beck)
  OrganImmune  -> 186112            -- 727 Hz (RIFE)
  OrganGeneral -> schumannFreq

-- | Modulate frequency by biometrics
modulateFrequency :: Unsigned 16 -> BioState -> Unsigned 16
modulateFrequency baseFreq bio =
  let -- Coherence factor: 0.9 + (coherence * 0.2 / 255)
      cohFactor = 230 + (resize (bsCoherence bio) :: Unsigned 16)  -- ~0.9 to 1.1 scaled

      -- HRV stability: 0.5 + (hrv * 0.5 / 255)
      hrvFactor = 128 + (resize (bsHRV bio) `shiftR` 1 :: Unsigned 16)

      -- Stress damping: 1.0 - (stress * 0.1 / 255)
      stressFactor = 256 - (resize (bsStressIndex bio) `shiftR` 4 :: Unsigned 16)

      -- Combine factors (all scaled by 256)
      combined = (cohFactor * hrvFactor * stressFactor) `shiftR` 16

  in (baseFreq * combined) `shiftR` 8

-- | Select waveform based on organ and coherence
selectWaveform :: OrganType -> BioState -> Waveform
selectWaveform organ bio
  | bsCoherence bio > 204 = WaveformSine  -- High coherence override
  | otherwise = case organ of
      OrganHeart  -> WaveformSine
      OrganBlood  -> WaveformSquare
      OrganImmune -> WaveformSquare
      OrganBrain  -> WaveformSine
      OrganLiver  -> WaveformTriangle
      _           -> if bsCoherence bio < 102
                     then WaveformTriangle
                     else WaveformSine

-- | Select phase lock based on state and chamber
selectPhaseLock :: BioState -> ChamberForm -> PhaseLock
selectPhaseLock bio chamber
  | bsHRV bio > 179          = PhaseCardiac
  | chamber == ChamberToroidal = PhaseSchumann
  | bsStressIndex bio < 77   = PhaseRespiratory
  | otherwise                = PhaseFree

-- | Compute chamber geometry modulation factor
chamberGeometryFactor :: ChamberForm -> Unsigned 16
chamberGeometryFactor chamber = case chamber of
  ChamberToroidal     -> phi16        -- 1.618 * 1024
  ChamberDodecahedral -> 1229         -- 1.2 * 1024
  ChamberSpherical    -> 1024         -- 1.0 * 1024
  ChamberPyramidal    -> 1448         -- sqrt(2) * 1024

-- | Apply chamber modulation for closed-loop mode
applyChamberModulation
  :: Unsigned 16
  -> BioState
  -> ChamberForm
  -> Unsigned 16
applyChamberModulation freq bio chamber =
  let geoFactor = chamberGeometryFactor chamber
      alignment = (resize (bsCoherence bio) * resize (bsHRV bio)) `shiftR` 8 :: Unsigned 16
      modulation = (alignment * (geoFactor - 1024)) `shiftR` 13  -- Small adjustment
  in freq + resize modulation

-- | Compute amplitude from coherence
computeAmplitude :: BioState -> Unsigned 8
computeAmplitude bio =
  128 + (bsCoherence bio `shiftR` 1)  -- 0.5 + (coherence * 0.5)

-- | Create tuning wave
createTuningWave
  :: OrganType
  -> BioState
  -> ChamberForm
  -> ChamberMode
  -> TuningWave
createTuningWave organ bio chamber mode =
  let baseFreq = getOrganBaseFreq organ
      modFreq = modulateFrequency baseFreq bio
      finalFreq = case mode of
        ModeClosedLoop -> applyChamberModulation modFreq bio chamber
        ModeReadOnly   -> modFreq
      waveform = selectWaveform organ bio
      phaseLock = selectPhaseLock bio chamber
      amplitude = computeAmplitude bio
  in TuningWave organ finalFreq waveform phaseLock amplitude

-- | Check if frequency is in therapeutic range
isTherapeuticFreq :: Unsigned 16 -> Bool
isTherapeuticFreq freq =
  let hz = freq `shiftR` 8  -- Convert to Hz
  in (hz >= 1 && hz <= 100) ||    -- Brainwave + low
     (hz >= 100 && hz <= 1000) ||  -- RIFE low
     (hz >= 1000 && hz <= 2000)    -- RIFE high / Kaali-Beck

-- | Tuning map pipeline state
data TuningState = TuningState
  { tsCurrentOrgan :: OrganType
  , tsWaveCount    :: Unsigned 4
  } deriving (Generic, NFDataX)

-- | Initial state
initialTuningState :: TuningState
initialTuningState = TuningState OrganGeneral 0

-- | Tuning pipeline input
data TuningInput = TuningInput
  { tiOrgan   :: OrganType
  , tiBioState :: BioState
  , tiChamber :: ChamberForm
  , tiMode    :: ChamberMode
  } deriving (Generic, NFDataX)

-- | Tuning map pipeline
tuningMapPipeline
  :: HiddenClockResetEnable dom
  => Signal dom TuningInput
  -> Signal dom TuningWave
tuningMapPipeline input =
  let mkWave inp = createTuningWave
        (tiOrgan inp)
        (tiBioState inp)
        (tiChamber inp)
        (tiMode inp)
  in mkWave <$> input

-- | Frequency modulation pipeline
freqModulationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 16, BioState)
  -> Signal dom Unsigned 16
freqModulationPipeline = fmap (uncurry modulateFrequency)

-- | Waveform selection pipeline
waveformSelectPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (OrganType, BioState)
  -> Signal dom Waveform
waveformSelectPipeline = fmap (uncurry selectWaveform)
