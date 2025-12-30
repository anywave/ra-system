{-|
Module      : Ra.Resonator.MusicChamber
Description : Biometric-triggered spatial audio modulator
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Transforms biometric field resonance into spatially immersive soundscapes
inside a Ra Chamber. Provides real-time audio feedback reflecting user's
field state, harmonic alignment, and chamber configuration.

== Sound Field Design

=== Biometric Mapping

* Heart rate → rhythm modulation
* HRV → pitch variance
* Respiration → envelope timing
* Skin conductance → timbre intensity
* Coherence → harmonic consonance

=== Avatar Signatures

Each avatar has a distinct tonal signature:

* Metallic: Bright, resonant overtones
* Crystalline: Pure, bell-like tones
* Hollow: Reverberant, spacious
* Warm: Rich, fundamental-heavy
-}
module Ra.Resonator.MusicChamber
  ( -- * Core Types
    ChamberSoundField(..)
  , HarmonicScale(..)
  , TemporalEnvelope(..)
  , SoundSpatialProfile(..)

    -- * Resonance Modulation
  , ResonanceMod(..)
  , InversionState(..)
  , AvatarTone(..)

    -- * Biometric Input
  , BiometricInput(..)
  , defaultBiometric

    -- * Sound Generation
  , generateChamberSound
  , applySoundMods
  , soundToOutput

    -- * Scale Systems
  , scaleFromCoherence
  , scaleFrequencies
  , scaleRoot

    -- * Envelope Shaping
  , envelopeFromBreath
  , envelopeDuration
  , envelopePhase

    -- * Spatial Mapping
  , spatialFromField
  , stereoPosition
  , depthPosition

    -- * Avatar Timbres
  , toneToTimbre
  , timbreSpectrum
  , toneBlend

    -- * Real-Time Output
  , SoundOutput(..)
  , outputParameters
  , mixSounds
  ) where

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Chamber sound field configuration
data ChamberSoundField = ChamberSoundField
  { csfBaseScale     :: !HarmonicScale
  , csfPulseEnvelope :: !TemporalEnvelope
  , csfSpatialProfile :: !SoundSpatialProfile
  , csfModulations   :: ![ResonanceMod]
  , csfLoopDuration  :: !Double         -- ^ φ^n seconds
  } deriving (Eq, Show)

-- | Harmonic scale (based on Keely)
data HarmonicScale
  = ScalePentatonic     -- ^ 5-note resonant
  | ScaleDiatonic       -- ^ 7-note natural
  | ScaleChromatic      -- ^ 12-tone full
  | ScaleOvertone       -- ^ Natural harmonic series
  | ScaleSolfeggio      -- ^ Sacred solfeggio
  | ScaleCustom ![Double]  -- ^ Custom frequencies
  deriving (Eq, Show)

-- | Temporal envelope for sound shaping
data TemporalEnvelope = TemporalEnvelope
  { teAttack    :: !Double          -- ^ Attack time (s)
  , teDecay     :: !Double          -- ^ Decay time (s)
  , teSustain   :: !Double          -- ^ Sustain level [0, 1]
  , teRelease   :: !Double          -- ^ Release time (s)
  , teBreathMod :: !Double          -- ^ Breath modulation depth
  } deriving (Eq, Show)

-- | Spatial sound profile
data SoundSpatialProfile = SoundSpatialProfile
  { sspPan       :: !Double          -- ^ Stereo position [-1, 1]
  , sspDepth     :: !Double          -- ^ Distance/reverb [0, 1]
  , sspHeight    :: !Double          -- ^ Vertical position [-1, 1]
  , sspRotation  :: !Double          -- ^ Rotation angle [0, 2*pi]
  , sspSpread    :: !Double          -- ^ Spatial width [0, 1]
  } deriving (Eq, Show)

-- =============================================================================
-- Resonance Modulation
-- =============================================================================

-- | Resonance modulation types
data ResonanceMod
  = CoherencePulse !Double           -- ^ Modulates rhythm
  | FluxDrift !Double                -- ^ Affects pitch bend
  | InversionShift !InversionState   -- ^ Flips waveform
  | AvatarOverlay !AvatarTone        -- ^ Timbral signature
  deriving (Eq, Show)

-- | Inversion state
data InversionState
  = Normal          -- ^ Standard polarity
  | Inverted        -- ^ Inverted polarity
  | Transitioning   -- ^ Between states
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Avatar tonal signature
data AvatarTone
  = ToneMetallic     -- ^ Bright, resonant overtones
  | ToneCrystalline  -- ^ Pure, bell-like
  | ToneHollow       -- ^ Reverberant, spacious
  | ToneWarm         -- ^ Rich, fundamental-heavy
  | ToneEthereal     -- ^ Airy, high harmonics
  | ToneGrounded     -- ^ Deep, bass-focused
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Biometric Input
-- =============================================================================

-- | Biometric input data
data BiometricInput = BiometricInput
  { biHeartRate       :: !Double      -- ^ BPM
  , biHrv             :: !Double      -- ^ Heart rate variability
  , biRespiration     :: !Double      -- ^ Breaths per minute
  , biSkinConductance :: !Double      -- ^ Microsiemens
  , biCoherence       :: !Double      -- ^ Coherence [0, 1]
  } deriving (Eq, Show)

-- | Default biometric values
defaultBiometric :: BiometricInput
defaultBiometric = BiometricInput
  { biHeartRate = 70.0
  , biHrv = 50.0
  , biRespiration = 12.0
  , biSkinConductance = 2.0
  , biCoherence = 0.5
  }

-- =============================================================================
-- Sound Generation
-- =============================================================================

-- | Generate chamber sound from inputs
generateChamberSound :: BiometricInput -> ScalarField -> AvatarProfile -> ChamberConfig -> ChamberSoundField
generateChamberSound bio field avatar chamber =
  let -- Scale from coherence
      scale = scaleFromCoherence (biCoherence bio) (ccBaseFreq chamber)

      -- Envelope from breath
      envelope = envelopeFromBreath (biRespiration bio) (biHrv bio)

      -- Spatial from field
      spatial = spatialFromField field

      -- Modulations
      mods = buildModulations bio field avatar

      -- Loop duration follows φ^n
      loopN = round (biCoherence bio * 4) :: Int
      loopDur = phi ** fromIntegral loopN

  in ChamberSoundField
      { csfBaseScale = scale
      , csfPulseEnvelope = envelope
      , csfSpatialProfile = spatial
      , csfModulations = mods
      , csfLoopDuration = loopDur
      }

-- Build modulation list
buildModulations :: BiometricInput -> ScalarField -> AvatarProfile -> [ResonanceMod]
buildModulations bio field avatar =
  [ CoherencePulse (biCoherence bio)
  , FluxDrift (sfFlux field * 0.5)
  , InversionShift (if sfInverted field then Inverted else Normal)
  , AvatarOverlay (apTone avatar)
  ]

-- | Apply modulations to sound field
applySoundMods :: [ResonanceMod] -> ChamberSoundField -> ChamberSoundField
applySoundMods mods field =
  foldr applyMod field mods
  where
    applyMod :: ResonanceMod -> ChamberSoundField -> ChamberSoundField
    applyMod (CoherencePulse c) f =
      let env = csfPulseEnvelope f
          newEnv = env { teSustain = teSustain env * c }
      in f { csfPulseEnvelope = newEnv }
    applyMod (FluxDrift d) f =
      let sp = csfSpatialProfile f
          newSp = sp { sspRotation = sspRotation sp + d }
      in f { csfSpatialProfile = newSp }
    applyMod (InversionShift Inverted) f =
      -- Add timbre shift for inversion
      f { csfModulations = AvatarOverlay ToneHollow : csfModulations f }
    applyMod _ f = f

-- | Convert sound field to output
soundToOutput :: ChamberSoundField -> SoundOutput
soundToOutput field =
  let freqs = scaleFrequencies (csfBaseScale field)
      env = csfPulseEnvelope field
      spatial = csfSpatialProfile field
  in SoundOutput
      { soFrequencies = freqs
      , soAmplitude = teSustain env
      , soPan = sspPan spatial
      , soReverb = sspDepth spatial
      , soDuration = csfLoopDuration field
      }

-- =============================================================================
-- Scale Systems
-- =============================================================================

-- | Select scale based on coherence
scaleFromCoherence :: Double -> Double -> HarmonicScale
scaleFromCoherence coherence _baseFreq
  | coherence >= 0.85 = ScaleSolfeggio
  | coherence >= 0.7 = ScaleOvertone
  | coherence >= 0.5 = ScaleDiatonic
  | coherence >= 0.3 = ScalePentatonic
  | otherwise = ScalePentatonic

-- | Get frequencies for scale (in Hz)
scaleFrequencies :: HarmonicScale -> [Double]
scaleFrequencies scale = case scale of
  ScalePentatonic ->
    [256.0, 288.0, 320.0, 384.0, 426.67]  -- C D E G A

  ScaleDiatonic ->
    [256.0, 288.0, 320.0, 341.33, 384.0, 426.67, 480.0]  -- C D E F G A B

  ScaleChromatic ->
    [256.0 * (2 ** (fromIntegral n / 12)) | n <- [0..11 :: Int]]

  ScaleOvertone ->
    [256.0 * fromIntegral n | n <- [1..8 :: Int]]  -- Natural harmonics

  ScaleSolfeggio ->
    [174.0, 285.0, 396.0, 417.0, 528.0, 639.0, 741.0, 852.0, 963.0]

  ScaleCustom freqs -> freqs

-- | Get root frequency
scaleRoot :: HarmonicScale -> Double
scaleRoot scale = head (scaleFrequencies scale)

-- =============================================================================
-- Envelope Shaping
-- =============================================================================

-- | Create envelope from breath data
envelopeFromBreath :: Double -> Double -> TemporalEnvelope
envelopeFromBreath breathRate hrv =
  let -- Breath cycle time
      cycleTime = 60.0 / breathRate

      -- Attack/release follow breath phase
      attack = cycleTime * 0.3
      release = cycleTime * 0.4

      -- HRV modulates sustain variation
      hrvMod = hrv / 100.0

  in TemporalEnvelope
      { teAttack = attack
      , teDecay = cycleTime * 0.1
      , teSustain = 0.7 + hrvMod * 0.2
      , teRelease = release
      , teBreathMod = hrvMod
      }

-- | Get envelope total duration
envelopeDuration :: TemporalEnvelope -> Double
envelopeDuration env =
  teAttack env + teDecay env + teRelease env

-- | Get current envelope phase
envelopePhase :: TemporalEnvelope -> Double -> Double
envelopePhase env time =
  let total = envelopeDuration env
      phase = (time / total) * 2 * pi
  in phase

-- =============================================================================
-- Spatial Mapping
-- =============================================================================

-- | Create spatial profile from field
spatialFromField :: ScalarField -> SoundSpatialProfile
spatialFromField field =
  let -- Pan from field phase
      pan' = sin (sfPhase field)

      -- Depth from coherence
      depth = 1.0 - sfCoherence field

      -- Height from harmonic level
      height = (fromIntegral (sfHarmonic field) - 3.0) / 3.0

      -- Rotation from phase
      rotation = sfPhase field

      -- Spread from flux
      spread = sfFlux field

  in SoundSpatialProfile
      { sspPan = pan'
      , sspDepth = depth
      , sspHeight = clamp11 height
      , sspRotation = rotation
      , sspSpread = spread
      }

-- | Get stereo position
stereoPosition :: SoundSpatialProfile -> (Double, Double)
stereoPosition sp =
  let pan' = sspPan sp
      left = (1.0 - pan') / 2.0
      right = (1.0 + pan') / 2.0
  in (left, right)

-- | Get depth position
depthPosition :: SoundSpatialProfile -> Double
depthPosition = sspDepth

-- =============================================================================
-- Avatar Timbres
-- =============================================================================

-- | Convert avatar tone to timbre parameters
toneToTimbre :: AvatarTone -> TimbreParams
toneToTimbre tone = case tone of
  ToneMetallic -> TimbreParams
    { tpBrightness = 0.9
    , tpRichness = 0.7
    , tpAttackSharp = 0.8
    , tpDecayLong = 0.9
    , tpHarmonicBias = 2.0
    }
  ToneCrystalline -> TimbreParams
    { tpBrightness = 1.0
    , tpRichness = 0.3
    , tpAttackSharp = 0.9
    , tpDecayLong = 0.6
    , tpHarmonicBias = 3.0
    }
  ToneHollow -> TimbreParams
    { tpBrightness = 0.4
    , tpRichness = 0.8
    , tpAttackSharp = 0.3
    , tpDecayLong = 1.0
    , tpHarmonicBias = 0.5
    }
  ToneWarm -> TimbreParams
    { tpBrightness = 0.5
    , tpRichness = 1.0
    , tpAttackSharp = 0.4
    , tpDecayLong = 0.7
    , tpHarmonicBias = 1.0
    }
  ToneEthereal -> TimbreParams
    { tpBrightness = 0.8
    , tpRichness = 0.4
    , tpAttackSharp = 0.2
    , tpDecayLong = 0.8
    , tpHarmonicBias = 4.0
    }
  ToneGrounded -> TimbreParams
    { tpBrightness = 0.3
    , tpRichness = 0.9
    , tpAttackSharp = 0.5
    , tpDecayLong = 0.6
    , tpHarmonicBias = 0.3
    }

-- | Get harmonic spectrum for timbre
timbreSpectrum :: TimbreParams -> [Double]
timbreSpectrum tp =
  let bias = tpHarmonicBias tp
      richness = tpRichness tp
  in [ richness / (fromIntegral n ** (1.0 / bias))
     | n <- [1..8 :: Int]
     ]

-- | Blend two avatar tones
toneBlend :: AvatarTone -> AvatarTone -> Double -> TimbreParams
toneBlend t1 t2 blend =
  let p1 = toneToTimbre t1
      p2 = toneToTimbre t2
      lerp a b x = a + (b - a) * x
  in TimbreParams
      { tpBrightness = lerp (tpBrightness p1) (tpBrightness p2) blend
      , tpRichness = lerp (tpRichness p1) (tpRichness p2) blend
      , tpAttackSharp = lerp (tpAttackSharp p1) (tpAttackSharp p2) blend
      , tpDecayLong = lerp (tpDecayLong p1) (tpDecayLong p2) blend
      , tpHarmonicBias = lerp (tpHarmonicBias p1) (tpHarmonicBias p2) blend
      }

-- | Timbre parameters
data TimbreParams = TimbreParams
  { tpBrightness    :: !Double      -- ^ High frequency content [0, 1]
  , tpRichness      :: !Double      -- ^ Harmonic richness [0, 1]
  , tpAttackSharp   :: !Double      -- ^ Attack sharpness [0, 1]
  , tpDecayLong     :: !Double      -- ^ Decay length [0, 1]
  , tpHarmonicBias  :: !Double      -- ^ Harmonic bias (1 = equal, >1 = higher)
  } deriving (Eq, Show)

-- =============================================================================
-- Real-Time Output
-- =============================================================================

-- | Sound output for synthesis
data SoundOutput = SoundOutput
  { soFrequencies :: ![Double]      -- ^ Active frequencies
  , soAmplitude   :: !Double        -- ^ Master amplitude [0, 1]
  , soPan         :: !Double        -- ^ Stereo pan [-1, 1]
  , soReverb      :: !Double        -- ^ Reverb amount [0, 1]
  , soDuration    :: !Double        -- ^ Duration (s)
  } deriving (Eq, Show)

-- | Get output parameters as tuple
outputParameters :: SoundOutput -> (Double, Double, Double)
outputParameters out = (soAmplitude out, soPan out, soReverb out)

-- | Mix multiple sound outputs
mixSounds :: [SoundOutput] -> SoundOutput
mixSounds [] = SoundOutput [] 0.0 0.0 0.0 0.0
mixSounds sounds =
  let allFreqs = concatMap soFrequencies sounds
      avgAmp = sum (map soAmplitude sounds) / fromIntegral (length sounds)
      avgPan = sum (map soPan sounds) / fromIntegral (length sounds)
      avgReverb = sum (map soReverb sounds) / fromIntegral (length sounds)
      maxDur = maximum (map soDuration sounds)
  in SoundOutput
      { soFrequencies = allFreqs
      , soAmplitude = avgAmp
      , soPan = avgPan
      , soReverb = avgReverb
      , soDuration = maxDur
      }

-- =============================================================================
-- Internal Types
-- =============================================================================

-- | Scalar field (simplified)
data ScalarField = ScalarField
  { sfCoherence :: !Double
  , sfFlux      :: !Double
  , sfPhase     :: !Double
  , sfHarmonic  :: !Int
  , sfInverted  :: !Bool
  } deriving (Eq, Show)

-- | Avatar profile (simplified)
data AvatarProfile = AvatarProfile
  { apId    :: !String
  , apTone  :: !AvatarTone
  } deriving (Eq, Show)

-- | Chamber config (simplified)
data ChamberConfig = ChamberConfig
  { ccBaseFreq :: !Double
  } deriving (Eq, Show)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp to [-1, 1]
clamp11 :: Double -> Double
clamp11 x = max (-1.0) (min 1.0 x)
