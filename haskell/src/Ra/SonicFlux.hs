{-|
Module      : Ra.SonicFlux
Description : Real-time harmonic sonification of scalar emergence
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Transforms scalar emergence events into rich harmonic sound signatures,
enabling experiential feedback loops for coherence training, holographic
tuning, and musical synthesis from fragment interactions.

== Sonification Principles

=== Coordinate to Audio Mapping

* theta, phi → Spatialization (L/R pan, 3D positioning)
* h, radius → Pitch base and modulation depth
* harmonic (l, m) → Timbre spectrum shape (overtone profiles)
* alpha → Volume and brightness
* Inversion → Phase inversion and detuning
* TemporalPhase → LFO modulation and beat sync

=== Keely Sympathetic Influence

Based on Keely's sympathetic vibratory physics:
* Additive synthesis using overtone series
* Triple structure: fundamental, overtone, undertone
* 21-octave frequency bands mapped to emergence states

=== Output Channels

SonicFlux outputs can be:
* Streamed as JSON to external synthesizers (SuperCollider, Max/MSP)
* Converted to Ra.Music notation
* Fed to binaural beat generators
* Used for haptic vibrotactile feedback
-}
module Ra.SonicFlux
  ( -- * Core Output Types
    SonicOutput(..)
  , EnvelopeKind(..)
  , TimbreProfile(..)
  , mkSonicOutput

    -- * Emergence Context
  , EmergenceContext(..)
  , mkEmergenceContext
  , contextCoherence

    -- * Sonification Functions
  , sonifyEmergence
  , mapPitch
  , mapAmplitude
  , mapPan
  , mapTimbre

    -- * Harmonic Mapping
  , HarmonicMode(..)
  , harmonicTimbre
  , overtoneWeights
  , keelyOctave

    -- * Temporal Modulation
  , TemporalMod(..)
  , temporalModulation
  , lfoRate
  , beatSync

    -- * Spatial Positioning
  , SpatialPosition(..)
  , sphericalToPan
  , panToStereo
  , pan3D

    -- * Phase and Inversion
  , PhaseState(..)
  , inversionToPhase
  , phaseDistortion
  , detuneAmount

    -- * Stream Generation
  , SonicStream(..)
  , StreamConfig(..)
  , initStream
  , pushEmergence
  , streamToJSON

    -- * Integration
  , SonicProfile(..)
  , profileForChamber
  , profileForAvatar
  , blendProfiles
  ) where

import Ra.Constants.Extended
  ( phiInverse )

-- =============================================================================
-- Core Output Types
-- =============================================================================

-- | Complete sonic output for an emergence event
data SonicOutput = SonicOutput
  { soFrequency      :: !Double           -- ^ Base frequency in Hz
  , soAmplitude      :: !Double           -- ^ Normalized amplitude [0,1]
  , soPanPosition    :: !(Double, Double) -- ^ Stereo or spatial coordinates
  , soTimbreProfile  :: !TimbreProfile    -- ^ Harmonic series weights
  , soPhaseMod       :: !(Maybe Double)   -- ^ Optional phase distortion
  , soEnvelopeType   :: !EnvelopeKind     -- ^ Envelope shape
  , soSourceFragment :: !(Maybe String)   -- ^ Origin fragment ID
  , soDuration       :: !Double           -- ^ Note duration in seconds
  } deriving (Eq, Show)

-- | Envelope shape for amplitude modulation
data EnvelopeKind
  = Smooth        -- ^ Gradual attack and release
  | Sharp         -- ^ Quick attack, quick release
  | Pulse         -- ^ Staccato, rhythmic
  | Pad           -- ^ Long sustain, ambient
  | Percussive    -- ^ Sharp attack, fast decay
  | Reverse       -- ^ Reversed envelope (swell)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Timbre profile as overtone weights
data TimbreProfile = TimbreProfile
  { tpFundamental  :: !Double      -- ^ Fundamental weight
  , tpOvertones    :: ![Double]    -- ^ Overtone series weights (up to 16)
  , tpUndertones   :: ![Double]    -- ^ Undertone weights (Keely influence)
  , tpBrightness   :: !Double      -- ^ Overall brightness [0,1]
  , tpRichness     :: !Double      -- ^ Harmonic richness [0,1]
  } deriving (Eq, Show)

-- | Create default sonic output
mkSonicOutput :: Double -> Double -> SonicOutput
mkSonicOutput freq amp = SonicOutput
  { soFrequency = freq
  , soAmplitude = clamp01 amp
  , soPanPosition = (0.5, 0.5)  -- Center
  , soTimbreProfile = defaultTimbre
  , soPhaseMod = Nothing
  , soEnvelopeType = Smooth
  , soSourceFragment = Nothing
  , soDuration = 1.0
  }

-- | Default timbre profile (sine-like)
defaultTimbre :: TimbreProfile
defaultTimbre = TimbreProfile
  { tpFundamental = 1.0
  , tpOvertones = [0.5, 0.25, 0.125, 0.0625]
  , tpUndertones = []
  , tpBrightness = 0.5
  , tpRichness = 0.3
  }

-- =============================================================================
-- Emergence Context
-- =============================================================================

-- | Context for sonifying an emergence event
data EmergenceContext = EmergenceContext
  { ecTheta       :: !Double      -- ^ Spherical theta [0, pi]
  , ecPhi         :: !Double      -- ^ Spherical phi [0, 2*pi]
  , ecRadius      :: !Double      -- ^ Radial distance [0, 1]
  , ecHarmonicL   :: !Int         -- ^ Harmonic degree l
  , ecHarmonicM   :: !Int         -- ^ Harmonic order m
  , ecAlpha       :: !Double      -- ^ Emergence alpha [0, 1]
  , ecInversion   :: !Bool        -- ^ Is inverted/shadow?
  , ecTemporalPhase :: !Double    -- ^ Temporal phase [0, 2*pi]
  , ecChamberType :: !(Maybe String) -- ^ Chamber type
  , ecAvatarId    :: !(Maybe String) -- ^ Avatar identifier
  , ecFragmentId  :: !(Maybe String) -- ^ Source fragment
  } deriving (Eq, Show)

-- | Create emergence context from parameters
mkEmergenceContext :: Double -> Double -> Double -> (Int, Int) -> Double -> EmergenceContext
mkEmergenceContext theta anglePhi radius (l, m) alpha = EmergenceContext
  { ecTheta = theta
  , ecPhi = anglePhi
  , ecRadius = radius
  , ecHarmonicL = l
  , ecHarmonicM = m
  , ecAlpha = alpha
  , ecInversion = False
  , ecTemporalPhase = 0.0
  , ecChamberType = Nothing
  , ecAvatarId = Nothing
  , ecFragmentId = Nothing
  }

-- | Get effective coherence from context
contextCoherence :: EmergenceContext -> Double
contextCoherence ec = ecAlpha ec * (1.0 - ecRadius ec * 0.3)

-- =============================================================================
-- Sonification Functions
-- =============================================================================

-- | Main sonification entry point
sonifyEmergence :: EmergenceContext -> SonicOutput
sonifyEmergence ctx =
  let -- Map coordinate to pitch
      freq = mapPitch ctx

      -- Map alpha to amplitude
      amp = mapAmplitude ctx

      -- Map angles to pan
      pan = mapPan ctx

      -- Generate timbre from harmonics
      timbre = mapTimbre ctx

      -- Handle inversion
      phaseMod = if ecInversion ctx
                 then Just (phaseDistortion ctx)
                 else Nothing

      -- Select envelope based on alpha
      envelope = selectEnvelope (ecAlpha ctx)

      -- Duration from temporal phase
      duration = 0.5 + ecAlpha ctx * 1.5  -- 0.5-2.0 seconds
  in SonicOutput
      { soFrequency = freq
      , soAmplitude = amp
      , soPanPosition = pan
      , soTimbreProfile = timbre
      , soPhaseMod = phaseMod
      , soEnvelopeType = envelope
      , soSourceFragment = ecFragmentId ctx
      , soDuration = duration
      }

-- | Map emergence coordinate to pitch
mapPitch :: EmergenceContext -> Double
mapPitch ctx =
  let -- Base frequency from radius (low radius = low pitch)
      baseFreq = 110.0 + ecRadius ctx * 880.0  -- A2 to A5 range

      -- Harmonic modification
      harmonicMod = fromIntegral (ecHarmonicL ctx) * 0.1 + 1.0

      -- Theta influence (higher theta = brighter)
      thetaMod = 1.0 + (ecTheta ctx / pi) * 0.2

      -- Phi influence (subtle detune for spatial width)
      phiMod = 1.0 + sin (ecPhi ctx) * 0.01
  in baseFreq * harmonicMod * thetaMod * phiMod

-- | Map alpha to amplitude
mapAmplitude :: EmergenceContext -> Double
mapAmplitude ctx =
  let baseAmp = ecAlpha ctx
      -- Boost near-threshold emergences slightly
      boosted = if baseAmp > 0.3 && baseAmp < 0.7
                then baseAmp * 1.1
                else baseAmp
      -- Inversion slightly reduces amplitude
      invMod = if ecInversion ctx then 0.8 else 1.0
  in clamp01 (boosted * invMod)

-- | Map spherical coordinates to stereo pan
mapPan :: EmergenceContext -> (Double, Double)
mapPan ctx = sphericalToPan (ecTheta ctx) (ecPhi ctx)

-- | Map harmonic mode to timbre profile
mapTimbre :: EmergenceContext -> TimbreProfile
mapTimbre ctx = harmonicTimbre (HarmonicMode (ecHarmonicL ctx) (ecHarmonicM ctx))

-- Select envelope based on alpha value
selectEnvelope :: Double -> EnvelopeKind
selectEnvelope alpha
  | alpha < 0.2 = Pulse
  | alpha < 0.4 = Percussive
  | alpha < 0.6 = Sharp
  | alpha < 0.8 = Smooth
  | otherwise = Pad

-- =============================================================================
-- Harmonic Mapping
-- =============================================================================

-- | Harmonic mode (l, m)
data HarmonicMode = HarmonicMode !Int !Int
  deriving (Eq, Show)

-- | Generate timbre profile from harmonic mode
harmonicTimbre :: HarmonicMode -> TimbreProfile
harmonicTimbre (HarmonicMode l m) =
  let -- Fundamental strength inversely related to l
      fundamental = 1.0 - fromIntegral l * 0.08

      -- Overtones shaped by l and m
      overtones = overtoneWeights l m

      -- Undertones for Keely influence (only for low m)
      undertones = if abs m < 3
                   then [0.2, 0.1]
                   else []

      -- Brightness from m (higher |m| = brighter)
      brightness = clamp01 (0.3 + fromIntegral (abs m) * 0.1)

      -- Richness from l (higher l = richer)
      richness = clamp01 (0.2 + fromIntegral l * 0.1)
  in TimbreProfile
      { tpFundamental = fundamental
      , tpOvertones = overtones
      , tpUndertones = undertones
      , tpBrightness = brightness
      , tpRichness = richness
      }

-- | Calculate overtone weights for harmonic mode
overtoneWeights :: Int -> Int -> [Double]
overtoneWeights l m =
  let -- Base decay rate
      decay = 0.5 + fromIntegral l * 0.05

      -- Generate 8 overtones
      harmonics = take 8 $ map (\n -> (1.0 / fromIntegral (n :: Int)) * decay) [2..]

      -- Modulate by m (odd harmonics emphasized for positive m)
      modulated = zipWith modByM harmonics ([2..] :: [Int])
      modByM h n = if m > 0 && odd n then h * 1.2 else h
  in modulated

-- | Map harmonic to Keely's 21-octave system
keelyOctave :: Int -> Int -> Int
keelyOctave l m =
  let baseOctave = (l `mod` 7) + 1  -- 1-7
      mOffset = (m + 7) `mod` 3     -- 0-2 (sub-octave offset)
  in baseOctave * 3 + mOffset       -- 3-21 range

-- =============================================================================
-- Temporal Modulation
-- =============================================================================

-- | Temporal modulation parameters
data TemporalMod = TemporalMod
  { tmLFORate     :: !Double      -- ^ LFO rate in Hz
  , tmLFODepth    :: !Double      -- ^ Modulation depth [0,1]
  , tmBeatDivision :: !Int        -- ^ Beat subdivision (1, 2, 4, 8...)
  , tmPhaseOffset :: !Double      -- ^ Phase offset [0, 2*pi]
  } deriving (Eq, Show)

-- | Compute temporal modulation from emergence phase
temporalModulation :: Double -> TemporalMod
temporalModulation phase =
  let -- LFO rate increases with phase
      rate = 0.5 + phase / (2 * pi) * 4.0  -- 0.5-4.5 Hz

      -- Depth from phi relationship
      depth = phiInverse * sin phase * 0.5 + 0.5

      -- Beat division from phase quadrant
      division = 2 ^ (floor (phase / (pi / 2)) `mod` 3 :: Int)
  in TemporalMod
      { tmLFORate = rate
      , tmLFODepth = depth
      , tmBeatDivision = division
      , tmPhaseOffset = phase
      }

-- | Get LFO rate from context
lfoRate :: EmergenceContext -> Double
lfoRate ctx = tmLFORate (temporalModulation (ecTemporalPhase ctx))

-- | Compute beat sync value
beatSync :: Double -> Int -> Double
beatSync bpm division =
  let beatDuration = 60.0 / bpm
      subdivided = beatDuration / fromIntegral division
  in subdivided

-- =============================================================================
-- Spatial Positioning
-- =============================================================================

-- | 3D spatial position
data SpatialPosition = SpatialPosition
  { spX :: !Double    -- ^ Left-right (-1 to 1)
  , spY :: !Double    -- ^ Front-back (-1 to 1)
  , spZ :: !Double    -- ^ Down-up (-1 to 1)
  } deriving (Eq, Show)

-- | Convert spherical coordinates to stereo pan
sphericalToPan :: Double -> Double -> (Double, Double)
sphericalToPan theta anglePhi =
  let -- Phi controls left-right
      lr = (cos anglePhi + 1.0) / 2.0  -- 0 = left, 1 = right

      -- Theta controls front-back (used as second coordinate)
      fb = (cos theta + 1.0) / 2.0     -- 0 = back, 1 = front
  in (lr, fb)

-- | Convert pan position to simple stereo
panToStereo :: (Double, Double) -> (Double, Double)
panToStereo (lr, _) =
  let left = 1.0 - lr
      right = lr
  in (left, right)

-- | Full 3D positioning
pan3D :: Double -> Double -> Double -> SpatialPosition
pan3D theta anglePhi radius =
  let x = sin theta * cos anglePhi * radius
      y = sin theta * sin anglePhi * radius
      z = cos theta * radius
  in SpatialPosition x y z

-- =============================================================================
-- Phase and Inversion
-- =============================================================================

-- | Phase state for inversion handling
data PhaseState = PhaseState
  { psPhase      :: !Double       -- ^ Current phase [0, 2*pi]
  , psInverted   :: !Bool         -- ^ Is inverted?
  , psDistortion :: !Double       -- ^ Distortion amount [0,1]
  , psDetune     :: !Double       -- ^ Detune in cents
  } deriving (Eq, Show)

-- | Convert inversion state to phase modification
inversionToPhase :: Bool -> Double -> Double
inversionToPhase isInverted basePhase =
  if isInverted
  then basePhase + pi  -- 180 degree shift
  else basePhase

-- | Calculate phase distortion for shadow fragments
phaseDistortion :: EmergenceContext -> Double
phaseDistortion ctx =
  let baseDistortion = if ecInversion ctx then 0.5 else 0.0
      alphaInfluence = (1.0 - ecAlpha ctx) * 0.3
  in baseDistortion + alphaInfluence

-- | Calculate detune amount for inverted fragments
detuneAmount :: EmergenceContext -> Double
detuneAmount ctx =
  if ecInversion ctx
  then 7.0 + ecAlpha ctx * 5.0  -- 7-12 cents detune
  else 0.0

-- =============================================================================
-- Stream Generation
-- =============================================================================

-- | Stream of sonic outputs
data SonicStream = SonicStream
  { ssConfig      :: !StreamConfig
  , ssOutputs     :: ![SonicOutput]
  , ssTimestamp   :: !Double
  , ssEventCount  :: !Int
  } deriving (Eq, Show)

-- | Stream configuration
data StreamConfig = StreamConfig
  { scSampleRate  :: !Int         -- ^ Output sample rate
  , scBufferSize  :: !Int         -- ^ Buffer size
  , scMaxEvents   :: !Int         -- ^ Max events to buffer
  , scLatency     :: !Double      -- ^ Target latency (ms)
  } deriving (Eq, Show)

-- | Initialize a new stream
initStream :: StreamConfig -> SonicStream
initStream config = SonicStream
  { ssConfig = config
  , ssOutputs = []
  , ssTimestamp = 0.0
  , ssEventCount = 0
  }

-- | Push emergence event to stream
pushEmergence :: EmergenceContext -> Double -> SonicStream -> SonicStream
pushEmergence ctx timestamp stream =
  let output = sonifyEmergence ctx
      maxEvents = scMaxEvents (ssConfig stream)
      newOutputs = take maxEvents (output : ssOutputs stream)
  in stream
      { ssOutputs = newOutputs
      , ssTimestamp = timestamp
      , ssEventCount = ssEventCount stream + 1
      }

-- | Convert stream to JSON representation
streamToJSON :: SonicStream -> String
streamToJSON stream =
  let outputs = ssOutputs stream
      eventStrings = map outputToJSON outputs
  in "{\"events\":[" ++ intercalateWith "," eventStrings ++ "],\"timestamp\":" ++ show (ssTimestamp stream) ++ "}"

-- Convert single output to JSON
outputToJSON :: SonicOutput -> String
outputToJSON so =
  "{\"freq\":" ++ show (soFrequency so) ++
  ",\"amp\":" ++ show (soAmplitude so) ++
  ",\"pan\":[" ++ show (fst (soPanPosition so)) ++ "," ++ show (snd (soPanPosition so)) ++ "]" ++
  ",\"dur\":" ++ show (soDuration so) ++
  ",\"env\":\"" ++ show (soEnvelopeType so) ++ "\"}"

-- Simple intercalate
intercalateWith :: String -> [String] -> String
intercalateWith _ [] = ""
intercalateWith _ [x] = x
intercalateWith sep (x:xs) = x ++ sep ++ intercalateWith sep xs

-- =============================================================================
-- Integration
-- =============================================================================

-- | Sonic profile for customization
data SonicProfile = SonicProfile
  { spPitchRange    :: !(Double, Double)  -- ^ Min/max frequency
  , spAmplitudeScale :: !Double           -- ^ Amplitude multiplier
  , spTimbreBase    :: !TimbreProfile     -- ^ Base timbre
  , spPreferredEnvelope :: !EnvelopeKind  -- ^ Default envelope
  , spSpatialMode   :: !String            -- ^ "stereo", "3d", "mono"
  } deriving (Eq, Show)

-- | Create profile for chamber type
profileForChamber :: String -> SonicProfile
profileForChamber chamberType = case chamberType of
  "pyramid" -> SonicProfile
    { spPitchRange = (220.0, 880.0)
    , spAmplitudeScale = 1.2
    , spTimbreBase = defaultTimbre { tpBrightness = 0.7 }
    , spPreferredEnvelope = Sharp
    , spSpatialMode = "3d"
    }
  "sphere" -> SonicProfile
    { spPitchRange = (110.0, 440.0)
    , spAmplitudeScale = 0.9
    , spTimbreBase = defaultTimbre { tpBrightness = 0.3, tpRichness = 0.6 }
    , spPreferredEnvelope = Pad
    , spSpatialMode = "stereo"
    }
  "torus" -> SonicProfile
    { spPitchRange = (165.0, 660.0)
    , spAmplitudeScale = 1.0
    , spTimbreBase = defaultTimbre { tpOvertones = [0.7, 0.5, 0.3, 0.2, 0.1] }
    , spPreferredEnvelope = Smooth
    , spSpatialMode = "stereo"
    }
  _ -> SonicProfile
    { spPitchRange = (110.0, 880.0)
    , spAmplitudeScale = 1.0
    , spTimbreBase = defaultTimbre
    , spPreferredEnvelope = Smooth
    , spSpatialMode = "stereo"
    }

-- | Create profile for avatar
profileForAvatar :: String -> SonicProfile
profileForAvatar avatarId =
  -- Generate consistent profile from avatar ID hash
  let hash = sum (map fromEnum avatarId) `mod` 100
      pitchOffset = fromIntegral hash * 2.0
      brightness = 0.3 + fromIntegral (hash `mod` 50) / 100.0
  in SonicProfile
      { spPitchRange = (110.0 + pitchOffset, 880.0 + pitchOffset)
      , spAmplitudeScale = 0.9 + fromIntegral (hash `mod` 20) / 100.0
      , spTimbreBase = defaultTimbre { tpBrightness = brightness }
      , spPreferredEnvelope = toEnum (hash `mod` 6)
      , spSpatialMode = "stereo"
      }

-- | Blend two profiles
blendProfiles :: Double -> SonicProfile -> SonicProfile -> SonicProfile
blendProfiles ratio p1 p2 =
  let blend a b = a * ratio + b * (1 - ratio)
      (minF1, maxF1) = spPitchRange p1
      (minF2, maxF2) = spPitchRange p2
  in SonicProfile
      { spPitchRange = (blend minF1 minF2, blend maxF1 maxF2)
      , spAmplitudeScale = blend (spAmplitudeScale p1) (spAmplitudeScale p2)
      , spTimbreBase = if ratio > 0.5 then spTimbreBase p1 else spTimbreBase p2
      , spPreferredEnvelope = if ratio > 0.5 then spPreferredEnvelope p1 else spPreferredEnvelope p2
      , spSpatialMode = if ratio > 0.5 then spSpatialMode p1 else spSpatialMode p2
      }

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
