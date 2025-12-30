{-|
Module      : Ra.Music.ChamberHarmonics
Description : Auditory feedback from scalar field HUD + coherence
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Converts scalar coherence data (from HUD overlays and avatar field states)
into auditory feedback — tuned harmonic loops, overtone sweeps, and interval
transitions — providing live chamber resonance matching internal emergence states.

This module is the sonic mirror of Ra.Visualizer.ShellHUD.

== Audio Mapping Theory

* Coherence spikes trigger chimes
* Aura bands map to overtone sweeps
* Consent gate states modulate pulse envelopes
* Vector appendage touches create tremolo effects
* Ankh Δ imbalances introduce dissonance
-}
module Ra.Music.ChamberHarmonics
  ( -- * Core Types
    HarmonyMode(..)
  , HarmonicFrame(..)
  , AudioMapping(..)

    -- * Frame Generation
  , generateHarmonicFrame
  , blendFrames
  , silentFrame

    -- * Audio Mapping
  , defaultAudioMapping

    -- * Frequency Mapping
  , coherenceToFrequency
  , auraToOvertones
  , ankhToDissonance

    -- * Envelope Shaping
  , EnvelopeShape(..)
  , shapeEnvelope
  , gateEnvelope

    -- * Effect Modulation
  , EffectType(..)
  , applyEffect
  , tremoloFromTouch

    -- * Solfeggio Integration
  , SolfeggioTone(..)
  , solfeggioFrequency
  , nearestSolfeggio

    -- * Stream Interface
  , HarmonicStream(..)
  , mkHarmonicStream
  , updateStream

    -- * HUD Integration
  , HUDPacket(..)
  , AuraPattern(..)
  , GateEffect(..)
  , frameFromHUD
  ) where

import Data.Time (UTCTime, getCurrentTime)

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Harmony mode for audio feedback intent
data HarmonyMode
  = CoherenceChime     -- ^ Chimes triggered by coherence spikes
  | AuraSweep          -- ^ Continuous overtone wave based on aura
  | GatedPulse         -- ^ Audio pulse modulated by consent state
  | FieldSymphony      -- ^ Complex multi-band harmonic chordscape
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Single frame of harmonic output
data HarmonicFrame = HarmonicFrame
  { hfRootFreq      :: !Double        -- ^ Root frequency (Hz)
  , hfOvertoneBands :: ![Double]      -- ^ Overtone multipliers [2x, 3x, 5x, etc.]
  , hfModulationAmt :: !Double        -- ^ Modulation depth [0, 1]
  , hfAmplitude     :: !Double        -- ^ Output volume [0, 1]
  , hfCoherenceTag  :: !(Maybe Double) -- ^ Optional coherence α
  , hfEnvelope      :: !EnvelopeShape
  , hfEffects       :: ![EffectType]
  } deriving (Eq, Show)

-- | Audio mapping configuration
data AudioMapping = AudioMapping
  { amBaseFrequency   :: !Double       -- ^ Base frequency (e.g., 528 Hz)
  , amCoherenceScale  :: !Double       -- ^ Coherence → frequency scaling
  , amOvertoneRatios  :: ![Double]     -- ^ Standard overtone series
  , amDissonanceMax   :: !Double       -- ^ Maximum dissonance amount
  , amVolumeRange     :: !(Double, Double) -- ^ (min, max) volume
  } deriving (Eq, Show)

-- =============================================================================
-- HUD Input Types (simplified)
-- =============================================================================

-- | HUD packet for audio generation
data HUDPacket = HUDPacket
  { hudCoherence   :: !Double
  , hudAura        :: !AuraPattern
  , hudAnkhDelta   :: !Double
  , hudTimestamp   :: !UTCTime
  } deriving (Eq, Show)

-- | Aura pattern for overtone mapping
data AuraPattern = AuraPattern
  { apBands     :: ![AuraBand]
  , apIntensity :: !Double
  } deriving (Eq, Show)

-- | Aura band
data AuraBand = AuraBand
  { abRadius    :: !Double
  , abIntensity :: !Double
  , abHue       :: !Double   -- ^ 0-360 degrees
  } deriving (Eq, Show)

-- | Consent gate effect
data GateEffect
  = GateOpening !Double    -- ^ Opening by factor
  | GateClosing !Double    -- ^ Closing by factor
  | GateHolding            -- ^ Steady state
  | GateFluctuating !Double -- ^ Oscillating
  deriving (Eq, Show)

-- =============================================================================
-- Frame Generation
-- =============================================================================

-- | Generate harmonic frame from HUD + aura + gate data
generateHarmonicFrame :: HUDPacket
                      -> AuraPattern
                      -> GateEffect
                      -> HarmonyMode
                      -> HarmonicFrame
generateHarmonicFrame hud aura gate mode =
  let coherence = hudCoherence hud
      ankhDelta = hudAnkhDelta hud

      -- Compute root frequency based on mode
      rootFreq = case mode of
        CoherenceChime -> coherenceToFrequency coherence
        AuraSweep -> auraToRootFreq aura
        GatedPulse -> gateToFrequency gate coherence
        FieldSymphony -> 432.0  -- Base A tuning

      -- Compute overtones from aura
      overtones = auraToOvertones aura

      -- Modulation from gate state
      modulation = gateToModulation gate

      -- Amplitude from coherence
      amplitude = coherence * 0.8 + 0.1

      -- Envelope shape
      envelope = modeToEnvelope mode

      -- Effects based on ankh delta and mode
      effects = buildEffects ankhDelta mode

  in HarmonicFrame
    { hfRootFreq = rootFreq
    , hfOvertoneBands = overtones
    , hfModulationAmt = modulation
    , hfAmplitude = amplitude
    , hfCoherenceTag = Just coherence
    , hfEnvelope = envelope
    , hfEffects = effects
    }

-- | Blend two harmonic frames
blendFrames :: Double -> HarmonicFrame -> HarmonicFrame -> HarmonicFrame
blendFrames factor f1 f2 = HarmonicFrame
  { hfRootFreq = lerp factor (hfRootFreq f1) (hfRootFreq f2)
  , hfOvertoneBands = zipWith (lerp factor) (hfOvertoneBands f1) (hfOvertoneBands f2)
  , hfModulationAmt = lerp factor (hfModulationAmt f1) (hfModulationAmt f2)
  , hfAmplitude = lerp factor (hfAmplitude f1) (hfAmplitude f2)
  , hfCoherenceTag = hfCoherenceTag f2  -- Use newer tag
  , hfEnvelope = hfEnvelope f2
  , hfEffects = hfEffects f2
  }

-- | Silent frame
silentFrame :: HarmonicFrame
silentFrame = HarmonicFrame
  { hfRootFreq = 0.0
  , hfOvertoneBands = []
  , hfModulationAmt = 0.0
  , hfAmplitude = 0.0
  , hfCoherenceTag = Nothing
  , hfEnvelope = EnvSilent
  , hfEffects = []
  }

-- =============================================================================
-- Frequency Mapping
-- =============================================================================

-- | Map coherence to frequency (using Solfeggio scale)
coherenceToFrequency :: Double -> Double
coherenceToFrequency coh
  | coh >= 0.95 = 963.0   -- Crown activation
  | coh >= 0.85 = 852.0   -- Intuition
  | coh >= 0.75 = 741.0   -- Expression
  | coh >= 0.65 = 639.0   -- Heart connection
  | coh >= 0.55 = 528.0   -- DNA repair / love
  | coh >= 0.45 = 417.0   -- Change facilitation
  | coh >= 0.35 = 396.0   -- Liberation from fear
  | coh >= 0.25 = 285.0   -- Quantum cognition
  | otherwise = 174.0    -- Foundation

-- | Map aura bands to overtone series
auraToOvertones :: AuraPattern -> [Double]
auraToOvertones aura =
  let bands = apBands aura
      baseOvertones = [2.0, 3.0, 5.0, 8.0, 13.0]  -- Fibonacci series
      intensity = apIntensity aura
      scaledOvertones = map (* intensity) baseOvertones
      bandContributions = map bandToOvertone bands
  in zipWith (+) scaledOvertones (bandContributions ++ repeat 0)

-- | Convert single band to overtone contribution
bandToOvertone :: AuraBand -> Double
bandToOvertone band =
  let hueContribution = hueToHarmonic (abHue band)
      intensityScale = abIntensity band
  in hueContribution * intensityScale

-- | Map hue (0-360) to harmonic ratio
hueToHarmonic :: Double -> Double
hueToHarmonic hue
  | hue < 60 = 2.0     -- Red → octave
  | hue < 120 = 3.0    -- Yellow → fifth
  | hue < 180 = 4.0    -- Green → double octave
  | hue < 240 = 5.0    -- Cyan → major third
  | hue < 300 = 6.0    -- Blue → fifth above octave
  | otherwise = 7.0    -- Violet → minor seventh

-- | Compute dissonance from Ankh delta
ankhToDissonance :: Double -> Double
ankhToDissonance delta =
  let absDelta = abs delta
      dissonance = absDelta * phi  -- Scale by golden ratio
  in min 1.0 dissonance

-- | Aura to root frequency
auraToRootFreq :: AuraPattern -> Double
auraToRootFreq aura =
  let baseFreq = 432.0  -- Base A
      shift = apIntensity aura * 96  -- Up to 528 Hz
  in baseFreq + shift

-- | Gate to frequency
gateToFrequency :: GateEffect -> Double -> Double
gateToFrequency gate coherence = case gate of
  GateOpening factor -> coherenceToFrequency coherence * (1 + factor * 0.1)
  GateClosing factor -> coherenceToFrequency coherence * (1 - factor * 0.1)
  GateHolding -> coherenceToFrequency coherence
  GateFluctuating amt -> coherenceToFrequency coherence * (1 + sin (amt * 2 * pi) * 0.05)

-- | Gate to modulation amount
gateToModulation :: GateEffect -> Double
gateToModulation gate = case gate of
  GateOpening factor -> factor * 0.3
  GateClosing factor -> factor * 0.5
  GateHolding -> 0.1
  GateFluctuating amt -> 0.2 + amt * 0.3

-- =============================================================================
-- Envelope Shaping
-- =============================================================================

-- | Envelope shape types
data EnvelopeShape
  = EnvADSR !Double !Double !Double !Double  -- ^ Attack, Decay, Sustain, Release
  | EnvPluck !Double                          -- ^ Pluck decay time
  | EnvPad !Double                            -- ^ Pad sustain level
  | EnvPulse !Double !Double                  -- ^ On time, Off time
  | EnvSilent
  deriving (Eq, Show)

-- | Mode to default envelope
modeToEnvelope :: HarmonyMode -> EnvelopeShape
modeToEnvelope mode = case mode of
  CoherenceChime -> EnvPluck 0.5
  AuraSweep -> EnvPad 0.8
  GatedPulse -> EnvPulse 0.3 0.7
  FieldSymphony -> EnvADSR 0.1 0.2 0.7 0.5

-- | Shape envelope based on coherence
shapeEnvelope :: Double -> EnvelopeShape -> EnvelopeShape
shapeEnvelope coherence env = case env of
  EnvADSR a d s r -> EnvADSR (a * phiInverse) (d * coherence) (s * coherence) r
  EnvPluck decay -> EnvPluck (decay * (1 + coherence))
  EnvPad sustain -> EnvPad (sustain * coherence)
  EnvPulse on off -> EnvPulse (on * coherence) (off * (1 - coherence))
  EnvSilent -> EnvSilent

-- | Apply gate effect to envelope
gateEnvelope :: GateEffect -> EnvelopeShape -> EnvelopeShape
gateEnvelope gate env = case gate of
  GateOpening factor ->
    case env of
      EnvADSR a d s r -> EnvADSR (a * 0.5) d (s * (1 + factor)) r
      other -> other
  GateClosing factor ->
    case env of
      EnvADSR a d s r -> EnvADSR (a * 2) d (s * (1 - factor)) (r * 2)
      other -> other
  GateHolding -> env
  GateFluctuating _ -> env

-- =============================================================================
-- Effect Modulation
-- =============================================================================

-- | Effect types
data EffectType
  = Tremolo !Double !Double    -- ^ Rate, Depth
  | Vibrato !Double !Double    -- ^ Rate, Depth
  | Chorus !Double !Double     -- ^ Rate, Mix
  | Reverb !Double !Double     -- ^ Size, Mix
  | Distortion !Double         -- ^ Amount
  | Filter !FilterType !Double -- ^ Type, Cutoff
  | Delay !Double !Double      -- ^ Time, Feedback
  deriving (Eq, Show)

-- | Filter types
data FilterType
  = LowPass
  | HighPass
  | BandPass
  | Notch
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Build effects list from ankh delta and mode
buildEffects :: Double -> HarmonyMode -> [EffectType]
buildEffects ankhDelta mode =
  let dissonance = ankhToDissonance ankhDelta

      -- Base effects for mode
      modeEffects = case mode of
        CoherenceChime -> [Reverb 0.5 0.3]
        AuraSweep -> [Chorus 0.3 0.2, Reverb 0.7 0.4]
        GatedPulse -> [Filter LowPass 0.6]
        FieldSymphony -> [Chorus 0.4 0.3, Reverb 0.8 0.5]

      -- Dissonance effects
      dissonanceEffects =
        if dissonance > 0.3
        then [Distortion (dissonance * 0.3)]
        else []

  in modeEffects ++ dissonanceEffects

-- | Apply effect to frame
applyEffect :: EffectType -> HarmonicFrame -> HarmonicFrame
applyEffect effect frame = case effect of
  Tremolo _rate depth ->
    frame { hfModulationAmt = hfModulationAmt frame + depth
          , hfEffects = effect : hfEffects frame }

  Vibrato _ _ ->
    frame { hfEffects = effect : hfEffects frame }

  Chorus _ mix ->
    frame { hfAmplitude = hfAmplitude frame * (1 - mix * 0.1)
          , hfEffects = effect : hfEffects frame }

  Reverb _ mix ->
    frame { hfAmplitude = hfAmplitude frame * (1 + mix * 0.1)
          , hfEffects = effect : hfEffects frame }

  Distortion amt ->
    frame { hfOvertoneBands = map (* (1 + amt)) (hfOvertoneBands frame)
          , hfEffects = effect : hfEffects frame }

  Filter _ _ ->
    frame { hfEffects = effect : hfEffects frame }

  Delay _ _ ->
    frame { hfEffects = effect : hfEffects frame }

-- | Create tremolo from vector appendage touch
tremoloFromTouch :: Double -> EffectType
tremoloFromTouch intensity =
  let rate = 4.0 + intensity * 8.0   -- 4-12 Hz
      depth = intensity * 0.5        -- Up to 50% depth
  in Tremolo rate depth

-- =============================================================================
-- Solfeggio Integration
-- =============================================================================

-- | Solfeggio tones
data SolfeggioTone
  = UT  -- ^ 396 Hz - Liberation
  | RE  -- ^ 417 Hz - Change
  | MI  -- ^ 528 Hz - Transformation
  | FA  -- ^ 639 Hz - Connection
  | SOL -- ^ 741 Hz - Expression
  | LA  -- ^ 852 Hz - Intuition
  | TI  -- ^ 963 Hz - Divine connection
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Get frequency for Solfeggio tone
solfeggioFrequency :: SolfeggioTone -> Double
solfeggioFrequency tone = case tone of
  UT  -> 396.0
  RE  -> 417.0
  MI  -> 528.0
  FA  -> 639.0
  SOL -> 741.0
  LA  -> 852.0
  TI  -> 963.0

-- | Find nearest Solfeggio tone
nearestSolfeggio :: Double -> SolfeggioTone
nearestSolfeggio freq =
  let tones = [UT, RE, MI, FA, SOL, LA, TI]
      distances = [(abs (freq - solfeggioFrequency t), t) | t <- tones]
      sorted = sortByFst distances
  in snd (head sorted)
  where
    sortByFst = foldr insertSorted []
    insertSorted x [] = [x]
    insertSorted x@(d, _) (y@(d', _):ys)
      | d <= d' = x : y : ys
      | otherwise = y : insertSorted x ys

-- =============================================================================
-- Stream Interface
-- =============================================================================

-- | Harmonic stream state
data HarmonicStream = HarmonicStream
  { hsCurrentFrame :: !HarmonicFrame
  , hsPreviousFrame :: !HarmonicFrame
  , hsMode :: !HarmonyMode
  , hsBlendFactor :: !Double
  , hsTimestamp :: !UTCTime
  } deriving (Eq, Show)

-- | Create harmonic stream
mkHarmonicStream :: HarmonyMode -> IO HarmonicStream
mkHarmonicStream mode = do
  now <- getCurrentTime
  return HarmonicStream
    { hsCurrentFrame = silentFrame
    , hsPreviousFrame = silentFrame
    , hsMode = mode
    , hsBlendFactor = 0.3
    , hsTimestamp = now
    }

-- | Update stream with new HUD data
updateStream :: HarmonicStream -> HUDPacket -> GateEffect -> IO HarmonicStream
updateStream stream hud gate = do
  now <- getCurrentTime
  let newFrame = generateHarmonicFrame hud (hudAura hud) gate (hsMode stream)
      blended = blendFrames (hsBlendFactor stream)
                            (hsCurrentFrame stream)
                            newFrame
  return stream
    { hsCurrentFrame = blended
    , hsPreviousFrame = hsCurrentFrame stream
    , hsTimestamp = now
    }

-- =============================================================================
-- HUD Integration
-- =============================================================================

-- | Generate frame directly from HUD packet
frameFromHUD :: HUDPacket -> HarmonyMode -> HarmonicFrame
frameFromHUD hud mode =
  generateHarmonicFrame hud (hudAura hud) GateHolding mode

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Linear interpolation
lerp :: Double -> Double -> Double -> Double
lerp t a b = a + t * (b - a)

-- | Default audio mapping
defaultAudioMapping :: AudioMapping
defaultAudioMapping = AudioMapping
  { amBaseFrequency = 528.0
  , amCoherenceScale = 1.0
  , amOvertoneRatios = [2.0, 3.0, 5.0, 8.0, 13.0]
  , amDissonanceMax = 0.5
  , amVolumeRange = (0.1, 0.9)
  }
