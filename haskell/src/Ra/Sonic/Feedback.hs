{-|
Module      : Ra.Sonic.Feedback
Description : Real-time audio cues for chamber synchronization
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Converts scalar sync events into real-time audio feedback for operator
awareness and biometric entrainment. Provides non-visual accessibility
for chamber state monitoring.

== Sound Design Principles

=== Harmonic Tones

Sync quality maps to pitch:

* High coherence (α ≥ 0.85): Emergence chord (multiple harmonics)
* Sync resonance delta < φ/10: Pure harmonic tone
* Desync events: Subtle pulse/click

=== Frequency Basis

Based on Keely sympathetic vibratory physics:

* Base frequency: 256 Hz (C4, middle C)
* Phi-scaled harmonics for resonance
* Solfeggio frequencies for healing tones
* Rife frequencies for therapeutic range

=== Accessibility

Audio feedback serves as:

* Non-visual system state indicator
* Meditative entrainment support
* Biometric rhythm sync tool
-}
module Ra.Sonic.Feedback
  ( -- * Core Sound Types
    SyncTone(..)
  , Pitch(..)
  , Hz
  , Volume

    -- * Tone Generation
  , generateSyncTones
  , toneFromCoherence
  , toneFromPhase

    -- * Playback Interface
  , SonicOutput(..)
  , playSyncTone
  , playToneSequence
  , sonicFeedback

    -- * Emergence Chords
  , emergenceChord
  , chordPitches
  , chordFromAlpha

    -- * Coherence Pulses
  , coherencePulse
  , pulseRate
  , pulseFromSync

    -- * Harmonic Mapping
  , harmonicToHz
  , alphaToFrequency
  , phaseToModulation

    -- * Sync Link Processing
  , SyncLink(..)
  , processSyncLinks
  , linkToTone

    -- * Biometric Integration
  , BiometricRhythm(..)
  , modulateWithHRV
  , entrain

    -- * Configuration
  , FeedbackConfig(..)
  , defaultConfig
  , enableSonicFeedback
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Frequency in Hertz
type Hz = Double

-- | Volume level [0, 1]
type Volume = Double

-- | Musical pitch representation
data Pitch = Pitch
  { pitchNote     :: !String        -- ^ Note name (C, D, E, etc.)
  , pitchOctave   :: !Int           -- ^ Octave number
  , pitchHz       :: !Hz            -- ^ Frequency in Hz
  , pitchCents    :: !Int           -- ^ Microtonal adjustment
  } deriving (Eq, Show)

-- | Sync tone types
data SyncTone
  = HarmonicTone !Pitch !Hz         -- ^ Pure harmonic tone at frequency
  | CoherencePulse !Double          -- ^ Alpha-modulated click/beat
  | EmergenceChord ![Pitch]         -- ^ Multi-tone resonance event
  | SilenceTone                     -- ^ No sound (placeholder)
  deriving (Eq, Show)

-- =============================================================================
-- Sonic Output
-- =============================================================================

-- | Sonic output specification
data SonicOutput = SonicOutput
  { soTone        :: !SyncTone
  , soVolume      :: !Volume
  , soDuration    :: !Double        -- ^ Duration in seconds
  , soPan         :: !Double        -- ^ Stereo pan [-1, 1]
  , soAttack      :: !Double        -- ^ Attack time (s)
  , soDecay       :: !Double        -- ^ Decay time (s)
  , soSustain     :: !Double        -- ^ Sustain level [0, 1]
  , soRelease     :: !Double        -- ^ Release time (s)
  } deriving (Eq, Show)

-- | Play a sync tone (returns description for external synthesis)
playSyncTone :: SyncTone -> IO String
playSyncTone tone = pure $ case tone of
  HarmonicTone pitch hz ->
    "TONE:" ++ show hz ++ "Hz:" ++ pitchNote pitch ++ show (pitchOctave pitch)
  CoherencePulse alpha ->
    "PULSE:alpha=" ++ show alpha ++ ":rate=" ++ show (pulseRate alpha)
  EmergenceChord pitches ->
    "CHORD:" ++ unwords [pitchNote p ++ show (pitchOctave p) | p <- pitches]
  SilenceTone ->
    "SILENCE"

-- | Play sequence of tones
playToneSequence :: [SonicOutput] -> IO [String]
playToneSequence = mapM (playSyncTone . soTone)

-- | Main sonic feedback function
sonicFeedback :: [SyncLink] -> IO [String]
sonicFeedback links = do
  let tones = generateSyncTones links
  mapM playSyncTone tones

-- =============================================================================
-- Tone Generation
-- =============================================================================

-- | Generate tones from sync links
generateSyncTones :: [SyncLink] -> [SyncTone]
generateSyncTones links =
  -- Limit to 1 tone per chamber per cycle (avoid audio spam)
  take (length links) $ map linkToTone (deduplicate links)
  where
    deduplicate = foldr insertUnique []
    insertUnique l acc
      | any (\x -> slChamberId x == slChamberId l) acc = acc
      | otherwise = l : acc

-- | Generate tone from coherence level
toneFromCoherence :: Double -> SyncTone
toneFromCoherence alpha
  | alpha >= 0.85 = EmergenceChord (chordFromAlpha alpha)
  | alpha >= 0.618 = HarmonicTone (pitchFromAlpha alpha) (alphaToFrequency alpha)
  | alpha >= 0.318 = CoherencePulse alpha
  | otherwise = SilenceTone

-- | Generate tone from phase alignment
toneFromPhase :: Double -> Double -> SyncTone
toneFromPhase phase delta
  | delta < phi / 10 = HarmonicTone (pitchFromPhase phase) (phaseToHz phase)
  | delta < 0.3 = CoherencePulse (1.0 - delta)
  | otherwise = SilenceTone

-- =============================================================================
-- Emergence Chords
-- =============================================================================

-- | Create emergence chord from alpha value
emergenceChord :: Double -> SyncTone
emergenceChord alpha = EmergenceChord (chordFromAlpha alpha)

-- | Get chord pitches from alpha
chordPitches :: Double -> [Pitch]
chordPitches = chordFromAlpha

-- | Build chord from alpha level
chordFromAlpha :: Double -> [Pitch]
chordFromAlpha alpha =
  let baseHz = 256.0 * (1.0 + alpha * phiInverse)  -- C4 shifted by alpha
      -- Perfect fifth = 3/2 ratio
      fifth = baseHz * 1.5
      -- Major third = 5/4 ratio
      third = baseHz * 1.25
      -- Octave for richness
      octave = baseHz * 2.0
  in [ Pitch "C" 4 baseHz 0
     , Pitch "E" 4 third 0
     , Pitch "G" 4 fifth 0
     , Pitch "C" 5 octave 0
     ]

-- =============================================================================
-- Coherence Pulses
-- =============================================================================

-- | Create coherence pulse
coherencePulse :: Double -> SyncTone
coherencePulse = CoherencePulse

-- | Calculate pulse rate from alpha
pulseRate :: Double -> Double
pulseRate alpha = 0.5 + alpha * 2.0  -- 0.5 to 2.5 Hz range

-- | Generate pulse from sync state
pulseFromSync :: Double -> Double -> SyncTone
pulseFromSync coherence _stability = CoherencePulse coherence

-- =============================================================================
-- Harmonic Mapping
-- =============================================================================

-- | Convert harmonic indices to Hz
harmonicToHz :: (Int, Int) -> Hz
harmonicToHz (l, m) =
  let baseHz = 256.0  -- C4
      lFactor = phi ** fromIntegral l
      mFactor = 1.0 + fromIntegral (abs m) * 0.1
  in baseHz * lFactor * mFactor

-- | Convert alpha to frequency
alphaToFrequency :: Double -> Hz
alphaToFrequency alpha =
  let minHz = 256.0   -- C4
      maxHz = 512.0   -- C5
  in minHz + alpha * (maxHz - minHz)

-- | Convert phase to modulation factor
phaseToModulation :: Double -> Double
phaseToModulation phase = 0.5 + 0.5 * sin phase

-- | Phase to Hz
phaseToHz :: Double -> Hz
phaseToHz phase = 256.0 * (1.0 + 0.1 * sin phase)

-- | Alpha to pitch
pitchFromAlpha :: Double -> Pitch
pitchFromAlpha alpha =
  let hz = alphaToFrequency alpha
      noteIdx = floor (12 * logBase 2 (hz / 256.0)) :: Int
      notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
      note = notes !! (noteIdx `mod` 12)
      octave = 4 + noteIdx `div` 12
  in Pitch note octave hz 0

-- | Phase to pitch
pitchFromPhase :: Double -> Pitch
pitchFromPhase phase = pitchFromAlpha (0.5 + 0.5 * sin phase)

-- =============================================================================
-- Sync Link Processing
-- =============================================================================

-- | Sync link representing chamber synchronization
data SyncLink = SyncLink
  { slChamberId   :: !String        -- ^ Chamber identifier
  , slCoherence   :: !Double        -- ^ Current coherence [0, 1]
  , slPhase       :: !Double        -- ^ Phase alignment [0, 2*pi]
  , slDelta       :: !Double        -- ^ Resonance delta
  , slTimestamp   :: !Double        -- ^ Event timestamp
  } deriving (Eq, Show)

-- | Process sync links into tones
processSyncLinks :: [SyncLink] -> [(SyncLink, SyncTone)]
processSyncLinks links = [(l, linkToTone l) | l <- links]

-- | Convert sync link to appropriate tone
linkToTone :: SyncLink -> SyncTone
linkToTone link
  -- High coherence triggers emergence chord
  | slCoherence link >= 0.85 =
      EmergenceChord (chordFromAlpha (slCoherence link))
  -- Good sync with low delta triggers harmonic
  | slDelta link < phi / 10 =
      HarmonicTone (pitchFromAlpha (slCoherence link)) (alphaToFrequency (slCoherence link))
  -- Desync or window closure triggers pulse
  | slDelta link > 0.3 =
      -- Subtle dissonance: minor 2nd offset
      let baseHz = alphaToFrequency (slCoherence link)
          dissonantHz = baseHz * (16.0 / 15.0)  -- Minor second ratio
      in HarmonicTone (pitchFromAlpha (slCoherence link)) dissonantHz
  -- Default: soft pulse
  | otherwise =
      CoherencePulse (slCoherence link)

-- =============================================================================
-- Biometric Integration
-- =============================================================================

-- | Biometric rhythm data
data BiometricRhythm = BiometricRhythm
  { brHRV          :: !Double        -- ^ Heart rate variability
  , brHeartRate    :: !Double        -- ^ Current heart rate (BPM)
  , brBreathRate   :: !Double        -- ^ Breath rate (per minute)
  , brPhase        :: !Double        -- ^ Current phase
  } deriving (Eq, Show)

-- | Modulate tone with HRV
modulateWithHRV :: BiometricRhythm -> SyncTone -> SonicOutput
modulateWithHRV bio tone =
  let hrvMod = brHRV bio / 100.0     -- Normalize HRV
      heartPhase = brPhase bio
      -- Duration follows breath cycle
      duration = 60.0 / brBreathRate bio
      -- Volume modulated by HRV
      volume = 0.3 + hrvMod * 0.5
  in SonicOutput
      { soTone = tone
      , soVolume = clamp01 volume
      , soDuration = duration
      , soPan = 0.3 * sin heartPhase  -- Gentle stereo movement
      , soAttack = 0.05
      , soDecay = 0.1
      , soSustain = 0.7
      , soRelease = 0.3
      }

-- | Entrain tones to biometric rhythm
entrain :: BiometricRhythm -> [SyncTone] -> [SonicOutput]
entrain bio = map (modulateWithHRV bio)

-- =============================================================================
-- Configuration
-- =============================================================================

-- | Feedback configuration
data FeedbackConfig = FeedbackConfig
  { fcEnabled       :: !Bool          -- ^ Enable sonic feedback
  , fcVolume        :: !Double        -- ^ Master volume [0, 1]
  , fcMinAlpha      :: !Double        -- ^ Minimum alpha to trigger sound
  , fcMaxTones      :: !Int           -- ^ Max simultaneous tones
  , fcPulseDuration :: !Double        -- ^ Pulse duration (s)
  , fcEntrainHRV    :: !Bool          -- ^ Entrain to HRV
  } deriving (Eq, Show)

-- | Default configuration
defaultConfig :: FeedbackConfig
defaultConfig = FeedbackConfig
  { fcEnabled = True
  , fcVolume = 0.5
  , fcMinAlpha = 0.318
  , fcMaxTones = 4
  , fcPulseDuration = 0.1
  , fcEntrainHRV = True
  }

-- | Check if sonic feedback is enabled
enableSonicFeedback :: FeedbackConfig -> Bool
enableSonicFeedback = fcEnabled

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
