{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : Ra.Music
Description : Ra Scalar Music Notation System (Keely-Inspired Harmonic Interface)
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Maps Ra System's scalar coordinate space to musical composition parameters
for use in coherence-responsive environments.

== Theoretical Background

=== Keely's Sympathetic Vibratory Physics

John Keely's 40 Laws of Sympathetic Vibration define force as transformation
through 21-octave bands:

* /Sonity/: Octaves 1-21, sound transmission, attraction\/repulsion
* /Sono-thermity/: Octaves 21-42, adhesion, molecular union
* /Thermism/: Octaves 42-63, cohesion, chemism
* /Electricity/: Octaves 63-84, induction, magnetism
* /Atomolity/: Octaves 84-105, gravism, gravity

=== Threefold Vibratory Structure

Keely's universal threefold order:

* 3 atoms per molecule (triangular arrangement)
* 3 atomoles per atom
* Triple-sympathetic resonance

This maps to our triple vibratory modes:
@Molecular@, @Atomic@, @Etheric@

=== Scalar-to-Music Mapping

@
Ra Dimension  →  Musical Parameter
────────────────────────────────────
θ (theta)     →  Pitch class (circle of fifths, 27→12 tone)
φ (phi)       →  Rhythmic signature (golden window subdivision)
h (harmonic)  →  Harmonic structure (interval sets)
r (radial)    →  Dynamic envelope (intensity/fade)
inversion     →  Contour direction (ascending/descending)
@

=== Safety Constraints

Musical output respects consent gating:

* FULL_CONSENT: All timbres available
* DIMINISHED: Limited dynamics, no extremes
* SUSPENDED: Drone/ambient only
* NO_CONSENT: Silence
-}
module Ra.Music
  ( -- * Core Types
    RaNoteline(..)
  , MusicNote(..)
  , PitchClass(..)
  , RhythmicValue(..)
  , Dynamic(..)
  , Contour(..)

    -- * Keely Vibratory Types
  , VibratorMode(..)
  , ForceOctave(..)
  , ResonatorClass(..)
  , SympatheticNode(..)

    -- * Timbral Types
  , Timbre(..)
  , TimbreFilter(..)
  , InstrumentArchetype(..)

    -- * Mapping Functions
  , thetaToPitch
  , phiToRhythm
  , harmonicToIntervalSet
  , radialToDynamic

    -- * Noteline Generation
  , generateNoteline
  , parseNoteline
  , renderNoteline
  , notelineTheta

    -- * Keely Integration
  , computeSympatheticResonance
  , classifyForceOctave
  , selectResonatorClass

    -- * Timbral Modulation
  , modulateTimbre
  , selectInstrument
  , applyTimbreFilter

    -- * Constants
  , keelyOctaveBandWidth
  , molecularDensityFactor
  , circleOfFifthsMapping
  , goldenRatio
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)
import Data.List (intercalate)
import Data.Maybe (fromMaybe)

import Ra.Repitans (Repitan, repitan, repitanIndex)

-- =============================================================================
-- Core Musical Types
-- =============================================================================

-- | Standard pitch classes (chromatic scale)
data PitchClass
  = C | Cs | D | Ds | E | F | Fs | G | Gs | A | As | B
  deriving (Eq, Ord, Enum, Bounded, Show, Generic, NFData)

-- | Rhythmic value with golden ratio subdivisions
data RhythmicValue
  = Whole           -- ^ φ^0 = 1
  | Half            -- ^ φ^1 = 0.618
  | Quarter         -- ^ φ^2 = 0.382
  | Eighth          -- ^ φ^3 = 0.236
  | Sixteenth       -- ^ φ^4 = 0.146
  | Triplet         -- ^ φ^2 triplet subdivision
  | Quintuplet      -- ^ φ^4 quintuplet
  | Polyrhythm Int  -- ^ φ^n complex polyrhythm
  deriving (Eq, Show, Generic, NFData)

-- | Dynamic marking (velocity/intensity)
data Dynamic
  = Pianissimo      -- ^ Very soft (r >= 7)
  | Piano           -- ^ Soft (r = 5-6)
  | MezzoPiano      -- ^ Medium soft (r = 4)
  | MezzoForte      -- ^ Medium loud (r = 3)
  | Forte           -- ^ Loud (r = 2)
  | Fortissimo      -- ^ Very loud (r = 1)
  | Sforzando       -- ^ Sudden accent (r = 0, core)
  deriving (Eq, Ord, Enum, Bounded, Show, Generic, NFData)

-- | Melodic contour (inversion state)
data Contour
  = Ascending       -- ^ Normal: upward motion
  | Descending      -- ^ Inverted: downward motion
  | Static          -- ^ Suspended: drone
  deriving (Eq, Show, Generic, NFData)

-- | A single music note with full Ra context
data MusicNote = MusicNote
  { mnPitch     :: !PitchClass      -- ^ Pitch class
  , mnOctave    :: !Int             -- ^ Octave number (0-9)
  , mnRhythm    :: !RhythmicValue   -- ^ Duration
  , mnDynamic   :: !Dynamic         -- ^ Intensity
  , mnContour   :: !Contour         -- ^ Direction
  , mnDotted    :: !Bool            -- ^ Dotted rhythm extension
  } deriving (Eq, Show, Generic, NFData)

-- | A Ra Noteline encoding scalar coordinates to music
--
-- Format: @R{shell}:H{harmonic}:θ{theta}:φ{phi} → Note Duration @ Dynamic (flags)@
data RaNoteline = RaNoteline
  { rnShell     :: !Int             -- ^ Radial shell depth (R0-R9)
  , rnHarmonic  :: !Int             -- ^ Harmonic channel (H1-H9)
  , rnThetaIdx  :: !Int             -- ^ Angular position (1-27), index for Repitan
  , rnPhi       :: !Double          -- ^ Golden phase depth
  , rnNote      :: !MusicNote       -- ^ Derived musical note
  , rnFlags     :: ![String]        -- ^ Modulation flags
  } deriving (Eq, Show, Generic, NFData)

-- | Get Repitan from noteline theta index
notelineTheta :: RaNoteline -> Maybe Repitan
notelineTheta nl = repitan (rnThetaIdx nl)

-- =============================================================================
-- Keely Vibratory Types
-- =============================================================================

-- | Keely's triple vibratory modes
--
-- From the threefold universal order:
-- * Molecular: baseline density, solid/liquid/gas
-- * Atomic: higher density, inter-molecular
-- * Etheric: 986,000× steel density, liquid ether
data VibratorMode
  = Molecular       -- ^ Baseline vibration, coarse
  | Atomic          -- ^ Inter-molecular, refined
  | Etheric         -- ^ Liquid ether, most subtle
  deriving (Eq, Ord, Enum, Bounded, Show, Generic, NFData)

-- | Keely's force octave bands (21 octaves each)
data ForceOctave
  = Sonity          -- ^ Octaves 1-21: Sound, sonism
  | SonoThermity    -- ^ Octaves 21-42: Sono-therm, adhesion
  | Thermism        -- ^ Octaves 42-63: Rad-energy, cohesion
  | Electricity     -- ^ Octaves 63-84: Induction, magnetism
  | Atomolity       -- ^ Octaves 84-105: Gravism, gravity
  deriving (Eq, Ord, Enum, Bounded, Show, Generic, NFData)

-- | Keely resonator classes for timbral selection
--
-- Based on Keely's compound resonators and sympathetic nodes.
data ResonatorClass
  = TripleSympathetic    -- ^ Three-pronged resonance (fundamental)
  | VibratorCore         -- ^ Central vibratory element
  | EnharmonicAmplifier  -- ^ Harmonic extension device
  | AtomolicTransducer   -- ^ Converts between force bands
  deriving (Eq, Show, Generic, NFData)

-- | Sympathetic node (harmonic anchor point)
data SympatheticNode = SympatheticNode
  { snFrequency  :: !Double         -- ^ Base frequency (Hz)
  , snHarmonics  :: ![Int]          -- ^ Harmonic ratios (1:2:3...)
  , snMode       :: !VibratorMode   -- ^ Vibratory mode
  , snResonance  :: !Double         -- ^ Resonance strength [0,1]
  } deriving (Eq, Show, Generic, NFData)

-- =============================================================================
-- Timbral Types
-- =============================================================================

-- | Timbre descriptor for sound color
data Timbre = Timbre
  { tmBrightness  :: !Double        -- ^ Spectral centroid [0,1]
  , tmWarmth      :: !Double        -- ^ Low-frequency emphasis [0,1]
  , tmAttack      :: !Double        -- ^ Attack time (ms)
  , tmSustain     :: !Double        -- ^ Sustain level [0,1]
  , tmDecay       :: !Double        -- ^ Decay time (ms)
  , tmRelease     :: !Double        -- ^ Release time (ms)
  } deriving (Eq, Show, Generic, NFData)

-- | Timbral filter types (chamber effects)
data TimbreFilter
  = LowPass Double      -- ^ Suppression (shadow)
  | HighPass Double     -- ^ Clarity (emergence)
  | BandPass Double Double  -- ^ Focus (coherence)
  | Phaser Double       -- ^ Shadow modulation
  | Chorus Double       -- ^ Fragment multiplicity
  | Reverb Double       -- ^ Chamber resonance
  deriving (Eq, Show, Generic, NFData)

-- | Instrument archetypes mapped to Ra states
data InstrumentArchetype
  = Flute             -- ^ Clarity, high coherence
  | Cello             -- ^ Density, depth
  | Gong              -- ^ Emergence, transformation
  | SingingBowl       -- ^ Sustained resonance
  | Synthesizer       -- ^ Synthetic fields
  | Voice             -- ^ Human coherence
  | Strings           -- ^ Triple-sympathetic (Keely)
  | Brass             -- ^ Force projection
  deriving (Eq, Show, Generic, NFData)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Keely's 21-octave band width
keelyOctaveBandWidth :: Int
keelyOctaveBandWidth = 21

-- | Atomolic density factor (986,000× steel)
molecularDensityFactor :: Double
molecularDensityFactor = 986000.0

-- | Circle of fifths mapping (27 Repitans → 12 pitches)
--
-- Maps the 27-fold theta space to the 12-note chromatic scale
-- using the circle of fifths progression.
--
-- Repitans 1-9: C → G → D → A → E → B → F# → C# → G#
-- Repitans 10-18: D# → A# → F → C → G → D → A → E → B
-- Repitans 19-27: F# → C# → G# → D# → A# → F → C → G → D
circleOfFifthsMapping :: [(Int, PitchClass)]
circleOfFifthsMapping =
  [ (1, C), (2, G), (3, D), (4, A), (5, E), (6, B)
  , (7, Fs), (8, Cs), (9, Gs)
  , (10, Ds), (11, As), (12, F), (13, C), (14, G), (15, D)
  , (16, A), (17, E), (18, B)
  , (19, Fs), (20, Cs), (21, Gs), (22, Ds), (23, As), (24, F)
  , (25, C), (26, G), (27, D)
  ]

-- | Golden ratio for phi calculations (φ)
goldenRatio :: Double
goldenRatio = 1.6180339887

-- =============================================================================
-- Mapping Functions
-- =============================================================================

-- | Map theta (Repitan 1-27) to pitch class
--
-- Uses the circle of fifths mapping, cycling through the 12-tone
-- chromatic scale in fifths progression.
thetaToPitch :: Repitan -> PitchClass
thetaToPitch rep =
  let val = repitanIndex rep
      -- Lookup or compute from modular arithmetic
      pitch = fromMaybe C (lookup val circleOfFifthsMapping)
  in pitch

-- | Map phi (golden phase depth) to rhythmic value
--
-- φ^0 = Whole
-- φ^1 = Half
-- φ^2 = Quarter (triplet feel)
-- φ^3 = Eighth
-- φ^4 = Sixteenth
-- φ^5+ = Polyrhythm
phiToRhythm :: Double -> RhythmicValue
phiToRhythm phiDepth
  | phiDepth < 0.5  = Whole
  | phiDepth < 1.0  = Half
  | phiDepth < 2.0  = Quarter
  | phiDepth < 3.0  = Triplet       -- φ^2 triplet subdivision
  | phiDepth < 4.0  = Eighth
  | phiDepth < 5.0  = Sixteenth
  | phiDepth < 6.0  = Quintuplet    -- φ^5 quintuplet
  | otherwise       = Polyrhythm (round phiDepth)

-- | Map harmonic channel to interval set
--
-- Based on Keely's use of thirds, sixths, and ninths:
-- * H1: Unison/Octave (1:1, 2:1)
-- * H2: Fifth (3:2)
-- * H3: Major Third (5:4) - Triple tier
-- * H4: Major Sixth (5:3)
-- * H5: Minor Third (6:5)
-- * H6: Minor Sixth (8:5)
-- * H7: Major Ninth (9:4)
-- * H8: Minor Ninth (16:9)
-- * H9: Compound (all intervals)
harmonicToIntervalSet :: Int -> [Double]
harmonicToIntervalSet h = case h of
  1 -> [1.0, 2.0]                   -- Unison, Octave
  2 -> [1.5, 2.0]                   -- Fifth, Octave
  3 -> [1.25, 1.5, 2.0]             -- Major third, Fifth, Octave
  4 -> [1.667, 2.0]                 -- Major sixth, Octave
  5 -> [1.2, 1.5, 2.0]              -- Minor third, Fifth, Octave
  6 -> [1.6, 2.0]                   -- Minor sixth, Octave
  7 -> [2.25, 3.0]                  -- Major ninth
  8 -> [1.778, 2.0]                 -- Minor ninth
  9 -> [1.25, 1.333, 1.5, 1.667, 2.0]  -- All consonant intervals
  _ -> [1.0, 2.0]                   -- Default: Unison/Octave

-- | Map radial shell depth to dynamic
--
-- Core (R0) = Maximum intensity (Sforzando)
-- Outer shells = Progressively softer
radialToDynamic :: Int -> Dynamic
radialToDynamic r = case r of
  0 -> Sforzando
  1 -> Fortissimo
  2 -> Forte
  3 -> MezzoForte
  4 -> MezzoPiano
  5 -> Piano
  6 -> Piano
  _ -> Pianissimo

-- =============================================================================
-- Noteline Generation
-- =============================================================================

-- | Generate a RaNoteline from scalar coordinates
generateNoteline
  :: Int              -- ^ Shell depth (R)
  -> Int              -- ^ Harmonic channel (H)
  -> Int              -- ^ Theta value (1-27)
  -> Double           -- ^ Phi phase depth
  -> Bool             -- ^ Is inverted?
  -> RaNoteline
generateNoteline shell harmonic thetaVal phiDepth isInverted =
  let
    -- Create Repitan (clamp to valid range)
    -- Create valid Repitan, clamped to 1-27 range
    rep = repitan (max 1 (min 27 thetaVal))

    -- Map to musical parameters
    pitch = case rep of
      Just r  -> thetaToPitch r
      Nothing -> C  -- Fallback

    rhythm = phiToRhythm phiDepth
    dynamic = radialToDynamic shell
    contour = if isInverted then Descending else Ascending

    -- Octave from shell depth (inverse: core = high, outer = low)
    octave = max 1 (8 - shell)

    -- Dotted if phi has fractional component
    dotted = phiDepth /= fromIntegral (round phiDepth :: Int)

    note = MusicNote
      { mnPitch = pitch
      , mnOctave = octave
      , mnRhythm = rhythm
      , mnDynamic = dynamic
      , mnContour = contour
      , mnDotted = dotted
      }

    -- Flags
    flags = if isInverted then ["inverted"] else []

    -- Clamped theta value for storage
    thetaClamped = max 1 (min 27 thetaVal)

  in RaNoteline
    { rnShell = shell
    , rnHarmonic = harmonic
    , rnThetaIdx = thetaClamped
    , rnPhi = phiDepth
    , rnNote = note
    , rnFlags = flags
    }

-- | Parse a RaNoteline from string format
--
-- Expected format: @R{n}:H{n}:θ{n}:φ{f}@
parseNoteline :: String -> Maybe RaNoteline
parseNoteline input =
  -- Simplified parser - in production would use a proper parser
  let parts = words input
  in if null parts
     then Nothing
     else Just $ generateNoteline 0 1 1 1.0 False  -- Placeholder

-- | Render a RaNoteline to string format
--
-- Output format:
-- @R0:H3:θ12:φ5.2 → C#5 dotted-eighth @ mezzo-forte (normal)@
renderNoteline :: RaNoteline -> String
renderNoteline RaNoteline{..} =
  let
    -- Coordinate part
    coords = "R" ++ show rnShell ++
             ":H" ++ show rnHarmonic ++
             ":θ" ++ show rnThetaIdx ++
             ":φ" ++ show rnPhi

    -- Note part
    pitchStr = showPitch (mnPitch rnNote)
    octStr = show (mnOctave rnNote)

    rhythmStr = case mnRhythm rnNote of
      Whole -> "whole"
      Half -> "half"
      Quarter -> "quarter"
      Eighth -> "eighth"
      Sixteenth -> "sixteenth"
      Triplet -> "triplet"
      Quintuplet -> "quintuplet"
      Polyrhythm n -> "poly-" ++ show n

    dottedStr = if mnDotted rnNote then "dotted-" else ""

    dynamicStr = case mnDynamic rnNote of
      Pianissimo -> "pianissimo"
      Piano -> "piano"
      MezzoPiano -> "mezzo-piano"
      MezzoForte -> "mezzo-forte"
      Forte -> "forte"
      Fortissimo -> "fortissimo"
      Sforzando -> "sforzando"

    contourStr = case mnContour rnNote of
      Ascending -> "normal"
      Descending -> "inverted"
      Static -> "drone"

    flagsStr = if null rnFlags
               then ""
               else ", " ++ intercalate ", " rnFlags

  in coords ++ " → " ++
     pitchStr ++ octStr ++ " " ++
     dottedStr ++ rhythmStr ++ " @ " ++
     dynamicStr ++ " (" ++ contourStr ++ flagsStr ++ ")"

-- | Show pitch class with sharps
showPitch :: PitchClass -> String
showPitch pc = case pc of
  C  -> "C"
  Cs -> "C#"
  D  -> "D"
  Ds -> "D#"
  E  -> "E"
  F  -> "F"
  Fs -> "F#"
  G  -> "G"
  Gs -> "G#"
  A  -> "A"
  As -> "A#"
  B  -> "B"

-- =============================================================================
-- Keely Integration
-- =============================================================================

-- | Compute sympathetic resonance between two frequencies
--
-- Based on Keely's Law 6: "Bodies whose atomic pitches are in harmonic
-- ratio oscillate in simultaneous resonance."
--
-- Returns resonance strength [0,1] where:
-- * 1.0 = Unison (1:1)
-- * 0.95 = Octave (2:1)
-- * 0.9 = Fifth (3:2)
-- * 0.85 = Third (5:4)
-- * Lower for more complex ratios
computeSympatheticResonance :: Double -> Double -> Double
computeSympatheticResonance freq1 freq2 =
  let
    -- Normalize to ratio
    ratio = if freq1 > freq2 then freq1 / freq2 else freq2 / freq1

    -- Simple harmonic ratios
    harmonics = [1.0, 2.0, 1.5, 4.0/3.0, 5.0/4.0, 6.0/5.0, 3.0, 4.0]

    -- Find closest harmonic
    distances = map (\h -> abs (ratio - h)) harmonics
    minDist = minimum distances

    -- Resonance inversely proportional to distance from harmonic
    resonance = max 0.0 (1.0 - minDist * 2.0)

  in resonance

-- | Classify frequency into Keely force octave band
--
-- Uses 21-octave bands from Keely's 40 Laws.
classifyForceOctave :: Double -> ForceOctave
classifyForceOctave freq =
  let
    -- Logarithmic octave from base frequency (20 Hz = octave 0)
    baseFreq = 20.0
    octave = round (logBase 2 (freq / baseFreq)) :: Int

    -- Map to force band (modulo 105 total octaves)
    band = (octave `mod` 105) `div` 21

  in case band of
    0 -> Sonity
    1 -> SonoThermity
    2 -> Thermism
    3 -> Electricity
    4 -> Atomolity
    _ -> Sonity  -- Wrap around

-- | Select resonator class based on vibratory mode and force octave
selectResonatorClass :: VibratorMode -> ForceOctave -> ResonatorClass
selectResonatorClass mode force = case (mode, force) of
  (Molecular, Sonity)     -> TripleSympathetic
  (Molecular, _)          -> VibratorCore
  (Atomic, Electricity)   -> AtomolicTransducer
  (Atomic, _)             -> EnharmonicAmplifier
  (Etheric, _)            -> AtomolicTransducer

-- =============================================================================
-- Timbral Modulation
-- =============================================================================

-- | Modulate timbre based on chamber attributes
modulateTimbre
  :: Double           -- ^ Coherence level [0,1]
  -> Bool             -- ^ Is inverted (shadow)?
  -> Int              -- ^ Shell depth
  -> Timbre
modulateTimbre coherence isInverted shell =
  let
    -- Base timbre
    brightness = coherence * 0.8 + 0.2
    warmth = if isInverted then 0.8 else 0.5

    -- Envelope from shell depth
    attack = fromIntegral shell * 50.0   -- ms
    sustain = coherence
    decay = fromIntegral (10 - shell) * 100.0
    release = decay * 1.5

  in Timbre
    { tmBrightness = brightness
    , tmWarmth = warmth
    , tmAttack = attack
    , tmSustain = sustain
    , tmDecay = decay
    , tmRelease = release
    }

-- | Select instrument archetype based on Ra state
selectInstrument
  :: Dynamic          -- ^ Dynamic level
  -> Contour          -- ^ Melodic contour
  -> VibratorMode     -- ^ Keely vibratory mode
  -> InstrumentArchetype
selectInstrument dynamic contour mode = case (dynamic, contour, mode) of
  (Sforzando, _, _)         -> Gong          -- Emergence
  (_, Static, _)            -> SingingBowl   -- Sustained
  (_, _, Etheric)           -> Voice         -- Human coherence
  (_, Ascending, Molecular) -> Flute         -- Clarity
  (_, Descending, _)        -> Cello         -- Depth/shadow
  (Forte, _, Atomic)        -> Brass         -- Force
  (_, Ascending, Atomic)    -> Strings       -- Triple-sympathetic (fallback for Ascending+Atomic not Forte)
  -- Note: All cases are covered - Molecular, Atomic, Etheric modes with Ascending, Descending, Static contours

-- | Apply timbral filter based on chamber effects
applyTimbreFilter
  :: Bool             -- ^ Is shadow/inverted?
  -> Double           -- ^ Coherence [0,1]
  -> Int              -- ^ Shell depth
  -> TimbreFilter
applyTimbreFilter isShadow coherence shell
  | isShadow && coherence < 0.3 = Phaser 0.5      -- Shadow modulation
  | isShadow                    = LowPass 500.0    -- Suppression
  | coherence > 0.8             = HighPass 200.0   -- Clarity
  | shell > 5                   = Reverb 0.7       -- Chamber resonance
  | otherwise                   = BandPass 200.0 5000.0  -- Balanced
