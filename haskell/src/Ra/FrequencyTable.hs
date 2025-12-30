{-|
Module      : Ra.FrequencyTable
Description : Ra-frequency mapping table for harmonics reference
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Comprehensive reference table linking Ra field harmonics to known frequency
systems including Rife, Tesla, Keely, chakra, and neural resonance states.

== Frequency Systems

=== Rife Frequencies

Dr. Royal Rife's therapeutic frequencies:

* 20 Hz - General tonic
* 728 Hz - Bacteria/virus
* 880 Hz - Tissue regeneration
* 5000 Hz - Pain relief

=== Tesla Impulse Ranges

Nikola Tesla's resonance research:

* 7.83 Hz - Schumann resonance
* 369-963 Hz - Sacred solfeggio
* 3.6 MHz - Radiant energy pulses

=== Keely Octaves

John Keely's 21-octave system:

* Octaves 1-7: Material/physical
* Octaves 8-14: Etheric/vital
* Octaves 15-21: Spiritual/mental

=== Chakra Resonance

Traditional chakra frequencies:

* Root: 256 Hz (C)
* Sacral: 288 Hz (D)
* Solar: 320 Hz (E)
* Heart: 341.3 Hz (F)
* Throat: 384 Hz (G)
* Third Eye: 426.7 Hz (A)
* Crown: 480 Hz (B)
-}
module Ra.FrequencyTable
  ( -- * Frequency Entry
    FrequencyEntry(..)
  , FrequencySystem(..)
  , FrequencyCategory(..)

    -- * Lookup Functions
  , lookupByHarmonic
  , lookupByFrequency
  , lookupByCategory
  , lookupBySystem

    -- * Rife Frequencies
  , rifeFrequencies
  , rifeLookup
  , rifeForCondition

    -- * Tesla Ranges
  , teslaRanges
  , teslaLookup
  , schumannBase

    -- * Keely Octaves
  , keelyOctaves
  , keelyLookup
  , harmonicToOctave

    -- * Chakra Frequencies
  , chakraFrequencies
  , chakraLookup
  , chakraForNote

    -- * Neural Resonance
  , neuralFrequencies
  , brainwaveState
  , neuralLookup

    -- * Solfeggio
  , solfeggioFrequencies
  , solfeggioLookup
  , solfeggioMeaning

    -- * Master Table
  , masterTable
  , allFrequencies
  , frequencyRange
  ) where

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Frequency table entry
data FrequencyEntry = FrequencyEntry
  { feFrequency   :: !Double          -- ^ Frequency in Hz
  , feHarmonic    :: !(Int, Int)      -- ^ Ra harmonic (l, m)
  , feSystem      :: !FrequencySystem -- ^ Source system
  , feCategory    :: !FrequencyCategory
  , feName        :: !String          -- ^ Entry name
  , feDescription :: !String          -- ^ Description
  } deriving (Eq, Show)

-- | Frequency system source
data FrequencySystem
  = Rife           -- ^ Rife therapeutic
  | Tesla          -- ^ Tesla resonance
  | Keely          -- ^ Keely octaves
  | Chakra         -- ^ Chakra system
  | Neural         -- ^ Brainwave states
  | Solfeggio      -- ^ Sacred solfeggio
  | Schumann       -- ^ Earth resonance
  | Custom         -- ^ User-defined
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Frequency category
data FrequencyCategory
  = Healing        -- ^ Therapeutic/healing
  | Consciousness  -- ^ Consciousness states
  | Energy         -- ^ Energy/vitality
  | Spiritual      -- ^ Spiritual/transcendent
  | Physical       -- ^ Physical body
  | Emotional      -- ^ Emotional balance
  | Mental         -- ^ Mental clarity
  | Universal      -- ^ Universal/cosmic
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Lookup Functions
-- =============================================================================

-- | Lookup by Ra harmonic
lookupByHarmonic :: (Int, Int) -> [FrequencyEntry]
lookupByHarmonic harm = filter (\e -> feHarmonic e == harm) masterTable

-- | Lookup by frequency (within tolerance)
lookupByFrequency :: Double -> Double -> [FrequencyEntry]
lookupByFrequency freq tolerance =
  filter (\e -> abs (feFrequency e - freq) <= tolerance) masterTable

-- | Lookup by category
lookupByCategory :: FrequencyCategory -> [FrequencyEntry]
lookupByCategory cat = filter (\e -> feCategory e == cat) masterTable

-- | Lookup by system
lookupBySystem :: FrequencySystem -> [FrequencyEntry]
lookupBySystem sys = filter (\e -> feSystem e == sys) masterTable

-- =============================================================================
-- Rife Frequencies
-- =============================================================================

-- | Rife therapeutic frequencies
rifeFrequencies :: [FrequencyEntry]
rifeFrequencies =
  [ FrequencyEntry 20.0 (0, 0) Rife Healing "General Tonic" "Overall wellness support"
  , FrequencyEntry 728.0 (2, 1) Rife Healing "Antibacterial" "Bacteria neutralization"
  , FrequencyEntry 880.0 (2, 2) Rife Healing "Regeneration" "Tissue regeneration"
  , FrequencyEntry 1550.0 (3, 1) Rife Physical "Bone Healing" "Bone and cartilage"
  , FrequencyEntry 2127.0 (3, 2) Rife Healing "Pain Relief" "Pain management"
  , FrequencyEntry 5000.0 (4, 0) Rife Physical "Inflammation" "Anti-inflammatory"
  , FrequencyEntry 10000.0 (5, 0) Rife Energy "Vitality" "Energy enhancement"
  ]

-- | Rife lookup by name
rifeLookup :: String -> Maybe FrequencyEntry
rifeLookup name =
  case filter (\e -> feName e == name) rifeFrequencies of
    (x:_) -> Just x
    [] -> Nothing

-- | Rife for condition
rifeForCondition :: String -> [FrequencyEntry]
rifeForCondition condition =
  filter (\e -> condition `isSubstringOf` feDescription e) rifeFrequencies

-- =============================================================================
-- Tesla Ranges
-- =============================================================================

-- | Tesla frequency ranges
teslaRanges :: [FrequencyEntry]
teslaRanges =
  [ FrequencyEntry 7.83 (0, 0) Tesla Universal "Schumann" "Earth resonance base"
  , FrequencyEntry 12.0 (1, 0) Tesla Consciousness "Alpha Peak" "Relaxed awareness"
  , FrequencyEntry 40.0 (1, 1) Tesla Mental "Gamma" "Peak cognition"
  , FrequencyEntry 369.0 (2, 0) Tesla Spiritual "369 Key" "Tesla's key frequency"
  , FrequencyEntry 432.0 (2, 1) Tesla Spiritual "Cosmic A" "Universal tuning"
  , FrequencyEntry 528.0 (2, 2) Tesla Healing "DNA Repair" "Transformation frequency"
  , FrequencyEntry 963.0 (3, 0) Tesla Spiritual "Divine" "Highest solfeggio"
  ]

-- | Tesla lookup
teslaLookup :: String -> Maybe FrequencyEntry
teslaLookup name =
  case filter (\e -> feName e == name) teslaRanges of
    (x:_) -> Just x
    [] -> Nothing

-- | Schumann base frequency
schumannBase :: Double
schumannBase = 7.83

-- =============================================================================
-- Keely Octaves
-- =============================================================================

-- | Keely's 21-octave system
keelyOctaves :: [FrequencyEntry]
keelyOctaves =
  [ -- Physical octaves (1-7)
    FrequencyEntry 256.0 (1, 0) Keely Physical "Octave 1" "Dense matter"
  , FrequencyEntry 512.0 (2, 0) Keely Physical "Octave 2" "Solid structure"
  , FrequencyEntry 1024.0 (3, 0) Keely Physical "Octave 3" "Molecular"
  , FrequencyEntry 2048.0 (4, 0) Keely Physical "Octave 4" "Cellular"
  , FrequencyEntry 4096.0 (5, 0) Keely Physical "Octave 5" "Tissue"
  , FrequencyEntry 8192.0 (6, 0) Keely Physical "Octave 6" "Organ"
  , FrequencyEntry 16384.0 (7, 0) Keely Physical "Octave 7" "System"
    -- Etheric octaves (8-14)
  , FrequencyEntry 32768.0 (8, 0) Keely Energy "Octave 8" "Vital force"
  , FrequencyEntry 65536.0 (9, 0) Keely Energy "Octave 9" "Pranic"
  , FrequencyEntry 131072.0 (10, 0) Keely Energy "Octave 10" "Etheric"
    -- Spiritual octaves (15-21)
  , FrequencyEntry 524288.0 (15, 0) Keely Spiritual "Octave 15" "Mental"
  , FrequencyEntry 2097152.0 (18, 0) Keely Spiritual "Octave 18" "Causal"
  , FrequencyEntry 8388608.0 (21, 0) Keely Spiritual "Octave 21" "Divine"
  ]

-- | Keely lookup by octave
keelyLookup :: Int -> Maybe FrequencyEntry
keelyLookup octave =
  let name = "Octave " ++ show octave
  in case filter (\e -> feName e == name) keelyOctaves of
       (x:_) -> Just x
       [] -> Nothing

-- | Map Ra harmonic to Keely octave
harmonicToOctave :: (Int, Int) -> Int
harmonicToOctave (l, m) = (l + abs m) `mod` 21 + 1

-- =============================================================================
-- Chakra Frequencies
-- =============================================================================

-- | Chakra frequencies (Western scale)
chakraFrequencies :: [FrequencyEntry]
chakraFrequencies =
  [ FrequencyEntry 256.0 (0, 0) Chakra Physical "Root" "Muladhara - Grounding"
  , FrequencyEntry 288.0 (1, 0) Chakra Emotional "Sacral" "Svadhisthana - Creativity"
  , FrequencyEntry 320.0 (2, 0) Chakra Energy "Solar" "Manipura - Power"
  , FrequencyEntry 341.33 (3, 0) Chakra Emotional "Heart" "Anahata - Love"
  , FrequencyEntry 384.0 (4, 0) Chakra Mental "Throat" "Vishuddha - Expression"
  , FrequencyEntry 426.67 (5, 0) Chakra Consciousness "Third Eye" "Ajna - Intuition"
  , FrequencyEntry 480.0 (6, 0) Chakra Spiritual "Crown" "Sahasrara - Unity"
  ]

-- | Chakra lookup by name
chakraLookup :: String -> Maybe FrequencyEntry
chakraLookup name =
  case filter (\e -> feName e == name) chakraFrequencies of
    (x:_) -> Just x
    [] -> Nothing

-- | Chakra for musical note
chakraForNote :: String -> Maybe FrequencyEntry
chakraForNote note = case note of
  "C" -> Just (head chakraFrequencies)
  "D" -> Just (chakraFrequencies !! 1)
  "E" -> Just (chakraFrequencies !! 2)
  "F" -> Just (chakraFrequencies !! 3)
  "G" -> Just (chakraFrequencies !! 4)
  "A" -> Just (chakraFrequencies !! 5)
  "B" -> Just (chakraFrequencies !! 6)
  _ -> Nothing

-- =============================================================================
-- Neural Resonance
-- =============================================================================

-- | Neural/brainwave frequencies
neuralFrequencies :: [FrequencyEntry]
neuralFrequencies =
  [ FrequencyEntry 0.5 (0, 0) Neural Consciousness "Delta Low" "Deep sleep"
  , FrequencyEntry 2.0 (0, 0) Neural Consciousness "Delta" "Healing sleep"
  , FrequencyEntry 4.0 (0, 0) Neural Consciousness "Theta Low" "Deep meditation"
  , FrequencyEntry 6.0 (1, 0) Neural Consciousness "Theta" "Creative insight"
  , FrequencyEntry 8.0 (1, 0) Neural Consciousness "Alpha Low" "Relaxed focus"
  , FrequencyEntry 10.0 (1, 1) Neural Consciousness "Alpha" "Calm awareness"
  , FrequencyEntry 12.0 (1, 1) Neural Consciousness "Alpha High" "Light meditation"
  , FrequencyEntry 15.0 (2, 0) Neural Mental "Beta Low" "Alert focus"
  , FrequencyEntry 20.0 (2, 1) Neural Mental "Beta" "Active thinking"
  , FrequencyEntry 30.0 (2, 2) Neural Mental "Beta High" "Intense focus"
  , FrequencyEntry 40.0 (3, 0) Neural Mental "Gamma" "Peak performance"
  , FrequencyEntry 100.0 (4, 0) Neural Mental "High Gamma" "Transcendence"
  ]

-- | Determine brainwave state from frequency
brainwaveState :: Double -> String
brainwaveState freq
  | freq < 4 = "Delta"
  | freq < 8 = "Theta"
  | freq < 13 = "Alpha"
  | freq < 30 = "Beta"
  | otherwise = "Gamma"

-- | Neural lookup
neuralLookup :: String -> Maybe FrequencyEntry
neuralLookup state =
  case filter (\e -> feName e == state) neuralFrequencies of
    (x:_) -> Just x
    [] -> Nothing

-- =============================================================================
-- Solfeggio
-- =============================================================================

-- | Sacred solfeggio frequencies
solfeggioFrequencies :: [FrequencyEntry]
solfeggioFrequencies =
  [ FrequencyEntry 174.0 (0, 0) Solfeggio Physical "UT quent" "Foundation/security"
  , FrequencyEntry 285.0 (1, 0) Solfeggio Healing "RE sonare" "Cellular healing"
  , FrequencyEntry 396.0 (1, 1) Solfeggio Emotional "MI ra" "Liberation from fear"
  , FrequencyEntry 417.0 (2, 0) Solfeggio Energy "FA muli" "Facilitating change"
  , FrequencyEntry 528.0 (2, 1) Solfeggio Healing "SOL ve" "DNA repair, transformation"
  , FrequencyEntry 639.0 (2, 2) Solfeggio Emotional "LA tere" "Relationships, connection"
  , FrequencyEntry 741.0 (3, 0) Solfeggio Mental "SI que" "Expression, solutions"
  , FrequencyEntry 852.0 (3, 1) Solfeggio Consciousness "LA" "Awakening intuition"
  , FrequencyEntry 963.0 (3, 2) Solfeggio Spiritual "SI" "Divine consciousness"
  ]

-- | Solfeggio lookup
solfeggioLookup :: Double -> Maybe FrequencyEntry
solfeggioLookup freq =
  case filter (\e -> abs (feFrequency e - freq) < 1.0) solfeggioFrequencies of
    (x:_) -> Just x
    [] -> Nothing

-- | Solfeggio meaning
solfeggioMeaning :: Double -> String
solfeggioMeaning freq = case solfeggioLookup freq of
  Just entry -> feDescription entry
  Nothing -> "Unknown solfeggio"

-- =============================================================================
-- Master Table
-- =============================================================================

-- | Complete master frequency table
masterTable :: [FrequencyEntry]
masterTable = concat
  [ rifeFrequencies
  , teslaRanges
  , keelyOctaves
  , chakraFrequencies
  , neuralFrequencies
  , solfeggioFrequencies
  , schumannHarmonics
  ]

-- Schumann resonance harmonics
schumannHarmonics :: [FrequencyEntry]
schumannHarmonics =
  [ FrequencyEntry 7.83 (0, 0) Schumann Universal "Schumann 1" "Fundamental"
  , FrequencyEntry 14.3 (1, 0) Schumann Universal "Schumann 2" "First harmonic"
  , FrequencyEntry 20.8 (1, 1) Schumann Universal "Schumann 3" "Second harmonic"
  , FrequencyEntry 27.3 (2, 0) Schumann Universal "Schumann 4" "Third harmonic"
  , FrequencyEntry 33.8 (2, 1) Schumann Universal "Schumann 5" "Fourth harmonic"
  , FrequencyEntry (7.83 * phi) (0, 0) Schumann Spiritual "Schumann Phi" "Golden Schumann"
  ]

-- | All unique frequencies
allFrequencies :: [Double]
allFrequencies = map feFrequency masterTable

-- | Get frequency range for system
frequencyRange :: FrequencySystem -> (Double, Double)
frequencyRange sys =
  let entries = lookupBySystem sys
      freqs = map feFrequency entries
  in if null freqs
     then (0, 0)
     else (minimum freqs, maximum freqs)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Check if string is substring
isSubstringOf :: String -> String -> Bool
isSubstringOf needle haystack = any (needle `isPrefixOf'`) (tails' haystack)

isPrefixOf' :: String -> String -> Bool
isPrefixOf' [] _ = True
isPrefixOf' _ [] = False
isPrefixOf' (x:xs) (y:ys) = x == y && isPrefixOf' xs ys

tails' :: [a] -> [[a]]
tails' [] = [[]]
tails' xs@(_:xs') = xs : tails' xs'
