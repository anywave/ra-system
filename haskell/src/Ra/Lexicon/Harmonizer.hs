{-|
Module      : Ra.Lexicon.Harmonizer
Description : Linguistic-coherence resonance engine
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Translates between human natural language input and scalar-resonant
linguistic tokens for Ra Field Modulation and Fragment Tuning.
Creates a real-time bridge between spoken/thought inputs and Ra
scalar field activations.

== Linguistic Resonance

=== Harmonic Tokens

Each word maps to:

* Tone profile (frequency class)
* Coherence band (intensity range)
* Optional fragment hint (for matched memories)

=== Avatar Modulation

User's avatar style affects token blending:

* Muted: Reduces intensity
* Amplified: Increases resonance
* Poetic: Adds harmonic overtones
* Technical: Precise frequency mapping
-}
module Ra.Lexicon.Harmonizer
  ( -- * Core Types
    HarmonicToken(..)
  , HarmonicTone(..)
  , BandClass(..)
  , UserPhrase(..)

    -- * Lexicon
  , LexiconMap
  , defaultLexicon
  , loadLexicon
  , addToken

    -- * Harmonization
  , harmonizePhrase
  , harmonizeWord
  , phraseResonance

    -- * Avatar Modulation
  , AvatarModulator(..)
  , AvatarProfile(..)
  , applyModulator

    -- * Fragment Linking
  , FragmentID
  , findFragmentHint
  , linkToFragment

    -- * Tone Profiles
  , toneFrequency
  , toneToHarmonic
  , harmonicToTone

    -- * Coherence Bands
  , bandToRange
  , classifyCoherence
  , bandWeight

    -- * Semantic Analysis
  , semanticPolarity
  , emotionalTone
  , intentCategory
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import qualified Data.Text as T
import Data.Text (Text)

-- Ra.Constants not needed here

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Fragment identifier
type FragmentID = String

-- | Harmonic token from lexicon
data HarmonicToken = HarmonicToken
  { htBaseWord      :: !Text            -- ^ Original input word
  , htToneProfile   :: !HarmonicTone    -- ^ Frequency class
  , htCoherenceBand :: !BandClass       -- ^ Coherence intensity range
  , htFragmentHint  :: !(Maybe FragmentID)  -- ^ Matched fragment
  } deriving (Eq, Show)

-- | Harmonic tone classes (based on Keely)
data HarmonicTone
  = ToneRoot         -- ^ Fundamental (C)
  | ToneSecond       -- ^ Second harmonic (D)
  | ToneThird        -- ^ Third harmonic (E)
  | ToneFourth       -- ^ Fourth harmonic (F)
  | ToneFifth        -- ^ Fifth harmonic (G)
  | ToneSixth        -- ^ Sixth harmonic (A)
  | ToneSeventh      -- ^ Seventh harmonic (B)
  | ToneOctave       -- ^ Octave (C')
  | ToneNeutral      -- ^ No specific tone
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Coherence band classification
data BandClass
  = BandLow          -- ^ α < 0.3
  | BandMid          -- ^ 0.3 ≤ α < 0.6
  | BandHigh         -- ^ 0.6 ≤ α < 0.85
  | BandPeak         -- ^ α ≥ 0.85
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | User phrase with harmonization
data UserPhrase = UserPhrase
  { upRawText     :: !Text
  , upTokenized   :: ![HarmonicToken]
  , upResonance   :: !Double          -- ^ Overall resonance [0, 1]
  , upAvatarMod   :: !AvatarModulator
  } deriving (Eq, Show)

-- =============================================================================
-- Lexicon
-- =============================================================================

-- | Lexicon mapping words to harmonic tokens
type LexiconMap = Map Text HarmonicToken

-- | Default lexicon with common resonant words
defaultLexicon :: LexiconMap
defaultLexicon = Map.fromList
  -- High resonance words
  [ (T.pack "love", HarmonicToken (T.pack "love") ToneFifth BandPeak Nothing)
  , (T.pack "peace", HarmonicToken (T.pack "peace") ToneFourth BandPeak Nothing)
  , (T.pack "harmony", HarmonicToken (T.pack "harmony") ToneFifth BandPeak Nothing)
  , (T.pack "light", HarmonicToken (T.pack "light") ToneOctave BandHigh Nothing)
  , (T.pack "heal", HarmonicToken (T.pack "heal") ToneSixth BandHigh Nothing)
  , (T.pack "grow", HarmonicToken (T.pack "grow") ToneThird BandHigh Nothing)
  , (T.pack "create", HarmonicToken (T.pack "create") ToneSecond BandHigh Nothing)
  , (T.pack "joy", HarmonicToken (T.pack "joy") ToneFifth BandPeak Nothing)

  -- Mid resonance words
  , (T.pack "think", HarmonicToken (T.pack "think") ToneThird BandMid Nothing)
  , (T.pack "move", HarmonicToken (T.pack "move") ToneSecond BandMid Nothing)
  , (T.pack "feel", HarmonicToken (T.pack "feel") ToneFourth BandMid Nothing)
  , (T.pack "know", HarmonicToken (T.pack "know") ToneSixth BandMid Nothing)
  , (T.pack "see", HarmonicToken (T.pack "see") ToneFifth BandMid Nothing)
  , (T.pack "hear", HarmonicToken (T.pack "hear") ToneFourth BandMid Nothing)

  -- Low resonance words (shadow/challenge)
  , (T.pack "fear", HarmonicToken (T.pack "fear") ToneRoot BandLow Nothing)
  , (T.pack "pain", HarmonicToken (T.pack "pain") ToneSecond BandLow Nothing)
  , (T.pack "dark", HarmonicToken (T.pack "dark") ToneRoot BandLow Nothing)
  , (T.pack "block", HarmonicToken (T.pack "block") ToneSecond BandLow Nothing)

  -- Neutral/functional words
  , (T.pack "the", HarmonicToken (T.pack "the") ToneNeutral BandMid Nothing)
  , (T.pack "and", HarmonicToken (T.pack "and") ToneNeutral BandMid Nothing)
  , (T.pack "is", HarmonicToken (T.pack "is") ToneNeutral BandMid Nothing)
  , (T.pack "a", HarmonicToken (T.pack "a") ToneNeutral BandMid Nothing)
  ]

-- | Load lexicon from external source (placeholder)
loadLexicon :: FilePath -> IO LexiconMap
loadLexicon _ = pure defaultLexicon  -- Would load from file

-- | Add token to lexicon
addToken :: Text -> HarmonicToken -> LexiconMap -> LexiconMap
addToken = Map.insert

-- =============================================================================
-- Harmonization
-- =============================================================================

-- | Harmonize a phrase
harmonizePhrase :: Text -> LexiconMap -> AvatarProfile -> ScalarField -> UserPhrase
harmonizePhrase rawText lexicon profile field =
  let -- Tokenize and lowercase
      wordList = T.words (T.toLower rawText)

      -- Harmonize each word
      tokens = map (harmonizeWord lexicon) wordList

      -- Apply avatar modulation
      modulator = apProfile profile
      modulatedTokens = map (applyModulatorToToken modulator) tokens

      -- Calculate overall resonance
      resonance = calculateResonance modulatedTokens field

  in UserPhrase
      { upRawText = rawText
      , upTokenized = modulatedTokens
      , upResonance = resonance
      , upAvatarMod = modulator
      }

-- | Harmonize single word
harmonizeWord :: LexiconMap -> Text -> HarmonicToken
harmonizeWord lexicon word =
  case Map.lookup word lexicon of
    Just token -> token
    Nothing ->
      -- Create neutral token for unknown words
      HarmonicToken word ToneNeutral BandMid Nothing

-- | Calculate phrase resonance
phraseResonance :: UserPhrase -> Double
phraseResonance = upResonance

-- Calculate resonance from tokens
calculateResonance :: [HarmonicToken] -> ScalarField -> Double
calculateResonance tokens field =
  let -- Base resonance from bands
      bandWeights = map (bandWeight . htCoherenceBand) tokens
      baseBand = if null bandWeights then 0.5
                 else sum bandWeights / fromIntegral (length bandWeights)

      -- Tone harmony bonus
      tones = map htToneProfile tokens
      toneHarmony = calculateToneHarmony tones

      -- Field coherence influence
      fieldMod = sfCoherence field * 0.2

  in clamp01 (baseBand * 0.6 + toneHarmony * 0.3 + fieldMod + 0.1)

-- Calculate tone harmony (how well tones work together)
calculateToneHarmony :: [HarmonicTone] -> Double
calculateToneHarmony tones =
  let -- Count harmonious pairs (fifths, fourths, thirds)
      pairs = [(t1, t2) | t1 <- tones, t2 <- tones, t1 /= t2]
      harmonious = length $ filter isHarmoniousPair pairs
      total = max 1 (length pairs)
  in fromIntegral harmonious / fromIntegral total

-- Check if two tones form harmonious pair
isHarmoniousPair :: (HarmonicTone, HarmonicTone) -> Bool
isHarmoniousPair (t1, t2) =
  let diff = abs (fromEnum t1 - fromEnum t2)
  in diff `elem` [4, 5, 3, 7]  -- Fifth, fourth, third, octave

-- =============================================================================
-- Avatar Modulation
-- =============================================================================

-- | Avatar modulator style
data AvatarModulator
  = ModMuted         -- ^ Reduces intensity
  | ModAmplified     -- ^ Increases resonance
  | ModPoetic        -- ^ Adds harmonic overtones
  | ModTechnical     -- ^ Precise frequency mapping
  | ModBalanced      -- ^ No modification
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Avatar profile
data AvatarProfile = AvatarProfile
  { apId          :: !String
  , apProfile     :: !AvatarModulator
  , apCoherence   :: !Double
  , apPreferences :: ![Text]
  } deriving (Eq, Show)

-- | Apply modulator to phrase
applyModulator :: AvatarModulator -> UserPhrase -> UserPhrase
applyModulator mod' phrase =
  let factor = modulatorFactor mod'
      newResonance = clamp01 (upResonance phrase * factor)
  in phrase { upResonance = newResonance }

-- Apply modulator to single token
applyModulatorToToken :: AvatarModulator -> HarmonicToken -> HarmonicToken
applyModulatorToToken mod' token =
  case mod' of
    ModPoetic ->
      -- Upgrade band if possible
      let newBand = if htCoherenceBand token < BandPeak
                    then succ (htCoherenceBand token)
                    else htCoherenceBand token
      in token { htCoherenceBand = newBand }
    ModMuted ->
      -- Downgrade band if possible
      let newBand = if htCoherenceBand token > BandLow
                    then pred (htCoherenceBand token)
                    else htCoherenceBand token
      in token { htCoherenceBand = newBand }
    _ -> token

-- Modulator intensity factor
modulatorFactor :: AvatarModulator -> Double
modulatorFactor mod' = case mod' of
  ModMuted -> 0.7
  ModAmplified -> 1.3
  ModPoetic -> 1.1
  ModTechnical -> 1.0
  ModBalanced -> 1.0

-- =============================================================================
-- Fragment Linking
-- =============================================================================

-- | Find fragment hint for word
findFragmentHint :: Text -> LexiconMap -> Maybe FragmentID
findFragmentHint word lexicon =
  case Map.lookup word lexicon of
    Just token -> htFragmentHint token
    Nothing -> Nothing

-- | Link token to fragment
linkToFragment :: HarmonicToken -> FragmentID -> HarmonicToken
linkToFragment token fid = token { htFragmentHint = Just fid }

-- =============================================================================
-- Tone Profiles
-- =============================================================================

-- | Get frequency for tone (Hz)
toneFrequency :: HarmonicTone -> Double
toneFrequency tone = case tone of
  ToneRoot -> 256.0      -- C4
  ToneSecond -> 288.0    -- D4
  ToneThird -> 320.0     -- E4
  ToneFourth -> 341.33   -- F4
  ToneFifth -> 384.0     -- G4
  ToneSixth -> 426.67    -- A4
  ToneSeventh -> 480.0   -- B4
  ToneOctave -> 512.0    -- C5
  ToneNeutral -> 256.0   -- Default to root

-- | Convert tone to harmonic (l, m)
toneToHarmonic :: HarmonicTone -> (Int, Int)
toneToHarmonic tone = case tone of
  ToneRoot -> (0, 0)
  ToneSecond -> (1, 0)
  ToneThird -> (1, 1)
  ToneFourth -> (2, 0)
  ToneFifth -> (2, 1)
  ToneSixth -> (2, 2)
  ToneSeventh -> (3, 0)
  ToneOctave -> (3, 1)
  ToneNeutral -> (0, 0)

-- | Convert harmonic to closest tone
harmonicToTone :: (Int, Int) -> HarmonicTone
harmonicToTone (l, m) = case (l `mod` 4, abs m `mod` 3) of
  (0, 0) -> ToneRoot
  (1, 0) -> ToneSecond
  (1, 1) -> ToneThird
  (2, 0) -> ToneFourth
  (2, 1) -> ToneFifth
  (2, 2) -> ToneSixth
  (3, 0) -> ToneSeventh
  (3, 1) -> ToneOctave
  _ -> ToneNeutral

-- =============================================================================
-- Coherence Bands
-- =============================================================================

-- | Get range for band class
bandToRange :: BandClass -> (Double, Double)
bandToRange band = case band of
  BandLow -> (0.0, 0.3)
  BandMid -> (0.3, 0.6)
  BandHigh -> (0.6, 0.85)
  BandPeak -> (0.85, 1.0)

-- | Classify coherence value to band
classifyCoherence :: Double -> BandClass
classifyCoherence c
  | c >= 0.85 = BandPeak
  | c >= 0.6 = BandHigh
  | c >= 0.3 = BandMid
  | otherwise = BandLow

-- | Get weight for band (for averaging)
bandWeight :: BandClass -> Double
bandWeight band = case band of
  BandLow -> 0.2
  BandMid -> 0.5
  BandHigh -> 0.8
  BandPeak -> 1.0

-- =============================================================================
-- Semantic Analysis
-- =============================================================================

-- | Compute semantic polarity [-1, 1]
semanticPolarity :: [HarmonicToken] -> Double
semanticPolarity tokens =
  let weights = map tokenPolarity tokens
      total = sum weights
      count = max 1 (length weights)
  in total / fromIntegral count

-- Token polarity
tokenPolarity :: HarmonicToken -> Double
tokenPolarity token =
  case htCoherenceBand token of
    BandPeak -> 1.0
    BandHigh -> 0.5
    BandMid -> 0.0
    BandLow -> -0.5

-- | Determine emotional tone
emotionalTone :: [HarmonicToken] -> String
emotionalTone tokens =
  let polarity = semanticPolarity tokens
      dominantBand = if null tokens then BandMid
                     else maximum (map htCoherenceBand tokens)
  in case (polarity > 0.3, dominantBand) of
       (True, BandPeak) -> "Joyful"
       (True, BandHigh) -> "Positive"
       (True, _) -> "Neutral-Positive"
       (False, BandLow) -> "Challenging"
       (False, _) -> "Neutral"

-- | Determine intent category
intentCategory :: [HarmonicToken] -> String
intentCategory tokens =
  let tones = map htToneProfile tokens
      hasFifth = ToneFifth `elem` tones
      hasRoot = ToneRoot `elem` tones
      hasSixth = ToneSixth `elem` tones
  in case (hasFifth, hasRoot, hasSixth) of
       (True, True, _) -> "Transforming"
       (True, False, _) -> "Harmonizing"
       (_, True, False) -> "Grounding"
       (_, _, True) -> "Healing"
       _ -> "General"

-- =============================================================================
-- Internal Types
-- =============================================================================

-- | Scalar field (simplified)
data ScalarField = ScalarField
  { sfCoherence :: !Double
  , sfIntensity :: !Double
  } deriving (Eq, Show)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
