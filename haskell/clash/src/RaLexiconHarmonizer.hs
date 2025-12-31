{-|
Module      : RaLexiconHarmonizer
Description : Linguistic-Resonance Token Translator
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 46: Dynamic linguistic-resonance translator that receives natural
language input, translates to HarmonicTokens, aligns with Scalar Field
logic, and adjusts output per Avatar resonance.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaLexiconHarmonizer where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Coherence band thresholds (8-bit scaled)
highCoherenceThreshold :: Unsigned 8
highCoherenceThreshold = 184   -- 0.72 * 255

midCoherenceThreshold :: Unsigned 8
midCoherenceThreshold = 102    -- 0.40 * 255

-- | Phrase mapping target (95%)
mappingTargetThreshold :: Unsigned 8
mappingTargetThreshold = 242   -- 0.95 * 255

-- | Coherence band classification
data CoherenceBand
  = BandHigh
  | BandMid
  | BandLow
  deriving (Generic, NFDataX, Eq, Show)

-- | Tone profile shapes
data ToneProfile
  = ToneFlat
  | ToneRising
  | ToneFalling
  | ToneWave
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Avatar modulation modes
data AvatarToneMode
  = ModeNeutral
  | ModeMuted
  | ModePoetic
  | ModeFormal
  | ModeBlunt
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Ra coordinate (for fragment anchoring)
data RaCoordinate = RaCoordinate
  { rcTheta :: Unsigned 5    -- 0-26
  , rcPhi   :: Unsigned 3    -- 0-5
  , rcH     :: Unsigned 3    -- 0-4
  } deriving (Generic, NFDataX, Eq)

-- | Harmonic token (synthesized from natural language)
data HarmonicToken = HarmonicToken
  { htHarmonicL     :: Unsigned 4     -- Spherical harmonic l (0-9)
  , htHarmonicM     :: Signed 8       -- Spherical harmonic m (-9 to 9)
  , htCoherenceBand :: CoherenceBand  -- High/Mid/Low
  , htToneProfile   :: ToneProfile    -- Flat/Rising/Falling/Wave
  , htAmplitude     :: Unsigned 8     -- 0-255
  , htFragmentHint  :: Unsigned 8     -- Fragment ID (0 = none)
  , htHasAnchor     :: Bool           -- Whether scalar anchor is valid
  , htAnchorCoord   :: RaCoordinate   -- Optional scalar anchor
  } deriving (Generic, NFDataX)

-- | Avatar profile for modulation
data AvatarProfile = AvatarProfile
  { apToneMode         :: AvatarToneMode
  , apResonanceAffinity :: Unsigned 8   -- 0-255 (0.0-1.0 scaled)
  } deriving (Generic, NFDataX)

-- | Scalar field state
data ScalarField = ScalarField
  { sfCoherence   :: Unsigned 8       -- 0-255
  , sfDominantL   :: Unsigned 4
  , sfDominantM   :: Signed 8
  , sfPhaseAngle  :: Unsigned 16      -- 0-65535 (0-2π)
  } deriving (Generic, NFDataX)

-- | Get coherence band from coherence value
getCoherenceBand :: Unsigned 8 -> CoherenceBand
getCoherenceBand coh
  | coh >= highCoherenceThreshold = BandHigh
  | coh >= midCoherenceThreshold  = BandMid
  | otherwise                      = BandLow

-- | Create neutral fallback token (for unknown words)
createFallbackToken :: Unsigned 8 -> HarmonicToken
createFallbackToken fieldCoherence = HarmonicToken
  { htHarmonicL = 0
  , htHarmonicM = 0
  , htCoherenceBand = BandMid
  , htToneProfile = ToneFlat
  , htAmplitude = 128  -- Reduced amplitude for unknown
  , htFragmentHint = 0
  , htHasAnchor = False
  , htAnchorCoord = RaCoordinate 0 0 0
  }

-- | Create token from lexicon entry
createTokenFromEntry
  :: Unsigned 4      -- Harmonic L
  -> Signed 8        -- Harmonic M
  -> ToneProfile     -- Tone profile
  -> Unsigned 8      -- Field coherence
  -> HarmonicToken
createTokenFromEntry l m tone fieldCoh = HarmonicToken
  { htHarmonicL = min 9 l
  , htHarmonicM = m
  , htCoherenceBand = getCoherenceBand fieldCoh
  , htToneProfile = tone
  , htAmplitude = 255
  , htFragmentHint = 0
  , htHasAnchor = False
  , htAnchorCoord = RaCoordinate 0 0 0
  }

-- | Apply avatar modulation to token
applyAvatarMod :: AvatarProfile -> HarmonicToken -> HarmonicToken
applyAvatarMod avatar token = case apToneMode avatar of
  ModeNeutral -> token

  ModeMuted -> HarmonicToken
    { htHarmonicL = htHarmonicL token
    , htHarmonicM = htHarmonicM token
    , htCoherenceBand = downgradeBand (htCoherenceBand token)
    , htToneProfile = ToneFlat
    , htAmplitude = htAmplitude token `shiftR` 1  -- Halve amplitude
    , htFragmentHint = htFragmentHint token
    , htHasAnchor = htHasAnchor token
    , htAnchorCoord = htAnchorCoord token
    }

  ModePoetic -> HarmonicToken
    { htHarmonicL = htHarmonicL token
    , htHarmonicM = htHarmonicM token `div` 2  -- Soften M
    , htCoherenceBand = htCoherenceBand token
    , htToneProfile = ToneWave
    , htAmplitude = min 255 (htAmplitude token + 25)  -- Boost amplitude
    , htFragmentHint = htFragmentHint token
    , htHasAnchor = htHasAnchor token
    , htAnchorCoord = htAnchorCoord token
    }

  ModeFormal -> HarmonicToken
    { htHarmonicL = htHarmonicL token
    , htHarmonicM = 0  -- Zero M variance
    , htCoherenceBand = htCoherenceBand token
    , htToneProfile = ToneFlat
    , htAmplitude = htAmplitude token
    , htFragmentHint = htFragmentHint token
    , htHasAnchor = htHasAnchor token
    , htAnchorCoord = htAnchorCoord token
    }

  ModeBlunt -> HarmonicToken
    { htHarmonicL = min 9 (htHarmonicL token + 1)  -- Increase L
    , htHarmonicM = htHarmonicM token
    , htCoherenceBand = htCoherenceBand token
    , htToneProfile = ToneFalling
    , htAmplitude = min 255 (htAmplitude token + 33)  -- Boost amplitude
    , htFragmentHint = htFragmentHint token
    , htHasAnchor = htHasAnchor token
    , htAnchorCoord = htAnchorCoord token
    }

-- | Downgrade coherence band
downgradeBand :: CoherenceBand -> CoherenceBand
downgradeBand BandHigh = BandMid
downgradeBand BandMid  = BandLow
downgradeBand BandLow  = BandLow

-- | Compute resonance score from token and field
computeTokenResonance :: HarmonicToken -> ScalarField -> Unsigned 8
computeTokenResonance token field =
  let -- Base score from coherence band
      bandScore = case htCoherenceBand token of
        BandHigh -> 230  -- ~0.9
        BandMid  -> 153  -- ~0.6
        BandLow  -> 77   -- ~0.3

      -- Harmonic alignment bonus
      lDiff = if htHarmonicL token > sfDominantL field
              then htHarmonicL token - sfDominantL field
              else sfDominantL field - htHarmonicL token
      mDiff = abs (htHarmonicM token - sfDominantM field)

      -- Alignment factor: 255 / (1 + diff)
      alignFactor = 255 `div` (1 + resize lDiff + resize mDiff `div` 2) :: Unsigned 8

      -- Combine: (bandScore * alignFactor * fieldCoherence) / 65536
      product1 = resize bandScore * resize alignFactor :: Unsigned 16
      product2 = (product1 * resize (sfCoherence field)) `shiftR` 16 :: Unsigned 16

  in resize $ min 255 product2

-- | Validate fragment hint coordinate
validateFragmentHint :: HarmonicToken -> Bool
validateFragmentHint token
  | not (htHasAnchor token) = True  -- No anchor = valid
  | otherwise =
      let coord = htAnchorCoord token
      in rcTheta coord < 27 && rcPhi coord < 6 && rcH coord < 5

-- | Compute token coverage (known/total * 255)
computeCoverage :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
computeCoverage knownCount totalCount
  | totalCount == 0 = 0
  | otherwise = resize ((resize knownCount * 255) `div` resize totalCount :: Unsigned 16)

-- | Check if coverage meets target
meetsCoverageTarget :: Unsigned 8 -> Bool
meetsCoverageTarget coverage = coverage >= mappingTargetThreshold

-- | Token modulation pipeline
tokenModPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (AvatarProfile, HarmonicToken)
  -> Signal dom HarmonicToken
tokenModPipeline input = uncurry applyAvatarMod <$> input

-- | Resonance computation pipeline
resonancePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (HarmonicToken, ScalarField)
  -> Signal dom (Unsigned 8)
resonancePipeline input = uncurry computeTokenResonance <$> input

-- | Coherence band pipeline
coherenceBandPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom CoherenceBand
coherenceBandPipeline = fmap getCoherenceBand

-- | Coverage validation pipeline
coveragePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8)
  -> Signal dom Bool
coveragePipeline input = meetsCoverageTarget . uncurry computeCoverage <$> input
