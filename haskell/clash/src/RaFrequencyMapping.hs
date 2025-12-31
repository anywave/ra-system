{-|
Module      : RaFrequencyMapping
Description : Ra-Frequency Mapping Table
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 39: Comprehensive frequency mapping linking Ra scalar field harmonics
to Rife, Tesla, Keely, and Chakra/Neural frequency systems.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaFrequencyMapping where

import Clash.Prelude

-- | Phi constant scaled to 16-bit fixed point (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Keely base unit (111 Hz)
keelyBase :: Unsigned 16
keelyBase = 111

-- | Neural/brainwave band
data NeuralBand = SubDelta | Delta | Theta | Alpha | Beta | Gamma | SuperGamma
  deriving (Generic, NFDataX, Eq, Show)

-- | Chakra center
data ChakraCenter = Root | Sacral | SolarPlexus | Heart | Throat | ThirdEye | Crown
  deriving (Generic, NFDataX, Eq, Show)

-- | Frequency match type
data MatchType = ExactMatch | NearestMatch | SymbolicMatch | RangeMatch
  deriving (Generic, NFDataX, Eq, Show)

-- | Ra harmonic type
data RaHarmonicType
  = TonicRoot | TonicSacral | TonicHeart | TonicConnection
  | TonicIntuition | TonicSpirit | TonicCrown
  | KeelyBase | KeelyTriple | KeelyHex | KeelyNines
  | Tesla369Low | Tesla369Mid | PhiBase | Schumann
  deriving (Generic, NFDataX, Eq, Show)

-- | Frequency mapping entry
data FrequencyMapping = FrequencyMapping
  { fmRaType       :: RaHarmonicType
  , fmFrequencyHz  :: Unsigned 16
  , fmNeuralBand   :: NeuralBand
  , fmChakra       :: Maybe ChakraCenter
  , fmMatchType    :: MatchType
  } deriving (Generic, NFDataX)

-- | Solfeggio frequencies (Hz)
solfeggioUT :: Unsigned 16
solfeggioUT = 396

solfeggioRE :: Unsigned 16
solfeggioRE = 417

solfeggioMI :: Unsigned 16
solfeggioMI = 528

solfeggioFA :: Unsigned 16
solfeggioFA = 639

solfeggioSOL :: Unsigned 16
solfeggioSOL = 741

solfeggioLA :: Unsigned 16
solfeggioLA = 852

-- | Crown frequency
crownFreq :: Unsigned 16
crownFreq = 963

-- | Schumann resonance (Hz * 100 for fixed point)
schumannHz100 :: Unsigned 16
schumannHz100 = 783  -- 7.83 Hz

-- | Get neural band for frequency
getNeuralBand :: Unsigned 16 -> NeuralBand
getNeuralBand hz
  | hz < 1    = SubDelta
  | hz < 4    = Delta
  | hz < 8    = Theta
  | hz < 13   = Alpha
  | hz < 30   = Beta
  | hz < 100  = Gamma
  | otherwise = SuperGamma

-- | Get chakra for solfeggio frequency
getChakraForFreq :: Unsigned 16 -> Maybe ChakraCenter
getChakraForFreq freq
  | freq == 396 = Just Root
  | freq == 417 = Just Sacral
  | freq == 528 = Just SolarPlexus
  | freq == 639 = Just Heart
  | freq == 741 = Just Throat
  | freq == 852 = Just ThirdEye
  | freq == 963 = Just Crown
  | otherwise   = Nothing

-- | Keely 3:6:9 ratio to Hz
-- 3:6:9 → 333, 666, 999
keelyRatioToHz :: Unsigned 4 -> Unsigned 16
keelyRatioToHz multiplier = resize multiplier * keelyBase

-- | Get Ra harmonic frequency
getRaHarmonicFreq :: RaHarmonicType -> Unsigned 16
getRaHarmonicFreq harmType = case harmType of
  TonicRoot       -> 396
  TonicSacral     -> 417
  TonicHeart      -> 528
  TonicConnection -> 639
  TonicIntuition  -> 741
  TonicSpirit     -> 852
  TonicCrown      -> 963
  KeelyBase       -> 111
  KeelyTriple     -> 333
  KeelyHex        -> 666
  KeelyNines      -> 999
  Tesla369Low     -> 6    -- Representative (3-9 Hz range)
  Tesla369Mid     -> 60   -- Representative (30-90 Hz range)
  PhiBase         -> 162  -- φ * 100
  Schumann        -> 8    -- ~7.83 rounded

-- | Check if frequency is Rife healing frequency
isRifeFrequency :: Unsigned 16 -> Bool
isRifeFrequency freq = freq `elem` rifeFreqs
  where
    rifeFreqs = 304 :> 320 :> 465 :> 528 :> 660 :>
                690 :> 728 :> 787 :> 880 :> 5000 :> Nil

-- | Check if frequencies are harmonically related
-- Returns True if ratio is close to common harmonic ratio
harmonicallyRelated :: Unsigned 16 -> Unsigned 16 -> Bool
harmonicallyRelated f1 f2
  | f1 == 0 || f2 == 0 = False
  | otherwise =
      let larger = max f1 f2
          smaller = max 1 (min f1 f2)
          ratio = (larger * 100) `div` smaller  -- Ratio * 100
      in ratio `elem` harmonicRatios
  where
    harmonicRatios = 100 :> 150 :> 200 :> 250 :> 300 :> 400 :> 500 :>
                     162 :> 324 :> 62 :> Nil  -- 1:1, 3:2, 2:1, etc., plus φ ratios

-- | Get Tesla range for frequency
data TeslaRange = LowImpulse | MidImpulse | HighImpulse | ScalarBase
  deriving (Generic, NFDataX, Eq, Show)

getTeslaRange :: Unsigned 16 -> TeslaRange
getTeslaRange hz
  | hz <= 10  = LowImpulse
  | hz <= 100 = MidImpulse
  | hz <= 1000 = HighImpulse
  | otherwise = HighImpulse

-- | Check if in Tesla 3:6:9 scalar base (3-9 Hz)
isTeslaScalarBase :: Unsigned 16 -> Bool
isTeslaScalarBase hz = hz >= 3 && hz <= 9

-- | Build frequency mapping for Ra harmonic
buildMapping :: RaHarmonicType -> FrequencyMapping
buildMapping harmType =
  let freq = getRaHarmonicFreq harmType
  in FrequencyMapping
    { fmRaType = harmType
    , fmFrequencyHz = freq
    , fmNeuralBand = getNeuralBand freq
    , fmChakra = getChakraForFreq freq
    , fmMatchType = if isRifeFrequency freq then ExactMatch else SymbolicMatch
    }

-- | Canonical Ra harmonics table (as ROM)
raHarmonicsTable :: Vec 14 FrequencyMapping
raHarmonicsTable = map buildMapping harmonicTypes
  where
    harmonicTypes = TonicRoot :> TonicSacral :> TonicHeart :> TonicConnection :>
                    TonicIntuition :> TonicSpirit :> TonicCrown :>
                    KeelyBase :> KeelyTriple :> KeelyHex :> KeelyNines :>
                    Tesla369Low :> Tesla369Mid :> Schumann :> Nil

-- | Lookup Ra harmonic by index
lookupHarmonic :: Unsigned 4 -> FrequencyMapping
lookupHarmonic idx = raHarmonicsTable !! idx

-- | Find neural band for arbitrary frequency pipeline
neuralBandPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 16)
  -> Signal dom NeuralBand
neuralBandPipeline = fmap getNeuralBand

-- | Chakra mapping pipeline
chakraPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 16)
  -> Signal dom (Maybe ChakraCenter)
chakraPipeline = fmap getChakraForFreq

-- | Harmonic lookup pipeline
harmonicLookupPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 4)
  -> Signal dom FrequencyMapping
harmonicLookupPipeline = fmap lookupHarmonic
