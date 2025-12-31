{-|
Module      : RaMemoryFragmentPalette
Description : Scalar Fragment Memory Palettes
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 67: Models memory fragments as scalar pigment nodes in a
phase-continuous coherence palette with recall strength based on
alpha depth using sigmoid transform.

Supports volatility calculation and inversion artifacts.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaMemoryFragmentPalette where

import Clash.Prelude

-- | Phi constant scaled
phi16 :: Unsigned 16
phi16 = 1657

-- | Volatility window size
volatilityWindowSize :: Unsigned 4
volatilityWindowSize = 8

-- | Sigmoid parameters (scaled)
sigmoidSteepness :: Unsigned 8
sigmoidSteepness = 6

sigmoidMidpoint :: Unsigned 8
sigmoidMidpoint = 128  -- 0.5 * 255

-- | Inversion artifact types
data InversionArtifact
  = ArtifactNone
  | ArtifactDreamBleed
  | ArtifactShadowEcho
  | ArtifactConsentDrift
  | ArtifactPhaseSlip
  | ArtifactMemoryFade
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Phase ID for Ra.Identity sync
data PhaseID = PhaseID
  { piNamespace  :: Unsigned 8   -- Namespace ID
  , piPhaseIndex :: Unsigned 8   -- Phase index
  } deriving (Generic, NFDataX, Eq, Show)

-- | Fragment ID
newtype FragmentID = FragmentID { unFragmentID :: Unsigned 16 }
  deriving (Generic, NFDataX, Eq, Show)

-- | Fragment node
data FragmentNode = FragmentNode
  { fnFragmentId       :: FragmentID
  , fnAlphaDepth       :: Unsigned 8    -- Transformed depth (0-255)
  , fnRawAlpha         :: Unsigned 8    -- Original alpha
  , fnHasPhaseLock     :: Bool
  , fnPhaseLock        :: PhaseID
  , fnInversionArtifact :: InversionArtifact
  } deriving (Generic, NFDataX, Eq, Show)

-- | Alpha trace for volatility (8-sample rolling window)
data AlphaTrace = AlphaTrace
  { atSamples   :: Vec 8 (Unsigned 8)
  , atCount     :: Unsigned 4
  , atSum       :: Unsigned 16
  , atSumSquares :: Unsigned 24
  } deriving (Generic, NFDataX, Eq, Show)

-- | Scalar field point
data ScalarFieldPoint = ScalarFieldPoint
  { sfpX     :: Signed 16
  , sfpY     :: Signed 16
  , sfpZ     :: Signed 16
  , sfpAlpha :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Sigmoid transform approximation
-- Maps alpha [0,255] to depth [0,255] with S-curve
sigmoidTransform :: Unsigned 8 -> Unsigned 8
sigmoidTransform alpha =
  let -- Approximate sigmoid using piecewise linear
      -- Low region: slow growth
      -- Mid region: fast growth
      -- High region: saturates
      diff = if alpha > sigmoidMidpoint
             then alpha - sigmoidMidpoint
             else sigmoidMidpoint - alpha

      steepFactor = resize sigmoidSteepness * resize diff :: Unsigned 16

      -- Sigmoid approximation
      depth = if alpha < 64 then
                (resize alpha `shiftL` 1) `shiftR` 2  -- Slow start
              else if alpha < 192 then
                64 + ((resize alpha - 64) * 3) `shiftR` 2  -- Fast middle
              else
                192 + (resize alpha - 192) `shiftR` 2  -- Slow end

  in resize $ min 255 depth

-- | Select inversion artifact based on alpha and phase offset
selectInversionArtifact :: Unsigned 8 -> Unsigned 8 -> InversionArtifact
selectInversionArtifact alpha phaseOffset
  | alpha < 38  = ArtifactMemoryFade    -- < 0.15
  | alpha < 77  = ArtifactDreamBleed    -- < 0.30
  | phaseOffset > 192 = ArtifactConsentDrift  -- > 0.75
  | phaseOffset > 153 = ArtifactPhaseSlip     -- > 0.60
  | alpha < 128 && phaseOffset > 77 = ArtifactShadowEcho
  | otherwise = ArtifactNone

-- | Create fragment node from alpha value
createFragmentNode
  :: FragmentID
  -> Unsigned 8     -- Raw alpha
  -> Maybe PhaseID  -- Phase lock
  -> Unsigned 8     -- Phase offset
  -> FragmentNode
createFragmentNode fragId alpha mPhaseLock phaseOffset =
  let depth = sigmoidTransform alpha
      artifact = selectInversionArtifact alpha phaseOffset
      (hasPhaseLock, phaseLock) = case mPhaseLock of
        Nothing -> (False, PhaseID 0 0)
        Just p  -> (True, p)
  in FragmentNode fragId depth alpha hasPhaseLock phaseLock artifact

-- | Initialize alpha trace
emptyAlphaTrace :: AlphaTrace
emptyAlphaTrace = AlphaTrace (repeat 0) 0 0 0

-- | Update alpha trace with new sample
updateAlphaTrace :: AlphaTrace -> Unsigned 8 -> AlphaTrace
updateAlphaTrace trace newAlpha =
  let idx = atCount trace .&. 7  -- Wrap at 8
      oldVal = atSamples trace !! idx
      newSamples = replace idx newAlpha (atSamples trace)

      -- Update running sums
      newSum = atSum trace - resize oldVal + resize newAlpha
      oldSq = resize oldVal * resize oldVal :: Unsigned 24
      newSq = resize newAlpha * resize newAlpha :: Unsigned 24
      newSumSq = atSumSquares trace - oldSq + newSq

      newCount = if atCount trace < 8 then atCount trace + 1 else 8

  in AlphaTrace newSamples newCount newSum newSumSq

-- | Calculate volatility from alpha trace
-- volatility = stddev(α) / mean(α)
calculateVolatility :: AlphaTrace -> Unsigned 8
calculateVolatility trace
  | atCount trace < 2 = 0
  | otherwise =
      let n = resize (atCount trace) :: Unsigned 16
          mean = atSum trace `div` n
          meanSq = mean * mean

          -- Variance = E[X^2] - E[X]^2
          eSq = resize (atSumSquares trace) `div` resize n :: Unsigned 16

          variance = if eSq > meanSq then eSq - meanSq else 0

          -- Approximate stddev (simplified)
          stddev = resize (variance `shiftR` 4) :: Unsigned 8

          -- volatility = stddev / mean (scaled)
          volatility = if mean > 0
                       then (resize stddev * 255) `div` resize mean
                       else 0

      in resize $ min 255 volatility

-- | Check phase lock sync
checkPhaseLockSync :: FragmentNode -> PhaseID -> Bool
checkPhaseLockSync node identityPhase =
  fnHasPhaseLock node &&
  piNamespace (fnPhaseLock node) == piNamespace identityPhase &&
  piPhaseIndex (fnPhaseLock node) == piPhaseIndex identityPhase

-- | Palette state
data PaletteState = PaletteState
  { psNodeCount :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Initial palette state
initialPaletteState :: PaletteState
initialPaletteState = PaletteState 0

-- | Palette input
data PaletteInput = PaletteInput
  { piRawAlpha    :: Unsigned 8
  , piFragmentIdx :: Unsigned 16
  , piPhaseLock   :: Maybe PhaseID
  , piPhaseOffset :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Fragment palette pipeline
fragmentPalettePipeline
  :: HiddenClockResetEnable dom
  => Signal dom PaletteInput
  -> Signal dom FragmentNode
fragmentPalettePipeline input =
  let mkNode inp = createFragmentNode
        (FragmentID (piFragmentIdx inp))
        (piRawAlpha inp)
        (piPhaseLock inp)
        (piPhaseOffset inp)
  in mkNode <$> input

-- | Volatility tracking pipeline
volatilityPipeline
  :: HiddenClockResetEnable dom
  => Signal dom Unsigned 8
  -> Signal dom (Unsigned 8, Unsigned 8)  -- (volatility, current alpha)
volatilityPipeline alphaSignal = mealy volatilityMealy emptyAlphaTrace alphaSignal
  where
    volatilityMealy trace alpha =
      let newTrace = updateAlphaTrace trace alpha
          vol = calculateVolatility newTrace
      in (newTrace, (vol, alpha))

-- | Sigmoid transform pipeline
sigmoidPipeline
  :: HiddenClockResetEnable dom
  => Signal dom Unsigned 8
  -> Signal dom Unsigned 8
sigmoidPipeline = fmap sigmoidTransform
