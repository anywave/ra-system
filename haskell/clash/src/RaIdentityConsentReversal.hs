{-|
Module      : RaIdentityConsentReversal
Description : Ankh-Phase Consent Reversal Detection
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 68: Detects consent reversal via ankh-phase signature changes,
triggers shadow glyph animation, and propagates recursive ripple events
to linked fragments with attenuation.

Uses Δ(ankh) threshold π/2, HRV collapse detection, and ripple attenuation.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaIdentityConsentReversal where

import Clash.Prelude

-- | Pi/2 scaled (1.5708 * 1024)
piOver2Scaled :: Unsigned 16
piOver2Scaled = 1608

-- | Pi scaled (3.1416 * 1024)
piScaled :: Unsigned 16
piScaled = 3217

-- | Ripple attenuation factor (0.7 * 256)
rippleAttenuation :: Unsigned 8
rippleAttenuation = 179

-- | Maximum ripple hops
rippleMaxHops :: Unsigned 2
rippleMaxHops = 3

-- | HRV drop threshold (0.25 * 256)
hrvDropThreshold :: Unsigned 8
hrvDropThreshold = 64

-- | HRV window size
hrvWindowSize :: Unsigned 2
hrvWindowSize = 3

-- | Shadow glyph indices (corresponding to Unicode symbols)
-- ◐=0, ◑=1, ◒=2, ◓=3, ●=4, ○=5, ◉=6, ◎=7
data ShadowGlyph
  = GlyphLeftHalf      -- ◐
  | GlyphRightHalf     -- ◑
  | GlyphLowerHalf     -- ◒
  | GlyphUpperHalf     -- ◓
  | GlyphFilled        -- ●
  | GlyphEmpty         -- ○
  | GlyphFisheye       -- ◉
  | GlyphBullseye      -- ◎
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Consent state
data ConsentState
  = ConsentActive
  | ConsentSuspended
  | ConsentRevoked
  | ConsentPending
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Ankh phase signature
data AnkhPhase = AnkhPhase
  { apAngle     :: Unsigned 16   -- Angle scaled by 1024
  , apMagnitude :: Unsigned 8    -- 0-255
  , apCoherence :: Unsigned 8    -- 0-255
  } deriving (Generic, NFDataX, Eq, Show)

-- | Fragment identity
data FragmentID = FragmentID
  { fidNamespace :: Unsigned 8
  , fidIndex     :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Fragment node
data FragmentNode = FragmentNode
  { fnFragmentId  :: FragmentID
  , fnAnkhPhase   :: AnkhPhase
  , fnAlphaDepth  :: Unsigned 8
  , fnConsent     :: ConsentState
  , fnLinkCount   :: Unsigned 4    -- Number of linked fragments
  } deriving (Generic, NFDataX, Eq, Show)

-- | HRV sample window (3-tick rolling)
data HRVWindow = HRVWindow
  { hwSamples :: Vec 3 (Unsigned 8)
  , hwIndex   :: Unsigned 2
  , hwCount   :: Unsigned 2
  } deriving (Generic, NFDataX, Eq, Show)

-- | Ripple event
data RippleEvent = RippleEvent
  { reSourceId       :: FragmentID
  , reTargetId       :: FragmentID
  , reAttenuatedDelta :: Unsigned 16  -- Scaled delta
  , reHopNumber      :: Unsigned 2
  , reShadowGlyph    :: ShadowGlyph
  } deriving (Generic, NFDataX, Eq, Show)

-- | Reversal detection result
data ReversalResult = ReversalResult
  { rrDetected      :: Bool
  , rrDeltaAnkh     :: Unsigned 16   -- Scaled
  , rrShadowGlyph   :: ShadowGlyph
  , rrHRVCollapse   :: Bool
  , rrNewConsent    :: ConsentState
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute angular difference normalized to [0, π]
-- Returns scaled value (multiply by 1024)
computeDeltaAnkh :: Unsigned 16 -> Unsigned 16 -> Unsigned 16
computeDeltaAnkh angle1 angle2 =
  let diff = if angle1 > angle2
             then angle1 - angle2
             else angle2 - angle1
      -- Normalize to [0, π] (if > π, use 2π - diff)
      normalized = if diff > piScaled
                   then 2 * piScaled - diff
                   else diff
  in min piScaled normalized

-- | Check if delta exceeds threshold (π/2)
exceedsThreshold :: Unsigned 16 -> Bool
exceedsThreshold delta = delta >= piOver2Scaled

-- | Select shadow glyph based on delta and alpha depth
selectShadowGlyph :: Unsigned 16 -> Unsigned 8 -> ShadowGlyph
selectShadowGlyph delta alphaDepth
  -- High delta (>= 0.75 * π)
  | delta >= (piScaled * 3) `shiftR` 2 =
      if alphaDepth >= 192 then GlyphFilled
      else if alphaDepth >= 128 then GlyphFisheye
      else GlyphBullseye
  -- Medium-high delta (>= 0.5 * π)
  | delta >= piOver2Scaled =
      if alphaDepth >= 192 then GlyphUpperHalf
      else if alphaDepth >= 128 then GlyphLowerHalf
      else GlyphRightHalf
  -- Lower delta
  | otherwise =
      if alphaDepth >= 128 then GlyphLeftHalf
      else GlyphEmpty

-- | Initialize HRV window
emptyHRVWindow :: HRVWindow
emptyHRVWindow = HRVWindow (repeat 0) 0 0

-- | Update HRV window with new sample
updateHRVWindow :: HRVWindow -> Unsigned 8 -> HRVWindow
updateHRVWindow window newSample =
  let idx = hwIndex window
      newSamples = replace idx newSample (hwSamples window)
      newIdx = if idx >= 2 then 0 else idx + 1
      newCount = if hwCount window < 3 then hwCount window + 1 else 3
  in HRVWindow newSamples newIdx newCount

-- | Detect HRV collapse (25% drop in 3-tick window)
detectHRVCollapse :: HRVWindow -> Bool
detectHRVCollapse window
  | hwCount window < 3 = False
  | otherwise =
      let samples = hwSamples window
          -- Get oldest and newest
          oldestIdx = if hwIndex window >= 2 then hwIndex window - 2 else hwIndex window + 1
          newestIdx = if hwIndex window == 0 then 2 else hwIndex window - 1
          oldest = samples !! oldestIdx
          newest = samples !! newestIdx
          -- Check 25% drop
          threshold = (resize oldest * hrvDropThreshold) `shiftR` 8 :: Unsigned 16
      in resize oldest - resize newest >= threshold

-- | Compute attenuated delta for ripple propagation
attenuateDelta :: Unsigned 16 -> Unsigned 16
attenuateDelta delta =
  (delta * resize rippleAttenuation) `shiftR` 8

-- | Check if ripple should propagate (delta >= threshold * 0.5)
shouldPropagate :: Unsigned 16 -> Bool
shouldPropagate delta = delta >= (piOver2Scaled `shiftR` 1)

-- | Detect consent reversal
detectReversal
  :: AnkhPhase      -- Previous phase
  -> AnkhPhase      -- Current phase
  -> Unsigned 8     -- Alpha depth
  -> HRVWindow      -- HRV history
  -> Unsigned 8     -- Current HRV
  -> ReversalResult
detectReversal prevPhase currPhase alphaDepth hrvWindow currHRV =
  let -- Compute delta
      delta = computeDeltaAnkh (apAngle prevPhase) (apAngle currPhase)

      -- Check threshold
      exceeds = exceedsThreshold delta

      -- Update HRV and check collapse
      newHRVWindow = updateHRVWindow hrvWindow currHRV
      hrvCollapse = detectHRVCollapse newHRVWindow

      -- Select glyph
      glyph = selectShadowGlyph delta alphaDepth

      -- Determine new consent state
      newConsent = if exceeds && hrvCollapse then ConsentRevoked
                   else if exceeds then ConsentSuspended
                   else if hrvCollapse then ConsentPending
                   else ConsentActive

      detected = exceeds || hrvCollapse

  in ReversalResult detected delta glyph hrvCollapse newConsent

-- | Reversal detector state
data ReversalState = ReversalState
  { rsPrevPhase  :: AnkhPhase
  , rsHRVWindow  :: HRVWindow
  , rsConsent    :: ConsentState
  , rsRippleHop  :: Unsigned 2
  } deriving (Generic, NFDataX)

-- | Initial reversal state
initialReversalState :: ReversalState
initialReversalState = ReversalState
  (AnkhPhase 0 0 0)
  emptyHRVWindow
  ConsentActive
  0

-- | Reversal detector input
data ReversalInput = ReversalInput
  { riCurrentPhase :: AnkhPhase
  , riAlphaDepth   :: Unsigned 8
  , riCurrentHRV   :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Reversal detector pipeline
reversalDetectorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom ReversalInput
  -> Signal dom ReversalResult
reversalDetectorPipeline input = mealy reversalMealy initialReversalState input
  where
    reversalMealy state inp =
      let result = detectReversal
            (rsPrevPhase state)
            (riCurrentPhase inp)
            (riAlphaDepth inp)
            (rsHRVWindow state)
            (riCurrentHRV inp)

          newState = ReversalState
            (riCurrentPhase inp)
            (updateHRVWindow (rsHRVWindow state) (riCurrentHRV inp))
            (rrNewConsent result)
            0

      in (newState, result)

-- | Delta computation pipeline
deltaComputePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 16, Unsigned 16)
  -> Signal dom Unsigned 16
deltaComputePipeline = fmap (uncurry computeDeltaAnkh)

-- | Shadow glyph selection pipeline
shadowGlyphPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 16, Unsigned 8)
  -> Signal dom ShadowGlyph
shadowGlyphPipeline = fmap (uncurry selectShadowGlyph)

-- | Ripple attenuation pipeline
rippleAttenuationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom Unsigned 16
  -> Signal dom Unsigned 16
rippleAttenuationPipeline = fmap attenuateDelta
