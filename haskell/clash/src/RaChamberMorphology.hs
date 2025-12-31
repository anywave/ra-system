{-|
Module      : RaChamberMorphology
Description : Resonance-Driven Chamber Form Modulation
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 44: Integrates scalar dynamics and user resonance to morph
chambers with φ-scaled timing transitions.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaChamberMorphology where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Transition timing (φ-scaled ticks)
transitionShort :: Unsigned 8
transitionShort = 13

transitionMedium :: Unsigned 8
transitionMedium = 21

transitionLong :: Unsigned 8
transitionLong = 34

-- | Coherence thresholds (scaled to 8-bit)
collapseThreshold :: Unsigned 8
collapseThreshold = 51    -- 0.20 * 255

hrvDeltaThreshold :: Unsigned 8
hrvDeltaThreshold = 51    -- 0.20 * 255

gradualThreshold :: Unsigned 8
gradualThreshold = 128    -- 0.50 * 255

stableThreshold :: Unsigned 8
stableThreshold = 184     -- 0.72 * 255

-- | Chamber form enumeration
data ChamberForm
  = FormEgg
  | FormDodecahedron
  | FormToroid
  | FormHelixSpindle
  | FormCaduceusAligned
  | FormHarmonicShell1
  | FormHarmonicShell2
  | FormHarmonicShell3
  | FormHarmonicShell4
  | FormHarmonicShell5
  | FormHarmonicShell6
  | FormHarmonicShell7
  | FormHarmonicShell8
  | FormHarmonicShell9
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Morph event type
data MorphEventType
  = GradualPhaseShift
  | RapidCollapse
  | ResonantExpansion
  deriving (Generic, NFDataX, Eq, Show)

-- | Morph event
data MorphEvent = MorphEvent
  { meEventType   :: MorphEventType
  , meTargetForm  :: ChamberForm
  , meDuration    :: Unsigned 8
  , meTriggerCoh  :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Harmonic shell configuration
data HarmonicShell = HarmonicShell
  { hsIndex       :: Unsigned 4    -- 1-9
  , hsRadialL     :: Unsigned 4    -- Spherical harmonic l
  , hsAngularM    :: Signed 8      -- Spherical harmonic m
  , hsRadiusFactor :: Unsigned 16  -- φ^index * 1024
  } deriving (Generic, NFDataX)

-- | Biometric snapshot
data BiometricSnapshot = BiometricSnapshot
  { bsCoherence       :: Unsigned 8
  , bsHrvValue        :: Unsigned 16
  , bsHrvDelta        :: Unsigned 8   -- Absolute delta
  , bsTicksSinceChange :: Unsigned 8
  , bsBreathPhase     :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Scalar resonance state
data ScalarResonance = ScalarResonance
  { srFieldStrength    :: Unsigned 8
  , srDominantL        :: Unsigned 4
  , srDominantM        :: Signed 8
  , srPhaseAngle       :: Unsigned 16
  } deriving (Generic, NFDataX)

-- | Chamber morph state
data ChamberMorphState = ChamberMorphState
  { cmsCurrentForm      :: ChamberForm
  , cmsTargetForm       :: Maybe ChamberForm
  , cmsTransitionProgress :: Unsigned 8  -- 0-255
  , cmsTransitionDuration :: Unsigned 8
  , cmsStability        :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Get shell form for index
getShellForm :: Unsigned 4 -> ChamberForm
getShellForm idx = case idx of
  1 -> FormHarmonicShell1
  2 -> FormHarmonicShell2
  3 -> FormHarmonicShell3
  4 -> FormHarmonicShell4
  5 -> FormHarmonicShell5
  6 -> FormHarmonicShell6
  7 -> FormHarmonicShell7
  8 -> FormHarmonicShell8
  9 -> FormHarmonicShell9
  _ -> FormHarmonicShell1

-- | Resolve form from coherence level
resolveFormFromCoherence :: Unsigned 8 -> ChamberForm
resolveFormFromCoherence coh
  | coh < 51   = FormEgg              -- < 0.20
  | coh < 102  = FormToroid           -- < 0.40
  | coh < 140  = FormHelixSpindle     -- < 0.55
  | coh < 184  = FormCaduceusAligned  -- < 0.72
  | coh < 217  = FormDodecahedron     -- < 0.85
  | otherwise  =                       -- >= 0.85: nested shells
      let shellIdx = min 9 (1 + (coh - 217) `div` 4)
      in getShellForm (resize shellIdx)

-- | Resolve form from dominant harmonic
resolveFormFromHarmonic :: Unsigned 4 -> ChamberForm
resolveFormFromHarmonic l
  | l == 0    = FormEgg
  | l == 1    = FormHelixSpindle
  | l == 2    = FormToroid
  | l == 3    = FormCaduceusAligned
  | l <= 6    = FormDodecahedron
  | otherwise = getShellForm (min 9 (l - 5))

-- | Resolve optimal chamber form
resolveChamberForm :: BiometricSnapshot -> ScalarResonance -> ChamberForm
resolveChamberForm bio res
  | bsCoherence bio >= stableThreshold = resolveFormFromCoherence (bsCoherence bio)
  | srFieldStrength res >= 179 = resolveFormFromHarmonic (srDominantL res)  -- 0.7 * 255
  | otherwise = resolveFormFromCoherence (bsCoherence bio)

-- | Check if collapse should trigger
shouldCollapse :: BiometricSnapshot -> Bool
shouldCollapse bio =
  bsCoherence bio < collapseThreshold ||
  (bsHrvDelta bio > hrvDeltaThreshold && bsTicksSinceChange bio < 5)

-- | Get transition duration based on coherence
getTransitionDuration :: Unsigned 8 -> Unsigned 8
getTransitionDuration coh
  | coh >= 204 = transitionShort   -- >= 0.80
  | coh >= 128 = transitionMedium  -- >= 0.50
  | otherwise  = transitionLong

-- | Generate morph event if needed
generateMorphEvent :: ChamberMorphState -> BiometricSnapshot -> ScalarResonance
                   -> Maybe MorphEvent
generateMorphEvent state bio res
  | shouldCollapse bio =
      Just $ MorphEvent RapidCollapse FormEgg 5 (bsCoherence bio)
  | otherwise =
      let targetForm = resolveChamberForm bio res
      in if targetForm == cmsCurrentForm state then Nothing
         else if cmsTargetForm state == Just targetForm then Nothing
         else
           let eventType = if bsCoherence bio >= stableThreshold
                           then ResonantExpansion
                           else GradualPhaseShift
               duration = getTransitionDuration (bsCoherence bio)
           in Just $ MorphEvent eventType targetForm duration (bsCoherence bio)

-- | Create harmonic shell
createHarmonicShell :: Unsigned 4 -> HarmonicShell
createHarmonicShell idx = HarmonicShell
  { hsIndex = idx
  , hsRadialL = idx
  , hsAngularM = 0
  , hsRadiusFactor = case idx of
      1 -> 1024
      2 -> phi16
      3 -> (phi16 * phi16) `shiftR` 10
      4 -> 2718  -- φ³ ≈ 4.236, but scaled
      5 -> 4400  -- φ⁴ ≈ 6.854
      6 -> 7119  -- φ⁵
      7 -> 11519 -- φ⁶
      8 -> 18638 -- φ⁷
      9 -> 30157 -- φ⁸
      _ -> 1024
  }

-- | Initialize chamber state
initChamberState :: ChamberMorphState
initChamberState = ChamberMorphState
  { cmsCurrentForm = FormEgg
  , cmsTargetForm = Nothing
  , cmsTransitionProgress = 0
  , cmsTransitionDuration = 0
  , cmsStability = 128
  }

-- | Update transition progress (one tick)
updateTransition :: ChamberMorphState -> ChamberMorphState
updateTransition state = case cmsTargetForm state of
  Nothing -> state
  Just target ->
    let dur = cmsTransitionDuration state
        prog = cmsTransitionProgress state
        step = if dur == 0 then 255 else 255 `div` dur
        newProg = min 255 (prog + step)
    in if newProg >= 255
       then ChamberMorphState target Nothing 0 0 (cmsStability state)
       else state { cmsTransitionProgress = newProg }

-- | Apply morph event to state
applyMorphEvent :: ChamberMorphState -> MorphEvent -> ChamberMorphState
applyMorphEvent state event = ChamberMorphState
  { cmsCurrentForm = cmsCurrentForm state
  , cmsTargetForm = Just (meTargetForm event)
  , cmsTransitionProgress = 0
  , cmsTransitionDuration = meDuration event
  , cmsStability = meTriggerCoh event
  }

-- | Compute stability score
computeStability :: BiometricSnapshot -> ScalarResonance -> Unsigned 8
computeStability bio res =
  let cohFactor = bsCoherence bio
      fieldFactor = srFieldStrength res
      hrvStability = if bsHrvDelta bio > 51
                     then 0
                     else 255 - (bsHrvDelta bio * 5)
      -- Weighted: 50% coherence, 30% field, 20% HRV
      weighted = (resize cohFactor * 5 + resize fieldFactor * 3 + resize hrvStability * 2) `div` 10
  in resize $ min 255 weighted

-- | Form resolution pipeline
formResolutionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BiometricSnapshot, ScalarResonance)
  -> Signal dom ChamberForm
formResolutionPipeline input = uncurry resolveChamberForm <$> input

-- | Collapse detection pipeline
collapseDetectionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom BiometricSnapshot
  -> Signal dom Bool
collapseDetectionPipeline = fmap shouldCollapse

-- | Morph state machine
morphStateMachine
  :: HiddenClockResetEnable dom
  => Signal dom (Maybe MorphEvent)
  -> Signal dom ChamberMorphState
morphStateMachine events = state
  where
    state = register initChamberState nextState
    nextState = update <$> state <*> events
    update s Nothing = updateTransition s
    update s (Just e) = applyMorphEvent s e
