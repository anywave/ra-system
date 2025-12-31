{-|
Module      : RaPipelineDreamShift
Description : Somniferous Chamber Shift Mechanism
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 76: Adapts chamber morphologies during sleep states using biometric
cues (HRV, REM cycles, breath coherence). Prevents coherence shattering
during vulnerable dream phases.

HRV drop >15% + α<0.3 = stress, Egg→Toroid→Caduceus tiers, biometric-locked gradient.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaPipelineDreamShift where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Thresholds
hrvDropThreshold :: Unsigned 8
hrvDropThreshold = 38      -- 0.15 * 255

alphaCollapseThreshold :: Unsigned 8
alphaCollapseThreshold = 77  -- 0.3 * 255

stressInterventionThreshold :: Unsigned 8
stressInterventionThreshold = 128  -- 0.5 * 255

morphGradientRate :: Unsigned 8
morphGradientRate = 26     -- 0.1 * 255

-- | Chamber morphology forms
data ChamberForm
  = FormEgg
  | FormToroid
  | FormCaduceusAligned
  | FormBuffer
  | FormShell
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Sleep phases
data SleepPhase
  = PhaseAwake
  | PhaseNREM1
  | PhaseNREM2
  | PhaseNREM3
  | PhaseREM
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Morph event types
data MorphEventType
  = EventGentleModulation
  | EventStressIntervention
  | EventInversionBuffering
  | EventCollapseFallback
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Biometric stream
data BioStream = BioStream
  { bsHRV            :: Unsigned 8   -- 0-255 normalized
  , bsHRVBaseline    :: Unsigned 8
  , bsAlpha          :: Unsigned 8
  , bsBreathCoherence :: Unsigned 8
  , bsSleepPhase     :: SleepPhase
  , bsCurrentTick    :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Chamber state
data ChamberState = ChamberState
  { csForm          :: ChamberForm
  , csAlpha         :: Unsigned 8
  , csStability     :: Unsigned 8
  , csMorphProgress :: Unsigned 8   -- 0-255 (0.0-1.0)
  } deriving (Generic, NFDataX, Eq, Show)

-- | Sleep morph event
data SleepMorphEvent = SleepMorphEvent
  { smeEventType  :: MorphEventType
  , smeTargetForm :: ChamberForm
  , smeUrgency    :: Unsigned 8     -- 0-255
  } deriving (Generic, NFDataX, Eq, Show)

-- | Nocturnal coherence profile
data NocturnalProfile = NocturnalProfile
  { npHRVBaseline   :: Unsigned 8
  , npDreamStress   :: Unsigned 8
  , npShiftGradient :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute HRV drop (as percentage of baseline * 255)
computeHRVDrop :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
computeHRVDrop current baseline =
  if baseline == 0 || current >= baseline then 0
  else
    let diff = baseline - current
        drop = (resize diff * 255) `div` resize baseline :: Unsigned 16
    in resize (min 255 drop)

-- | Detect alpha collapse
detectAlphaCollapse :: Unsigned 8 -> Bool
detectAlphaCollapse alpha = alpha < alphaCollapseThreshold

-- | Compute dream stress level (0-255)
computeDreamStress
  :: Unsigned 8    -- HRV drop
  -> Unsigned 8    -- Alpha
  -> Unsigned 8    -- Breath coherence
  -> SleepPhase
  -> Unsigned 8
computeDreamStress hrvDrop alpha breathCoh phase =
  let -- HRV contribution (0-102, ~0.4 * 255)
      hrvStress = min 102 (hrvDrop `shiftL` 1)

      -- Alpha contribution (0-102)
      alphaStress = if alpha < alphaCollapseThreshold then 102
                    else if alpha < 128 then 51
                    else 0

      -- Breath contribution (0-51, ~0.2 * 255)
      breathStress = (255 - breathCoh) `shiftR` 2

      -- REM amplification (1.2x)
      phaseFactor = if phase == PhaseREM then 307 else 256 :: Unsigned 16

      -- Combine
      total = resize hrvStress + resize alphaStress + resize breathStress :: Unsigned 16
      amplified = (total * phaseFactor) `shiftR` 8

  in min 255 (resize amplified)

-- | Select chamber form based on alpha
selectChamberForm :: Unsigned 8 -> ChamberForm
selectChamberForm alpha
  | alpha < 51  = FormShell           -- < 0.2
  | alpha < 102 = FormBuffer          -- 0.2 - 0.4
  | alpha < 153 = FormEgg             -- 0.4 - 0.6
  | alpha < 204 = FormToroid          -- 0.6 - 0.8
  | otherwise   = FormCaduceusAligned -- >= 0.8

-- | Compute shift gradient (morph rate)
computeShiftGradient :: BioStream -> ChamberForm -> Unsigned 8
computeShiftGradient bio targetForm =
  let -- Base rate from breath coherence
      baseRate = (bsBreathCoherence bio * morphGradientRate) `shiftR` 8

      -- Emergency forms morph faster (2x)
      emergencyFactor = case targetForm of
        FormShell  -> 2
        FormBuffer -> 2
        _          -> 1

      -- HRV factor (0.5 + hrv * 0.5)
      hrvFactor = 128 + (bsHRV bio `shiftR` 1)

      -- Combine
      result = (resize baseRate * resize emergencyFactor * resize hrvFactor) `shiftR` 8 :: Unsigned 16

  in min 255 (resize result)

-- | Generate sleep shift
generateSleepShift
  :: BioStream
  -> ChamberState
  -> (ChamberState, SleepMorphEvent)
generateSleepShift bio chamber =
  let -- Compute metrics
      hrvDrop = computeHRVDrop (bsHRV bio) (bsHRVBaseline bio)
      stress = computeDreamStress hrvDrop (bsAlpha bio) (bsBreathCoherence bio) (bsSleepPhase bio)

      -- Determine target form
      targetForm = selectChamberForm (bsAlpha bio)

      -- Determine event type
      isCollapse = hrvDrop > hrvDropThreshold && detectAlphaCollapse (bsAlpha bio)
      isStress = stress > stressInterventionThreshold

      (eventType, urgency) =
        if isCollapse then (EventStressIntervention, 230)
        else if isStress then (EventStressIntervention, stress)
        else if targetForm /= csForm chamber then (EventGentleModulation, 77)
        else (EventGentleModulation, 26)

      -- Compute gradient
      gradient = computeShiftGradient bio targetForm

      -- Update morph progress
      newProgress = if targetForm /= csForm chamber
                    then min 255 (csMorphProgress chamber + gradient)
                    else csMorphProgress chamber

      -- Complete morph if progress reaches max
      (newForm, finalProgress) =
        if newProgress >= 255
        then (targetForm, 0)
        else (csForm chamber, newProgress)

      -- Update stability
      newStability = if eventType == EventStressIntervention
                     then max 0 (csStability chamber - 26)
                     else min 255 (csStability chamber + 13)

      newChamber = ChamberState newForm (bsAlpha bio) newStability finalProgress
      event = SleepMorphEvent eventType targetForm urgency

  in (newChamber, event)

-- | Initial chamber state
initialChamberState :: ChamberState
initialChamberState = ChamberState FormEgg 128 204 0

-- | Dream shift pipeline
dreamShiftPipeline
  :: HiddenClockResetEnable dom
  => Signal dom BioStream
  -> Signal dom (ChamberState, SleepMorphEvent)
dreamShiftPipeline input = mealy shiftMealy initialChamberState input
  where
    shiftMealy chamber bio =
      let (newChamber, event) = generateSleepShift bio chamber
      in (newChamber, (newChamber, event))

-- | Stress computation pipeline
stressComputePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8, Unsigned 8, SleepPhase)
  -> Signal dom Unsigned 8
stressComputePipeline = fmap (\(h, a, b, p) -> computeDreamStress h a b p)

-- | Form selection pipeline
formSelectPipeline
  :: HiddenClockResetEnable dom
  => Signal dom Unsigned 8
  -> Signal dom ChamberForm
formSelectPipeline = fmap selectChamberForm

-- | HRV drop pipeline
hrvDropPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8)
  -> Signal dom Unsigned 8
hrvDropPipeline = fmap (uncurry computeHRVDrop)
