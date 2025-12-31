{-|
Module      : RaJoeBubble
Description : Joe Cell Scalar Bubble Emitter
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 37: Joe Cell with 5-stage progression and phi-scaled scalar shells.
Implements Y-Factor operator integration with polarity penalty.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaJoeBubble where

import Clash.Prelude

-- | Phi constant scaled to 16-bit fixed point (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Emission stage (1-5)
type EmissionStage = Unsigned 3

-- | Polarity state
data Polarity = PolarityPositive | PolarityNegative | PolarityNeutral
  deriving (Generic, NFDataX, Eq, Show)

-- | Stage progression thresholds (coherence * 255, duration in ticks)
-- Stage 2: 0.55/30s, Stage 3: 0.70/60s, Stage 4: 0.85/120s
stageThresholds :: Vec 5 (Unsigned 8, Unsigned 32)
stageThresholds =
  (0, 0)           :>  -- Stage 1: No requirements
  (140, 30000)     :>  -- Stage 2: 0.55 * 255, 30s at 1kHz
  (179, 60000)     :>  -- Stage 3: 0.70 * 255, 60s
  (217, 120000)    :>  -- Stage 4: 0.85 * 255, 120s
  (230, 0)         :>  -- Stage 5: Group sync/intent lock
  Nil

-- | Scalar shell configuration
data ScalarShell = ScalarShell
  { ssRadius        :: Unsigned 16   -- Phi-scaled radius (fixed point)
  , ssFrequencyBand :: Unsigned 12   -- Hz (scaled)
  , ssImpedance     :: Unsigned 8    -- 0-255 (high=resistive)
  } deriving (Generic, NFDataX)

-- | Bubble emitter state
data BubbleEmitter = BubbleEmitter
  { beCurrentStage      :: EmissionStage
  , beCoherenceLevel    :: Unsigned 8
  , beTimeInStage       :: Unsigned 32
  , beCurrentShells     :: Vec 5 ScalarShell
  , beOperatorPolarity  :: Polarity
  , beBaseFrequency     :: Unsigned 12  -- Hz
  , beResonanceScore    :: Unsigned 8   -- 0-255
  } deriving (Generic, NFDataX)

-- | Y-Factor operator state
data YOperator = YOperator
  { yoPolarity   :: Polarity
  , yoAlignment  :: Unsigned 8   -- 0-255
  , yoAmplitude  :: Unsigned 8   -- 0-255
  } deriving (Generic, NFDataX)

-- | Generate default phi-progression shell radii
-- Returns radius scaled to 16-bit (1024 = 1.0)
defaultShellRadius :: Unsigned 3 -> Unsigned 16
defaultShellRadius idx = case idx of
  0 -> 1024                     -- 1.0
  1 -> phi16                    -- φ
  2 -> (phi16 * phi16) `shiftR` 10  -- φ²
  3 -> (phi16 * phi16 * phi16) `shiftR` 20  -- φ³ (approximated)
  4 -> 4400                     -- φ⁴ (approximated to fit)
  _ -> 1024

-- | Generate scalar shells for stage
generateShells :: EmissionStage -> Unsigned 12 -> Vec 5 ScalarShell
generateShells stage baseFreq = map mkShell (0 :> 1 :> 2 :> 3 :> 4 :> Nil)
  where
    activeCount = resize stage :: Unsigned 3
    mkShell idx = ScalarShell
      { ssRadius = if idx < activeCount then defaultShellRadius idx else 0
      , ssFrequencyBand = if idx < activeCount
                          then baseFreq + resize (idx * 50)
                          else 0
      , ssImpedance = if idx < activeCount
                      then 255 - resize (idx * 40)
                      else 255
      }

-- | Check if can advance to next stage
canAdvanceStage :: BubbleEmitter -> Bool
canAdvanceStage emitter =
  let stage = beCurrentStage emitter
      nextStage = stage + 1
  in if nextStage > 4 then False
     else let (reqCoherence, reqDuration) = stageThresholds !! resize nextStage
          in beCoherenceLevel emitter >= reqCoherence &&
             beTimeInStage emitter >= reqDuration

-- | Advance emitter stage
advanceStage :: BubbleEmitter -> BubbleEmitter
advanceStage emitter =
  if canAdvanceStage emitter && beCurrentStage emitter < 5
  then emitter
    { beCurrentStage = beCurrentStage emitter + 1
    , beTimeInStage = 0
    , beCurrentShells = generateShells (beCurrentStage emitter + 1) (beBaseFrequency emitter)
    }
  else emitter

-- | Calculate polarity efficiency
-- Match = 1.0 (255), Mismatch = 0.6 (153)
polarityEfficiency :: Polarity -> Polarity -> Unsigned 8
polarityEfficiency operatorPol emitterPol
  | operatorPol == emitterPol = 255
  | operatorPol == PolarityNeutral = 255
  | emitterPol == PolarityNeutral = 255
  | otherwise = 153  -- 0.6 penalty

-- | Calculate operator resonance contribution
operatorResonance :: YOperator -> Unsigned 8 -> Unsigned 8
operatorResonance op emitterCoherence =
  let efficiency = polarityEfficiency (yoPolarity op) PolarityPositive
      alignBoost = yoAlignment op
      base = (resize emitterCoherence * resize (yoAmplitude op)) `shiftR` 8 :: Unsigned 16
  in resize $ (base * resize efficiency * resize alignBoost) `shiftR` 16

-- | Calculate bubble pulse frequency
-- Base rate synced to coherence, modified by stage
bubblePulseRate :: BubbleEmitter -> Unsigned 16
bubblePulseRate emitter =
  let baseRate = 60 + resize (beCoherenceLevel emitter `shiftR` 2) :: Unsigned 16  -- 60-120 BPM range
      stageMultiplier = 100 + resize (beCurrentStage emitter * 10) :: Unsigned 16
  in (baseRate * stageMultiplier) `div` 100

-- | Calculate emergence boost for stage
emergenceBoost :: EmissionStage -> Unsigned 8
emergenceBoost stage = case stage of
  1 -> 0    -- No boost
  2 -> 26   -- ~10% (0.1 * 255)
  3 -> 51   -- ~20%
  4 -> 77   -- ~30%
  5 -> 102  -- ~40%
  _ -> 0

-- | Update emitter state (called each tick)
updateEmitter :: BubbleEmitter -> Unsigned 8 -> BubbleEmitter
updateEmitter emitter newCoherence =
  let updated = emitter
        { beCoherenceLevel = newCoherence
        , beTimeInStage = beTimeInStage emitter + 1
        , beResonanceScore = calculateResonance emitter
        }
  in if canAdvanceStage updated
     then advanceStage updated
     else updated

-- | Calculate overall resonance score
calculateResonance :: BubbleEmitter -> Unsigned 8
calculateResonance emitter =
  let baseResonance = beCoherenceLevel emitter
      stageBoost = emergenceBoost (beCurrentStage emitter)
      total = resize baseResonance + resize stageBoost :: Unsigned 9
  in if total > 255 then 255 else resize total

-- | Initialize bubble emitter
initEmitter :: Unsigned 12 -> BubbleEmitter
initEmitter baseFreq = BubbleEmitter
  { beCurrentStage = 1
  , beCoherenceLevel = 0
  , beTimeInStage = 0
  , beCurrentShells = generateShells 1 baseFreq
  , beOperatorPolarity = PolarityNeutral
  , beBaseFrequency = baseFreq
  , beResonanceScore = 0
  }

-- | Top-level bubble emitter pipeline
bubbleEmitterPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)  -- Coherence input
  -> Signal dom BubbleEmitter
bubbleEmitterPipeline coherence = emitter
  where
    emitter = register (initEmitter 432) $ updateEmitter <$> emitter <*> coherence

-- | Stage progression pipeline
stagePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom EmissionStage
stagePipeline coherence = beCurrentStage <$> bubbleEmitterPipeline coherence

-- | Resonance output pipeline
resonancePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom (Unsigned 8)
resonancePipeline coherence = beResonanceScore <$> bubbleEmitterPipeline coherence
