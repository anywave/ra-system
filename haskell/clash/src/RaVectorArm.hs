{-|
Module      : RaVectorArm
Description : Biometric to Motion Transduction for Hands-Free UI
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 36: Ra vector arm with polar structure for intention projection.
13-zone theta, 12-band phi, coherence-gated motion.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaVectorArm where

import Clash.Prelude

-- | Motion permission level
data MotionPermission = FullMotion | PartialMotion | BlockedMotion
  deriving (Generic, NFDataX, Eq, Show)

-- | Breath/sync phase state
data PhaseState = Inhale | Peak | Exhale | Trough
  deriving (Generic, NFDataX, Eq, Show)

-- | Interaction form for UI elements
data InteractionForm = Idle | Hover | Active | Disabled
  deriving (Generic, NFDataX, Eq, Show)

-- | Vector arm configuration
data VectorArmConfig = VectorArmConfig
  { cfgFullThreshold    :: Unsigned 8   -- Scaled 0.82 * 255
  , cfgPartialThreshold :: Unsigned 8   -- Scaled 0.55 * 255
  , cfgBreathHoldTicks  :: Unsigned 16  -- 4.5s in ticks
  , cfgEegSpikeRatio    :: Unsigned 8   -- Scaled 1.5 * 100
  , cfgEegSpikeDuration :: Unsigned 16  -- 400ms in ticks
  } deriving (Generic, NFDataX)

-- | Default configuration (thresholds scaled to 8-bit)
defaultConfig :: VectorArmConfig
defaultConfig = VectorArmConfig
  { cfgFullThreshold    = 209  -- 0.82 * 255
  , cfgPartialThreshold = 140  -- 0.55 * 255
  , cfgBreathHoldTicks  = 4500 -- 4.5s at 1kHz
  , cfgEegSpikeRatio    = 150  -- 1.5 * 100
  , cfgEegSpikeDuration = 400  -- 400ms at 1kHz
  }

-- | Vector arm state (polar coordinates)
data VectorArm = VectorArm
  { vaAngleTheta    :: Unsigned 4  -- 1-13 azimuth zone
  , vaAnglePhi      :: Unsigned 4  -- 1-12 inclination band
  , vaReachDepth    :: Unsigned 8  -- 0-255 (scaled 0.0-1.0)
  , vaCoherencePulse :: Unsigned 8  -- 0-255 (scaled 0.0-1.0)
  , vaSyncPhase     :: PhaseState
  } deriving (Generic, NFDataX)

-- | Biometric input vector
data BiometricInput = BiometricInput
  { biHrvCoherence      :: Unsigned 8   -- 0-255
  , biBreathPhase       :: Unsigned 8   -- 0-255 (cycle position)
  , biBreathHoldTicks   :: Unsigned 16  -- Hold duration in ticks
  , biEegAmplitude      :: Unsigned 8   -- Current amplitude
  , biEegBaseline       :: Unsigned 8   -- Baseline amplitude
  , biEegSpikeDuration  :: Unsigned 16  -- Spike duration in ticks
  } deriving (Generic, NFDataX)

-- | UI element with scalar resonance
data UIElement = UIElement
  { uiElementId       :: Unsigned 8
  , uiPositionX       :: Signed 16
  , uiPositionY       :: Signed 16
  , uiScalarL         :: Unsigned 4  -- Harmonic l
  , uiScalarM         :: Signed 8    -- Harmonic m
  , uiInteractionForm :: InteractionForm
  } deriving (Generic, NFDataX)

-- | UI command output
data UICommand = NoCommand | MoveCommand | ActivateCommand | SelectCommand
  deriving (Generic, NFDataX, Eq, Show)

-- | Determine motion permission from coherence
getMotionPermission :: VectorArmConfig -> Unsigned 8 -> MotionPermission
getMotionPermission cfg coherence
  | coherence >= cfgFullThreshold cfg    = FullMotion
  | coherence >= cfgPartialThreshold cfg = PartialMotion
  | otherwise                            = BlockedMotion

-- | Calculate drag weight for partial permission
-- Returns 0-255 where 255 = full drag (frozen)
getDragWeight :: VectorArmConfig -> Unsigned 8 -> Unsigned 8
getDragWeight cfg coherence
  | coherence >= cfgFullThreshold cfg = 0
  | coherence < cfgPartialThreshold cfg = 255
  | otherwise =
      let range = cfgFullThreshold cfg - cfgPartialThreshold cfg
          offset = cfgFullThreshold cfg - coherence
      in resize $ (offset * 255) `div` max 1 range

-- | Detect breath hold trigger
detectBreathHold :: VectorArmConfig -> Unsigned 16 -> Bool
detectBreathHold cfg holdTicks = holdTicks >= cfgBreathHoldTicks cfg

-- | Detect EEG spike trigger
-- Spike when amplitude >= baseline * ratio and sustained
detectEegSpike :: VectorArmConfig -> Unsigned 8 -> Unsigned 8 -> Unsigned 16 -> Bool
detectEegSpike cfg amplitude baseline duration =
  let threshold = (resize baseline * resize (cfgEegSpikeRatio cfg)) `div` 100 :: Unsigned 16
  in resize amplitude >= threshold && duration >= cfgEegSpikeDuration cfg

-- | Check harmonic affinity (per-component ±1)
harmonicAffinity :: Unsigned 4 -> Signed 8 -> Unsigned 4 -> Signed 8 -> Unsigned 4 -> Bool
harmonicAffinity armL armM elemL elemM delta =
  let lDiff = if armL > elemL then armL - elemL else elemL - armL
      mDiff = if armM > elemM then armM - elemM else elemM - armM
  in lDiff <= resize delta && mDiff <= resize (resize delta :: Unsigned 8)

-- | Convert theta zone (1-13) to scaled angle (0-4095 for 12-bit)
thetaToScaled :: Unsigned 4 -> Unsigned 12
thetaToScaled zone =
  let zoneIdx = if zone > 0 then zone - 1 else 0
  in resize $ (resize zoneIdx * 4096) `div` 13

-- | Convert phi band (1-12) to scaled angle (0-2047 for 11-bit, representing 0-π)
phiToScaled :: Unsigned 4 -> Unsigned 11
phiToScaled band =
  let bandIdx = if band > 0 then band - 1 else 0
  in resize $ (resize bandIdx * 2048) `div` 12

-- | Evaluate biometric input to vector arm
evaluateVectorIntent :: BiometricInput -> VectorArm
evaluateVectorIntent bio = VectorArm
  { vaAngleTheta = resize $ (biBreathPhase bio * 13) `shiftR` 8 + 1
  , vaAnglePhi = 7  -- Default middle band
  , vaReachDepth = biHrvCoherence bio
  , vaCoherencePulse = biHrvCoherence bio
  , vaSyncPhase = phaseFromBreath (biBreathPhase bio)
  }

-- | Determine phase from breath position
phaseFromBreath :: Unsigned 8 -> PhaseState
phaseFromBreath phase
  | phase < 64  = Inhale
  | phase < 128 = Peak
  | phase < 192 = Exhale
  | otherwise   = Trough

-- | Generate UI command from vector arm and element
vectorToCommand :: VectorArmConfig -> VectorArm -> UIElement -> UICommand
vectorToCommand cfg arm elem =
  let perm = getMotionPermission cfg (vaCoherencePulse arm)
  in case perm of
    BlockedMotion -> NoCommand
    _ -> if harmonicAffinity (vaAngleTheta arm) (resize $ vaAnglePhi arm)
                             (uiScalarL elem) (uiScalarM elem) 2
         then MoveCommand
         else NoCommand

-- | Top-level vector arm pipeline
vectorArmPipeline
  :: HiddenClockResetEnable dom
  => Signal dom BiometricInput
  -> Signal dom VectorArm
vectorArmPipeline = fmap evaluateVectorIntent

-- | Motion permission pipeline
motionPermissionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom BiometricInput
  -> Signal dom MotionPermission
motionPermissionPipeline bio =
  getMotionPermission defaultConfig . biHrvCoherence <$> bio

-- | Trigger detection pipeline
triggerPipeline
  :: HiddenClockResetEnable dom
  => Signal dom BiometricInput
  -> Signal dom (Bool, Bool)  -- (breathHold, eegSpike)
triggerPipeline bio = bundle (breathHold, eegSpike)
  where
    breathHold = detectBreathHold defaultConfig . biBreathHoldTicks <$> bio
    eegSpike = (\b -> detectEegSpike defaultConfig
                        (biEegAmplitude b)
                        (biEegBaseline b)
                        (biEegSpikeDuration b)) <$> bio
