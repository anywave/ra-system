{-|
Module      : RaFragmentEchoField
Description : Scalar Harmonic Reverberation and Echo Fields
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 50: Models scalar harmonic reverberation and latent fragment echoes,
with exponential decay, trigger chance computation, and gating influence.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaFragmentEchoField where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Echo thresholds (8-bit scaled)
influenceThreshold :: Unsigned 8
influenceThreshold = 26        -- 0.10 * 255

deactivationThreshold :: Unsigned 8
deactivationThreshold = 13     -- 0.05 * 255

-- | Trigger chance factors (8-bit scaled)
triggerBaseFactor :: Unsigned 8
triggerBaseFactor = 77         -- 0.3 * 255

harmonicMatchBonus :: Unsigned 8
harmonicMatchBonus = 13        -- 0.05 * 255

-- | Boost factor scale (8-bit)
boostFactorScale :: Unsigned 8
boostFactorScale = 128         -- 0.5 * 255

-- | Max echo age for evaluation
maxEchoAgePhiN :: Unsigned 8
maxEchoAgePhiN = 10

-- | High intensity threshold for skip exception
highIntensityThreshold :: Unsigned 8
highIntensityThreshold = 242   -- 0.95 * 255

-- | Omega format (harmonic signature)
data OmegaFormat = OmegaFormat
  { ofOmegaL     :: Unsigned 4
  , ofOmegaM     :: Signed 8
  , ofPhaseAngle :: Unsigned 16
  , ofAmplitude  :: Unsigned 8
  } deriving (Generic, NFDataX, Eq)

-- | Echo field
data EchoField = EchoField
  { efFragmentId     :: Unsigned 8
  , efDecayRate      :: Unsigned 8      -- 0-255 (0=constant, 255=instant)
  , efBaseIntensity  :: Unsigned 8      -- 0-255
  , efHarmonicMemory :: OmegaFormat
  , efTimestampPhiN  :: Unsigned 16     -- φ^n tick at creation
  } deriving (Generic, NFDataX)

-- | Echo influence
data EchoInfluence = EchoInfluence
  { eiBoostFactor   :: Unsigned 8      -- Coherence uplift
  , eiHasHarmonicPull :: Bool
  , eiHarmonicPull  :: OmegaFormat     -- Bias toward similar fragment
  , eiTriggerChance :: Unsigned 8      -- Ghost activation chance
  } deriving (Generic, NFDataX)

-- | Gate state
data GateState = GateState
  { gsCoherenceThreshold :: Unsigned 8
  , gsCurrentCoherence   :: Unsigned 8
  , gsHasFragmentBias    :: Bool
  , gsNextFragmentBias   :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Exponential decay lookup table
-- exp(-rate * dt / 255) * 255 for rate=128 (mid decay), dt=0..15
expDecayLUT :: Vec 16 (Unsigned 8)
expDecayLUT = $(listToVecTH
  [255, 195, 149, 114, 87, 67, 51, 39, 30, 23, 18, 14, 10, 8, 6, 5])

-- | Compute echo intensity using exponential decay
-- intensity = baseIntensity * exp(-decayRate * dt / 255)
computeEchoIntensity :: EchoField -> Unsigned 16 -> Unsigned 8
computeEchoIntensity echo currentPhiN =
  let dt = if currentPhiN >= efTimestampPhiN echo
           then currentPhiN - efTimestampPhiN echo
           else 0

      -- Clamp dt to table size
      dtClamped = min 15 (resize dt) :: Unsigned 4

      -- Get decay factor from LUT (assumes mid-range decay)
      -- For different decay rates, scale the result
      baseDecay = expDecayLUT !! dtClamped

      -- Scale by actual decay rate (higher rate = faster decay)
      decayScaled = if efDecayRate echo == 0
                    then 255  -- No decay
                    else (resize baseDecay * (256 - resize (efDecayRate echo))) `shiftR` 8 :: Unsigned 16

      -- Apply to base intensity
      intensity = (resize (efBaseIntensity echo) * decayScaled) `shiftR` 8 :: Unsigned 16

  in resize $ min 255 intensity

-- | Check if echo is still active
isEchoActive :: Unsigned 8 -> Bool
isEchoActive intensity = intensity >= deactivationThreshold

-- | Check if echo has influence
hasEchoInfluence :: Unsigned 8 -> Bool
hasEchoInfluence intensity = intensity >= influenceThreshold

-- | Compute boost factor from intensity
-- boostFactor = intensity * 0.5
computeBoostFactor :: Unsigned 8 -> Unsigned 8
computeBoostFactor intensity = intensity `shiftR` 1

-- | Compute trigger chance
-- triggerChance = intensity * 0.3 + (harmonicMatch ? 0.05 : 0)
computeTriggerChance :: Unsigned 8 -> Bool -> Unsigned 8
computeTriggerChance intensity harmonicMatch =
  let baseChance = (resize intensity * resize triggerBaseFactor) `shiftR` 8 :: Unsigned 16
      bonus = if harmonicMatch then harmonicMatchBonus else 0
      total = resize baseChance + bonus :: Unsigned 8
  in min 255 total

-- | Check harmonic match between echo and field
checkHarmonicMatch :: OmegaFormat -> OmegaFormat -> Bool
checkHarmonicMatch echoH fieldH =
  let lMatch = ofOmegaL echoH == ofOmegaL fieldH
      mDiff = if ofOmegaM echoH > ofOmegaM fieldH
              then ofOmegaM echoH - ofOmegaM fieldH
              else ofOmegaM fieldH - ofOmegaM echoH
      mMatch = mDiff <= 1
      -- Amplitude similarity (within 20%)
      ampDiff = if ofAmplitude echoH > ofAmplitude fieldH
                then ofAmplitude echoH - ofAmplitude fieldH
                else ofAmplitude fieldH - ofAmplitude echoH
      ampMatch = ampDiff < 51  -- ~0.2 * 255
  in lMatch && mMatch && ampMatch

-- | Should skip echo evaluation for efficiency
shouldSkipEchoEvaluation :: EchoField -> Unsigned 16 -> Bool
shouldSkipEchoEvaluation echo currentPhiN =
  let age = if currentPhiN >= efTimestampPhiN echo
            then currentPhiN - efTimestampPhiN echo
            else 0
  in age > resize maxEchoAgePhiN && efBaseIntensity echo <= highIntensityThreshold

-- | Evaluate echo influence
evaluateEchoInfluence :: EchoField -> Unsigned 16 -> Maybe OmegaFormat -> Maybe EchoInfluence
evaluateEchoInfluence echo currentPhiN maybeFieldH =
  let intensity = computeEchoIntensity echo currentPhiN

      -- Check if active and has influence
      active = isEchoActive intensity
      hasInfluence = hasEchoInfluence intensity

  in if not active || not hasInfluence
     then Nothing
     else
       let boost = computeBoostFactor intensity

           -- Check harmonic match
           (harmonicMatch, harmonicPull) = case maybeFieldH of
             Nothing -> (False, efHarmonicMemory echo)
             Just fieldH ->
               let match = checkHarmonicMatch (efHarmonicMemory echo) fieldH
               in (match, if match then efHarmonicMemory echo else efHarmonicMemory echo)

           trigger = computeTriggerChance intensity harmonicMatch

       in Just $ EchoInfluence boost harmonicMatch harmonicPull trigger

-- | Apply echo influence to gate state
applyEchoInfluence :: EchoInfluence -> GateState -> GateState
applyEchoInfluence influence gate =
  let newCoherence = min 255 (gsCurrentCoherence gate + eiBoostFactor influence)
      (hasBias, bias) = if eiHasHarmonicPull influence
                        then (True, resize (ofOmegaL (eiHarmonicPull influence)) * 10 +
                                    resize (if ofOmegaM (eiHarmonicPull influence) < 0
                                            then -(ofOmegaM (eiHarmonicPull influence))
                                            else ofOmegaM (eiHarmonicPull influence)))
                        else (gsHasFragmentBias gate, gsNextFragmentBias gate)
  in GateState (gsCoherenceThreshold gate) newCoherence hasBias bias

-- | Generate echo field from content
generateEchoField
  :: Unsigned 8      -- Fragment ID
  -> Unsigned 8      -- Content stability (0-255)
  -> Unsigned 8      -- Content duration (ticks)
  -> OmegaFormat     -- Harmonic signature
  -> Unsigned 16     -- Current φ^n tick
  -> EchoField
generateEchoField fragId stability duration harmonic currentPhiN =
  let -- Decay rate inversely related to stability
      decayRate = 255 - ((resize stability * 179) `shiftR` 8)  -- * 0.7

      -- Base intensity from stability and duration
      baseIntensity = (resize stability * min 255 (duration * 25)) `shiftR` 8

  in EchoField fragId decayRate baseIntensity harmonic currentPhiN

-- | Echo intensity pipeline
echoIntensityPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (EchoField, Unsigned 16)
  -> Signal dom (Unsigned 8)
echoIntensityPipeline input = uncurry computeEchoIntensity <$> input

-- | Boost factor pipeline
boostFactorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom (Unsigned 8)
boostFactorPipeline = fmap computeBoostFactor

-- | Trigger chance pipeline
triggerChancePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Bool)
  -> Signal dom (Unsigned 8)
triggerChancePipeline input = uncurry computeTriggerChance <$> input

-- | Harmonic match pipeline
harmonicMatchPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (OmegaFormat, OmegaFormat)
  -> Signal dom Bool
harmonicMatchPipeline input = uncurry checkHarmonicMatch <$> input

-- | Gate influence pipeline
gateInfluencePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (EchoInfluence, GateState)
  -> Signal dom GateState
gateInfluencePipeline input = uncurry applyEchoInfluence <$> input
