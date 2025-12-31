{-|
Module      : RaAppendageVectorRemap
Description : Non-Physical Limb Vector Remapping
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 71: Defines non-physical limb structures linked to scalar field
intent pathways. Maps intent → vector → gesture → effect for post-physical
avatar interfaces.

Supports layered routing (gradient descent + shell topology) and symbolic effects.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaAppendageVectorRemap where

import Clash.Prelude

-- | Phi constant scaled
phi16 :: Unsigned 16
phi16 = 1657

-- | Pi/2 scaled (1.5708 * 1024)
piOver2 :: Unsigned 16
piOver2 = 1608

-- | Emotion tags
data EmotionTag
  = EmotionNeutral
  | EmotionFocused
  | EmotionProtective
  | EmotionExpansive
  | EmotionReceptive
  | EmotionProjective
  | EmotionHealing
  | EmotionShielding
  | EmotionRevealing
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Effect type
data EffectType
  = EffectSymbolic
  | EffectBound
  | EffectComposite
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Symbolic effect names (indexed)
data SymbolicEffect
  = EffPush
  | EffPull
  | EffShield
  | EffReveal
  | EffHide
  | EffHeal
  | EffCharge
  | EffDischarge
  | EffConnect
  | EffSever
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Harmonic shell index
data ShellIndex
  = Shell1
  | Shell2
  | Shell3
  | Shell4
  | Shell5
  | Shell6
  | Shell7
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | 3D coordinate (fixed-point, 8.8 format)
data RaCoordinate = RaCoordinate
  { rcX :: Signed 16
  , rcY :: Signed 16
  , rcZ :: Signed 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Chamber vector
data ChamberVector = ChamberVector
  { cvOrigin    :: RaCoordinate
  , cvDirection :: RaCoordinate
  , cvMagnitude :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Intent signature
data IntentSignature = IntentSignature
  { isEmotionTag  :: EmotionTag
  , isTargetCoord :: ChamberVector
  , isPhaseAngle  :: Unsigned 16   -- Radians * 1024
  } deriving (Generic, NFDataX, Eq, Show)

-- | Avatar effect
data AvatarEffect = AvatarEffect
  { aeEffectType    :: EffectType
  , aeSymbolicName  :: SymbolicEffect
  , aeHasBinding    :: Bool
  , aeMagnitude     :: Unsigned 16
  , aePhase         :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Avatar state
data AvatarState = AvatarState
  { asPosition    :: RaCoordinate
  , asActiveShell :: ShellIndex
  , asCoherence   :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Vector limb
data VectorLimb = VectorLimb
  { vlIntentPhase :: IntentSignature
  , vlActiveShell :: ShellIndex
  , vlEffect      :: AvatarEffect
  , vlRouteLength :: Unsigned 4   -- Number of waypoints
  } deriving (Generic, NFDataX, Eq, Show)

-- | Map emotion to symbolic effect
emotionToEffect :: EmotionTag -> SymbolicEffect
emotionToEffect emotion = case emotion of
  EmotionNeutral    -> EffReveal
  EmotionFocused    -> EffPush
  EmotionProtective -> EffShield
  EmotionExpansive  -> EffCharge
  EmotionReceptive  -> EffPull
  EmotionProjective -> EffPush
  EmotionHealing    -> EffHeal
  EmotionShielding  -> EffShield
  EmotionRevealing  -> EffReveal

-- | Check if effect has method binding
effectHasBinding :: SymbolicEffect -> Bool
effectHasBinding eff = case eff of
  EffPush      -> True   -- Ra.Chamber.ForceProject
  EffPull      -> True   -- Ra.Chamber.ForceAttract
  EffShield    -> True   -- Ra.Chamber.BarrierCreate
  EffReveal    -> True   -- Ra.Visualizer.FieldShow
  EffHide      -> True   -- Ra.Visualizer.FieldMask
  EffHeal      -> True   -- Ra.BioField.HealingPulse
  EffCharge    -> True   -- Ra.Energy.ChargeAccumulate
  EffDischarge -> True   -- Ra.Energy.ChargeRelease
  EffConnect   -> True   -- Ra.Network.LinkEstablish
  EffSever     -> True   -- Ra.Network.LinkBreak

-- | Select shell based on intent
selectShellForIntent :: IntentSignature -> AvatarState -> ShellIndex
selectShellForIntent intent avatar = case isEmotionTag intent of
  EmotionProtective -> Shell6
  EmotionShielding  -> Shell6
  EmotionHealing    -> Shell2
  EmotionProjective -> Shell5
  EmotionExpansive  -> Shell5
  EmotionReceptive  -> Shell3
  _                 -> asActiveShell avatar

-- | Compute distance between coordinates
coordinateDistance :: RaCoordinate -> RaCoordinate -> Unsigned 16
coordinateDistance c1 c2 =
  let dx = resize (rcX c1 - rcX c2) :: Signed 32
      dy = resize (rcY c1 - rcY c2) :: Signed 32
      dz = resize (rcZ c1 - rcZ c2) :: Signed 32
      sumSq = dx * dx + dy * dy + dz * dz
      -- Approximate sqrt using bitshift
  in resize (sumSq `shiftR` 8)

-- | Compute route length (simplified - based on distance)
computeRouteLength :: RaCoordinate -> ChamberVector -> Unsigned 4
computeRouteLength start target =
  let dist = coordinateDistance start (cvOrigin target)
      -- Roughly 1 waypoint per 256 units
      steps = dist `shiftR` 8
  in min 15 (resize steps + 2)  -- At least 2 (start + end)

-- | Create avatar effect
createAvatarEffect :: IntentSignature -> AvatarEffect
createAvatarEffect intent =
  let emotion = isEmotionTag intent
      symbolic = emotionToEffect emotion
      hasBinding = effectHasBinding symbolic
      effectType = if hasBinding then EffectBound else EffectSymbolic
  in AvatarEffect
       effectType
       symbolic
       hasBinding
       (cvMagnitude (isTargetCoord intent))
       (isPhaseAngle intent)

-- | Create vector limb
createVectorLimb
  :: IntentSignature
  -> AvatarState
  -> VectorLimb
createVectorLimb intent avatar =
  let shell = selectShellForIntent intent avatar
      effect = createAvatarEffect intent
      routeLen = computeRouteLength (asPosition avatar) (isTargetCoord intent)
  in VectorLimb intent shell effect routeLen

-- | Check if secondary intents should be processed
shouldProcessSecondary :: Unsigned 8 -> Bool
shouldProcessSecondary intensity = intensity > 77  -- > 0.3 * 255

-- | Compute total intensity
computeTotalIntensity :: Unsigned 8 -> Unsigned 8 -> Unsigned 4 -> Unsigned 8
computeTotalIntensity intensity coherence limbCount =
  let -- total = intensity * coherence * limbCount
      product = resize intensity * resize coherence :: Unsigned 16
      scaled = (product * resize limbCount) `shiftR` 8
  in min 255 (resize scaled)

-- | Validate limb coherence
validateLimbCoherence :: VectorLimb -> Bool
validateLimbCoherence limb =
  vlRouteLength limb > 0 &&
  aeMagnitude (vlEffect limb) > 0

-- | Remap input
data RemapInput = RemapInput
  { riIntent     :: IntentSignature
  , riAvatar     :: AvatarState
  , riIntensity  :: Unsigned 8
  , riCoherence  :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Remap output
data RemapOutput = RemapOutput
  { roLimb           :: VectorLimb
  , roTotalIntensity :: Unsigned 8
  , roIsValid        :: Bool
  } deriving (Generic, NFDataX)

-- | Vector remap pipeline
vectorRemapPipeline
  :: HiddenClockResetEnable dom
  => Signal dom RemapInput
  -> Signal dom RemapOutput
vectorRemapPipeline input =
  let remap inp =
        let limb = createVectorLimb (riIntent inp) (riAvatar inp)
            totalInt = computeTotalIntensity
              (riIntensity inp)
              (riCoherence inp)
              1  -- Single limb
            valid = validateLimbCoherence limb
        in RemapOutput limb totalInt valid
  in remap <$> input

-- | Shell selection pipeline
shellSelectPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (IntentSignature, AvatarState)
  -> Signal dom ShellIndex
shellSelectPipeline = fmap (uncurry selectShellForIntent)

-- | Effect creation pipeline
effectCreatePipeline
  :: HiddenClockResetEnable dom
  => Signal dom IntentSignature
  -> Signal dom AvatarEffect
effectCreatePipeline = fmap createAvatarEffect

-- | Emotion to effect mapping pipeline
emotionEffectPipeline
  :: HiddenClockResetEnable dom
  => Signal dom EmotionTag
  -> Signal dom SymbolicEffect
emotionEffectPipeline = fmap emotionToEffect
