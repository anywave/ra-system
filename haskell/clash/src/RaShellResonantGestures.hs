{-|
Module      : RaShellResonantGestures
Description : Biometric-Attuned Gesture Access
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 59: Biometric-attuned gesture access using scalar resonance,
coherence gating, and field phase alignment.

Prevents gesture misfires from non-intentional movements.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaShellResonantGestures where

import Clash.Prelude

-- | Phi constant scaled
phi16 :: Unsigned 16
phi16 = 1657

-- | Default resonance threshold (0.65 * 255)
defaultResonanceThreshold :: Unsigned 8
defaultResonanceThreshold = 166

-- | Phase alignment threshold (0.3 * 255)
phaseAlignmentThreshold :: Unsigned 8
phaseAlignmentThreshold = 77

-- | Gesture types
data GestureType
  = GestNone
  | GestReachForward
  | GestPullBack
  | GestPushOut
  | GestGraspClose
  | GestReleaseOpen
  | GestSwipeLeft
  | GestSwipeRight
  | GestSwipeUp
  | GestSwipeDown
  | GestCircleCW
  | GestCircleCCW
  | GestHoldSteady
  | GestPoint
  | GestEmergencyStop
  | GestCloseHand
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Control intent types
data ControlIntent
  = IntentNone
  | IntentReach
  | IntentPull
  | IntentPush
  | IntentGrasp
  | IntentRelease
  | IntentMoveTo
  | IntentHoverAt
  | IntentPointAt
  | IntentStop
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Torsion state
data TorsionState
  = TorsionNormal
  | TorsionInverted
  | TorsionNull
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Rejection reasons
data RejectionReason
  = RejectNone
  | RejectLowCoherence
  | RejectTorsionBlocked
  | RejectPhaseMisaligned
  | RejectNoMatchFound
  | RejectWindowClosed
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Flux coherence
data FluxCoherence = FluxCoherence
  { fcValue     :: Unsigned 8   -- 0-255 coherence
  , fcStability :: Unsigned 8   -- 0-255 stability
  , fcPhase     :: Unsigned 16  -- 0-65535 = 0-2π
  } deriving (Generic, NFDataX)

-- | Temporal window state
data TemporalWindow = TemporalWindow
  { twIsActive       :: Bool
  , twWindowIndex    :: Unsigned 4
  , twPhaseAlignment :: Unsigned 8  -- 0-255 alignment
  } deriving (Generic, NFDataX)

-- | Resonant gesture result
data ResonantGesture = ResonantGesture
  { rgUserId          :: Unsigned 8
  , rgGesture         :: GestureType
  , rgResonance       :: FluxCoherence
  , rgTorsionState    :: TorsionState
  , rgMatched         :: Bool
  , rgAuthorized      :: Bool
  , rgRejectionReason :: RejectionReason
  , rgIntent          :: ControlIntent
  } deriving (Generic, NFDataX)

-- | User biometric state
data UserBiometricState = UserBiometricState
  { ubsUserId         :: Unsigned 8
  , ubsResonance      :: FluxCoherence
  , ubsTorsionState   :: TorsionState
  , ubsTemporalWindow :: TemporalWindow
  } deriving (Generic, NFDataX)

-- | Resonance gate entry
data ResonanceGate = ResonanceGate
  { rgGesture      :: GestureType
  , rgMinCoherence :: Unsigned 8
  , rgIsSet        :: Bool
  } deriving (Generic, NFDataX)

-- | Check if gesture is always allowed
isAlwaysAllowed :: GestureType -> Bool
isAlwaysAllowed gesture = case gesture of
  GestHoldSteady    -> True
  GestPoint         -> True
  GestEmergencyStop -> True
  _                 -> False

-- | Check if gesture is allowed under inversion
isInversionAllowed :: GestureType -> Bool
isInversionAllowed gesture = case gesture of
  GestCloseHand   -> True
  GestReleaseOpen -> True
  _               -> False

-- | Get gesture intent
getGestureIntent :: GestureType -> ControlIntent
getGestureIntent gesture = case gesture of
  GestNone          -> IntentNone
  GestReachForward  -> IntentReach
  GestPullBack      -> IntentPull
  GestPushOut       -> IntentPush
  GestGraspClose    -> IntentGrasp
  GestReleaseOpen   -> IntentRelease
  GestSwipeLeft     -> IntentMoveTo
  GestSwipeRight    -> IntentMoveTo
  GestSwipeUp       -> IntentPush
  GestSwipeDown     -> IntentPull
  GestCircleCW      -> IntentHoverAt
  GestCircleCCW     -> IntentHoverAt
  GestHoldSteady    -> IntentHoverAt
  GestPoint         -> IntentPointAt
  GestEmergencyStop -> IntentStop
  GestCloseHand     -> IntentGrasp

-- | Check coherence threshold
checkCoherence
  :: Unsigned 8          -- User ID
  -> GestureType
  -> FluxCoherence
  -> Vec 16 ResonanceGate  -- Custom gates
  -> (Bool, RejectionReason)
checkCoherence userId gesture resonance gates
  | isAlwaysAllowed gesture = (True, RejectNone)
  | otherwise =
      let -- Find custom gate
          findGate :: Unsigned 8 -> Maybe Unsigned 8
          findGate i =
            let gate = gates !! i
            in if rgIsSet gate && rgGesture gate == gesture
               then Just (rgMinCoherence gate)
               else Nothing

          customThreshold = foldl
            (\acc i -> case findGate i of
                         Just t -> Just t
                         Nothing -> acc)
            Nothing
            $(listToVecTH [0..15 :: Unsigned 8])

          threshold = case customThreshold of
                        Just t  -> t
                        Nothing -> defaultResonanceThreshold

      in if fcValue resonance >= threshold
         then (True, RejectNone)
         else (False, RejectLowCoherence)

-- | Check torsion state
checkTorsion :: GestureType -> TorsionState -> (Bool, RejectionReason)
checkTorsion gesture torsion =
  case torsion of
    TorsionInverted ->
      if isInversionAllowed gesture
      then (True, RejectNone)
      else (False, RejectTorsionBlocked)
    _ -> (True, RejectNone)

-- | Check temporal window
checkTemporalWindow :: TemporalWindow -> (Bool, RejectionReason)
checkTemporalWindow window =
  if twIsActive window
  then (True, RejectNone)
  else (False, RejectWindowClosed)

-- | Check phase alignment
checkPhaseAlignment :: GestureType -> TemporalWindow -> (Bool, RejectionReason)
checkPhaseAlignment gesture window
  | isAlwaysAllowed gesture = (True, RejectNone)
  | twPhaseAlignment window >= phaseAlignmentThreshold = (True, RejectNone)
  | otherwise = (False, RejectPhaseMisaligned)

-- | Authorize resonant gesture
authorizeResonantGesture
  :: Unsigned 8           -- User ID
  -> GestureType
  -> Bool                 -- Matched from LimbLearning
  -> FluxCoherence
  -> TorsionState
  -> TemporalWindow
  -> Vec 16 ResonanceGate
  -> ResonantGesture
authorizeResonantGesture userId gesture matched resonance torsion window gates =
  let -- Check match first
      (authorized1, reason1) =
        if matched
        then (True, RejectNone)
        else (False, RejectNoMatchFound)

      -- Check coherence
      (authorized2, reason2) =
        if authorized1
        then checkCoherence userId gesture resonance gates
        else (False, reason1)

      -- Check torsion
      (authorized3, reason3) =
        if authorized2
        then checkTorsion gesture torsion
        else (False, reason2)

      -- Check temporal window
      (authorized4, reason4) =
        if authorized3
        then checkTemporalWindow window
        else (False, reason3)

      -- Check phase alignment
      (authorized5, reason5) =
        if authorized4
        then checkPhaseAlignment gesture window
        else (False, reason4)

      -- Final result
      finalIntent = if authorized5
                    then getGestureIntent gesture
                    else IntentNone

  in ResonantGesture
       userId
       gesture
       resonance
       torsion
       matched
       authorized5
       reason5
       finalIntent

-- | Authorizer state
data AuthorizerState = AuthorizerState
  { asGates :: Vec 16 ResonanceGate
  , asUsers :: Vec 8 UserBiometricState
  } deriving (Generic, NFDataX)

-- | Initial authorizer state
initialAuthorizerState :: AuthorizerState
initialAuthorizerState = AuthorizerState
  (repeat (ResonanceGate GestNone 0 False))
  (repeat (UserBiometricState 0
    (FluxCoherence 0 0 0)
    TorsionNormal
    (TemporalWindow True 0 128)))

-- | Set resonance gate
setResonanceGate
  :: AuthorizerState
  -> Unsigned 4       -- Gate index
  -> GestureType
  -> Unsigned 8       -- Min coherence
  -> AuthorizerState
setResonanceGate state idx gesture minCoh =
  let gate = ResonanceGate gesture minCoh True
      newGates = replace idx gate (asGates state)
  in state { asGates = newGates }

-- | Update user state
updateUserState
  :: AuthorizerState
  -> Unsigned 3       -- User index
  -> UserBiometricState
  -> AuthorizerState
updateUserState state idx userState =
  let newUsers = replace idx userState (asUsers state)
  in state { asUsers = newUsers }

-- | Get user state
getUserState
  :: AuthorizerState
  -> Unsigned 3
  -> UserBiometricState
getUserState state idx = asUsers state !! idx

-- | Authorization input
data AuthInput = AuthInput
  { aiUserId     :: Unsigned 8
  , aiGesture    :: GestureType
  , aiMatched    :: Bool
  , aiResonance  :: FluxCoherence
  , aiTorsion    :: TorsionState
  , aiWindow     :: TemporalWindow
  } deriving (Generic, NFDataX)

-- | Authorization pipeline
authorizationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom AuthInput
  -> Signal dom ResonantGesture
authorizationPipeline input = mealy authMealy initialAuthorizerState input
  where
    authMealy state inp =
      let result = authorizeResonantGesture
            (aiUserId inp)
            (aiGesture inp)
            (aiMatched inp)
            (aiResonance inp)
            (aiTorsion inp)
            (aiWindow inp)
            (asGates state)
      in (state, result)
