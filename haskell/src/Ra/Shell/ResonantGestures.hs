{-|
Module      : Ra.Shell.ResonantGestures
Description : Biometric-attuned gesture access control
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Extends Ra.Shell.LimbLearning to conditionally unlock gestures based on
real-time coherence, torsion polarity, and user field phase alignment.
Prevents accidental activation from muscle spasms, mimicry, or "hollow intent".

== Resonant Access Theory

=== Coherence Gating

Gestures require minimum coherence:

* Default floor: φ / ANKH ≈ 0.318
* Can be customized per gesture
* High-consequence actions require higher thresholds

=== Torsion Filtering

* Normal torsion: All gestures allowed
* Inverted torsion: Only whitelisted gestures
* Null torsion: Neutral state, most gestures allowed

=== Phase Alignment

Gestures may require alignment with:

* φ^n emergence windows
* User's biometric phase
* Chamber resonance cycle
-}
module Ra.Shell.ResonantGestures
  ( -- * Core Types
    ResonantGesture(..)
  , TorsionState(..)
  , FluxCoherence
  , AuthorizationResult(..)

    -- * Authorization
  , authorizeResonantGesture
  , checkAuthorization
  , isAuthorized

    -- * Gate Configuration
  , GestureGate(..)
  , setGestureResonanceGate
  , getGestureGate
  , removeGestureGate

    -- * Torsion Whitelist
  , TorsionWhitelist
  , addToWhitelist
  , removeFromWhitelist
  , isWhitelisted

    -- * Phase Alignment
  , PhaseRequirement(..)
  , checkPhaseAlignment
  , phaseAlignmentScore

    -- * Biometric Integration
  , BiometricState(..)
  , getResonanceScore
  , checkBiometricGate

    -- * Audit Logging
  , GestureAuditLog(..)
  , logGestureAttempt
  , getAuditLog

    -- * Library Integration
  , ResonantLibrary(..)
  , mkResonantLibrary
  , authorizeFromLibrary
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import qualified Data.Set as Set
import Data.Set (Set)

import Ra.Constants.Extended
  ( phiInverse )

-- Import gesture types
import Ra.Shell.LimbGestures
  ( Gesture(..)
  , GestureEvent(..)
  , ScalarVectorTrack(..)
  )

import Ra.Shell.LimbLearning
  ( UserID
  , UserGestureLibrary
  , matchUserGesture
  )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Flux coherence measurement
type FluxCoherence = Double

-- | Resonant gesture with authorization data
data ResonantGesture = ResonantGesture
  { rgUserId       :: !UserID
  , rgGesture      :: !Gesture
  , rgGestureData  :: ![ScalarVectorTrack]
  , rgResonance    :: !FluxCoherence
  , rgTorsionPhase :: !TorsionState
  , rgMatched      :: !Bool
  , rgAuthorized   :: !Bool
  , rgPhaseScore   :: !Double
  } deriving (Eq, Show)

-- | Torsion state
data TorsionState
  = TorsionNormal     -- ^ Standard polarity
  | TorsionInverted   -- ^ Inverted polarity (shadow state)
  | TorsionNull       -- ^ Neutral/no torsion
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Authorization result
data AuthorizationResult
  = Authorized !ResonantGesture
  | DeniedLowCoherence !Double !Double  -- ^ (current, required)
  | DeniedInversion !Gesture
  | DeniedNoMatch
  | DeniedPhaseAlignment !Double        -- ^ Phase score too low
  deriving (Eq, Show)

-- =============================================================================
-- Authorization
-- =============================================================================

-- | Authorize gesture against biometric thresholds
authorizeResonantGesture :: UserID
                         -> [ScalarVectorTrack]
                         -> UserGestureLibrary
                         -> GestureGateConfig
                         -> BiometricState
                         -> AuthorizationResult
authorizeResonantGesture userId tracks library gateConfig bioState =
  let -- Check coherence first
      coherence = bsCoherence bioState
      torsion = bsTorsionState bioState
      phase = bsPhase bioState

      -- Get default coherence floor
      defaultFloor = phiInverse  -- ~0.618

  in case matchUserGesture library tracks of
       Nothing -> DeniedNoMatch

       Just event ->
         let gesture = geGestureType event
             requiredCoh = getGestureThreshold gesture gateConfig defaultFloor

             -- Check coherence gate
             cohPass = coherence >= requiredCoh

             -- Check torsion gate
             torsPass = checkTorsionGate gesture torsion (ggcWhitelist gateConfig)

             -- Check phase alignment
             phaseReq = getPhaseRequirement gesture gateConfig
             phaseScore = checkPhaseAlignment phase phaseReq
             phasePass = phaseScore >= 0.5

             -- Build resonant gesture
             resonantGesture = ResonantGesture
               { rgUserId = userId
               , rgGesture = gesture
               , rgGestureData = tracks
               , rgResonance = coherence
               , rgTorsionPhase = torsion
               , rgMatched = True
               , rgAuthorized = cohPass && torsPass && phasePass
               , rgPhaseScore = phaseScore
               }

         in if not cohPass then
              DeniedLowCoherence coherence requiredCoh
            else if not torsPass then
              DeniedInversion gesture
            else if not phasePass then
              DeniedPhaseAlignment phaseScore
            else
              Authorized resonantGesture

-- | Check if authorization result is positive
checkAuthorization :: AuthorizationResult -> Bool
checkAuthorization (Authorized _) = True
checkAuthorization _ = False

-- | Extract authorized gesture if any
isAuthorized :: AuthorizationResult -> Maybe ResonantGesture
isAuthorized (Authorized g) = Just g
isAuthorized _ = Nothing

-- =============================================================================
-- Gate Configuration
-- =============================================================================

-- | Gesture gate requirements
data GestureGate = GestureGate
  { ggCoherence    :: !FluxCoherence    -- ^ Required coherence [0, 1]
  , ggPhase        :: !(Maybe PhaseRequirement)
  , ggAllowInverted :: !Bool            -- ^ Allow in inverted state
  } deriving (Eq, Show)

-- | Full gate configuration
data GestureGateConfig = GestureGateConfig
  { ggcGates    :: !(Map Gesture GestureGate)
  , ggcWhitelist :: !TorsionWhitelist
  , ggcDefault  :: !GestureGate
  } deriving (Eq, Show)

-- | Default gate configuration
defaultGateConfig :: GestureGateConfig
defaultGateConfig = GestureGateConfig
  { ggcGates = Map.empty
  , ggcWhitelist = defaultWhitelist
  , ggcDefault = GestureGate phiInverse Nothing False
  }

-- | Set resonance gate for specific gesture
setGestureResonanceGate :: Gesture -> FluxCoherence -> GestureGateConfig -> GestureGateConfig
setGestureResonanceGate gesture coherence config =
  let gate = case Map.lookup gesture (ggcGates config) of
               Nothing -> GestureGate coherence Nothing False
               Just existing -> existing { ggCoherence = coherence }
  in config { ggcGates = Map.insert gesture gate (ggcGates config) }

-- | Get gate for gesture
getGestureGate :: Gesture -> GestureGateConfig -> GestureGate
getGestureGate gesture config =
  Map.findWithDefault (ggcDefault config) gesture (ggcGates config)

-- | Remove custom gate (revert to default)
removeGestureGate :: Gesture -> GestureGateConfig -> GestureGateConfig
removeGestureGate gesture config =
  config { ggcGates = Map.delete gesture (ggcGates config) }

-- Get threshold for gesture
getGestureThreshold :: Gesture -> GestureGateConfig -> FluxCoherence -> FluxCoherence
getGestureThreshold gesture config defaultVal =
  case Map.lookup gesture (ggcGates config) of
    Nothing -> defaultVal
    Just gate -> ggCoherence gate

-- Get phase requirement
getPhaseRequirement :: Gesture -> GestureGateConfig -> Maybe PhaseRequirement
getPhaseRequirement gesture config =
  case Map.lookup gesture (ggcGates config) of
    Nothing -> Nothing
    Just gate -> ggPhase gate

-- =============================================================================
-- Torsion Whitelist
-- =============================================================================

-- | Gestures allowed in inverted state
type TorsionWhitelist = Set Gesture

-- | Default whitelist (safe gestures in inversion)
defaultWhitelist :: TorsionWhitelist
defaultWhitelist = Set.fromList
  [ HoldStill    -- Safe to hold in any state
  , OpenHand     -- Release is always safe
  ]

-- | Add gesture to whitelist
addToWhitelist :: Gesture -> TorsionWhitelist -> TorsionWhitelist
addToWhitelist = Set.insert

-- | Remove from whitelist
removeFromWhitelist :: Gesture -> TorsionWhitelist -> TorsionWhitelist
removeFromWhitelist = Set.delete

-- | Check if gesture is whitelisted for inversion
isWhitelisted :: Gesture -> TorsionWhitelist -> Bool
isWhitelisted = Set.member

-- Check torsion gate
checkTorsionGate :: Gesture -> TorsionState -> TorsionWhitelist -> Bool
checkTorsionGate gesture torsion whitelist = case torsion of
  TorsionNormal -> True  -- All gestures allowed
  TorsionNull -> True    -- Neutral state allows most
  TorsionInverted -> isWhitelisted gesture whitelist

-- =============================================================================
-- Phase Alignment
-- =============================================================================

-- | Phase alignment requirement
data PhaseRequirement = PhaseRequirement
  { prPhase       :: !Double          -- ^ Required phase [0, 2*pi]
  , prTolerance   :: !Double          -- ^ Allowed deviation
  , prPhiWindow   :: !Int             -- ^ φ^n window alignment
  } deriving (Eq, Show)

-- | Check phase alignment
checkPhaseAlignment :: Double -> Maybe PhaseRequirement -> Double
checkPhaseAlignment _ Nothing = 1.0  -- No requirement = perfect alignment
checkPhaseAlignment userPhase (Just req) =
  let phaseDelta = abs (userPhase - prPhase req)
      normalizedDelta = min phaseDelta (2 * pi - phaseDelta)
      score = 1.0 - normalizedDelta / pi
  in if normalizedDelta <= prTolerance req
     then max 0.5 score
     else score * 0.5

-- | Get phase alignment score
phaseAlignmentScore :: Double -> Double -> Double
phaseAlignmentScore userPhase targetPhase =
  let delta = abs (userPhase - targetPhase)
      normalizedDelta = min delta (2 * pi - delta)
  in 1.0 - normalizedDelta / pi

-- =============================================================================
-- Biometric Integration
-- =============================================================================

-- | Biometric state for authorization
data BiometricState = BiometricState
  { bsCoherence    :: !FluxCoherence   -- ^ Current coherence
  , bsTorsionState :: !TorsionState    -- ^ Current torsion
  , bsPhase        :: !Double          -- ^ Current phase [0, 2*pi]
  , bsHRV          :: !Double          -- ^ Heart rate variability
  , bsBreathPhase  :: !Double          -- ^ Breath cycle phase
  } deriving (Eq, Show)

-- | Get resonance score from biometric state
getResonanceScore :: BiometricState -> FluxCoherence
getResonanceScore = bsCoherence

-- | Check if biometrics meet gate requirements
checkBiometricGate :: BiometricState -> GestureGate -> Bool
checkBiometricGate bio gate =
  let cohPass = bsCoherence bio >= ggCoherence gate
      torsPass = case bsTorsionState bio of
                   TorsionInverted -> ggAllowInverted gate
                   _ -> True
      phasePass = case ggPhase gate of
                    Nothing -> True
                    Just req -> checkPhaseAlignment (bsPhase bio) (Just req) >= 0.5
  in cohPass && torsPass && phasePass

-- =============================================================================
-- Audit Logging
-- =============================================================================

-- | Audit log entry for gesture attempt
data GestureAuditLog = GestureAuditLog
  { galUserId      :: !UserID
  , galGesture     :: !(Maybe Gesture)
  , galResult      :: !String
  , galCoherence   :: !FluxCoherence
  , galTorsion     :: !TorsionState
  , galTimestamp   :: !Int              -- ^ φ^n tick
  } deriving (Eq, Show)

-- | Create audit log from authorization result
logGestureAttempt :: UserID -> AuthorizationResult -> BiometricState -> Int -> GestureAuditLog
logGestureAttempt userId result bioState timestamp =
  let (gesture, resultStr) = case result of
        Authorized rg -> (Just (rgGesture rg), "AUTHORIZED")
        DeniedLowCoherence cur req -> (Nothing, "DENIED:LOW_COHERENCE:" ++ show cur ++ "<" ++ show req)
        DeniedInversion g -> (Just g, "DENIED:INVERSION")
        DeniedNoMatch -> (Nothing, "DENIED:NO_MATCH")
        DeniedPhaseAlignment score -> (Nothing, "DENIED:PHASE:" ++ show score)
  in GestureAuditLog
      { galUserId = userId
      , galGesture = gesture
      , galResult = resultStr
      , galCoherence = bsCoherence bioState
      , galTorsion = bsTorsionState bioState
      , galTimestamp = timestamp
      }

-- | Get formatted audit log
getAuditLog :: GestureAuditLog -> String
getAuditLog log' =
  "[" ++ show (galTimestamp log') ++ "] " ++
  galUserId log' ++ ": " ++
  maybe "no-gesture" show (galGesture log') ++ " -> " ++
  galResult log' ++
  " (coh=" ++ show (galCoherence log') ++ ", tor=" ++ show (galTorsion log') ++ ")"

-- =============================================================================
-- Library Integration
-- =============================================================================

-- | Resonant gesture library with gates
data ResonantLibrary = ResonantLibrary
  { rlGestureLib :: !UserGestureLibrary
  , rlGateConfig :: !GestureGateConfig
  , rlAuditLog   :: ![GestureAuditLog]
  } deriving (Eq, Show)

-- | Create resonant library
mkResonantLibrary :: UserGestureLibrary -> ResonantLibrary
mkResonantLibrary lib = ResonantLibrary
  { rlGestureLib = lib
  , rlGateConfig = defaultGateConfig
  , rlAuditLog = []
  }

-- | Authorize gesture from resonant library
authorizeFromLibrary :: UserID
                     -> [ScalarVectorTrack]
                     -> BiometricState
                     -> Int              -- ^ Timestamp
                     -> ResonantLibrary
                     -> (AuthorizationResult, ResonantLibrary)
authorizeFromLibrary userId tracks bioState timestamp lib =
  let result = authorizeResonantGesture userId tracks
                 (rlGestureLib lib)
                 (rlGateConfig lib)
                 bioState
      logEntry = logGestureAttempt userId result bioState timestamp
      updatedLib = lib { rlAuditLog = logEntry : take 1000 (rlAuditLog lib) }
  in (result, updatedLib)
