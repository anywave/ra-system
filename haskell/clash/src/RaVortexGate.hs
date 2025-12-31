{-|
Module      : RaVortexGate
Description : Scalar Vortex Gate as Consent-Based Portal
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 38: Consent-gated scalar vortex portal with spin direction
and phase opposition detection.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaVortexGate where

import Clash.Prelude

-- | Consent similarity threshold (0.93 * 255 = 237)
consentThreshold :: Unsigned 8
consentThreshold = 237

-- | Default coherence threshold (0.72 * 255 = 184)
coherenceThreshold :: Unsigned 8
coherenceThreshold = 184

-- | Spin direction
data SpinDirection = Clockwise | CounterClockwise
  deriving (Generic, NFDataX, Eq, Show)

-- | Gate activation state
data GateState = Locked | Spinning | Traversable
  deriving (Generic, NFDataX, Eq, Show)

-- | Gate polarity configuration
data GatePolarity = StandardPolarity | InvertedPolarity
  deriving (Generic, NFDataX, Eq, Show)

-- | Consent hash (8 bytes stored as Vec 8)
type ConsentHash = Vec 8 (Unsigned 8)

-- | Phase vector (16-bit fixed point for magnitude and angle)
data PhaseVector = PhaseVector
  { pvMagnitude :: Unsigned 16  -- Scaled 0-65535
  , pvAngle     :: Signed 16    -- Radians * 4096
  , pvSpin      :: SpinDirection
  } deriving (Generic, NFDataX)

-- | Torsion field visual state
data TorsionField = TorsionField
  { tfSpinVelocity   :: Unsigned 16  -- Radians/sec * 256
  , tfFieldRadius    :: Unsigned 16  -- Scaled radius
  , tfCollapseFactor :: Unsigned 8   -- 0=open, 255=collapsed
  , tfGlowIntensity  :: Unsigned 8   -- 0-255
  } deriving (Generic, NFDataX)

-- | Vortex gate configuration
data VortexGate = VortexGate
  { vgGateId           :: Unsigned 8
  , vgTorsionSpin      :: SpinDirection
  , vgRequiredCoherence :: Unsigned 8
  , vgConsentSignature :: ConsentHash
  , vgCurrentState     :: GateState
  , vgFieldAlignment   :: PhaseVector
  , vgAuraField        :: TorsionField
  , vgPolarityConfig   :: GatePolarity
  } deriving (Generic, NFDataX)

-- | User state for gate passage evaluation
data UserState = UserState
  { usCurrentCoherence :: Unsigned 8
  , usConsentHash      :: ConsentHash
  , usSpinPolarity     :: SpinDirection
  , usPhaseVec         :: PhaseVector
  } deriving (Generic, NFDataX)

-- | Gate evaluation result
data GateResult = GateResult
  { grCanPass  :: Bool
  , grNewState :: GateState
  , grReason   :: Unsigned 4  -- 0=OK, 1=LowCoherence, 2=ConsentMismatch, 3=PhaseOpposed
  } deriving (Generic, NFDataX)

-- | Compute cosine similarity between consent hashes
-- Simplified: sum of element-wise products / (mag1 * mag2)
-- Returns 0-255 similarity score
matchConsent :: ConsentHash -> ConsentHash -> Unsigned 8
matchConsent h1 h2 =
  let products = zipWith (\a b -> resize a * resize b :: Unsigned 16) h1 h2
      dotProduct = fold (+) products
      -- Simplified normalization (assumes similar magnitudes)
      normalized = dotProduct `shiftR` 8
  in resize $ min 255 normalized

-- | Check if phase vectors are opposed (dot product < 0)
-- Uses simplified 2D approximation: cos(angle_diff)
isOpposed :: PhaseVector -> PhaseVector -> Bool
isOpposed v1 v2 =
  let angleDiff = pvAngle v1 - pvAngle v2
      -- Opposed if angle difference > π/2 (4096 * π/2 ≈ 6434)
      -- or < -π/2
  in angleDiff > 6434 || angleDiff < (-6434)

-- | Get spin polarity value
getSpinPolarity :: SpinDirection -> GatePolarity -> Signed 8
getSpinPolarity spin config = case (spin, config) of
  (Clockwise, StandardPolarity)        -> 1
  (CounterClockwise, StandardPolarity) -> -1
  (Clockwise, InvertedPolarity)        -> -1
  (CounterClockwise, InvertedPolarity) -> 1

-- | Check if user can pass through gate
canPassThrough :: VortexGate -> UserState -> Bool
canPassThrough gate user =
  let coherenceOk = usCurrentCoherence user >= vgRequiredCoherence gate
      consentSim = matchConsent (vgConsentSignature gate) (usConsentHash user)
      consentOk = consentSim >= consentThreshold
      phaseOk = not $ isOpposed (vgFieldAlignment gate) (usPhaseVec user)
  in coherenceOk && consentOk && phaseOk

-- | Evaluate gate and return detailed result
evaluateGate :: VortexGate -> UserState -> GateResult
evaluateGate gate user
  | usCurrentCoherence user < vgRequiredCoherence gate =
      GateResult False Locked 1  -- Low coherence
  | matchConsent (vgConsentSignature gate) (usConsentHash user) < consentThreshold =
      GateResult False Locked 2  -- Consent mismatch
  | isOpposed (vgFieldAlignment gate) (usPhaseVec user) =
      GateResult False Locked 3  -- Phase opposed
  | otherwise =
      GateResult True Traversable 0  -- All conditions met

-- | Update gate torsion field visuals based on state
updateVisuals :: VortexGate -> UserState -> TorsionField
updateVisuals gate user =
  if canPassThrough gate user
  then TorsionField
    { tfSpinVelocity = 128     -- Slow (0.5 rad/s * 256)
    , tfFieldRadius = 512      -- Expanded (2.0 * 256)
    , tfCollapseFactor = 0     -- Open
    , tfGlowIntensity = 204    -- Bright (0.8 * 255)
    }
  else TorsionField
    { tfSpinVelocity = 1280    -- Fast (5.0 rad/s * 256)
    , tfFieldRadius = 128      -- Contracted (0.5 * 256)
    , tfCollapseFactor = 204   -- Nearly collapsed (0.8 * 255)
    , tfGlowIntensity = 77     -- Dim (0.3 * 255)
    }

-- | Create default vortex gate
defaultGate :: Unsigned 8 -> ConsentHash -> VortexGate
defaultGate gateId consent = VortexGate
  { vgGateId = gateId
  , vgTorsionSpin = Clockwise
  , vgRequiredCoherence = coherenceThreshold
  , vgConsentSignature = consent
  , vgCurrentState = Locked
  , vgFieldAlignment = PhaseVector 1024 0 Clockwise
  , vgAuraField = TorsionField 256 256 128 128
  , vgPolarityConfig = StandardPolarity
  }

-- | Create default user state
defaultUser :: Unsigned 8 -> ConsentHash -> UserState
defaultUser coherence consent = UserState
  { usCurrentCoherence = coherence
  , usConsentHash = consent
  , usSpinPolarity = Clockwise
  , usPhaseVec = PhaseVector 1024 0 Clockwise
  }

-- | Top-level gate evaluation pipeline
gateEvaluationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (VortexGate, UserState)
  -> Signal dom GateResult
gateEvaluationPipeline input = uncurry evaluateGate <$> input

-- | Gate state pipeline
gateStatePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (VortexGate, UserState)
  -> Signal dom GateState
gateStatePipeline input = grNewState <$> gateEvaluationPipeline input

-- | Visual update pipeline
visualPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (VortexGate, UserState)
  -> Signal dom TorsionField
visualPipeline input = uncurry updateVisuals <$> input
