{-|
Module      : Ra.Consent.Transformer
Description : Multi-core consent transformer using Hubbard + Tesla principles
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Hybrid system modeling consent state amplification and transduction using
principles from the Hubbard Energy Transformer (core-coil geometries, 3:1
step-up resonance) and Tesla Boundary Layer Turbine (disk-layer adhesion,
reversible state pathways, valvular conduits).

== Transformer Theory

=== Hubbard Core Geometry

* Central chamber tube with harmonic rods
* Surrounding mesh of outer coils (state amplifiers)
* Radioactive→electrical state mapping analogy
* Ground-based resonance wire networks

=== Tesla Boundary Dynamics

* Disk-layer adhesion modeling for state flow
* Reversible state pathways through valvular conduits
* High-frequency frictionless resonance flow
* Asymmetric flow for consent amplification
-}
module Ra.Consent.Transformer
  ( -- * Core Types
    ConsentTransformer(..)
  , TransformerCore(..)
  , CoilMesh(..)
  , ValvularConduit(..)

    -- * Transformer Construction
  , createTransformer
  , configureCore
  , addCoilLayer
  , setResonanceRatio

    -- * State Transduction
  , TransductionResult(..)
  , transduce
  , consentInputSignal
  , permissionOutput

    -- * Harmonic Operations
  , HarmonicResonance(..)
  , harmonicExcitation
  , stepUpConsent
  , resonanceLock

    -- * Flow Control
  , FlowState(..)
  , initiateFlow
  , reverseFlow
  , valvularGating

    -- * Biometric Integration
  , BiometricPulse(..)
  , processPulse
  , pulseToCoherence
  , feedbackLoop
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete consent transformer system
data ConsentTransformer = ConsentTransformer
  { ctCore         :: !TransformerCore     -- ^ Central chamber
  , ctCoilMesh     :: !CoilMesh            -- ^ Outer coil array
  , ctConduits     :: ![ValvularConduit]   -- ^ Flow conduits
  , ctResonanceRatio :: !Double            -- ^ Step-up ratio (default 3:1)
  , ctCoherence    :: !Double              -- ^ Current coherence [0, 1]
  , ctFlowState    :: !FlowState           -- ^ Current flow state
  , ctActive       :: !Bool                -- ^ Transformer active
  } deriving (Eq, Show)

-- | Central transformer core (Hubbard-style)
data TransformerCore = TransformerCore
  { tcHarmonicRods :: !Int                 -- ^ Number of harmonic rods
  , tcRodMaterial  :: !RodMaterial         -- ^ Rod material type
  , tcCoreRadius   :: !Double              -- ^ Core radius (normalized)
  , tcCoreLength   :: !Double              -- ^ Core length (normalized)
  , tcExcitation   :: !Double              -- ^ Excitation level [0, 1]
  , tcFrequency    :: !Double              -- ^ Operating frequency (Hz)
  } deriving (Eq, Show)

-- | Rod material types (analogous to Hubbard's materials)
data RodMaterial
  = MaterialFerrous      -- ^ Iron-based (standard)
  | MaterialCrystalline  -- ^ Quartz/crystal resonance
  | MaterialCopper       -- ^ Copper conductivity
  | MaterialGold         -- ^ Gold coherence
  | MaterialComposite    -- ^ Multi-material
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Outer coil mesh (state amplifiers)
data CoilMesh = CoilMesh
  { cmLayers       :: !Int                 -- ^ Number of coil layers
  , cmWindingsPerLayer :: !Int             -- ^ Windings per layer
  , cmLayerSpacing :: !Double              -- ^ Spacing between layers
  , cmAmplification :: !Double             -- ^ Amplification factor
  , cmPhaseOffset  :: !Double              -- ^ Phase offset [0, 2pi]
  } deriving (Eq, Show)

-- | Valvular conduit (Tesla-style asymmetric flow)
data ValvularConduit = ValvularConduit
  { vcId           :: !String              -- ^ Conduit identifier
  , vcDirection    :: !FlowDirection       -- ^ Preferred flow direction
  , vcResistance   :: !Double              -- ^ Flow resistance [0, 1]
  , vcThreshold    :: !Double              -- ^ Activation threshold
  , vcOpen         :: !Bool                -- ^ Conduit open state
  } deriving (Eq, Show)

-- | Flow direction
data FlowDirection
  = FlowForward       -- ^ Normal consent flow
  | FlowReverse       -- ^ Reverse/withdrawal flow
  | FlowBidirectional -- ^ Both directions
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Flow state
data FlowState
  = FlowIdle          -- ^ No active flow
  | FlowCharging      -- ^ Building charge
  | FlowDischarging   -- ^ Releasing energy
  | FlowSteady        -- ^ Stable continuous flow
  | FlowReversing     -- ^ Direction reversal
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Transformer Construction
-- =============================================================================

-- | Create default consent transformer
createTransformer :: ConsentTransformer
createTransformer = ConsentTransformer
  { ctCore = defaultCore
  , ctCoilMesh = defaultCoilMesh
  , ctConduits = defaultConduits
  , ctResonanceRatio = 3.0  -- Classic Hubbard 3:1
  , ctCoherence = 0.5
  , ctFlowState = FlowIdle
  , ctActive = False
  }

-- | Configure transformer core
configureCore :: ConsentTransformer -> Int -> RodMaterial -> Double -> ConsentTransformer
configureCore trans rodCount material freq =
  let core = (ctCore trans)
        { tcHarmonicRods = rodCount
        , tcRodMaterial = material
        , tcFrequency = freq
        }
  in trans { ctCore = core }

-- | Add coil layer to mesh
addCoilLayer :: ConsentTransformer -> Int -> ConsentTransformer
addCoilLayer trans windings =
  let mesh = ctCoilMesh trans
      newMesh = mesh
        { cmLayers = cmLayers mesh + 1
        , cmWindingsPerLayer = windings
        , cmAmplification = cmAmplification mesh * phi
        }
  in trans { ctCoilMesh = newMesh }

-- | Set resonance step-up ratio
setResonanceRatio :: ConsentTransformer -> Double -> ConsentTransformer
setResonanceRatio trans ratio =
  trans { ctResonanceRatio = max 1.0 ratio }

-- =============================================================================
-- State Transduction
-- =============================================================================

-- | Transduction result
data TransductionResult = TransductionResult
  { trPermission   :: !PermissionState     -- ^ Output permission level
  , trAmplification :: !Double             -- ^ Amplification achieved
  , trCoherence    :: !Double              -- ^ Resulting coherence
  , trStable       :: !Bool                -- ^ Stability indicator
  , trFeedback     :: !Double              -- ^ Feedback signal
  } deriving (Eq, Show)

-- | Permission state output
data PermissionState
  = PermissionDenied      -- ^ No permission granted
  | PermissionLimited     -- ^ Partial/conditional permission
  | PermissionGranted     -- ^ Full permission
  | PermissionAmplified   -- ^ Enhanced/amplified permission
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Main transduction function
transduce :: ConsentTransformer -> CoherenceState -> HarmonicResonance -> TransductionResult
transduce trans coherence resonance =
  let -- Calculate gate opening based on coherence
      gateOpen = csLevel coherence >= gateThreshold trans

      -- Apply step-up if gate is open
      amplified = if gateOpen
                  then csLevel coherence * ctResonanceRatio trans * hrStrength resonance
                  else csLevel coherence * 0.5

      -- Determine permission level
      permission = calculatePermission amplified (ctCoherence trans)

      -- Stability check
      stable = amplified < 2.0 && csStability coherence > 0.3

      -- Feedback for loop closure
      feedback = (amplified - ctCoherence trans) * phiInverse

  in TransductionResult
    { trPermission = permission
    , trAmplification = amplified
    , trCoherence = min 1.0 amplified
    , trStable = stable
    , trFeedback = feedback
    }

-- | Consent input signal processing
consentInputSignal :: CoherenceState -> HarmonicResonance -> Double
consentInputSignal coherence resonance =
  csLevel coherence * hrStrength resonance * (1 + fromIntegral (hrHarmonic resonance) * 0.1)

-- | Permission output calculation
permissionOutput :: TransductionResult -> (PermissionState, Double)
permissionOutput result = (trPermission result, trAmplification result)

-- =============================================================================
-- Harmonic Operations
-- =============================================================================

-- | Harmonic resonance parameters
data HarmonicResonance = HarmonicResonance
  { hrFrequency    :: !Double              -- ^ Base frequency (Hz)
  , hrHarmonic     :: !Int                 -- ^ Harmonic number
  , hrStrength     :: !Double              -- ^ Resonance strength [0, 1]
  , hrPhase        :: !Double              -- ^ Phase [0, 2pi]
  } deriving (Eq, Show)

-- | Coherence state input
data CoherenceState = CoherenceState
  { csLevel        :: !Double              -- ^ Coherence level [0, 1]
  , csStability    :: !Double              -- ^ Stability factor [0, 1]
  , csAnkhDelta    :: !Double              -- ^ Delta(ankh) [-1, 1]
  } deriving (Eq, Show)

-- | Apply harmonic excitation to core
harmonicExcitation :: ConsentTransformer -> HarmonicResonance -> ConsentTransformer
harmonicExcitation trans resonance =
  let core = ctCore trans
      newExcitation = tcExcitation core + hrStrength resonance * 0.2
      newCore = core { tcExcitation = min 1.0 newExcitation }
  in trans { ctCore = newCore }

-- | Step up consent through coil mesh
stepUpConsent :: ConsentTransformer -> Double -> Double
stepUpConsent trans inputLevel =
  let meshAmp = cmAmplification (ctCoilMesh trans)
      ratio = ctResonanceRatio trans
      layers = fromIntegral (cmLayers (ctCoilMesh trans))
  in inputLevel * meshAmp * (ratio / 3) * (1 + layers * 0.1)

-- | Check for resonance lock condition
resonanceLock :: ConsentTransformer -> HarmonicResonance -> Bool
resonanceLock trans resonance =
  let coreFreq = tcFrequency (ctCore trans)
      resFreq = hrFrequency resonance * fromIntegral (hrHarmonic resonance)
      freqMatch = abs (coreFreq - resFreq) / max 1 coreFreq < 0.05
      strengthOk = hrStrength resonance > phiInverse
  in freqMatch && strengthOk

-- =============================================================================
-- Flow Control
-- =============================================================================

-- | Initiate consent flow
initiateFlow :: ConsentTransformer -> ConsentTransformer
initiateFlow trans =
  let openConduits = map (\c -> c { vcOpen = True }) (ctConduits trans)
  in trans
    { ctConduits = openConduits
    , ctFlowState = FlowCharging
    , ctActive = True
    }

-- | Reverse flow direction
reverseFlow :: ConsentTransformer -> ConsentTransformer
reverseFlow trans =
  let reversedConduits = map reverseConduit (ctConduits trans)
  in trans
    { ctConduits = reversedConduits
    , ctFlowState = FlowReversing
    }

-- | Apply valvular gating
valvularGating :: ConsentTransformer -> Double -> ConsentTransformer
valvularGating trans pressure =
  let gatedConduits = map (gateConduit pressure) (ctConduits trans)
      anyOpen = any vcOpen gatedConduits
      newFlow = if anyOpen then FlowSteady else FlowIdle
  in trans { ctConduits = gatedConduits, ctFlowState = newFlow }

-- =============================================================================
-- Biometric Integration
-- =============================================================================

-- | Biometric pulse from portal
data BiometricPulse = BiometricPulse
  { bpSource       :: !String              -- ^ Source identifier
  , bpHRV          :: !Double              -- ^ Heart rate variability
  , bpCoherence    :: !Double              -- ^ Measured coherence
  , bpIntensity    :: !Double              -- ^ Pulse intensity
  , bpTimestamp    :: !Int                 -- ^ Pulse timestamp
  } deriving (Eq, Show)

-- | Process biometric pulse through transformer
processPulse :: ConsentTransformer -> BiometricPulse -> (ConsentTransformer, TransductionResult)
processPulse trans pulse =
  let coherence = CoherenceState
        { csLevel = bpCoherence pulse
        , csStability = bpHRV pulse / 100
        , csAnkhDelta = 0
        }
      resonance = HarmonicResonance
        { hrFrequency = 7.83
        , hrHarmonic = 1
        , hrStrength = bpIntensity pulse
        , hrPhase = 0
        }
      result = transduce trans coherence resonance
      newTrans = trans { ctCoherence = trCoherence result }
  in (newTrans, result)

-- | Convert pulse to coherence state
pulseToCoherence :: BiometricPulse -> CoherenceState
pulseToCoherence pulse = CoherenceState
  { csLevel = bpCoherence pulse
  , csStability = min 1.0 (bpHRV pulse / 100)
  , csAnkhDelta = (bpIntensity pulse - 0.5) * 2
  }

-- | Complete feedback loop cycle
feedbackLoop :: ConsentTransformer -> BiometricPulse -> (ConsentTransformer, Double)
feedbackLoop trans pulse =
  let (newTrans, result) = processPulse trans pulse
      feedback = trFeedback result
      adjustedTrans = if trStable result
                      then newTrans { ctFlowState = FlowSteady }
                      else newTrans { ctFlowState = FlowCharging }
  in (adjustedTrans, feedback)

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Default transformer core
defaultCore :: TransformerCore
defaultCore = TransformerCore
  { tcHarmonicRods = 8
  , tcRodMaterial = MaterialFerrous
  , tcCoreRadius = 0.1
  , tcCoreLength = 0.5
  , tcExcitation = 0
  , tcFrequency = 528
  }

-- | Default coil mesh
defaultCoilMesh :: CoilMesh
defaultCoilMesh = CoilMesh
  { cmLayers = 3
  , cmWindingsPerLayer = 100
  , cmLayerSpacing = 0.02
  , cmAmplification = phi
  , cmPhaseOffset = 0
  }

-- | Default conduits
defaultConduits :: [ValvularConduit]
defaultConduits =
  [ ValvularConduit "primary" FlowForward 0.2 0.3 False
  , ValvularConduit "secondary" FlowForward 0.3 0.4 False
  , ValvularConduit "reverse" FlowReverse 0.5 0.6 False
  ]

-- | Gate threshold calculation
gateThreshold :: ConsentTransformer -> Double
gateThreshold trans =
  phiInverse / ctResonanceRatio trans

-- | Calculate permission from amplified level
calculatePermission :: Double -> Double -> PermissionState
calculatePermission amplified _currentCoherence
  | amplified < 0.2 = PermissionDenied
  | amplified < phiInverse = PermissionLimited
  | amplified < phi = PermissionGranted
  | otherwise = PermissionAmplified

-- | Reverse a conduit
reverseConduit :: ValvularConduit -> ValvularConduit
reverseConduit c = c
  { vcDirection = case vcDirection c of
      FlowForward -> FlowReverse
      FlowReverse -> FlowForward
      FlowBidirectional -> FlowBidirectional
  }

-- | Gate conduit based on pressure
gateConduit :: Double -> ValvularConduit -> ValvularConduit
gateConduit pressure c =
  c { vcOpen = pressure > vcThreshold c }
