{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

-- | Prompt 19: Ra.DomainExtensions - Safety Module for Undefined Operations
-- FPGA module for handling domain boundaries, division by zero,
-- zero coherence states, and graceful recovery strategies.
--
-- Codex References:
-- - Ra.Emergence: emergenceScore modulation
-- - Ra.Coherence: CoherenceAbyss handling
-- - Ra.Shadow: ShadowSurface liminal layer
-- - P18: Attractor type integration
--
-- Features:
-- - safeDivide with epsilon substitution (SFixed 8 24 precision)
-- - Zero coherence (CoherenceAbyss) handling
-- - Semantic reframing for different operation types
-- - Multiplicative potential lift recovery
-- - shinigamiApple attractor in ROM

module RaDomainExtensions where

import Clash.Prelude
import GHC.Generics (Generic)

-- ============================================================================
-- Constants (Codex-aligned)
-- ============================================================================

-- | Epsilon for division safety (SFixed 8 24 representation)
-- Represents approximately 5.96e-8 (closest to 1e-12 in fixed point)
epsilon :: SFixed 8 24
epsilon = 0.000000059604644775390625  -- 2^-24

-- | Golden ratio
phiConst :: SFixed 8 24
phiConst = 1.618033988749895

-- | Shadow surface depth floor (0.42)
shadowSurfaceDepth :: Unsigned 12
shadowSurfaceDepth = 1720  -- 0.42 * 4096

-- | Maximum emergence score before overflow
maxEmergence :: Unsigned 24
maxEmergence = 16777215  -- 2^24 - 1

-- ============================================================================
-- Types
-- ============================================================================

-- | Domain boundary conditions
data DomainBoundary
  = BoundarySafe           -- Within valid domain
  | BoundaryZeroDivision   -- Division by zero attempted
  | BoundaryCoherenceAbyss -- Coherence exactly 0.0
  | BoundaryShadowSurface  -- In liminal echo layer
  | BoundaryNegative       -- Emergence score went negative
  | BoundaryOverflow       -- Value exceeded range
  deriving (Generic, NFDataX, Eq, Show)

-- | Recovery strategies for domain violations
data RecoveryStrategy
  = RecoveryNone           -- No recovery needed
  | RecoveryEpsilon        -- Replace zero with epsilon
  | RecoveryPotentialLift  -- Apply multiplicative lift
  | RecoveryShadowEcho     -- Return shadow surface echo
  | RecoveryClamp          -- Clamp to valid range
  deriving (Generic, NFDataX, Eq, Show)

-- | Domain result from safe operations
data DomainResult = DomainResult
  { drValue         :: SFixed 8 24      -- Result value
  , drBoundary      :: DomainBoundary   -- Boundary condition hit
  , drRecovery      :: RecoveryStrategy -- Recovery applied
  , drOriginal      :: SFixed 8 24      -- Original value before recovery
  , drPotentialLift :: SFixed 4 12      -- Lift factor applied
  } deriving (Generic, NFDataX, Show)

-- | Attractor type (isomorphic with P18)
data Attractor = Attractor
  { attId          :: Unsigned 8        -- Attractor ID
  , attEnticement  :: Unsigned 12       -- Enticement (0-4095)
  , attFlavor      :: Unsigned 2        -- 0=emotional, 1=sensory, 2=archetypal, 3=unknown
  , attPotential   :: SFixed 4 12       -- Potential lift factor
  } deriving (Generic, NFDataX, Show)

-- | Emergence condition state
data EmergenceCondition = EmergenceCondition
  { ecCoherence      :: Unsigned 12     -- Coherence (0-4095)
  , ecFlux           :: Unsigned 12     -- Flux (0-4095)
  , ecShadowThresh   :: Unsigned 12     -- Shadow threshold
  , ecEmergenceScore :: SFixed 8 24     -- Current emergence score
  } deriving (Generic, NFDataX, Show)

-- | Operation type for semantic reframing
data OperationType
  = OpFluxRatio
  | OpCoherenceFactor
  | OpEmergenceScale
  | OpGeneric
  deriving (Generic, NFDataX, Eq, Show)

-- | Domain input bundle
data DomainInput = DomainInput
  { diNumerator   :: SFixed 8 24
  , diDenominator :: SFixed 8 24
  , diOperation   :: OperationType
  , diCondition   :: EmergenceCondition
  , diAttractor   :: Maybe Attractor
  } deriving (Generic, NFDataX)

-- ============================================================================
-- ROM Attractors
-- ============================================================================

-- | Shinigami Apple (Death Note reference) - ROM attractor
shinigamiApple :: Attractor
shinigamiApple = Attractor
  { attId         = 1
  , attEnticement = 4091  -- 0.999 * 4096
  , attFlavor     = 2     -- Archetypal
  , attPotential  = 0.42  -- The answer to everything
  }

-- | Void Crystal - negative lift attractor
voidCrystal :: Attractor
voidCrystal = Attractor
  { attId         = 2
  , attEnticement = 41   -- 0.01 * 4096
  , attFlavor     = 3    -- Unknown
  , attPotential  = -0.15
  }

-- | Echo Mirror - neutral attractor
echoMirror :: Attractor
echoMirror = Attractor
  { attId         = 3
  , attEnticement = 2048  -- 0.5 * 4096
  , attFlavor     = 1     -- Sensory
  , attPotential  = 0.0
  }

-- ============================================================================
-- Core Domain Extension Functions
-- ============================================================================

-- | Check if value is effectively zero (within epsilon)
isEffectivelyZero :: SFixed 8 24 -> Bool
isEffectivelyZero x = abs x < epsilon

-- | Safe division with epsilon substitution
safeDivide :: SFixed 8 24 -> SFixed 8 24 -> DomainResult
safeDivide num denom
  | isEffectivelyZero denom =
      let safeDenom = if denom >= 0 then epsilon else -epsilon
          result = num / safeDenom
      in DomainResult
           { drValue = result
           , drBoundary = BoundaryZeroDivision
           , drRecovery = RecoveryEpsilon
           , drOriginal = 0  -- Would be infinity
           , drPotentialLift = 0
           }
  | otherwise =
      DomainResult
        { drValue = num / denom
        , drBoundary = BoundarySafe
        , drRecovery = RecoveryNone
        , drOriginal = num / denom
        , drPotentialLift = 0
        }

-- | Handle zero coherence (CoherenceAbyss)
handleZeroCoherence :: EmergenceCondition -> Maybe Attractor -> DomainResult
handleZeroCoherence EmergenceCondition{..} mAttractor
  | ecCoherence == 0 =
      case mAttractor of
        Just att | attPotential att > 0 ->
          -- Apply potential lift: multiplicative recovery
          let lifted = ecEmergenceScore * (1 + unSFixed (attPotential att))
          in DomainResult
               { drValue = lifted
               , drBoundary = BoundaryCoherenceAbyss
               , drRecovery = RecoveryPotentialLift
               , drOriginal = ecEmergenceScore
               , drPotentialLift = attPotential att
               }
        _ ->
          -- No attractor or negative lift - return shadow surface echo
          DomainResult
            { drValue = fromIntegral shadowSurfaceDepth / 4096.0
            , drBoundary = BoundaryCoherenceAbyss
            , drRecovery = RecoveryShadowEcho
            , drOriginal = ecEmergenceScore
            , drPotentialLift = 0
            }
  | otherwise =
      DomainResult
        { drValue = ecEmergenceScore
        , drBoundary = BoundarySafe
        , drRecovery = RecoveryNone
        , drOriginal = ecEmergenceScore
        , drPotentialLift = 0
        }
  where
    unSFixed :: SFixed 4 12 -> SFixed 8 24
    unSFixed x = fromIntegral (toInteger x)

-- | Reframe division by zero with semantic meaning
reframeDivisionByZero :: OperationType -> SFixed 8 24 -> SFixed 8 24 -> DomainResult
reframeDivisionByZero op num denom
  | not (isEffectivelyZero denom) =
      -- Normal division
      DomainResult
        { drValue = num / denom
        , drBoundary = BoundarySafe
        , drRecovery = RecoveryNone
        , drOriginal = num / denom
        , drPotentialLift = 0
        }
  | otherwise =
      case op of
        OpFluxRatio ->
          -- Flux ratio with zero = max flux potential (1.0)
          DomainResult
            { drValue = 1.0
            , drBoundary = BoundaryZeroDivision
            , drRecovery = RecoveryClamp
            , drOriginal = 0
            , drPotentialLift = 0
            }
        OpCoherenceFactor ->
          -- Coherence factor with zero = abyss
          DomainResult
            { drValue = 0.0
            , drBoundary = BoundaryCoherenceAbyss
            , drRecovery = RecoveryShadowEcho
            , drOriginal = 0
            , drPotentialLift = 0
            }
        OpEmergenceScale ->
          -- Emergence scale with zero = phi (divine proportion)
          DomainResult
            { drValue = phiConst
            , drBoundary = BoundaryZeroDivision
            , drRecovery = RecoveryEpsilon
            , drOriginal = 0
            , drPotentialLift = 0
            }
        OpGeneric ->
          -- Generic: use safe divide
          safeDivide num denom

-- | Apply potential lift multiplicatively
applyPotentialLift :: SFixed 8 24 -> SFixed 4 12 -> DomainResult
applyPotentialLift score lift =
  let lifted = score * (1 + unSFixed lift)
  in
    if lifted < 0
    then DomainResult
           { drValue = 0
           , drBoundary = BoundaryNegative
           , drRecovery = RecoveryClamp
           , drOriginal = lifted
           , drPotentialLift = lift
           }
    else if lifted > 1000000  -- Overflow threshold
    then DomainResult
           { drValue = 1000000
           , drBoundary = BoundaryOverflow
           , drRecovery = RecoveryClamp
           , drOriginal = lifted
           , drPotentialLift = lift
           }
    else DomainResult
           { drValue = lifted
           , drBoundary = BoundarySafe
           , drRecovery = RecoveryNone
           , drOriginal = lifted
           , drPotentialLift = lift
           }
  where
    unSFixed :: SFixed 4 12 -> SFixed 8 24
    unSFixed x = fromIntegral (toInteger x)

-- | Shadow surface echo for liminal states
shadowSurfaceEcho :: EmergenceCondition -> DomainResult
shadowSurfaceEcho EmergenceCondition{..}
  | ecCoherence < ecShadowThresh =
      -- In shadow region
      let depth = max shadowSurfaceDepth
                      (truncateB $ (fromIntegral ecCoherence :: SFixed 8 24) *
                                   (fromIntegral ecShadowThresh / 4096.0))
      in DomainResult
           { drValue = fromIntegral depth / 4096.0
           , drBoundary = BoundaryShadowSurface
           , drRecovery = RecoveryShadowEcho
           , drOriginal = ecEmergenceScore
           , drPotentialLift = 0
           }
  | otherwise =
      DomainResult
        { drValue = ecEmergenceScore
        , drBoundary = BoundarySafe
        , drRecovery = RecoveryNone
        , drOriginal = ecEmergenceScore
        , drPotentialLift = 0
        }

-- ============================================================================
-- Top-Level Entity
-- ============================================================================

-- | Process domain-safe operation
processDomain :: DomainInput -> DomainResult
processDomain DomainInput{..} =
  let
    -- First check for zero coherence
    zeroCheck = handleZeroCoherence diCondition diAttractor
  in
    if drBoundary zeroCheck /= BoundarySafe
    then zeroCheck
    else
      -- Check for division by zero with semantic reframing
      reframeDivisionByZero diOperation diNumerator diDenominator

-- | Top entity for domain extensions
{-# ANN domainExtensionsTop
  (Synthesize
    { t_name   = "domain_extensions_top"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "numerator"
                 , PortName "denominator"
                 , PortName "operation"
                 , PortName "coherence"
                 , PortName "flux"
                 , PortName "shadow_thresh"
                 , PortName "emergence_score"
                 , PortName "attractor_present"
                 , PortName "attractor_id"
                 , PortName "attractor_enticement"
                 , PortName "attractor_flavor"
                 , PortName "attractor_potential"
                 ]
    , t_output = PortProduct "output"
                 [ PortName "result_value"
                 , PortName "boundary"
                 , PortName "recovery"
                 , PortName "original"
                 , PortName "potential_lift"
                 ]
    }) #-}
domainExtensionsTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (SFixed 8 24)   -- numerator
  -> Signal System (SFixed 8 24)   -- denominator
  -> Signal System (Unsigned 2)    -- operation (0=flux, 1=coherence, 2=emergence, 3=generic)
  -> Signal System (Unsigned 12)   -- coherence
  -> Signal System (Unsigned 12)   -- flux
  -> Signal System (Unsigned 12)   -- shadow_thresh
  -> Signal System (SFixed 8 24)   -- emergence_score
  -> Signal System Bool            -- attractor_present
  -> Signal System (Unsigned 8)    -- attractor_id
  -> Signal System (Unsigned 12)   -- attractor_enticement
  -> Signal System (Unsigned 2)    -- attractor_flavor
  -> Signal System (SFixed 4 12)   -- attractor_potential
  -> Signal System ( SFixed 8 24   -- result_value
                   , Unsigned 3    -- boundary (encoded)
                   , Unsigned 3    -- recovery (encoded)
                   , SFixed 8 24   -- original
                   , SFixed 4 12   -- potential_lift
                   )
domainExtensionsTop clk rst en
                    num denom op
                    coh flux shadThresh emScore
                    attPresent attId attEnt attFlav attPot =
  withClockResetEnable clk rst en $
    let
      -- Decode operation
      decOp o = case o of
        0 -> OpFluxRatio
        1 -> OpCoherenceFactor
        2 -> OpEmergenceScale
        _ -> OpGeneric

      -- Build attractor
      buildAtt present aId aEnt aFlav aPot =
        if present
        then Just $ Attractor aId aEnt aFlav aPot
        else Nothing

      -- Build condition
      buildCond c f st es = EmergenceCondition c f st es

      -- Build input
      input = DomainInput
        <$> num
        <*> denom
        <*> fmap decOp op
        <*> (buildCond <$> coh <*> flux <*> shadThresh <*> emScore)
        <*> (buildAtt <$> attPresent <*> attId <*> attEnt <*> attFlav <*> attPot)

      -- Process and extract output
      output = fmap processDomain input

      -- Encode boundary
      encodeBoundary b = case b of
        BoundarySafe -> 0
        BoundaryZeroDivision -> 1
        BoundaryCoherenceAbyss -> 2
        BoundaryShadowSurface -> 3
        BoundaryNegative -> 4
        BoundaryOverflow -> 5

      -- Encode recovery
      encodeRecovery r = case r of
        RecoveryNone -> 0
        RecoveryEpsilon -> 1
        RecoveryPotentialLift -> 2
        RecoveryShadowEcho -> 3
        RecoveryClamp -> 4

      extractOut DomainResult{..} =
        ( drValue
        , encodeBoundary drBoundary
        , encodeRecovery drRecovery
        , drOriginal
        , drPotentialLift
        )
    in fmap extractOut output

-- ============================================================================
-- Testbench
-- ============================================================================

-- | Test condition (zero coherence)
testConditionAbyss :: EmergenceCondition
testConditionAbyss = EmergenceCondition
  { ecCoherence = 0
  , ecFlux = 2048
  , ecShadowThresh = 2949  -- 0.72 * 4096
  , ecEmergenceScore = 0.5
  }

-- | Test condition (normal)
testConditionNormal :: EmergenceCondition
testConditionNormal = EmergenceCondition
  { ecCoherence = 3277  -- 0.8 * 4096
  , ecFlux = 2048
  , ecShadowThresh = 2949
  , ecEmergenceScore = 0.7
  }

-- | Test inputs
testInputs :: Vec 4 DomainInput
testInputs =
     -- Test 1: Safe division
     DomainInput 10.0 2.0 OpGeneric testConditionNormal Nothing
  :> -- Test 2: Zero division with flux ratio
     DomainInput 5.0 0.0 OpFluxRatio testConditionNormal Nothing
  :> -- Test 3: Zero coherence with shinigami apple
     DomainInput 1.0 1.0 OpGeneric testConditionAbyss (Just shinigamiApple)
  :> -- Test 4: Zero coherence without attractor
     DomainInput 1.0 1.0 OpGeneric testConditionAbyss Nothing
  :> Nil

-- | Testbench entity
testBench :: Signal System DomainResult
testBench = withClockResetEnable systemClockGen systemResetGen enableGen $
  fmap processDomain (fromList (toList testInputs))
