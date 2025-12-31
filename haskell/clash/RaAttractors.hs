{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

-- | Prompt 18: Ra Attractors and Emergence Modulation
-- FPGA module for metaphysical attractors that modulate Ra fragment
-- emergence thresholds by influencing coherence fields and resonance.
--
-- Codex References:
-- - KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md: Sympathetic resonance
-- - RA_SCALAR.hs: Fragment emergence logic
-- - Prompt 12: ShadowConsent
--
-- Features:
-- - Attractor type with fragment targeting
-- - Phi-scaled enticement curve (enticement^1.618)
-- - PhaseComponent flavor effects
-- - AttractorEffect with priority ordering
-- - Competing attractors resolution
-- - 8 concurrent attractors supported

module RaAttractors where

import Clash.Prelude
import GHC.Generics (Generic)

-- ============================================================================
-- Constants (Codex-aligned)
-- ============================================================================

-- | Golden ratio (SFixed 4 12)
phiConst :: SFixed 4 12
phiConst = 1.618

-- | Minimum enticement to activate (0.58 * 4096)
enticementMin :: Unsigned 12
enticementMin = 2376  -- 0.58

-- | Coherence floor (0.22 * 4096)
coherenceFloor :: Unsigned 12
coherenceFloor = 901  -- 0.22

-- | Shadow threshold normal (0.72 * 4096)
shadowThresholdNormal :: Unsigned 12
shadowThresholdNormal = 2950

-- | Shadow threshold lowered (0.61 * 4096)
shadowThresholdLowered :: Unsigned 12
shadowThresholdLowered = 2499

-- | Maximum concurrent effects
maxEffects :: Unsigned 2
maxEffects = 2

-- ============================================================================
-- Types
-- ============================================================================

-- | Phase component (attractor flavor)
data PhaseComponent
  = Emotional      -- Affects flux, variability
  | Sensory        -- Modulates temporal phase
  | Archetypal     -- Potential guardian gate bypass
  | UnknownPhase   -- ±25% chaos modulation
  deriving (Generic, NFDataX, Eq, Show)

-- | Attractor effect type
data AttractorEffect
  = EffectNone
  | GatingOverride    -- Priority 1
  | InversionFlip     -- Priority 2
  | ResonanceBoost    -- Priority 3
  deriving (Generic, NFDataX, Eq, Show)

-- | Fragment target type
data FragmentTarget
  = TargetById (Unsigned 12)      -- Explicit fragment ID
  | TargetShadow                  -- shadow:*
  | TargetGuardian                -- guardian:*
  | TargetShellGE (Unsigned 3)    -- h>=n
  | TargetSymbolic                -- emergence_form=SYMBOLIC
  deriving (Generic, NFDataX, Eq, Show)

-- | Fragment descriptor
data Fragment = Fragment
  { fragId          :: Unsigned 12
  , fragCoherence   :: Unsigned 12
  , fragPotential   :: Unsigned 12
  , fragFlux        :: Signed 12
  , fragPhase       :: Unsigned 12
  , fragShell       :: Unsigned 3
  , fragIsShadow    :: Bool
  , fragIsGuardian  :: Bool
  , fragIsSymbolic  :: Bool
  } deriving (Generic, NFDataX, Show)

-- | Attractor descriptor
data Attractor = Attractor
  { attSymbol      :: Unsigned 8      -- Symbol ID
  , attFlavor      :: PhaseComponent  -- Phase component
  , attEnticement  :: Unsigned 12     -- Enticement level (0-4095)
  , attTarget      :: FragmentTarget  -- Target selector
  , attActive      :: Bool            -- Is attractor active
  } deriving (Generic, NFDataX, Show)

-- | Emergence condition
data EmergenceCondition = EmergenceCondition
  { ecPotential       :: Unsigned 12
  , ecFlux            :: Signed 12
  , ecPhase           :: Unsigned 12
  , ecInversion       :: Unsigned 12   -- Inversion likelihood
  , ecThreshold       :: Unsigned 12   -- Coherence threshold
  , ecGatingActive    :: Bool
  , ecShadowAccess    :: Bool
  } deriving (Generic, NFDataX, Show)

-- | Attractor result
data AttractorResult = AttractorResult
  { arEffect1         :: AttractorEffect
  , arEffect2         :: AttractorEffect
  , arCoherenceDelta  :: Unsigned 12
  , arPotentialBoost  :: Unsigned 12
  , arGatingOverride  :: Bool
  , arInversionFlip   :: Bool
  , arMatched         :: Bool
  } deriving (Generic, NFDataX, Show)

-- | Combined output
data AttractorOutput = AttractorOutput
  { aoCondition :: EmergenceCondition
  , aoResult    :: AttractorResult
  } deriving (Generic, NFDataX, Show)

-- ============================================================================
-- Phi-Scaled Enticement
-- ============================================================================

-- | Approximate enticement^phi using lookup table
-- For FPGA efficiency, use piecewise linear approximation
phiScaledEnticement :: Unsigned 12 -> Unsigned 12
phiScaledEnticement ent
  | ent < 1024  = ent `shiftR` 1              -- ~0.5x for low values
  | ent < 2048  = ent - (ent `shiftR` 2)      -- ~0.75x for mid values
  | ent < 3072  = ent - (ent `shiftR` 3)      -- ~0.875x for high values
  | otherwise   = ent - (ent `shiftR` 4)      -- ~0.9375x for very high

-- ============================================================================
-- Fragment Targeting
-- ============================================================================

-- | Check if attractor targets fragment
matchesTarget :: FragmentTarget -> Fragment -> Bool
matchesTarget (TargetById fid) frag = fragId frag == fid
matchesTarget TargetShadow frag = fragIsShadow frag
matchesTarget TargetGuardian frag = fragIsGuardian frag
matchesTarget (TargetShellGE n) frag = fragShell frag >= n
matchesTarget TargetSymbolic frag = fragIsSymbolic frag

-- ============================================================================
-- Flavor Effects
-- ============================================================================

-- | Apply flavor-specific modulation
applyFlavor :: PhaseComponent -> Unsigned 12 -> EmergenceCondition -> EmergenceCondition
applyFlavor flavor enticement cond =
  let
    entScaled = resize enticement `shiftR` 3  -- /8 for scaling
  in case flavor of
    Emotional ->
      cond { ecFlux = ecFlux cond + resize entScaled
           , ecInversion = ecInversion cond + (enticement `shiftR` 4)
           }
    Sensory ->
      cond { ecPhase = ecPhase cond + (enticement `shiftR` 3)
           , ecFlux = ecFlux cond + resize (enticement `shiftR` 5)
           }
    Archetypal ->
      cond { ecGatingActive = enticement < 3482  -- < 0.85
           , ecPotential = ecPotential cond + (enticement `shiftR` 4)
           }
    UnknownPhase ->
      -- Chaos modulation (simplified: use enticement bits as "random")
      let chaos = resize (enticement .&. 0x1FF) :: Signed 12
      in cond { ecPotential = if chaos > 0
                              then ecPotential cond + resize (enticement `shiftR` 4)
                              else satSub SatBound (ecPotential cond) (enticement `shiftR` 4)
              }

-- ============================================================================
-- Effect Determination
-- ============================================================================

-- | Determine effects based on attractor and fragment
determineEffects :: Attractor -> Fragment -> EmergenceCondition
                 -> (AttractorEffect, AttractorEffect, Bool, Bool)
determineEffects Attractor{..} Fragment{..} cond =
  let
    -- GatingOverride: Archetypal + high enticement
    gatingOverride = attFlavor == Archetypal && attEnticement >= 3482

    -- InversionFlip: Shadow targeting + sufficient enticement
    inversionFlip = fragIsShadow && attEnticement >= 3277  -- 0.80

    -- ResonanceBoost: Always if active
    resonanceBoost = attEnticement >= enticementMin

    -- Assign effects by priority (max 2)
    effect1 = if gatingOverride then GatingOverride
              else if inversionFlip then InversionFlip
              else if resonanceBoost then ResonanceBoost
              else EffectNone

    effect2 = if gatingOverride && inversionFlip then InversionFlip
              else if gatingOverride && resonanceBoost then ResonanceBoost
              else if inversionFlip && resonanceBoost then ResonanceBoost
              else EffectNone
  in (effect1, effect2, gatingOverride, inversionFlip)

-- ============================================================================
-- Core Attractor Application
-- ============================================================================

-- | Apply single attractor to emergence condition
applyAttractor :: Attractor -> Fragment -> EmergenceCondition -> AttractorOutput
applyAttractor att@Attractor{..} frag cond =
  let
    -- Check if active and targets fragment
    targets = matchesTarget attTarget frag
    active = attActive && attEnticement >= enticementMin && targets

    -- Phi-scaled enticement
    phiEnt = phiScaledEnticement attEnticement

    -- Apply flavor effects
    flavorCond = if active then applyFlavor attFlavor phiEnt cond else cond

    -- Determine effects
    (eff1, eff2, gateOver, invFlip) = determineEffects att frag cond

    -- Calculate coherence delta
    cohDelta = phiEnt `shiftR` 3  -- ~15% reduction
    newThreshold = if active
                   then max coherenceFloor (ecThreshold flavorCond - cohDelta)
                   else ecThreshold flavorCond

    -- Apply shadow threshold lowering
    shadowThreshold = if fragIsShadow frag && active
                      then min newThreshold shadowThresholdLowered
                      else newThreshold

    -- Apply potential boost: potential *= (1 + phiEnt * 0.2)
    potBoost = phiEnt `shiftR` 3  -- ~12.5%
    newPotential = if active
                   then satAdd SatBound (ecPotential flavorCond) potBoost
                   else ecPotential flavorCond

    -- Build output condition
    outCond = flavorCond
      { ecThreshold = shadowThreshold
      , ecPotential = newPotential
      , ecGatingActive = not gateOver && ecGatingActive flavorCond
      , ecShadowAccess = invFlip || ecShadowAccess flavorCond
      , ecInversion = if invFlip
                      then 4095 - ecInversion flavorCond
                      else ecInversion flavorCond
      }

    -- Build result
    result = AttractorResult
      { arEffect1 = if active then eff1 else EffectNone
      , arEffect2 = if active then eff2 else EffectNone
      , arCoherenceDelta = if active then cohDelta else 0
      , arPotentialBoost = if active then potBoost else 0
      , arGatingOverride = gateOver && active
      , arInversionFlip = invFlip && active
      , arMatched = active
      }
  in AttractorOutput outCond result

-- ============================================================================
-- Multi-Attractor Resolution (8 concurrent)
-- ============================================================================

-- | Resolve up to 8 competing attractors
resolveAttractors :: Vec 8 Attractor -> Fragment -> EmergenceCondition -> AttractorOutput
resolveAttractors attractors frag initCond =
  let
    -- Apply each attractor sequentially
    applyOne cond att = aoCondition (applyAttractor att frag cond)
    finalCond = foldl applyOne initCond attractors

    -- Collect results (just track if any matched)
    results = map (\att -> applyAttractor att frag initCond) attractors
    anyMatched = or $ map (arMatched . aoResult) results

    -- Find dominant effects (from highest enticement attractor)
    sorted = map attEnticement attractors
    maxEnt = fold max sorted
    dominant = head $ filter (\att -> attEnticement att == maxEnt) (toList attractors)
    domResult = aoResult $ applyAttractor dominant frag initCond
  in AttractorOutput finalCond domResult

-- ============================================================================
-- Default Attractors (ROM)
-- ============================================================================

-- | Apple attractor (shinigami)
appleAttractor :: Attractor
appleAttractor = Attractor
  { attSymbol     = 1
  , attFlavor     = Archetypal
  , attEnticement = 3768  -- 0.92 * 4096
  , attTarget     = TargetShadow
  , attActive     = True
  }

-- | Tuning fork attractor
tuningForkAttractor :: Attractor
tuningForkAttractor = Attractor
  { attSymbol     = 2
  , attFlavor     = Sensory
  , attEnticement = 3604  -- 0.88 * 4096
  , attTarget     = TargetGuardian
  , attActive     = True
  }

-- | Mirror attractor
mirrorAttractor :: Attractor
mirrorAttractor = Attractor
  { attSymbol     = 3
  , attFlavor     = Emotional
  , attEnticement = 3482  -- 0.85 * 4096
  , attTarget     = TargetShadow
  , attActive     = True
  }

-- | Quartz attractor
quartzAttractor :: Attractor
quartzAttractor = Attractor
  { attSymbol     = 4
  , attFlavor     = Sensory
  , attEnticement = 3195  -- 0.78 * 4096
  , attTarget     = TargetShellGE 3
  , attActive     = True
  }

-- | Sun glyph attractor
sunGlyphAttractor :: Attractor
sunGlyphAttractor = Attractor
  { attSymbol     = 5
  , attFlavor     = Archetypal
  , attEnticement = 3891  -- 0.95 * 4096
  , attTarget     = TargetSymbolic
  , attActive     = True
  }

-- ============================================================================
-- Top-Level Entity
-- ============================================================================

-- | Single attractor FSM
attractorFSM :: HiddenClockResetEnable dom
             => Signal dom (Attractor, Fragment, EmergenceCondition)
             -> Signal dom AttractorOutput
attractorFSM = fmap (\(att, frag, cond) -> applyAttractor att frag cond)

-- | Top entity with port annotations
{-# ANN attractorTop
  (Synthesize
    { t_name   = "attractor_top"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "att_symbol"
                 , PortName "att_flavor"
                 , PortName "att_enticement"
                 , PortName "att_target_type"
                 , PortName "att_target_id"
                 , PortName "att_active"
                 , PortName "frag_id"
                 , PortName "frag_coherence"
                 , PortName "frag_shell"
                 , PortName "frag_is_shadow"
                 , PortName "frag_is_guardian"
                 , PortName "frag_is_symbolic"
                 , PortName "cond_potential"
                 , PortName "cond_threshold"
                 , PortName "cond_gating"
                 ]
    , t_output = PortProduct "output"
                 [ PortName "effect1"
                 , PortName "effect2"
                 , PortName "coh_delta"
                 , PortName "pot_boost"
                 , PortName "gating_override"
                 , PortName "inversion_flip"
                 , PortName "matched"
                 , PortName "new_threshold"
                 , PortName "new_potential"
                 ]
    }) #-}
attractorTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 8)    -- att_symbol
  -> Signal System (Unsigned 2)    -- att_flavor
  -> Signal System (Unsigned 12)   -- att_enticement
  -> Signal System (Unsigned 3)    -- att_target_type
  -> Signal System (Unsigned 12)   -- att_target_id
  -> Signal System Bool            -- att_active
  -> Signal System (Unsigned 12)   -- frag_id
  -> Signal System (Unsigned 12)   -- frag_coherence
  -> Signal System (Unsigned 3)    -- frag_shell
  -> Signal System Bool            -- frag_is_shadow
  -> Signal System Bool            -- frag_is_guardian
  -> Signal System Bool            -- frag_is_symbolic
  -> Signal System (Unsigned 12)   -- cond_potential
  -> Signal System (Unsigned 12)   -- cond_threshold
  -> Signal System Bool            -- cond_gating
  -> Signal System ( Unsigned 2    -- effect1
                   , Unsigned 2    -- effect2
                   , Unsigned 12   -- coh_delta
                   , Unsigned 12   -- pot_boost
                   , Bool          -- gating_override
                   , Bool          -- inversion_flip
                   , Bool          -- matched
                   , Unsigned 12   -- new_threshold
                   , Unsigned 12   -- new_potential
                   )
attractorTop clk rst en
             attSym attFlav attEnt attTgtType attTgtId attAct
             fragId fragCoh fragShell fragShadow fragGuard fragSymb
             condPot condThresh condGate =
  withClockResetEnable clk rst en $
    let
      -- Decode flavor
      decFlavor f = case f of
        0 -> Emotional
        1 -> Sensory
        2 -> Archetypal
        _ -> UnknownPhase

      -- Decode target
      decTarget t tid = case t of
        0 -> TargetById tid
        1 -> TargetShadow
        2 -> TargetGuardian
        3 -> TargetShellGE (resize tid)
        _ -> TargetSymbolic

      -- Build attractor
      attractor = Attractor
        <$> attSym
        <*> fmap decFlavor attFlav
        <*> attEnt
        <*> (decTarget <$> attTgtType <*> attTgtId)
        <*> attAct

      -- Build fragment
      fragment = Fragment
        <$> fragId <*> fragCoh <*> pure 2048 <*> pure 0 <*> pure 0
        <*> fragShell <*> fragShadow <*> fragGuard <*> fragSymb

      -- Build condition
      condition = EmergenceCondition
        <$> condPot <*> pure 0 <*> pure 0 <*> pure 0
        <*> condThresh <*> condGate <*> pure False

      -- Apply attractor
      output = attractorFSM $ (,,) <$> attractor <*> fragment <*> condition

      -- Encode effect
      encEffect eff = case eff of
        EffectNone     -> 0
        GatingOverride -> 1
        InversionFlip  -> 2
        ResonanceBoost -> 3

      -- Extract output
      extractOut (AttractorOutput cond result) =
        ( encEffect (arEffect1 result)
        , encEffect (arEffect2 result)
        , arCoherenceDelta result
        , arPotentialBoost result
        , arGatingOverride result
        , arInversionFlip result
        , arMatched result
        , ecThreshold cond
        , ecPotential cond
        )
    in fmap extractOut output

-- ============================================================================
-- Testbench
-- ============================================================================

-- | Test fragment (shadow)
testFragment :: Fragment
testFragment = Fragment
  { fragId        = 144
  , fragCoherence = 2048
  , fragPotential = 2048
  , fragFlux      = 0
  , fragPhase     = 0
  , fragShell     = 4
  , fragIsShadow  = True
  , fragIsGuardian = False
  , fragIsSymbolic = False
  }

-- | Test condition
testCondition :: EmergenceCondition
testCondition = EmergenceCondition
  { ecPotential    = 2048
  , ecFlux         = 0
  , ecPhase        = 0
  , ecInversion    = 820
  , ecThreshold    = 2950
  , ecGatingActive = True
  , ecShadowAccess = False
  }

-- | Test inputs
testInputs :: Vec 3 (Attractor, Fragment, EmergenceCondition)
testInputs =
     (appleAttractor, testFragment, testCondition)
  :> (mirrorAttractor, testFragment, testCondition)
  :> (sunGlyphAttractor, testFragment { fragIsSymbolic = True }, testCondition)
  :> Nil

-- | Testbench entity
testBench :: Signal System AttractorOutput
testBench = withClockResetEnable systemClockGen systemResetGen enableGen $
  attractorFSM (fromList (toList testInputs))
