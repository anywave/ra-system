{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

-- | Prompt 21: Ra.Biotransduction - Scalar-to-Somatic Transduction Engine
-- FPGA module for real-time mapping of scalar harmonics and coherence
-- to biophysical effects like warmth, pulses, and breath entrainment.
--
-- Codex References:
-- - Ra.Emergence: Field emergence modulation
-- - Ra.Coherence: Coherence-driven effect intensity
-- - P20: Chamber tuning integration
-- - P19: Domain safety
--
-- Features:
-- - Radial depth → body locus mapping
-- - Harmonic (l,m) → effect type mapping
-- - Phase gain with 0.1 minimum clamping
-- - Smooth coherence interpolation
-- - Unified ChamberFlavor (7 variants)

module RaBiotransduction where

import Clash.Prelude
import GHC.Generics (Generic)

-- ============================================================================
-- Constants (Codex-aligned)
-- ============================================================================

-- | Minimum phase gain (clamped)
minPhaseGain :: SFixed 4 12
minPhaseGain = 0.1

-- | Coherence thresholds (12-bit: 4096 = 1.0)
coherenceNullThreshold :: Unsigned 12
coherenceNullThreshold = 819    -- 0.2 * 4096

coherenceWeakThreshold :: Unsigned 12
coherenceWeakThreshold = 1638   -- 0.4 * 4096

coherenceFullThreshold :: Unsigned 12
coherenceFullThreshold = 3277   -- 0.8 * 4096

-- | Pi approximation for sin calculation
piConst :: SFixed 4 12
piConst = 3.14159

-- ============================================================================
-- Types - Body & Axis
-- ============================================================================

-- | Body regions for somatic targeting
data BodyRegion
  = RegionChest
  | RegionCrown
  | RegionSpine
  | RegionLeftHand
  | RegionRightHand
  | RegionFeet
  | RegionAbdomen
  deriving (Generic, NFDataX, Eq, Show)

-- | 3D axes for proprioceptive effects
data Axis3D
  = AxisX
  | AxisY
  | AxisZ
  deriving (Generic, NFDataX, Eq, Show)

-- | Unified chamber flavor (P20 + P21)
data ChamberFlavor
  = FlavorHealing
  | FlavorRetrieval
  | FlavorNavigation
  | FlavorDream
  | FlavorArchive
  | FlavorTherapeutic
  | FlavorExploratory
  deriving (Generic, NFDataX, Eq, Show)

-- ============================================================================
-- Types - Biophysical Effects
-- ============================================================================

-- | Effect type enumeration
data EffectType
  = EffThermalShift
  | EffTactilePulse
  | EffProprioceptiveWarp
  | EffAffectiveOverlay
  | EffBreathEntrainment
  | EffInnerResonance
  | EffNull
  deriving (Generic, NFDataX, Eq, Show)

-- | Biophysical effect output
data BiophysicalEffect = BiophysicalEffect
  { beType       :: EffectType
  , beRegion     :: BodyRegion
  , beAxis       :: Axis3D
  , beMagnitude  :: SFixed 4 12      -- Effect intensity/delta
  , beValence    :: SFixed 4 12      -- For affective (-1 to +1)
  , beArousal    :: Unsigned 12      -- For affective (0-4095)
  , beFrequency  :: SFixed 8 8       -- For breath (Hz)
  , beHarmonicL  :: Unsigned 4       -- For resonance
  , beHarmonicM  :: Signed 4         -- For resonance
  } deriving (Generic, NFDataX, Show)

-- | Null effect constant
nullEffect :: BiophysicalEffect
nullEffect = BiophysicalEffect
  { beType = EffNull
  , beRegion = RegionChest
  , beAxis = AxisY
  , beMagnitude = 0
  , beValence = 0
  , beArousal = 0
  , beFrequency = 0
  , beHarmonicL = 0
  , beHarmonicM = 0
  }

-- ============================================================================
-- Types - Field & Context
-- ============================================================================

-- | Window phase for modulation
data WindowPhase = WindowPhase
  { wpPhiDepth :: Unsigned 4
  , wpPhase    :: Unsigned 12        -- 0-4095 (0.0-1.0)
  } deriving (Generic, NFDataX, Show)

-- | Scalar field parameters
data ScalarField = ScalarField
  { sfRadius      :: Unsigned 12     -- 0-4095 (0.0-1.0)
  , sfWindowPhase :: WindowPhase
  } deriving (Generic, NFDataX, Show)

-- | Ra coordinate with harmonics
data RaCoordinate = RaCoordinate
  { rcHarmonicL :: Unsigned 4        -- l index
  , rcHarmonicM :: Signed 4          -- m index
  , rcAngle     :: Unsigned 12       -- angle (0-4095)
  } deriving (Generic, NFDataX, Show)

-- | Emergence condition
data EmergenceCondition = EmergenceCondition
  { ecFlux            :: Unsigned 12
  , ecShadowThreshold :: Unsigned 12
  } deriving (Generic, NFDataX, Show)

-- | Transduction input bundle
data TransductionInput = TransductionInput
  { tiField      :: ScalarField
  , tiCoord      :: RaCoordinate
  , tiEmergence  :: EmergenceCondition
  , tiCoherence  :: Unsigned 12
  , tiChamber    :: ChamberFlavor
  } deriving (Generic, NFDataX)

-- | Transduction output (up to 4 effects)
data TransductionOutput = TransductionOutput
  { toEffect0    :: BiophysicalEffect
  , toEffect1    :: BiophysicalEffect
  , toEffect2    :: BiophysicalEffect
  , toEffect3    :: BiophysicalEffect
  , toEffectCount :: Unsigned 3
  } deriving (Generic, NFDataX, Show)

-- ============================================================================
-- Phase Gain Calculation
-- ============================================================================

-- | Approximate sin using Taylor series (good for 0-1 range)
approxSinPi :: Unsigned 12 -> SFixed 4 12
approxSinPi phase =
  let
    -- Normalize to 0-1 range
    x = fromIntegral phase / 4096.0 :: SFixed 4 12
    -- sin(pi*x) ≈ pi*x - (pi*x)^3/6 for small x
    -- For x around 0.5, sin(pi*x) ≈ 1
    -- Use quadratic approximation: 4*x*(1-x) peaks at x=0.5
    sinApprox = 4.0 * x * (1.0 - x)
  in sinApprox

-- | Calculate phase gain with minimum clamping
calculatePhaseGain :: Unsigned 12 -> SFixed 4 12
calculatePhaseGain phase =
  let rawGain = approxSinPi phase
  in max minPhaseGain rawGain

-- ============================================================================
-- Coherence Envelope
-- ============================================================================

-- | Coherence envelope level
data CoherenceLevel
  = CohNull
  | CohWeak
  | CohModerate
  | CohFull
  deriving (Generic, NFDataX, Eq, Show)

-- | Get coherence level and intensity
coherenceEnvelope :: Unsigned 12 -> (CoherenceLevel, Unsigned 12)
coherenceEnvelope coh
  | coh < coherenceNullThreshold = (CohNull, 0)
  | coh < coherenceWeakThreshold =
      -- Interpolate 0.2-0.4 range
      let range = coherenceWeakThreshold - coherenceNullThreshold
          offset = coh - coherenceNullThreshold
          intensity = (offset * 4096) `div` range
      in (CohWeak, truncateB intensity)
  | coh < coherenceFullThreshold =
      -- Interpolate 0.4-0.8 range, start at 0.5
      let range = coherenceFullThreshold - coherenceWeakThreshold
          offset = coh - coherenceWeakThreshold
          baseIntensity = 2048  -- 0.5
          addIntensity = (offset * 2048) `div` range
      in (CohModerate, truncateB (baseIntensity + addIntensity))
  | otherwise = (CohFull, 4095)

-- ============================================================================
-- Body Region Mapping
-- ============================================================================

-- | Map radial depth to body region
radiusToRegion :: Unsigned 12 -> BodyRegion
radiusToRegion radius
  | radius < 614  = RegionCrown    -- < 0.15
  | radius < 1843 = RegionChest    -- < 0.45
  | radius < 3072 = RegionAbdomen  -- < 0.75
  | otherwise     = RegionFeet

-- ============================================================================
-- Core Transduction
-- ============================================================================

-- | Build thermal shift effect
buildThermalEffect :: SFixed 4 12 -> Unsigned 12 -> BiophysicalEffect
buildThermalEffect phaseGain intensity =
  let
    -- delta = 2.0 * phaseGain * (intensity/4096)
    intensityF = fromIntegral intensity / 4096.0 :: SFixed 4 12
    delta = 2.0 * phaseGain * intensityF
  in nullEffect { beType = EffThermalShift, beMagnitude = delta }

-- | Build breath entrainment effect
buildBreathEffect :: Unsigned 12 -> BiophysicalEffect
buildBreathEffect intensity =
  let
    -- freq = 0.1 + 0.3 * intensity
    intensityF = fromIntegral intensity / 4096.0 :: SFixed 8 8
    freq = 0.1 + 0.3 * intensityF
  in nullEffect { beType = EffBreathEntrainment, beFrequency = freq }

-- | Build tactile pulse effect
buildTactileEffect :: BodyRegion -> SFixed 4 12 -> Unsigned 12 -> BiophysicalEffect
buildTactileEffect region phaseGain intensity =
  let
    intensityF = fromIntegral intensity / 4096.0 :: SFixed 4 12
    magnitude = 0.5 * intensityF * phaseGain
  in nullEffect { beType = EffTactilePulse, beRegion = region, beMagnitude = magnitude }

-- | Build inner resonance effect
buildResonanceEffect :: Unsigned 4 -> Signed 4 -> BiophysicalEffect
buildResonanceEffect l m =
  nullEffect { beType = EffInnerResonance, beHarmonicL = l, beHarmonicM = m }

-- | Build proprioceptive warp effect
buildProprioEffect :: Axis3D -> SFixed 4 12 -> BiophysicalEffect
buildProprioEffect axis magnitude =
  nullEffect { beType = EffProprioceptiveWarp, beAxis = axis, beMagnitude = magnitude }

-- | Build affective overlay effect
buildAffectiveEffect :: SFixed 4 12 -> Unsigned 12 -> BiophysicalEffect
buildAffectiveEffect valence arousal =
  nullEffect { beType = EffAffectiveOverlay, beValence = valence, beArousal = arousal }

-- | Get chamber-specific effects
getChamberEffects :: ChamberFlavor -> Unsigned 12 -> BiophysicalEffect
getChamberEffects chamber coherence =
  case chamber of
    FlavorHealing     -> buildAffectiveEffect 0.8 coherence
    FlavorTherapeutic -> buildAffectiveEffect 0.8 coherence
    FlavorRetrieval   -> buildThermalEffect (-1.0) coherence  -- negative = chills
    FlavorDream       -> buildProprioEffect AxisZ 0.4
    FlavorArchive     -> buildResonanceEffect 3 0
    FlavorNavigation  -> buildTactileEffect RegionSpine 1.0 (coherence `shiftR` 2)
    FlavorExploratory -> buildTactileEffect RegionSpine 1.0 (coherence `shiftR` 2)

-- | Main transduction function
transduceField :: TransductionInput -> TransductionOutput
transduceField TransductionInput{..} =
  let
    -- Get coherence envelope
    (cohLevel, intensity) = coherenceEnvelope tiCoherence

    -- Check for null coherence
    isNull = cohLevel == CohNull

    -- Calculate phase gain
    phaseGain = calculatePhaseGain (wpPhase (sfWindowPhase tiField))

    -- Get body region from radius
    region = radiusToRegion (sfRadius tiField)

    -- Get harmonic indices
    l = rcHarmonicL tiCoord
    m = rcHarmonicM tiCoord

    -- Build base effects from harmonic
    baseEffect0 = case l of
      0 -> buildThermalEffect phaseGain intensity
      1 -> buildBreathEffect intensity
      2 -> buildTactileEffect RegionLeftHand phaseGain intensity
      _ -> buildResonanceEffect l m

    baseEffect1 = case l of
      2 -> buildTactileEffect RegionRightHand phaseGain intensity
      _ -> buildTactileEffect region 1.0 (intensity `shiftR` 1)

    -- Chamber effect
    chamberEffect = getChamberEffects tiChamber tiCoherence

    -- Count effects
    effectCount = if isNull then 1
                  else if l == 2 then 3  -- bilateral + chamber
                  else 2                  -- base + chamber
  in
    if isNull
    then TransductionOutput nullEffect nullEffect nullEffect nullEffect 1
    else TransductionOutput
           { toEffect0 = baseEffect0
           , toEffect1 = if l == 2 then baseEffect1 else chamberEffect
           , toEffect2 = if l == 2 then chamberEffect else nullEffect
           , toEffect3 = nullEffect
           , toEffectCount = effectCount
           }

-- ============================================================================
-- Top-Level Entity
-- ============================================================================

-- | Biotransduction FSM
biotransductionFSM :: HiddenClockResetEnable dom
                   => Signal dom TransductionInput
                   -> Signal dom TransductionOutput
biotransductionFSM = fmap transduceField

-- | Top entity for biotransduction
{-# ANN biotransductionTop
  (Synthesize
    { t_name   = "biotransduction_top"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "radius"
                 , PortName "phase_depth"
                 , PortName "phase"
                 , PortName "harmonic_l"
                 , PortName "harmonic_m"
                 , PortName "angle"
                 , PortName "flux"
                 , PortName "shadow_thresh"
                 , PortName "coherence"
                 , PortName "chamber"
                 ]
    , t_output = PortProduct "output"
                 [ PortName "effect0_type"
                 , PortName "effect0_magnitude"
                 , PortName "effect0_region"
                 , PortName "effect1_type"
                 , PortName "effect1_magnitude"
                 , PortName "effect_count"
                 ]
    }) #-}
biotransductionTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 12)    -- radius
  -> Signal System (Unsigned 4)     -- phase_depth
  -> Signal System (Unsigned 12)    -- phase
  -> Signal System (Unsigned 4)     -- harmonic_l
  -> Signal System (Signed 4)       -- harmonic_m
  -> Signal System (Unsigned 12)    -- angle
  -> Signal System (Unsigned 12)    -- flux
  -> Signal System (Unsigned 12)    -- shadow_thresh
  -> Signal System (Unsigned 12)    -- coherence
  -> Signal System (Unsigned 3)     -- chamber (encoded)
  -> Signal System ( Unsigned 3     -- effect0_type
                   , SFixed 4 12    -- effect0_magnitude
                   , Unsigned 3     -- effect0_region
                   , Unsigned 3     -- effect1_type
                   , SFixed 4 12    -- effect1_magnitude
                   , Unsigned 3     -- effect_count
                   )
biotransductionTop clk rst en
                   radius phaseDepth phase
                   harmonicL harmonicM angle
                   flux shadowThresh
                   coherence chamber =
  withClockResetEnable clk rst en $
    let
      -- Decode chamber
      decodeChamber c = case c of
        0 -> FlavorHealing
        1 -> FlavorRetrieval
        2 -> FlavorNavigation
        3 -> FlavorDream
        4 -> FlavorArchive
        5 -> FlavorTherapeutic
        _ -> FlavorExploratory

      -- Build window phase
      windowPhase = WindowPhase <$> phaseDepth <*> phase

      -- Build scalar field
      field = ScalarField <$> radius <*> windowPhase

      -- Build coordinate
      coord = RaCoordinate <$> harmonicL <*> harmonicM <*> angle

      -- Build emergence
      emergence = EmergenceCondition <$> flux <*> shadowThresh

      -- Build input
      input = TransductionInput
        <$> field
        <*> coord
        <*> emergence
        <*> coherence
        <*> fmap decodeChamber chamber

      -- Transduce
      output = biotransductionFSM input

      -- Encode effect type
      encodeType t = case t of
        EffThermalShift       -> 0
        EffTactilePulse       -> 1
        EffProprioceptiveWarp -> 2
        EffAffectiveOverlay   -> 3
        EffBreathEntrainment  -> 4
        EffInnerResonance     -> 5
        EffNull               -> 6

      -- Encode body region
      encodeRegion r = case r of
        RegionChest     -> 0
        RegionCrown     -> 1
        RegionSpine     -> 2
        RegionLeftHand  -> 3
        RegionRightHand -> 4
        RegionFeet      -> 5
        RegionAbdomen   -> 6

      -- Extract output
      extractOut TransductionOutput{..} =
        ( encodeType (beType toEffect0)
        , beMagnitude toEffect0
        , encodeRegion (beRegion toEffect0)
        , encodeType (beType toEffect1)
        , beMagnitude toEffect1
        , toEffectCount
        )
    in fmap extractOut output

-- ============================================================================
-- Testbench
-- ============================================================================

-- | Test input: high coherence, H_{0,0}, healing
testInputThermal :: TransductionInput
testInputThermal = TransductionInput
  { tiField = ScalarField 2048 (WindowPhase 1 2048)  -- radius=0.5, phase=0.5
  , tiCoord = RaCoordinate 0 0 0                      -- H_{0,0}
  , tiEmergence = EmergenceCondition 2048 2949
  , tiCoherence = 3686                                -- 0.9
  , tiChamber = FlavorHealing
  }

-- | Test input: mid coherence, H_{2,1}, navigation
testInputBilateral :: TransductionInput
testInputBilateral = TransductionInput
  { tiField = ScalarField 2048 (WindowPhase 1 2048)
  , tiCoord = RaCoordinate 2 1 0                      -- H_{2,1}
  , tiEmergence = EmergenceCondition 2048 2949
  , tiCoherence = 2867                                -- 0.7
  , tiChamber = FlavorNavigation
  }

-- | Test input: low coherence (null)
testInputNull :: TransductionInput
testInputNull = TransductionInput
  { tiField = ScalarField 2048 (WindowPhase 1 2048)
  , tiCoord = RaCoordinate 1 0 0
  , tiEmergence = EmergenceCondition 2048 2949
  , tiCoherence = 410                                 -- 0.1
  , tiChamber = FlavorHealing
  }

-- | Testbench inputs
testInputs :: Vec 3 TransductionInput
testInputs = testInputThermal :> testInputBilateral :> testInputNull :> Nil

-- | Testbench entity
testBench :: Signal System TransductionOutput
testBench = withClockResetEnable systemClockGen systemResetGen enableGen $
  biotransductionFSM (fromList (toList testInputs))
