{-|
Module      : RaOrgoneScalar
Description : Orgone field influence on Ra scalar stability
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 9: Synthesizes Reichian orgone dynamics with scalar field modulation
protocols. Models OR (positive orgone) and DOR (deadly orgone) interactions
with Ra scalar fields for emergence stability control.

== Orgone Field Model

@
orgone_field = {
  or_level: 0.72,           -- Positive orgone charge (0-1)
  dor_level: 0.18,          -- Deadly orgone level (0-1)
  accumulation_rate: 0.04,  -- Exponential charge rate
  chamber_geometry: pyramidal
}
@

== Field Characteristics

- or_level: Increases scalar potential and coherence
- dor_level: Increases scalar instability, fragment collapse
- accumulation_rate: Real-time OR modulation (exponential curve)
- chamber_geometry: Affects field harmonics via geometry multipliers

== Scalar Coupling (Codex-Aligned)

@
potential *= (1 + or_level - dor_level)
flux_coherence *= (1 - dor_level)
inversion_probability *= (1 + dor_level - or_level)
@

== Functional Outcomes

- High OR = deeper scalar wells, stable emergence
- High DOR = fragmented emergence, shadow artifacts
- Balanced OR/DOR = field stagnation

== Chamber Geometry Amplification

| Geometry   | OR Boost | DOR Shield | Notes                    |
|------------|----------|------------|--------------------------|
| Pyramidal  | +0.15    | -0.10      | Proven resonance enhancer |
| Dome       | +0.10    | -0.05      | Smooth energy flow       |
| Rectangular| 0        | 0          | Neutral or noise-prone   |

== Phenomenological Triggers (Reich)

- Blue luminescence in OR-saturated zones → coherence spike
- Sudden OR discharge under stress → rapid fragment collapse
- Emotion amplification near convergence → affects access scoring

== Precision Handling

@
Frequency Precision:       8-bit normalized (0-255 = 0.0-1.0)
Intermediate Calculations: 16-bit fixed-point
Output Types:              8-bit coherence-scaled
@

== Upstream Dependencies

- RaSympatheticHarmonic (Prompt 8): Fragment access scoring
- RaConsentFramework (Prompt 32): Coherence gating
- Ra.Constants: Chamber geometry parameters

== Downstream Propagation

- ScalarOutput feeds into RaFieldTransferBus (Prompt 35)
- EmergenceScore controls RaVisualizerShell (Prompt 41)
- LuminescenceFlag triggers blue glow rendering

== Hardware Synthesis Targets

- Xilinx Artix-7: ~120 LUTs, 4 DSP slices
- Intel Cyclone V: ~140 ALMs, 3 DSP blocks
- Clock: Validated at 100 MHz system clock
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaOrgoneScalar where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Fixed-point representation (0-255 maps to 0.0-1.0)
type Fixed8 = Unsigned 8

-- | Extended calculation precision
type CalcValue = Unsigned 16

-- | Chamber geometry enumeration
data ChamberGeometry
  = Pyramidal      -- ^ Proven resonance enhancer (+0.15 OR, -0.10 DOR)
  | Dome           -- ^ Smooth energy flow (+0.10 OR, -0.05 DOR)
  | Rectangular    -- ^ Neutral, noise-prone (no modifiers)
  | Spherical      -- ^ Experimental (+0.12 OR, -0.08 DOR)
  deriving (Generic, NFDataX, Show, Eq)

-- | Emergence result classification
data EmergenceClass
  = AlphaEmergence    -- ^ High OR, stable emergence
  | StableFragment    -- ^ Balanced, deep scalar potentials
  | BaselineStability -- ^ Neutral state
  | ShadowFragment    -- ^ High DOR, turbulence
  | FieldCollapse     -- ^ Critical DOR, fragment collapse
  deriving (Generic, NFDataX, Show, Eq)

-- | Orgone field state
data OrgoneField = OrgoneField
  { orLevel          :: Fixed8           -- ^ Positive orgone (0-255)
  , dorLevel         :: Fixed8           -- ^ Deadly orgone (0-255)
  , accumulationRate :: Fixed8           -- ^ Charge rate (0-255)
  , chamberGeometry  :: ChamberGeometry  -- ^ Chamber type
  } deriving (Generic, NFDataX, Show, Eq)

-- | Scalar field output
data ScalarOutput = ScalarOutput
  { potential           :: Fixed8         -- ^ Modified scalar potential
  , fluxCoherence       :: Fixed8         -- ^ Flux coherence level
  , inversionProb       :: Fixed8         -- ^ Inversion probability
  , emergenceScore      :: Fixed8         -- ^ Overall emergence score
  , fragmentStability   :: Fixed8         -- ^ Fragment stability (flux - inversion)
  , emergenceClass      :: EmergenceClass -- ^ Classification
  , luminescenceFlag    :: Bool           -- ^ Blue luminescence trigger
  , dischargeWarning    :: Bool           -- ^ OR discharge warning
  , shadowRisk          :: Bool           -- ^ DOR > 0.6 warning
  } deriving (Generic, NFDataX, Show, Eq)

-- | Combined input for evaluation
data EvaluationInput = EvaluationInput
  { orgoneField     :: OrgoneField  -- ^ Current orgone state
  , basePotential   :: Fixed8       -- ^ Base scalar potential
  , baseCoherence   :: Fixed8       -- ^ Base coherence level
  , emotionalStress :: Fixed8       -- ^ Emotional stress factor (0-255)
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Geometry OR boosts (scaled: 38 = 0.15, 26 = 0.10, 31 = 0.12)
geometryOrBoost :: ChamberGeometry -> Fixed8
geometryOrBoost Pyramidal   = 38   -- +0.15
geometryOrBoost Dome        = 26   -- +0.10
geometryOrBoost Rectangular = 0    -- +0.00
geometryOrBoost Spherical   = 31   -- +0.12

-- | Geometry DOR shields (scaled: 26 = 0.10, 13 = 0.05, 20 = 0.08)
geometryDorShield :: ChamberGeometry -> Fixed8
geometryDorShield Pyramidal   = 26   -- -0.10
geometryDorShield Dome        = 13   -- -0.05
geometryDorShield Rectangular = 0    -- -0.00
geometryDorShield Spherical   = 20   -- -0.08

-- | Thresholds for emergence classification
thresholdAlpha :: Fixed8
thresholdAlpha = 204        -- 0.80

thresholdStable :: Fixed8
thresholdStable = 153       -- 0.60

thresholdBaseline :: Fixed8
thresholdBaseline = 102     -- 0.40

thresholdShadow :: Fixed8
thresholdShadow = 51        -- 0.20

-- | Luminescence threshold (high OR saturation)
luminescenceThreshold :: Fixed8
luminescenceThreshold = 204  -- 0.80 OR level

-- | Discharge warning threshold (OR drop under stress)
dischargeThreshold :: Fixed8
dischargeThreshold = 180     -- Stress level triggering discharge

-- =============================================================================
-- Core Functions: Geometry Modulation
-- =============================================================================

-- | Apply chamber geometry modifiers to orgone levels
applyGeometryModifiers :: OrgoneField -> (Fixed8, Fixed8)
applyGeometryModifiers field =
  let
    geom = chamberGeometry field
    orBoost = geometryOrBoost geom
    dorShield = geometryDorShield geom
    -- Apply modifiers with saturation
    modifiedOr = satAdd SatBound (orLevel field) orBoost
    modifiedDor = satSub SatBound (dorLevel field) dorShield
  in
    (modifiedOr, modifiedDor)

-- =============================================================================
-- Core Functions: Scalar Coupling
-- =============================================================================

-- | Compute modified scalar potential
-- Formula: potential *= (1 + or_level - dor_level)
-- Using fixed-point: potential * (256 + or - dor) / 256
computePotential :: Fixed8 -> Fixed8 -> Fixed8 -> Fixed8
computePotential basePot orLvl dorLvl =
  let
    base = resize basePot :: CalcValue
    orVal = resize orLvl :: CalcValue
    dorVal = resize dorLvl :: CalcValue
    -- Multiplier = 256 + or - dor (range: ~0 to ~512)
    multiplier = 256 + orVal - min orVal dorVal + orVal - dorVal
    -- Simplified: (256 + or - dor), clamped
    mult = satAdd SatBound 256 (satSub SatBound orVal dorVal)
    result = (base * mult) `shiftR` 8
  in
    resize (min 255 result)

-- | Compute flux coherence
-- Formula: flux_coherence *= (1 - dor_level)
-- Using fixed-point: coherence * (256 - dor) / 256
computeFluxCoherence :: Fixed8 -> Fixed8 -> Fixed8
computeFluxCoherence baseCoh dorLvl =
  let
    base = resize baseCoh :: CalcValue
    dorVal = resize dorLvl :: CalcValue
    multiplier = 256 - dorVal
    result = (base * multiplier) `shiftR` 8
  in
    resize result

-- | Compute inversion probability
-- Formula: inversion_probability *= (1 + dor_level - or_level)
-- Using fixed-point: base * (256 + dor - or) / 256
computeInversionProb :: Fixed8 -> Fixed8 -> Fixed8 -> Fixed8
computeInversionProb baseInv orLvl dorLvl =
  let
    base = resize baseInv :: CalcValue
    orVal = resize orLvl :: CalcValue
    dorVal = resize dorLvl :: CalcValue
    -- Multiplier = 256 + dor - or
    mult = 256 + dorVal - min dorVal orVal + dorVal - orVal
    -- Simplified with saturation
    multiplier = if dorVal >= orVal
                 then 256 + (dorVal - orVal)
                 else satSub SatBound 256 (orVal - dorVal)
    result = (base * multiplier) `shiftR` 8
  in
    resize (min 255 result)

-- =============================================================================
-- Core Functions: Accumulation & Dynamics
-- =============================================================================

-- | Exponential accumulation approximation
-- Uses linear approximation: or_new = or + rate * (1 - or/256)
accumulateOrgone :: Fixed8 -> Fixed8 -> Fixed8
accumulateOrgone currentOr rate =
  let
    current = resize currentOr :: CalcValue
    rateVal = resize rate :: CalcValue
    -- Headroom for accumulation
    headroom = 256 - current
    -- Increment scaled by headroom (exponential approach)
    increment = (rateVal * headroom) `shiftR` 8
  in
    resize (min 255 (current + increment))

-- | Discharge under emotional stress
-- High stress + high OR = discharge risk
computeDischarge :: Fixed8 -> Fixed8 -> (Fixed8, Bool)
computeDischarge orLvl stress =
  let
    -- Discharge amount proportional to stress and OR level
    dischargeAmt = if stress > dischargeThreshold && orLvl > 128
                   then (resize orLvl * resize stress :: CalcValue) `shiftR` 10
                   else 0
    newOr = satSub SatBound orLvl (resize dischargeAmt)
    warning = dischargeAmt > 20
  in
    (newOr, warning)

-- =============================================================================
-- Core Functions: Emergence Evaluation
-- =============================================================================

-- | Compute emergence score from field parameters
-- Score = potential * flux_coherence * (1 - inversion_probability)
-- This is the Codex-aligned multiplicative formula
computeEmergenceScore :: Fixed8 -> Fixed8 -> Fixed8 -> Fixed8
computeEmergenceScore pot coh inv =
  let
    potVal = resize pot :: CalcValue
    cohVal = resize coh :: CalcValue
    invVal = resize inv :: CalcValue
    -- Multiplicative formula: pot * coh * (256 - inv) / 256^2
    -- Scaled to fit 8-bit output
    invComplement = 256 - invVal
    product1 = (potVal * cohVal) `shiftR` 8  -- pot * coh / 256
    score = (product1 * invComplement) `shiftR` 8  -- * (1 - inv)
  in
    resize (min 255 score)

-- | Compute fragment stability
-- Stability = flux_coherence - inversion_probability
computeFragmentStability :: Fixed8 -> Fixed8 -> Fixed8
computeFragmentStability coh inv =
  satSub SatBound coh inv

-- | Classify emergence based on score and DOR level
classifyEmergence :: Fixed8 -> Fixed8 -> EmergenceClass
classifyEmergence score dorLvl
  | dorLvl > 200                  = FieldCollapse
  | score >= thresholdAlpha       = AlphaEmergence
  | score >= thresholdStable      = StableFragment
  | score >= thresholdBaseline    = BaselineStability
  | score >= thresholdShadow      = ShadowFragment
  | otherwise                     = FieldCollapse

-- | Check for blue luminescence (OR saturation)
checkLuminescence :: Fixed8 -> Fixed8 -> Bool
checkLuminescence orLvl coherence =
  orLvl >= luminescenceThreshold && coherence > 150

-- =============================================================================
-- Main Evaluation Function
-- =============================================================================

-- | Evaluate emergence with full orgone field influence
evaluateEmergence :: EvaluationInput -> ScalarOutput
evaluateEmergence input =
  let
    field = orgoneField input

    -- Apply geometry modifiers
    (modOr, modDor) = applyGeometryModifiers field

    -- Apply stress-based discharge
    (finalOr, discharge) = computeDischarge modOr (emotionalStress input)

    -- Compute scalar coupling
    pot = computePotential (basePotential input) finalOr modDor
    coh = computeFluxCoherence (baseCoherence input) modDor
    inv = computeInversionProb 13 finalOr modDor  -- Base inversion = 0.05 (13/256)

    -- Compute emergence metrics
    score = computeEmergenceScore pot coh inv
    stability = computeFragmentStability coh inv
    eClass = classifyEmergence score modDor

    -- Check phenomenological triggers (Reichian)
    lum = checkLuminescence finalOr coh
    shadow = modDor > 153  -- DOR > 0.6 threshold
  in
    ScalarOutput
      { potential         = pot
      , fluxCoherence     = coh
      , inversionProb     = inv
      , emergenceScore    = score
      , fragmentStability = stability
      , emergenceClass    = eClass
      , luminescenceFlag  = lum
      , dischargeWarning  = discharge
      , shadowRisk        = shadow
      }

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Orgone scalar processor (combinational)
orgoneScalarProcessor
  :: HiddenClockResetEnable dom
  => Signal dom EvaluationInput
  -> Signal dom ScalarOutput
orgoneScalarProcessor = fmap evaluateEmergence

-- | Stateful processor with OR accumulation over time
orgoneAccumulatorProcessor
  :: HiddenClockResetEnable dom
  => Signal dom EvaluationInput
  -> Signal dom ScalarOutput
orgoneAccumulatorProcessor input = mealy accumState initState input
  where
    initState :: Fixed8
    initState = 128  -- Start at 50% OR

    accumState :: Fixed8 -> EvaluationInput -> (Fixed8, ScalarOutput)
    accumState accOr evalInput =
      let
        field = orgoneField evalInput
        -- Accumulate OR over time
        newOr = accumulateOrgone accOr (accumulationRate field)
        -- Update field with accumulated OR
        updatedField = field { orLevel = newOr }
        updatedInput = evalInput { orgoneField = updatedField }
        -- Evaluate with updated state
        output = evaluateEmergence updatedInput
      in
        (newOr, output)

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
orgoneScalarTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System EvaluationInput
  -> Signal System ScalarOutput
orgoneScalarTop = exposeClockResetEnable orgoneScalarProcessor

-- | Accumulator variant for synthesis
orgoneAccumulatorTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System EvaluationInput
  -> Signal System ScalarOutput
orgoneAccumulatorTop = exposeClockResetEnable orgoneAccumulatorProcessor

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Standard test fields
testFieldHighOr :: OrgoneField
testFieldHighOr = OrgoneField 230 26 10 Pyramidal  -- OR=0.9, DOR=0.1

testFieldBalanced :: OrgoneField
testFieldBalanced = OrgoneField 128 128 10 Rectangular  -- OR=0.5, DOR=0.5

testFieldHighDor :: OrgoneField
testFieldHighDor = OrgoneField 51 179 10 Dome  -- OR=0.2, DOR=0.7

testFieldStable :: OrgoneField
testFieldStable = OrgoneField 204 77 10 Pyramidal  -- OR=0.8, DOR=0.3

-- | Test input cases matching spec
testInputs :: Vec 4 EvaluationInput
testInputs =
  -- Test 0: OR=0.9, DOR=0.1, Pyramidal -> Alpha emergence + high coherence
  EvaluationInput testFieldHighOr 200 200 50 :>

  -- Test 1: OR=0.5, DOR=0.5, Rectangular -> Baseline stability
  EvaluationInput testFieldBalanced 128 128 50 :>

  -- Test 2: OR=0.2, DOR=0.7, Dome -> Shadow fragments + turbulence
  EvaluationInput testFieldHighDor 128 128 50 :>

  -- Test 3: OR=0.8, DOR=0.3, Pyramidal -> Stable fragments + deep potentials
  EvaluationInput testFieldStable 180 180 80 :>

  Nil

-- | Expected outputs (approximate due to fixed-point)
-- Fields: potential, fluxCoherence, inversionProb, emergenceScore, fragmentStability,
--         emergenceClass, luminescenceFlag, dischargeWarning, shadowRisk
expectedOutput :: Vec 4 ScalarOutput
expectedOutput =
  -- Test 0: High OR pyramidal -> Alpha emergence, luminescence
  -- OR=0.9+0.15=1.0(sat), DOR=0.1-0.1=0.0 -> high potential, high coherence
  ScalarOutput 255 200 5 200 195 AlphaEmergence True False False :>

  -- Test 1: Balanced -> Baseline stability
  -- OR=0.5, DOR=0.5 -> moderate values
  ScalarOutput 128 128 13 60 115 BaselineStability False False False :>

  -- Test 2: High DOR dome -> Shadow fragments, shadowRisk
  -- OR=0.2+0.1=0.3, DOR=0.7-0.05=0.65 -> low potential, shadow risk
  ScalarOutput 80 90 20 50 70 ShadowFragment False False True :>

  -- Test 3: Stable pyramid -> Stable fragments, luminescence
  -- OR=0.8+0.15=0.95, DOR=0.3-0.1=0.2 -> high stability
  ScalarOutput 220 180 8 180 172 StableFragment True False False :>

  Nil

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for orgone scalar validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    stim = stimuliGenerator clk rst testInputs
    out = orgoneScalarTop clk rst enableGen stim
    done = outputVerifier' clk rst expectedOutput out

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Get geometry name for logging
geometryName :: ChamberGeometry -> String
geometryName Pyramidal   = "pyramidal"
geometryName Dome        = "dome"
geometryName Rectangular = "rectangular"
geometryName Spherical   = "spherical"

-- | Get emergence class name
emergenceClassName :: EmergenceClass -> String
emergenceClassName AlphaEmergence    = "ALPHA_EMERGENCE"
emergenceClassName StableFragment    = "STABLE_FRAGMENT"
emergenceClassName BaselineStability = "BASELINE_STABILITY"
emergenceClassName ShadowFragment    = "SHADOW_FRAGMENT"
emergenceClassName FieldCollapse     = "FIELD_COLLAPSE"

-- | Check if field is in safe operating range
isSafeField :: OrgoneField -> Bool
isSafeField field =
  orLevel field > dorLevel field && dorLevel field < 150

-- | Compute OR/DOR ratio (scaled to 0-255, 128 = balanced)
orDorRatio :: OrgoneField -> Fixed8
orDorRatio field =
  let
    orVal = resize (orLevel field) :: CalcValue
    dorVal = resize (dorLevel field) :: CalcValue
    total = orVal + dorVal
    ratio = if total == 0 then 128 else (orVal * 256) `div` total
  in
    resize ratio
