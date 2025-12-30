{-|
Module      : Ra.Orgone
Description : Orgone field influence modeling per Reich's principles
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Models orgone field dynamics based on Wilhelm Reich's research,
integrating with Ra System scalar emergence.

== Reich Orgone Principles

From Reich's orgone accumulator research:

* OR (Positive Orgone): Life-enhancing, coherence-building field
* DOR (Deadly Orgone): Life-diminishing, stagnant/toxic field
* Accumulator layers: Metal/organic alternation amplifies field
* Temperature differential: Orgone concentration measurable by ~0.3-0.4°C delta
* Pulsation: Orgone exhibits rhythmic expansion/contraction

== Integration with Ra System

* OR maps to coherence above POR floor (1/φ ≈ 0.618)
* DOR maps to coherence below DOR floor (1/π ≈ 0.318)
* Accumulator layers modeled as shell depth in Ra.Scalar
* Temperature delta correlates with emergence intensity

== Biometric Mapping

Reich observed orgone charge correlates with:
* Skin conductivity (GSR) - higher charge = lower resistance
* Respiratory depth - deeper breath = more charge
* Muscle tension - relaxed muscles = better flow
-}
module Ra.Orgone
  ( -- * Orgone Polarity
    OrgonePolarity(..)
  , polarityFromCoherence
  , polarityToMultiplier

    -- * Orgone Field
  , OrgoneField(..)
  , mkOrgoneField
  , fieldStrength
  , fieldCharge
  , fieldDecay

    -- * Accumulator Model
  , AccumulatorLayer(..)
  , Accumulator(..)
  , mkAccumulator
  , accumulatorCharge
  , accumulatorTemperatureDelta
  , optimalLayerCount

    -- * OR/DOR Dynamics
  , OrgoneDynamics(..)
  , computeDynamics
  , dynamicsToCoherence
  , isStagnant

    -- * Charge/Discharge Cycles
  , ChargeState(..)
  , ChargeCycle(..)
  , cyclePhase
  , chargeRate
  , dischargeRate
  , pulsationFrequency

    -- * Field Interaction
  , FieldInteraction(..)
  , computeInteraction
  , interactionToEmergence

    -- * Biometric Correlation
  , OrgoneBiometrics(..)
  , biometricsToCharge
  , chargeIndicators

    -- * Reich Constants
  , reichLayerCount
  , reichMaxAccumulation
  , reichChargeDecayRate
  , reichTemperatureDelta
  , dorFloor
  , porFloor
  ) where

import Ra.Constants.Extended
  ( coherenceFloorDOR, coherenceFloorPOR
  , reichLayerCount, reichAccumulationMax
  , reichChargeDecay, reichTemperatureDelta
  )

-- =============================================================================
-- Orgone Polarity
-- =============================================================================

-- | Orgone polarity: OR (positive) or DOR (negative/stagnant)
data OrgonePolarity
  = OR    -- ^ Positive Orgone - life-enhancing
  | DOR   -- ^ Deadly Orgone - life-diminishing
  | Mixed -- ^ Transition zone between OR and DOR
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Determine polarity from coherence level
--
-- * coherence >= POR floor (0.618) -> OR
-- * coherence <= DOR floor (0.318) -> DOR
-- * between floors -> Mixed
polarityFromCoherence :: Double -> OrgonePolarity
polarityFromCoherence c
  | c >= coherenceFloorPOR = OR
  | c <= coherenceFloorDOR = DOR
  | otherwise = Mixed

-- | Convert polarity to emergence multiplier
--
-- * OR: +1.0 (full positive emergence)
-- * Mixed: 0.5 (partial emergence)
-- * DOR: -0.3 (shadow/inverted emergence)
polarityToMultiplier :: OrgonePolarity -> Double
polarityToMultiplier OR = 1.0
polarityToMultiplier Mixed = 0.5
polarityToMultiplier DOR = -0.3

-- =============================================================================
-- Orgone Field
-- =============================================================================

-- | Orgone field model
data OrgoneField = OrgoneField
  { ofPolarity  :: !OrgonePolarity  -- ^ Current field polarity
  , ofIntensity :: !Double          -- ^ Field intensity [0,1]
  , ofCharge    :: !Double          -- ^ Accumulated charge [0, maxAccumulation]
  , ofPulsation :: !Double          -- ^ Pulsation phase [0, 2π)
  , ofDecayRate :: !Double          -- ^ Charge decay rate per second
  } deriving (Eq, Show)

-- | Create orgone field from coherence level
mkOrgoneField :: Double -> OrgoneField
mkOrgoneField coherence = OrgoneField
  { ofPolarity = polarityFromCoherence coherence
  , ofIntensity = clamp01 coherence
  , ofCharge = coherence * reichAccumulationMax
  , ofPulsation = 0.0
  , ofDecayRate = reichChargeDecay
  }

-- | Compute field strength (intensity * polarity multiplier)
fieldStrength :: OrgoneField -> Double
fieldStrength of' =
  ofIntensity of' * polarityToMultiplier (ofPolarity of')

-- | Get current charge level
fieldCharge :: OrgoneField -> Double
fieldCharge = ofCharge

-- | Apply decay over time interval
fieldDecay :: Double -> OrgoneField -> OrgoneField
fieldDecay dt of' =
  let newCharge = max 0 (ofCharge of' - ofDecayRate of' * dt)
      newIntensity = newCharge / reichAccumulationMax
      newPolarity = polarityFromCoherence newIntensity
  in of' { ofCharge = newCharge
         , ofIntensity = newIntensity
         , ofPolarity = newPolarity
         }

-- =============================================================================
-- Accumulator Model
-- =============================================================================

-- | Single accumulator layer (metal/organic pair)
data AccumulatorLayer = AccumulatorLayer
  { alMetalThickness   :: !Double  -- ^ Metal layer thickness (mm)
  , alOrganicThickness :: !Double  -- ^ Organic layer thickness (mm)
  , alEfficiency       :: !Double  -- ^ Layer efficiency [0,1]
  } deriving (Eq, Show)

-- | Multi-layer accumulator
data Accumulator = Accumulator
  { accLayers        :: ![AccumulatorLayer]
  , accTotalCharge   :: !Double  -- ^ Total accumulated charge
  , accSaturation    :: !Double  -- ^ Saturation level [0,1]
  } deriving (Eq, Show)

-- | Create accumulator with standard layer count
mkAccumulator :: Int -> Accumulator
mkAccumulator n =
  let layers = replicate (max 1 n) defaultLayer
  in Accumulator
      { accLayers = layers
      , accTotalCharge = 0.0
      , accSaturation = 0.0
      }
  where
    defaultLayer = AccumulatorLayer 0.5 10.0 1.0

-- | Compute layer efficiency based on count
-- Reich found 7 layers optimal; more layers have diminishing returns
layerEfficiency :: Int -> Double
layerEfficiency n =
  let optimal = fromIntegral reichLayerCount
      actual = fromIntegral n
  in if n <= reichLayerCount
     then actual / optimal
     else optimal / actual * 0.9  -- Diminishing returns past 7

-- | Compute total charge from field exposure
accumulatorCharge :: OrgoneField -> Accumulator -> Double
accumulatorCharge field acc =
  let baseCharge = fieldCharge field
      layerCount = length (accLayers acc)
      efficiency = layerEfficiency layerCount
      saturationFactor = 1.0 - accSaturation acc
  in min reichAccumulationMax (baseCharge * efficiency * saturationFactor)

-- | Compute temperature differential (Reich's Einstein experiment)
--
-- Reich demonstrated 0.3-0.4°C temperature rise inside accumulator
-- relative to ambient temperature.
accumulatorTemperatureDelta :: Accumulator -> Double
accumulatorTemperatureDelta acc =
  let saturation = accSaturation acc
  in reichTemperatureDelta * saturation

-- | Optimal layer count for maximum efficiency
optimalLayerCount :: Int
optimalLayerCount = reichLayerCount

-- =============================================================================
-- OR/DOR Dynamics
-- =============================================================================

-- | Orgone dynamics state
data OrgoneDynamics = OrgoneDynamics
  { odPolarity    :: !OrgonePolarity
  , odVelocity    :: !Double  -- ^ Rate of polarity change
  , odStagnation  :: !Double  -- ^ Stagnation level [0,1]
  , odPulsation   :: !Double  -- ^ Current pulsation amplitude
  } deriving (Eq, Show)

-- | Compute dynamics from field state
computeDynamics :: OrgoneField -> OrgoneField -> Double -> OrgoneDynamics
computeDynamics prev curr dt =
  let prevIntensity = ofIntensity prev
      currIntensity = ofIntensity curr
      velocity = (currIntensity - prevIntensity) / max 0.001 dt

      -- Stagnation increases when charge is high but intensity is low
      stagnation = if ofCharge curr > reichAccumulationMax / 2
                   && currIntensity < 0.5
                   then 0.5 + 0.5 * (1 - currIntensity)
                   else 0.0

      -- Pulsation from field state
      pulsation = sin (ofPulsation curr)
  in OrgoneDynamics
      { odPolarity = ofPolarity curr
      , odVelocity = velocity
      , odStagnation = clamp01 stagnation
      , odPulsation = pulsation
      }

-- | Convert dynamics to coherence adjustment
dynamicsToCoherence :: OrgoneDynamics -> Double
dynamicsToCoherence od =
  let base = case odPolarity od of
        OR -> 1.0
        Mixed -> 0.5
        DOR -> 0.0
      stagnationPenalty = odStagnation od * 0.3
      pulsationBonus = (odPulsation od + 1) / 2 * 0.1
  in clamp01 (base - stagnationPenalty + pulsationBonus)

-- | Check if field is stagnant (DOR accumulation risk)
isStagnant :: OrgoneDynamics -> Bool
isStagnant od = odStagnation od > 0.5 || odPolarity od == DOR

-- =============================================================================
-- Charge/Discharge Cycles
-- =============================================================================

-- | Charge state in pulsation cycle
data ChargeState
  = Charging     -- ^ Field is accumulating charge
  | Holding      -- ^ Field at peak charge
  | Discharging  -- ^ Field releasing charge
  | Rest         -- ^ Field at minimum charge
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Complete charge cycle
data ChargeCycle = ChargeCycle
  { ccState      :: !ChargeState
  , ccPhase      :: !Double  -- ^ Phase within current state [0,1]
  , ccCharge     :: !Double  -- ^ Current charge level
  , ccPeakCharge :: !Double  -- ^ Maximum charge this cycle
  } deriving (Eq, Show)

-- | Determine cycle phase from charge trajectory
cyclePhase :: Double -> Double -> ChargeState
cyclePhase charge velocity
  | velocity > 0.1 = Charging
  | velocity < -0.1 = Discharging
  | charge > 0.7 = Holding
  | otherwise = Rest

-- | Compute charge rate (Reich: faster charge = healthier organism)
chargeRate :: OrgoneBiometrics -> Double
chargeRate bio =
  let breathFactor = obBreathDepth bio
      relaxFactor = 1.0 - obMuscleTension bio
  in breathFactor * relaxFactor * reichAccumulationMax / 60.0  -- Per second

-- | Compute discharge rate
dischargeRate :: OrgoneBiometrics -> Double
dischargeRate bio =
  let tensionFactor = obMuscleTension bio
      gsrFactor = 1.0 - obGSR bio  -- Lower GSR = more discharge
  in (tensionFactor + gsrFactor) / 2 * reichChargeDecay

-- | Natural pulsation frequency (Reich: ~7-8 Hz for healthy organism)
pulsationFrequency :: OrgoneDynamics -> Double
pulsationFrequency od =
  let basePulsation = 7.5  -- Hz
      stagnationFactor = 1.0 - odStagnation od
  in basePulsation * stagnationFactor

-- =============================================================================
-- Field Interaction
-- =============================================================================

-- | Result of orgone field interaction
data FieldInteraction = FieldInteraction
  { fiResultPolarity :: !OrgonePolarity
  , fiCombinedCharge :: !Double
  , fiResonance      :: !Double  -- ^ How well fields harmonize
  , fiEmergence      :: !Double  -- ^ Emergence potential
  } deriving (Eq, Show)

-- | Compute interaction between two orgone fields
computeInteraction :: OrgoneField -> OrgoneField -> FieldInteraction
computeInteraction f1 f2 =
  let p1 = ofPolarity f1
      p2 = ofPolarity f2

      -- Polarity combination
      resultPolarity = case (p1, p2) of
        (OR, OR) -> OR
        (DOR, DOR) -> DOR
        (OR, DOR) -> Mixed
        (DOR, OR) -> Mixed
        (Mixed, _) -> Mixed
        (_, Mixed) -> Mixed

      -- Charge combination (geometric mean for resonance)
      c1 = ofCharge f1
      c2 = ofCharge f2
      combinedCharge = sqrt (c1 * c2)

      -- Resonance based on pulsation phase alignment
      phaseDiff = abs (ofPulsation f1 - ofPulsation f2)
      resonance = cos phaseDiff  -- 1.0 when in phase, -1.0 when anti-phase

      -- Emergence potential
      emergence = combinedCharge / reichAccumulationMax * (resonance + 1) / 2
  in FieldInteraction
      { fiResultPolarity = resultPolarity
      , fiCombinedCharge = combinedCharge
      , fiResonance = resonance
      , fiEmergence = clamp01 emergence
      }

-- | Convert interaction to emergence intensity
interactionToEmergence :: FieldInteraction -> Double
interactionToEmergence fi =
  let polarityFactor = polarityToMultiplier (fiResultPolarity fi)
      resonanceFactor = (fiResonance fi + 1) / 2
  in fiEmergence fi * polarityFactor * resonanceFactor

-- =============================================================================
-- Biometric Correlation
-- =============================================================================

-- | Biometric inputs correlated with orgone charge
data OrgoneBiometrics = OrgoneBiometrics
  { obGSR          :: !Double  -- ^ Galvanic skin response [0,1] (normalized)
  , obBreathDepth  :: !Double  -- ^ Respiratory depth [0,1]
  , obMuscleTension :: !Double -- ^ Overall muscle tension [0,1]
  , obHeartCoherence :: !Double -- ^ HRV-derived coherence [0,1]
  } deriving (Eq, Show)

-- | Convert biometrics to orgone charge estimate
--
-- Reich's observations:
-- * Lower GSR = higher charge (skin becomes more conductive)
-- * Deeper breath = more charge accumulation
-- * Relaxed muscles = better energy flow
-- * Higher HRV coherence = better orgone pulsation
biometricsToCharge :: OrgoneBiometrics -> Double
biometricsToCharge bio =
  let gsrFactor = 1.0 - obGSR bio       -- Invert: low GSR = high charge
      breathFactor = obBreathDepth bio
      tensionFactor = 1.0 - obMuscleTension bio  -- Invert: low tension = high flow
      hrvFactor = obHeartCoherence bio

      -- Weighted combination
      charge = 0.25 * gsrFactor
             + 0.30 * breathFactor
             + 0.20 * tensionFactor
             + 0.25 * hrvFactor
  in clamp01 charge * reichAccumulationMax

-- | Indicators of orgone charge from biometrics
chargeIndicators :: OrgoneBiometrics -> (Bool, Bool, Bool, Bool)
chargeIndicators bio =
  ( obGSR bio < 0.4           -- Low skin resistance
  , obBreathDepth bio > 0.6   -- Deep breathing
  , obMuscleTension bio < 0.4 -- Relaxed muscles
  , obHeartCoherence bio > 0.6 -- High HRV coherence
  )

-- =============================================================================
-- Reich Constants (re-exported from Extended)
-- =============================================================================

-- | Standard accumulator layer count
reichMaxAccumulation :: Double
reichMaxAccumulation = reichAccumulationMax

-- | Charge decay rate per second
reichChargeDecayRate :: Double
reichChargeDecayRate = reichChargeDecay

-- | DOR coherence floor (1/π ≈ 0.318)
dorFloor :: Double
dorFloor = coherenceFloorDOR

-- | POR coherence floor (1/φ ≈ 0.618)
porFloor :: Double
porFloor = coherenceFloorPOR

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
