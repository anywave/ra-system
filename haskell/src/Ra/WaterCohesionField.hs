{-|
Module      : Ra.WaterCohesionField
Description : Fluidic medium for coherence propagation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Models coherence as a scalar field propagating through a fluidic substrate
(blood, lymph, bio-aura) to simulate biofield effects based on water science
and blood electrification research.

== Water Science Principles

=== Structured Water

Water can exist in multiple phase states:

* Bulk water - Normal H2O clusters
* Exclusion zone (EZ) water - H3O2 structured layers
* Coherent domains - Quantum-coherent water clusters

=== Coherence Propagation

Scalar coherence propagates through water via:

* Ionic conductivity pathways
* Hydrogen bond network resonance
* EZ water coherent domains
* Electromagnetic coupling

=== Biofield Integration

The water-based biofield includes:

* Blood plasma - Primary ionic medium
* Lymphatic fluid - Secondary coherence channel
* Interstitial water - Tissue-level propagation
* Cellular water - Intracellular coherence
-}
module Ra.WaterCohesionField
  ( -- * Core Types
    CohesionField(..)
  , FluidPhase(..)
  , WaterState(..)
  , mkCohesionField

    -- * Fluid Parameters
  , FluidParams(..)
  , defaultPlasma
  , defaultLymph
  , defaultInterstitial

    -- * Propagation
  , PropagationResult(..)
  , propagateCoherence
  , propagationSpeed
  , attenuationFactor

    -- * Ionic Balance
  , IonicBalance(..)
  , computeIonicBalance
  , hydrationLevel
  , conductivity

    -- * Plasmic Resonance
  , PlasmicResonance(..)
  , initResonance
  , resonanceStrength
  , resonancePhase

    -- * Avatar Mapping
  , WaterPhaseColor(..)
  , phaseToColor
  , avatarWaterMap
  , coherenceGradient

    -- * Chamber Integration
  , ChamberFluid(..)
  , fluidInChamber
  , chamberResonance
  ) where

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Cohesion field in fluidic medium
data CohesionField = CohesionField
  { cfPhase         :: !FluidPhase      -- ^ Primary fluid phase
  , cfWaterState    :: !WaterState      -- ^ Water structure state
  , cfCoherence     :: !Double          -- ^ Field coherence [0,1]
  , cfViscosity     :: !Double          -- ^ Fluid viscosity
  , cfSalinity      :: !Double          -- ^ Ionic concentration
  , cfTemperature   :: !Double          -- ^ Temperature (Celsius)
  , cfPropagation   :: !Double          -- ^ Propagation coefficient
  } deriving (Eq, Show)

-- | Fluid phase type
data FluidPhase
  = BloodPlasma     -- ^ Primary ionic medium
  | LymphaticFluid  -- ^ Secondary coherence channel
  | Interstitial    -- ^ Tissue-level propagation
  | Cellular        -- ^ Intracellular water
  | Cerebrospinal   -- ^ Brain/spine fluid
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Water structure state
data WaterState
  = BulkWater       -- ^ Normal H2O clusters
  | EZWater         -- ^ Exclusion zone (H3O2)
  | CoherentDomain  -- ^ Quantum-coherent clusters
  | Structured      -- ^ Hexagonal structured water
  | Chaotic         -- ^ Disordered state
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create cohesion field
mkCohesionField :: FluidPhase -> Double -> CohesionField
mkCohesionField phase coherence =
  let params = paramsForPhase phase
  in CohesionField
      { cfPhase = phase
      , cfWaterState = stateFromCoherence coherence
      , cfCoherence = clamp01 coherence
      , cfViscosity = fpViscosity params
      , cfSalinity = fpSalinity params
      , cfTemperature = 37.0  -- Body temperature
      , cfPropagation = fpPropagation params
      }

-- State from coherence level
stateFromCoherence :: Double -> WaterState
stateFromCoherence c
  | c > 0.9 = CoherentDomain
  | c > 0.7 = Structured
  | c > 0.5 = EZWater
  | c > 0.3 = BulkWater
  | otherwise = Chaotic

-- =============================================================================
-- Fluid Parameters
-- =============================================================================

-- | Fluid parameters
data FluidParams = FluidParams
  { fpViscosity    :: !Double    -- ^ Dynamic viscosity (mPa·s)
  , fpSalinity     :: !Double    -- ^ Salt concentration (g/L)
  , fpPropagation  :: !Double    -- ^ Base propagation coefficient
  , fpDensity      :: !Double    -- ^ Density (kg/m³)
  } deriving (Eq, Show)

-- | Default blood plasma parameters
defaultPlasma :: FluidParams
defaultPlasma = FluidParams
  { fpViscosity = 1.5
  , fpSalinity = 9.0      -- ~0.9% NaCl
  , fpPropagation = 0.85
  , fpDensity = 1025.0
  }

-- | Default lymphatic fluid
defaultLymph :: FluidParams
defaultLymph = FluidParams
  { fpViscosity = 1.2
  , fpSalinity = 8.5
  , fpPropagation = 0.75
  , fpDensity = 1020.0
  }

-- | Default interstitial fluid
defaultInterstitial :: FluidParams
defaultInterstitial = FluidParams
  { fpViscosity = 1.0
  , fpSalinity = 8.0
  , fpPropagation = 0.70
  , fpDensity = 1010.0
  }

-- Get params for phase
paramsForPhase :: FluidPhase -> FluidParams
paramsForPhase phase = case phase of
  BloodPlasma -> defaultPlasma
  LymphaticFluid -> defaultLymph
  Interstitial -> defaultInterstitial
  Cellular -> FluidParams 0.8 7.0 0.60 1000.0
  Cerebrospinal -> FluidParams 0.7 7.5 0.90 1007.0

-- =============================================================================
-- Propagation
-- =============================================================================

-- | Propagation result
data PropagationResult = PropagationResult
  { prFinalCoherence :: !Double     -- ^ Coherence after propagation
  , prDistance       :: !Double     -- ^ Distance traveled
  , prTime           :: !Double     -- ^ Time elapsed (ms)
  , prAttenuation    :: !Double     -- ^ Total attenuation
  , prPhaseShift     :: !Double     -- ^ Phase shift accumulated
  } deriving (Eq, Show)

-- | Propagate coherence through field
propagateCoherence :: CohesionField -> Double -> Double -> PropagationResult
propagateCoherence field distance inputCoherence =
  let speed = propagationSpeed field
      time = distance / speed * 1000  -- Convert to ms

      -- Attenuation based on viscosity and salinity
      atten = attenuationFactor field distance

      -- Final coherence
      finalCoh = inputCoherence * atten * cfCoherence field

      -- Phase shift from water state
      phaseShift = phaseShiftForState (cfWaterState field) distance
  in PropagationResult
      { prFinalCoherence = clamp01 finalCoh
      , prDistance = distance
      , prTime = time
      , prAttenuation = atten
      , prPhaseShift = phaseShift
      }

-- | Calculate propagation speed (m/s)
propagationSpeed :: CohesionField -> Double
propagationSpeed field =
  let baseSpeed = 1500.0  -- Sound speed in water
      viscosityMod = 1.0 - cfViscosity field * 0.1
      salinityMod = 1.0 + cfSalinity field * 0.01
      stateMod = case cfWaterState field of
        CoherentDomain -> 1.2
        Structured -> 1.1
        EZWater -> 1.05
        BulkWater -> 1.0
        Chaotic -> 0.9
  in baseSpeed * viscosityMod * salinityMod * stateMod * cfPropagation field

-- | Calculate attenuation factor
attenuationFactor :: CohesionField -> Double -> Double
attenuationFactor field distance =
  let -- Attenuation per meter
      baseAtten = 0.1
      viscosityFactor = cfViscosity field * 0.05
      salinityBoost = cfSalinity field * 0.01  -- Salinity improves conduction
      stateFactor = case cfWaterState field of
        CoherentDomain -> 0.02
        Structured -> 0.04
        EZWater -> 0.06
        BulkWater -> 0.10
        Chaotic -> 0.20

      totalAtten = baseAtten + viscosityFactor + stateFactor - salinityBoost
  in exp (-totalAtten * distance)

-- Phase shift for water state
phaseShiftForState :: WaterState -> Double -> Double
phaseShiftForState state distance = case state of
  CoherentDomain -> distance * 0.01 * phi
  Structured -> distance * 0.02 * phi
  EZWater -> distance * 0.05
  BulkWater -> distance * 0.1
  Chaotic -> distance * 0.2

-- =============================================================================
-- Ionic Balance
-- =============================================================================

-- | Ionic balance state
data IonicBalance = IonicBalance
  { ibSodium      :: !Double    -- ^ Na+ concentration
  , ibPotassium   :: !Double    -- ^ K+ concentration
  , ibCalcium     :: !Double    -- ^ Ca2+ concentration
  , ibMagnesium   :: !Double    -- ^ Mg2+ concentration
  , ibChloride    :: !Double    -- ^ Cl- concentration
  , ibHydration   :: !Double    -- ^ Overall hydration [0,1]
  } deriving (Eq, Show)

-- | Compute ionic balance from field
computeIonicBalance :: CohesionField -> IonicBalance
computeIonicBalance field =
  let salinity = cfSalinity field
      -- Standard plasma ratios scaled by salinity
      na = salinity * 15.5   -- ~140 mEq/L at normal
      k = salinity * 0.5     -- ~4.5 mEq/L
      ca = salinity * 0.11   -- ~1 mEq/L
      mg = salinity * 0.22   -- ~2 mEq/L
      cl = salinity * 11.0   -- ~100 mEq/L

      hydration = clamp01 (cfCoherence field * 0.8 + 0.2)
  in IonicBalance
      { ibSodium = na
      , ibPotassium = k
      , ibCalcium = ca
      , ibMagnesium = mg
      , ibChloride = cl
      , ibHydration = hydration
      }

-- | Get hydration level
hydrationLevel :: IonicBalance -> Double
hydrationLevel = ibHydration

-- | Calculate ionic conductivity
conductivity :: IonicBalance -> Double
conductivity ib =
  let -- Weighted sum of ion contributions
      naCond = ibSodium ib * 0.05
      kCond = ibPotassium ib * 0.07
      caCond = ibCalcium ib * 0.06
      mgCond = ibMagnesium ib * 0.05
      clCond = ibChloride ib * 0.08
  in (naCond + kCond + caCond + mgCond + clCond) * ibHydration ib

-- =============================================================================
-- Plasmic Resonance
-- =============================================================================

-- | Plasmic resonance state
data PlasmicResonance = PlasmicResonance
  { prStrength    :: !Double      -- ^ Resonance strength [0,1]
  , prFrequency   :: !Double      -- ^ Resonance frequency (Hz)
  , prPhase       :: !Double      -- ^ Current phase [0, 2*pi]
  , prQFactor     :: !Double      -- ^ Quality factor
  , prDamping     :: !Double      -- ^ Damping coefficient
  } deriving (Eq, Show)

-- | Initialize plasmic resonance
initResonance :: CohesionField -> Double -> PlasmicResonance
initResonance field inputFreq =
  let -- Natural frequency depends on water state
      naturalFreq = case cfWaterState field of
        CoherentDomain -> 7.83 * phi  -- Schumann * phi
        Structured -> 7.83
        EZWater -> 10.0
        BulkWater -> 15.0
        Chaotic -> 20.0

      -- Resonance strength from frequency match
      freqRatio = inputFreq / naturalFreq
      strength = if freqRatio > 0.5 && freqRatio < 2.0
                 then 1.0 - abs (1.0 - freqRatio) * 0.5
                 else 0.2

      -- Q factor from coherence
      qFactor = 10.0 + cfCoherence field * 90.0

      -- Damping inversely related to coherence
      damping = 0.1 - cfCoherence field * 0.08
  in PlasmicResonance
      { prStrength = clamp01 strength
      , prFrequency = naturalFreq
      , prPhase = 0.0
      , prQFactor = qFactor
      , prDamping = max 0.01 damping
      }

-- | Get resonance strength
resonanceStrength :: PlasmicResonance -> Double
resonanceStrength = prStrength

-- | Get current phase
resonancePhase :: PlasmicResonance -> Double
resonancePhase = prPhase

-- =============================================================================
-- Avatar Mapping
-- =============================================================================

-- | Water phase color mapping
data WaterPhaseColor = WaterPhaseColor
  { wpcRed     :: !Int
  , wpcGreen   :: !Int
  , wpcBlue    :: !Int
  , wpcAlpha   :: !Double
  } deriving (Eq, Show)

-- | Map water state to color
phaseToColor :: WaterState -> Double -> WaterPhaseColor
phaseToColor state coherence = case state of
  CoherentDomain -> WaterPhaseColor
    { wpcRed = 100
    , wpcGreen = 200
    , wpcBlue = 255
    , wpcAlpha = 0.8 + coherence * 0.2
    }
  Structured -> WaterPhaseColor
    { wpcRed = 80
    , wpcGreen = 180
    , wpcBlue = 240
    , wpcAlpha = 0.7 + coherence * 0.2
    }
  EZWater -> WaterPhaseColor
    { wpcRed = 60
    , wpcGreen = 150
    , wpcBlue = 220
    , wpcAlpha = 0.6 + coherence * 0.2
    }
  BulkWater -> WaterPhaseColor
    { wpcRed = 40
    , wpcGreen = 120
    , wpcBlue = 200
    , wpcAlpha = 0.5 + coherence * 0.2
    }
  Chaotic -> WaterPhaseColor
    { wpcRed = 100
    , wpcGreen = 80
    , wpcBlue = 150
    , wpcAlpha = 0.3 + coherence * 0.2
    }

-- | Generate avatar water map
avatarWaterMap :: CohesionField -> [(String, WaterPhaseColor)]
avatarWaterMap field =
  let coherence = cfCoherence field
      state = cfWaterState field
      baseColor = phaseToColor state coherence

      -- Zone-specific colors
      crownColor = modColor baseColor 1.2 1.1 1.0
      heartColor = modColor baseColor 1.0 1.2 1.1
      solarColor = modColor baseColor 1.1 1.2 0.9
      sacralColor = modColor baseColor 1.0 1.0 1.2
      rootColor = modColor baseColor 0.9 1.0 1.1
  in [ ("crown", crownColor)
     , ("third_eye", baseColor)
     , ("throat", baseColor)
     , ("heart", heartColor)
     , ("solar", solarColor)
     , ("sacral", sacralColor)
     , ("root", rootColor)
     ]

-- Modify color
modColor :: WaterPhaseColor -> Double -> Double -> Double -> WaterPhaseColor
modColor c rm gm bm = c
  { wpcRed = min 255 (round (fromIntegral (wpcRed c) * rm))
  , wpcGreen = min 255 (round (fromIntegral (wpcGreen c) * gm))
  , wpcBlue = min 255 (round (fromIntegral (wpcBlue c) * bm))
  }

-- | Generate coherence gradient
coherenceGradient :: CohesionField -> Int -> [Double]
coherenceGradient field steps =
  let baseCoherence = cfCoherence field
      stepSize = 1.0 / fromIntegral (max 1 steps)
  in [baseCoherence * (1.0 - fromIntegral i * stepSize * 0.5) | i <- [0..steps-1]]

-- =============================================================================
-- Chamber Integration
-- =============================================================================

-- | Fluid state in chamber
data ChamberFluid = ChamberFluid
  { chfField       :: !CohesionField
  , chfVolume      :: !Double         -- ^ Volume in liters
  , chfCirculation :: !Double         -- ^ Circulation rate
  , chfResonance   :: !PlasmicResonance
  } deriving (Eq, Show)

-- | Create fluid in chamber
fluidInChamber :: FluidPhase -> Double -> Double -> ChamberFluid
fluidInChamber phase coherence volume =
  let field = mkCohesionField phase coherence
      resonance = initResonance field 7.83
  in ChamberFluid
      { chfField = field
      , chfVolume = volume
      , chfCirculation = 1.0
      , chfResonance = resonance
      }

-- | Get chamber resonance state
chamberResonance :: ChamberFluid -> Double
chamberResonance = prStrength . chfResonance

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
