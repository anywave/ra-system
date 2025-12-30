{-|
Module      : Ra.ORME.FluxChamber
Description : ORME flux chamber for monoatomic element processing
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements flux chamber mechanics for Orbitally Rearranged Monoatomic Element
(ORME) processing. The chamber creates controlled scalar field conditions
for element transmutation and energy extraction.

== ORME Theory

=== Monoatomic States

ORME elements exist in high-spin monoatomic states:

* Single atoms with rearranged electron orbitals
* High-spin nucleus coupling to scalar field
* Zero-point energy interaction

=== Flux Chamber Operation

1. Coherence Priming: Establish baseline field coherence
2. Element Introduction: Insert target material
3. Flux Cycling: Apply oscillating scalar flux
4. Transmutation Window: Monitor phase transition
5. Energy Harvest: Extract excess energy
-}
module Ra.ORME.FluxChamber
  ( -- * Core Types
    FluxChamber(..)
  , ChamberState(..)
  , ORMEElement(..)
  , FluxCycle(..)

    -- * Chamber Operations
  , initializeChamber
  , activateChamber
  , deactivateChamber
  , chamberStatus

    -- * Flux Control
  , startFluxCycle
  , modulateFlux
  , stabilizeFlux
  , fluxIntensity

    -- * Element Processing
  , introduceElement
  , elementCoherence
  , transmutationReady
  , extractEnergy

    -- * Monitoring
  , ChamberMetrics(..)
  , measureChamber
  , fluxHistory
  , energyYield

    -- * Safety
  , SafetyState(..)
  , checkSafety
  , emergencyShutdown
  , coolingCycle
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Flux chamber configuration
data FluxChamber = FluxChamber
  { fcState        :: !ChamberState          -- ^ Current chamber state
  , fcElement      :: !(Maybe ORMEElement)   -- ^ Element in chamber
  , fcFluxLevel    :: !Double                -- ^ Current flux level [0, 1]
  , fcCoherence    :: !Double                -- ^ Field coherence [0, 1]
  , fcTemperature  :: !Double                -- ^ Chamber temperature (K)
  , fcCycles       :: !Int                   -- ^ Completed flux cycles
  , fcSafety       :: !SafetyState           -- ^ Safety status
  } deriving (Eq, Show)

-- | Chamber operational state
data ChamberState
  = ChamberIdle        -- ^ Ready, no activity
  | ChamberPriming     -- ^ Establishing coherence
  | ChamberActive      -- ^ Processing active
  | ChamberHarvesting  -- ^ Energy extraction
  | ChamberCooling     -- ^ Cooling down
  | ChamberError       -- ^ Error state
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | ORME element types
data ORMEElement = ORMEElement
  { oeType       :: !ElementType    -- ^ Base element
  , oeMass       :: !Double         -- ^ Atomic mass
  , oeSpinState  :: !SpinState      -- ^ Current spin state
  , oeCoherence  :: !Double         -- ^ Element coherence
  , oeEnergy     :: !Double         -- ^ Stored energy
  } deriving (Eq, Show)

-- | Base element types for ORME
data ElementType
  = ElemGold        -- ^ Au - primary ORME candidate
  | ElemPlatinum    -- ^ Pt
  | ElemIridium     -- ^ Ir
  | ElemRhodium     -- ^ Rh
  | ElemPalladium   -- ^ Pd
  | ElemRuthenium   -- ^ Ru
  | ElemOsmium      -- ^ Os
  | ElemSilver      -- ^ Ag
  | ElemCopper      -- ^ Cu
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Spin state enumeration
data SpinState
  = SpinLow        -- ^ Normal spin state
  | SpinMedium     -- ^ Elevated spin
  | SpinHigh       -- ^ High-spin ORME state
  | SpinSuper      -- ^ Superconducting spin
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Flux cycle specification
data FluxCycle = FluxCycle
  { fcyAmplitude  :: !Double    -- ^ Cycle amplitude [0, 1]
  , fcyFrequency  :: !Double    -- ^ Cycle frequency (Hz)
  , fcyPhase      :: !Double    -- ^ Phase offset [0, 2π]
  , fcyDuration   :: !Int       -- ^ Duration in ticks
  , fcyWaveform   :: !Waveform  -- ^ Flux waveform type
  } deriving (Eq, Show)

-- | Waveform types
data Waveform
  = WaveSine       -- ^ Sinusoidal
  | WaveSquare     -- ^ Square wave
  | WaveTriangle   -- ^ Triangle wave
  | WaveSawtooth   -- ^ Sawtooth
  | WaveGolden     -- ^ Golden ratio modulated
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Chamber Operations
-- =============================================================================

-- | Initialize flux chamber
initializeChamber :: FluxChamber
initializeChamber = FluxChamber
  { fcState = ChamberIdle
  , fcElement = Nothing
  , fcFluxLevel = 0
  , fcCoherence = 0
  , fcTemperature = 293  -- Room temperature (K)
  , fcCycles = 0
  , fcSafety = SafetyNormal
  }

-- | Activate chamber for processing
activateChamber :: FluxChamber -> FluxChamber
activateChamber fc
  | fcSafety fc /= SafetyNormal = fc { fcState = ChamberError }
  | fcState fc == ChamberIdle =
      fc { fcState = ChamberPriming, fcCoherence = phiInverse * 0.5 }
  | otherwise = fc

-- | Deactivate chamber
deactivateChamber :: FluxChamber -> FluxChamber
deactivateChamber fc =
  fc { fcState = ChamberCooling, fcFluxLevel = 0 }

-- | Get chamber status string
chamberStatus :: FluxChamber -> String
chamberStatus fc =
  "Chamber: " ++ show (fcState fc) ++
  " | Flux: " ++ show (round (fcFluxLevel fc * 100) :: Int) ++ "%" ++
  " | Coherence: " ++ show (round (fcCoherence fc * 100) :: Int) ++ "%" ++
  " | Cycles: " ++ show (fcCycles fc)

-- =============================================================================
-- Flux Control
-- =============================================================================

-- | Start a flux cycle
startFluxCycle :: FluxChamber -> FluxCycle -> FluxChamber
startFluxCycle fc cycle'
  | fcState fc /= ChamberActive && fcState fc /= ChamberPriming = fc
  | otherwise =
      let newFlux = fcFluxLevel fc + fcyAmplitude cycle' * phiInverse
          newCoherence = min 1.0 (fcCoherence fc + fcyAmplitude cycle' * 0.1)
          newCycles = fcCycles fc + 1
      in fc
        { fcState = ChamberActive
        , fcFluxLevel = min 1.0 newFlux
        , fcCoherence = newCoherence
        , fcCycles = newCycles
        }

-- | Modulate flux level
modulateFlux :: FluxChamber -> Double -> FluxChamber
modulateFlux fc modulation =
  let newFlux = fcFluxLevel fc * (1 + modulation * phiInverse)
  in fc { fcFluxLevel = max 0 (min 1 newFlux) }

-- | Stabilize flux to golden ratio
stabilizeFlux :: FluxChamber -> FluxChamber
stabilizeFlux fc =
  let targetFlux = phiInverse
      diff = targetFlux - fcFluxLevel fc
      adjustment = diff * 0.1
  in fc { fcFluxLevel = fcFluxLevel fc + adjustment }

-- | Get current flux intensity
fluxIntensity :: FluxChamber -> Double
fluxIntensity fc = fcFluxLevel fc * fcCoherence fc * phi

-- =============================================================================
-- Element Processing
-- =============================================================================

-- | Introduce element to chamber
introduceElement :: FluxChamber -> ORMEElement -> FluxChamber
introduceElement fc elem'
  | fcState fc /= ChamberPriming && fcState fc /= ChamberIdle = fc
  | otherwise = fc { fcElement = Just elem' }

-- | Get element coherence (chamber + element)
elementCoherence :: FluxChamber -> Double
elementCoherence fc =
  case fcElement fc of
    Nothing -> 0
    Just elem' ->
      let chamberEffect = fcCoherence fc * fcFluxLevel fc
          elementBase = oeCoherence elem'
      in (chamberEffect + elementBase) / 2

-- | Check if ready for transmutation
transmutationReady :: FluxChamber -> Bool
transmutationReady fc =
  fcState fc == ChamberActive &&
  fcCoherence fc > phiInverse &&
  fcFluxLevel fc > phiInverse &&
  case fcElement fc of
    Nothing -> False
    Just elem' -> oeSpinState elem' >= SpinHigh

-- | Extract energy from chamber
extractEnergy :: FluxChamber -> (FluxChamber, Double)
extractEnergy fc
  | fcState fc /= ChamberActive = (fc, 0)
  | otherwise =
      let elementEnergy = maybe 0 oeEnergy (fcElement fc)
          fluxContribution = fcFluxLevel fc * fcCoherence fc * phi
          totalEnergy = elementEnergy * 0.1 + fluxContribution
          newFlux = fcFluxLevel fc * phiInverse  -- Flux decreases
          newCoherence = fcCoherence fc * 0.95   -- Slight coherence loss
      in (fc { fcState = ChamberHarvesting
             , fcFluxLevel = newFlux
             , fcCoherence = newCoherence
             }, totalEnergy)

-- =============================================================================
-- Monitoring
-- =============================================================================

-- | Chamber performance metrics
data ChamberMetrics = ChamberMetrics
  { cmFluxAverage    :: !Double    -- ^ Average flux level
  , cmCoherenceAvg   :: !Double    -- ^ Average coherence
  , cmCycleCount     :: !Int       -- ^ Total cycles
  , cmEnergyTotal    :: !Double    -- ^ Total energy extracted
  , cmEfficiency     :: !Double    -- ^ Processing efficiency
  } deriving (Eq, Show)

-- | Measure chamber metrics
measureChamber :: FluxChamber -> ChamberMetrics
measureChamber fc =
  let efficiency = if fcCycles fc > 0
                   then fcCoherence fc * fcFluxLevel fc / fromIntegral (fcCycles fc)
                   else 0
  in ChamberMetrics
    { cmFluxAverage = fcFluxLevel fc
    , cmCoherenceAvg = fcCoherence fc
    , cmCycleCount = fcCycles fc
    , cmEnergyTotal = fluxIntensity fc * fromIntegral (fcCycles fc)
    , cmEfficiency = efficiency
    }

-- | Get flux history (simplified - returns current state)
fluxHistory :: FluxChamber -> [(Int, Double)]
fluxHistory fc =
  [(fcCycles fc, fcFluxLevel fc)]

-- | Calculate energy yield
energyYield :: FluxChamber -> Double
energyYield fc =
  let baseYield = fcFluxLevel fc * fcCoherence fc
      elementBonus = case fcElement fc of
        Nothing -> 0
        Just elem' -> oeEnergy elem' * elementMultiplier (oeType elem')
  in baseYield + elementBonus

-- | Element-specific multipliers
elementMultiplier :: ElementType -> Double
elementMultiplier ElemGold = phi
elementMultiplier ElemPlatinum = phi * phiInverse
elementMultiplier ElemIridium = phi * phiInverse * phiInverse
elementMultiplier ElemRhodium = phiInverse
elementMultiplier ElemPalladium = phiInverse * phiInverse
elementMultiplier ElemRuthenium = 0.5
elementMultiplier ElemOsmium = 0.6
elementMultiplier ElemSilver = 0.4
elementMultiplier ElemCopper = 0.3

-- =============================================================================
-- Safety
-- =============================================================================

-- | Safety state enumeration
data SafetyState
  = SafetyNormal     -- ^ All systems normal
  | SafetyWarning    -- ^ Warning conditions
  | SafetyCritical   -- ^ Critical - intervention needed
  | SafetyShutdown   -- ^ Emergency shutdown active
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Check safety conditions
checkSafety :: FluxChamber -> SafetyState
checkSafety fc
  | fcTemperature fc > 1000 = SafetyCritical
  | fcTemperature fc > 500 = SafetyWarning
  | fcFluxLevel fc > 0.95 && fcCoherence fc < 0.3 = SafetyCritical
  | fcFluxLevel fc > 0.9 = SafetyWarning
  | otherwise = SafetyNormal

-- | Emergency shutdown procedure
emergencyShutdown :: FluxChamber -> FluxChamber
emergencyShutdown fc =
  fc { fcState = ChamberError
     , fcFluxLevel = 0
     , fcSafety = SafetyShutdown
     }

-- | Run cooling cycle
coolingCycle :: FluxChamber -> Int -> FluxChamber
coolingCycle fc ticks =
  let coolingRate = fromIntegral ticks * 10  -- K per tick
      newTemp = max 293 (fcTemperature fc - coolingRate)
      newState = if newTemp <= 300 then ChamberIdle else ChamberCooling
      newSafety = if newTemp <= 400 then SafetyNormal else fcSafety fc
  in fc { fcTemperature = newTemp, fcState = newState, fcSafety = newSafety }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Create default element
_defaultElement :: ElementType -> ORMEElement
_defaultElement elemType = ORMEElement
  { oeType = elemType
  , oeMass = elementMass elemType
  , oeSpinState = SpinLow
  , oeCoherence = 0.5
  , oeEnergy = 0
  }

-- | Element atomic mass (approximate)
elementMass :: ElementType -> Double
elementMass ElemGold = 197.0
elementMass ElemPlatinum = 195.0
elementMass ElemIridium = 192.0
elementMass ElemRhodium = 103.0
elementMass ElemPalladium = 106.0
elementMass ElemRuthenium = 101.0
elementMass ElemOsmium = 190.0
elementMass ElemSilver = 108.0
elementMass ElemCopper = 64.0
