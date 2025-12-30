{-|
Module      : Ra.Energy.RadiantTap
Description : Radiant energy extraction from scalar field fluctuations
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements radiant energy tapping mechanics for extracting usable energy
from scalar field fluctuations. Based on concepts of zero-point energy
and coherence-driven extraction.

== Radiant Energy Theory

=== Extraction Principle

Scalar field fluctuations contain extractable energy:

* Coherent fluctuations concentrate energy
* Tap resonance creates extraction window
* Golden ratio harmonics maximize efficiency

=== Tap Architecture

1. Resonator: Tunes to field fluctuations
2. Concentrator: Focuses coherent energy
3. Converter: Transforms to usable form
4. Buffer: Stores extracted energy
-}
module Ra.Energy.RadiantTap
  ( -- * Core Types
    RadiantTap(..)
  , TapState(..)
  , EnergyBuffer(..)
  , TapResonator(..)

    -- * Tap Operations
  , initializeTap
  , activateTap
  , deactivateTap
  , tapStatus

    -- * Energy Extraction
  , extractEnergy
  , concentrateField
  , convertEnergy
  , bufferEnergy

    -- * Resonator Control
  , tuneResonator
  , resonatorFrequency
  , resonanceQuality

    -- * Efficiency
  , TapEfficiency(..)
  , measureEfficiency
  , optimizeTap
  , efficiencyHistory

    -- * Output
  , EnergyOutput(..)
  , getOutput
  , sustainedOutput
  , peakOutput
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Radiant energy tap
data RadiantTap = RadiantTap
  { rtState      :: !TapState           -- ^ Current state
  , rtResonator  :: !TapResonator       -- ^ Resonator configuration
  , rtBuffer     :: !EnergyBuffer       -- ^ Energy storage
  , rtCoherence  :: !Double             -- ^ Field coherence [0, 1]
  , rtOutput     :: !Double             -- ^ Current output level
  , rtCycles     :: !Int                -- ^ Operation cycles
  } deriving (Eq, Show)

-- | Tap operational state
data TapState
  = TapIdle        -- ^ Not operating
  | TapTuning      -- ^ Tuning resonator
  | TapExtracting  -- ^ Active extraction
  | TapBuffering   -- ^ Buffering energy
  | TapOutputting  -- ^ Delivering output
  | TapError       -- ^ Error state
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Resonator configuration
data TapResonator = TapResonator
  { trFrequency   :: !Double    -- ^ Resonant frequency (Hz)
  , trQFactor     :: !Double    -- ^ Quality factor
  , trBandwidth   :: !Double    -- ^ Resonance bandwidth
  , trPhase       :: !Double    -- ^ Phase alignment [0, 2π]
  , trGain        :: !Double    -- ^ Resonator gain
  } deriving (Eq, Show)

-- | Energy buffer storage
data EnergyBuffer = EnergyBuffer
  { ebCapacity    :: !Double    -- ^ Maximum capacity
  , ebStored      :: !Double    -- ^ Current stored energy
  , ebChargeRate  :: !Double    -- ^ Charge rate
  , ebDischarge   :: !Double    -- ^ Discharge rate
  , ebEfficiency  :: !Double    -- ^ Storage efficiency [0, 1]
  } deriving (Eq, Show)

-- =============================================================================
-- Tap Operations
-- =============================================================================

-- | Initialize radiant tap
initializeTap :: Double -> RadiantTap
initializeTap capacity = RadiantTap
  { rtState = TapIdle
  , rtResonator = defaultResonator
  , rtBuffer = createBuffer capacity
  , rtCoherence = 0
  , rtOutput = 0
  , rtCycles = 0
  }

-- | Default resonator settings
defaultResonator :: TapResonator
defaultResonator = TapResonator
  { trFrequency = 528  -- Solfeggio frequency
  , trQFactor = phi * 100  -- Golden Q factor
  , trBandwidth = 10
  , trPhase = 0
  , trGain = 1.0
  }

-- | Create energy buffer
createBuffer :: Double -> EnergyBuffer
createBuffer capacity = EnergyBuffer
  { ebCapacity = capacity
  , ebStored = 0
  , ebChargeRate = capacity * 0.1
  , ebDischarge = capacity * 0.05
  , ebEfficiency = phiInverse
  }

-- | Activate tap for extraction
activateTap :: RadiantTap -> Double -> RadiantTap
activateTap tap coherence
  | rtState tap == TapError = tap
  | otherwise =
      tap { rtState = TapTuning
          , rtCoherence = coherence
          }

-- | Deactivate tap
deactivateTap :: RadiantTap -> RadiantTap
deactivateTap tap =
  tap { rtState = TapIdle, rtOutput = 0 }

-- | Get tap status string
tapStatus :: RadiantTap -> String
tapStatus tap =
  "Tap: " ++ show (rtState tap) ++
  " | Stored: " ++ show (round (ebStored (rtBuffer tap)) :: Int) ++
  " | Output: " ++ show (round (rtOutput tap * 100) :: Int) ++ "%" ++
  " | Cycles: " ++ show (rtCycles tap)

-- =============================================================================
-- Energy Extraction
-- =============================================================================

-- | Extract energy from field
extractEnergy :: RadiantTap -> Double -> (RadiantTap, Double)
extractEnergy tap fieldStrength
  | rtState tap /= TapExtracting && rtState tap /= TapTuning = (tap, 0)
  | otherwise =
      let resonator = rtResonator tap
          coherence = rtCoherence tap
          -- Energy extracted based on resonance and coherence
          resonanceBoost = trQFactor resonator / 100 * trGain resonator
          extracted = fieldStrength * coherence * resonanceBoost * phiInverse
          newTap = tap
            { rtState = TapExtracting
            , rtCycles = rtCycles tap + 1
            }
      in (newTap, extracted)

-- | Concentrate field energy
concentrateField :: RadiantTap -> Double -> RadiantTap
concentrateField tap concentration =
  let newCoherence = min 1.0 (rtCoherence tap + concentration * phiInverse)
  in tap { rtCoherence = newCoherence }

-- | Convert extracted energy to buffer
convertEnergy :: RadiantTap -> Double -> RadiantTap
convertEnergy tap extracted =
  let buffer = rtBuffer tap
      efficiency = ebEfficiency buffer
      converted = extracted * efficiency
      newStored = min (ebCapacity buffer) (ebStored buffer + converted)
      newBuffer = buffer { ebStored = newStored }
  in tap { rtBuffer = newBuffer, rtState = TapBuffering }

-- | Buffer energy for output
bufferEnergy :: RadiantTap -> RadiantTap
bufferEnergy tap =
  let buffer = rtBuffer tap
      newState = if ebStored buffer > ebCapacity buffer * 0.5
                 then TapOutputting
                 else TapBuffering
  in tap { rtState = newState }

-- =============================================================================
-- Resonator Control
-- =============================================================================

-- | Tune resonator to frequency
tuneResonator :: RadiantTap -> Double -> RadiantTap
tuneResonator tap targetFreq =
  let resonator = rtResonator tap
      -- Golden ratio frequency adjustment
      goldenFreq = targetFreq * phi / phi  -- Normalized
      newResonator = resonator { trFrequency = goldenFreq }
  in tap { rtResonator = newResonator, rtState = TapTuning }

-- | Get resonator frequency
resonatorFrequency :: RadiantTap -> Double
resonatorFrequency tap = trFrequency (rtResonator tap)

-- | Compute resonance quality
resonanceQuality :: RadiantTap -> Double
resonanceQuality tap =
  let resonator = rtResonator tap
      qFactor = trQFactor resonator
      bandwidth = trBandwidth resonator
      quality = if bandwidth > 0 then qFactor / bandwidth else qFactor
  in quality * phiInverse

-- =============================================================================
-- Efficiency
-- =============================================================================

-- | Tap efficiency metrics
data TapEfficiency = TapEfficiency
  { teExtraction   :: !Double    -- ^ Extraction efficiency [0, 1]
  , teConversion   :: !Double    -- ^ Conversion efficiency [0, 1]
  , teStorage      :: !Double    -- ^ Storage efficiency [0, 1]
  , teOutput       :: !Double    -- ^ Output efficiency [0, 1]
  , teOverall      :: !Double    -- ^ Overall efficiency [0, 1]
  } deriving (Eq, Show)

-- | Measure tap efficiency
measureEfficiency :: RadiantTap -> TapEfficiency
measureEfficiency tap =
  let buffer = rtBuffer tap
      resonator = rtResonator tap
      extraction = rtCoherence tap * trGain resonator
      conversion = ebEfficiency buffer
      storage = if ebCapacity buffer > 0
                then ebStored buffer / ebCapacity buffer
                else 0
      outputEff = if ebStored buffer > 0
                  then rtOutput tap / ebStored buffer
                  else 0
      overall = extraction * conversion * storage * max 0.1 outputEff
  in TapEfficiency
    { teExtraction = extraction
    , teConversion = conversion
    , teStorage = storage
    , teOutput = outputEff
    , teOverall = overall
    }

-- | Optimize tap settings
optimizeTap :: RadiantTap -> RadiantTap
optimizeTap tap =
  let resonator = rtResonator tap
      -- Optimize Q factor for golden ratio
      optimalQ = phi * 100
      -- Optimize gain
      optimalGain = phiInverse + rtCoherence tap * (phi - phiInverse)
      newResonator = resonator
        { trQFactor = optimalQ
        , trGain = optimalGain
        }
      buffer = rtBuffer tap
      newBuffer = buffer
        { ebEfficiency = min 1.0 (ebEfficiency buffer + 0.01)
        }
  in tap { rtResonator = newResonator, rtBuffer = newBuffer }

-- | Get efficiency history (simplified)
efficiencyHistory :: RadiantTap -> [(Int, Double)]
efficiencyHistory tap =
  [(rtCycles tap, teOverall (measureEfficiency tap))]

-- =============================================================================
-- Output
-- =============================================================================

-- | Energy output specification
data EnergyOutput = EnergyOutput
  { eoInstant     :: !Double    -- ^ Instantaneous output
  , eoSustained   :: !Double    -- ^ Sustainable output
  , eoPeak        :: !Double    -- ^ Peak output capability
  , eoStability   :: !Double    -- ^ Output stability [0, 1]
  } deriving (Eq, Show)

-- | Get current output
getOutput :: RadiantTap -> EnergyOutput
getOutput tap =
  let buffer = rtBuffer tap
      instant = rtOutput tap
      sustained = ebDischarge buffer * ebEfficiency buffer
      peak = ebStored buffer * phi
      stability = if rtCycles tap > 0
                  then rtCoherence tap * phiInverse
                  else 0
  in EnergyOutput
    { eoInstant = instant
    , eoSustained = sustained
    , eoPeak = peak
    , eoStability = stability
    }

-- | Get sustained output level
sustainedOutput :: RadiantTap -> Double
sustainedOutput tap =
  let buffer = rtBuffer tap
      baseOutput = ebDischarge buffer * ebEfficiency buffer
      coherenceBoost = rtCoherence tap * phiInverse
  in baseOutput * (1 + coherenceBoost)

-- | Get peak output level
peakOutput :: RadiantTap -> Double
peakOutput tap =
  let buffer = rtBuffer tap
      storedEnergy = ebStored buffer
      resonanceBoost = trQFactor (rtResonator tap) / 100
  in storedEnergy * resonanceBoost * phi

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Discharge buffer to output
_dischargeBuffer :: RadiantTap -> Double -> (RadiantTap, Double)
_dischargeBuffer tap amount =
  let buffer = rtBuffer tap
      available = min amount (ebStored buffer)
      newStored = ebStored buffer - available
      newBuffer = buffer { ebStored = newStored }
  in (tap { rtBuffer = newBuffer, rtOutput = available }, available)

-- | Charge buffer
_chargeBuffer :: RadiantTap -> Double -> RadiantTap
_chargeBuffer tap amount =
  let buffer = rtBuffer tap
      chargeAmount = min amount (ebChargeRate buffer)
      newStored = min (ebCapacity buffer) (ebStored buffer + chargeAmount)
      newBuffer = buffer { ebStored = newStored }
  in tap { rtBuffer = newBuffer }
