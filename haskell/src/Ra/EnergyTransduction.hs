{-|
Module      : Ra.EnergyTransduction
Description : Fragment emergence to power output transduction
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Models transduction of emergence events into electric or radiant energy
fields for use in physical or virtual systems, based on Tesla radiant
energy and Moray converter principles.

== Transduction Theory

=== Emergence to Energy Mapping

Scalar emergence can be transduced to:

* Pseudo-voltage (scalar tension differential)
* Radiant energy pulses (non-Hertzian)
* Coherent field states (for device interfaces)

=== Tesla Radiant Principles

* Abrupt discharge creates radiant pulses
* Pulse sharpness determines energy extraction
* Longitudinal waves carry energy without loss

=== Moray Converter Model

* Resonant extraction from scalar flux
* Antenna-like coupling to field
* Valve tubes for energy rectification
-}
module Ra.EnergyTransduction
  ( -- * Core Types
    TransductionField(..)
  , EnergyOutput(..)
  , OutputMode(..)
  , mkTransductionField

    -- * Emergence Conversion
  , convertEmergence
  , emergenceToVoltage
  , emergenceToRadiant
  , emergenceToCoherent

    -- * Device Interface
  , DeviceInterface(..)
  , InterfaceType(..)
  , connectDevice
  , outputToDevice

    -- * Power Bloom
  , PowerBloom(..)
  , BloomState(..)
  , initiateBloom
  , bloomIntensity
  , decayBloom

    -- * Safety Clamping
  , SafetyClamp(..)
  , clampOutput
  , checkStability
  , emergencyShutdown

    -- * Visualization
  , BloomVisual(..)
  , visualizeBloom
  , fieldMapColors

    -- * Efficiency
  , TransductionEfficiency(..)
  , computeEfficiency
  , optimizeTransduction
  ) where

import Ra.Constants.Extended
  ( phi, coherenceFloorPOR )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Transduction field state
data TransductionField = TransductionField
  { tfPotential     :: !Double        -- ^ Field potential [0,1]
  , tfFluxCoherence :: !Double        -- ^ Flux coherence [0,1]
  , tfOutputMode    :: !OutputMode    -- ^ Current output mode
  , tfCapacity      :: !Double        -- ^ Energy capacity
  , tfCharge        :: !Double        -- ^ Current charge level
  , tfEnabled       :: !Bool          -- ^ Transduction enabled
  } deriving (Eq, Show)

-- | Energy output
data EnergyOutput = EnergyOutput
  { eoVoltage     :: !Double          -- ^ Pseudo-voltage (V)
  , eoRadiance    :: !Double          -- ^ Radiant intensity [0,1]
  , eoCoherence   :: !Double          -- ^ Output coherence [0,1]
  , eoPulseRate   :: !Double          -- ^ Pulse frequency (Hz)
  , eoStable      :: !Bool            -- ^ Output stable
  } deriving (Eq, Show)

-- | Output mode
data OutputMode
  = VoltageMode     -- ^ Pseudo-electrical output
  | RadiantMode     -- ^ Radiant energy pulses
  | CoherentMode    -- ^ Coherent field state
  | HybridMode      -- ^ Combined outputs
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create transduction field
mkTransductionField :: Double -> OutputMode -> TransductionField
mkTransductionField potential mode = TransductionField
  { tfPotential = clamp01 potential
  , tfFluxCoherence = 0.5
  , tfOutputMode = mode
  , tfCapacity = 1.0
  , tfCharge = 0.0
  , tfEnabled = True
  }

-- =============================================================================
-- Emergence Conversion
-- =============================================================================

-- | Convert emergence score to energy output
convertEmergence :: Double -> Double -> TransductionField -> EnergyOutput
convertEmergence alpha flux field =
  case tfOutputMode field of
    VoltageMode  -> emergenceToVoltage alpha flux field
    RadiantMode  -> emergenceToRadiant alpha flux field
    CoherentMode -> emergenceToCoherent alpha flux field
    HybridMode   -> hybridConversion alpha flux field

-- | Convert to pseudo-voltage
emergenceToVoltage :: Double -> Double -> TransductionField -> EnergyOutput
emergenceToVoltage alpha flux field =
  let -- Voltage proportional to alpha and potential
      voltage = alpha * tfPotential field * 12.0  -- Scale to 0-12V range

      -- Radiance minimal in voltage mode
      radiance = flux * 0.1

      -- Coherence from flux
      coherence = flux * tfFluxCoherence field

      -- Pulse rate from coherence
      pulseRate = coherence * 100.0 + 10.0  -- 10-110 Hz

      -- Stability check
      stable = flux > coherenceFloorPOR && alpha > 0.3
  in EnergyOutput
      { eoVoltage = voltage
      , eoRadiance = radiance
      , eoCoherence = coherence
      , eoPulseRate = pulseRate
      , eoStable = stable
      }

-- | Convert to radiant energy
emergenceToRadiant :: Double -> Double -> TransductionField -> EnergyOutput
emergenceToRadiant alpha flux field =
  let -- Radiance from alpha squared (nonlinear)
      radiance = alpha * alpha * tfPotential field

      -- Voltage minimal in radiant mode
      voltage = alpha * 2.0

      -- Coherence critical for radiant
      coherence = flux * phi

      -- High-frequency pulses
      pulseRate = radiance * 1000.0 + 100.0  -- 100-1100 Hz

      stable = coherence > 0.5
  in EnergyOutput
      { eoVoltage = voltage
      , eoRadiance = clamp01 radiance
      , eoCoherence = clamp01 coherence
      , eoPulseRate = pulseRate
      , eoStable = stable
      }

-- | Convert to coherent field state
emergenceToCoherent :: Double -> Double -> TransductionField -> EnergyOutput
emergenceToCoherent alpha flux field =
  let -- Pure coherence output
      coherence = alpha * flux * tfFluxCoherence field * phi

      -- Minimal voltage/radiance
      voltage = coherence * 3.0
      radiance = coherence * 0.3

      -- Slow, stable pulses
      pulseRate = 7.83 * (1 + coherence)  -- Schumann base

      stable = coherence > coherenceFloorPOR
  in EnergyOutput
      { eoVoltage = voltage
      , eoRadiance = radiance
      , eoCoherence = clamp01 coherence
      , eoPulseRate = pulseRate
      , eoStable = stable
      }

-- Hybrid conversion
hybridConversion :: Double -> Double -> TransductionField -> EnergyOutput
hybridConversion alpha flux field =
  let v = emergenceToVoltage alpha flux field
      r = emergenceToRadiant alpha flux field
      c = emergenceToCoherent alpha flux field
  in EnergyOutput
      { eoVoltage = (eoVoltage v + eoVoltage r + eoVoltage c) / 3
      , eoRadiance = max (eoRadiance v) (max (eoRadiance r) (eoRadiance c))
      , eoCoherence = eoCoherence c  -- Use coherent mode coherence
      , eoPulseRate = (eoPulseRate v + eoPulseRate r) / 2
      , eoStable = eoStable v && eoStable r && eoStable c
      }

-- =============================================================================
-- Device Interface
-- =============================================================================

-- | Device interface for energy output
data DeviceInterface = DeviceInterface
  { diId          :: !String
  , diType        :: !InterfaceType
  , diMaxVoltage  :: !Double        -- ^ Maximum safe voltage
  , diMaxRadiance :: !Double        -- ^ Maximum radiance
  , diConnected   :: !Bool
  , diLastOutput  :: !(Maybe EnergyOutput)
  } deriving (Eq, Show)

-- | Interface type
data InterfaceType
  = BiometricFeedback   -- ^ Wearable/feedback device
  | VisualDisplay       -- ^ Visual output
  | HapticActuator      -- ^ Vibration/haptic
  | AudioTransducer     -- ^ Sound output
  | FieldEmitter        -- ^ Scalar field device
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Connect device
connectDevice :: String -> InterfaceType -> DeviceInterface
connectDevice deviceId devType = DeviceInterface
  { diId = deviceId
  , diType = devType
  , diMaxVoltage = maxVoltageFor devType
  , diMaxRadiance = maxRadianceFor devType
  , diConnected = True
  , diLastOutput = Nothing
  }

-- | Output to device
outputToDevice :: EnergyOutput -> DeviceInterface -> (DeviceInterface, Bool)
outputToDevice output device =
  if not (diConnected device)
  then (device, False)
  else
    let -- Clamp to device limits
        clampedVoltage = min (eoVoltage output) (diMaxVoltage device)
        clampedRadiance = min (eoRadiance output) (diMaxRadiance device)

        clampedOutput = output
          { eoVoltage = clampedVoltage
          , eoRadiance = clampedRadiance
          }

        success = eoStable output
    in (device { diLastOutput = Just clampedOutput }, success)

-- Device type limits
maxVoltageFor :: InterfaceType -> Double
maxVoltageFor t = case t of
  BiometricFeedback -> 5.0
  VisualDisplay     -> 24.0
  HapticActuator    -> 12.0
  AudioTransducer   -> 48.0
  FieldEmitter      -> 100.0

maxRadianceFor :: InterfaceType -> Double
maxRadianceFor t = case t of
  BiometricFeedback -> 0.5
  VisualDisplay     -> 1.0
  HapticActuator    -> 0.3
  AudioTransducer   -> 0.8
  FieldEmitter      -> 1.0

-- =============================================================================
-- Power Bloom
-- =============================================================================

-- | Power bloom visualization state
data PowerBloom = PowerBloom
  { pbIntensity   :: !Double        -- ^ Current intensity [0,1]
  , pbRadius      :: !Double        -- ^ Bloom radius
  , pbState       :: !BloomState
  , pbColor       :: !(Int, Int, Int) -- ^ RGB color
  , pbPulsePhase  :: !Double        -- ^ Pulse phase [0, 2*pi]
  } deriving (Eq, Show)

-- | Bloom state
data BloomState
  = BloomIdle       -- ^ No bloom
  | BloomBuilding   -- ^ Intensity increasing
  | BloomPeak       -- ^ Maximum intensity
  | BloomFading     -- ^ Intensity decreasing
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Initiate power bloom from energy output
initiateBloom :: EnergyOutput -> PowerBloom
initiateBloom output =
  let intensity = eoRadiance output * eoCoherence output
      radius = intensity * 2.0 + 0.5
      color = bloomColorFromOutput output
      state = if intensity > 0.1 then BloomBuilding else BloomIdle
  in PowerBloom
      { pbIntensity = intensity
      , pbRadius = radius
      , pbState = state
      , pbColor = color
      , pbPulsePhase = 0.0
      }

-- | Get bloom intensity
bloomIntensity :: PowerBloom -> Double
bloomIntensity = pbIntensity

-- | Decay bloom over time
decayBloom :: Double -> PowerBloom -> PowerBloom
decayBloom dt bloom =
  let decayRate = 0.5  -- 50% per second
      newIntensity = max 0 (pbIntensity bloom * (1 - decayRate * dt))
      newRadius = newIntensity * 2.0 + 0.5
      newState = if newIntensity < 0.05 then BloomIdle
                 else if newIntensity < pbIntensity bloom then BloomFading
                 else pbState bloom
      newPhase = wrapPhase (pbPulsePhase bloom + dt * 2 * pi * 0.5)
  in bloom
      { pbIntensity = newIntensity
      , pbRadius = newRadius
      , pbState = newState
      , pbPulsePhase = newPhase
      }

-- Color from output characteristics
bloomColorFromOutput :: EnergyOutput -> (Int, Int, Int)
bloomColorFromOutput output =
  let r = round (eoVoltage output / 12.0 * 255) `min` 255
      g = round (eoCoherence output * 255)
      b = round (eoRadiance output * 255)
  in (r, g, b)

-- =============================================================================
-- Safety Clamping
-- =============================================================================

-- | Safety clamp configuration
data SafetyClamp = SafetyClamp
  { scMaxVoltage    :: !Double      -- ^ Maximum voltage
  , scMaxRadiance   :: !Double      -- ^ Maximum radiance
  , scMinCoherence  :: !Double      -- ^ Minimum coherence for output
  , scMaxPulseRate  :: !Double      -- ^ Maximum pulse rate
  , scEnabled       :: !Bool
  } deriving (Eq, Show)

-- | Clamp output to safety limits
clampOutput :: SafetyClamp -> EnergyOutput -> EnergyOutput
clampOutput clamp output =
  if not (scEnabled clamp)
  then output
  else output
      { eoVoltage = min (scMaxVoltage clamp) (eoVoltage output)
      , eoRadiance = min (scMaxRadiance clamp) (eoRadiance output)
      , eoPulseRate = min (scMaxPulseRate clamp) (eoPulseRate output)
      , eoStable = eoStable output && eoCoherence output >= scMinCoherence clamp
      }

-- | Check output stability
checkStability :: EnergyOutput -> SafetyClamp -> Bool
checkStability output clamp =
  eoVoltage output <= scMaxVoltage clamp &&
  eoRadiance output <= scMaxRadiance clamp &&
  eoCoherence output >= scMinCoherence clamp &&
  eoPulseRate output <= scMaxPulseRate clamp

-- | Emergency shutdown - zero all outputs
emergencyShutdown :: TransductionField -> TransductionField
emergencyShutdown field = field
  { tfEnabled = False
  , tfCharge = 0.0
  }

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Bloom visualization data
data BloomVisual = BloomVisual
  { bvCenter      :: !(Double, Double)  -- ^ Center position
  , bvRadius      :: !Double
  , bvColor       :: !(Int, Int, Int)
  , bvGlow        :: !Double
  , bvPulsePhase  :: !Double
  } deriving (Eq, Show)

-- | Create visual from bloom
visualizeBloom :: PowerBloom -> (Double, Double) -> BloomVisual
visualizeBloom bloom center = BloomVisual
  { bvCenter = center
  , bvRadius = pbRadius bloom
  , bvColor = pbColor bloom
  , bvGlow = pbIntensity bloom
  , bvPulsePhase = pbPulsePhase bloom
  }

-- | Get field map colors
fieldMapColors :: TransductionField -> [(Double, Double, (Int, Int, Int))]
fieldMapColors field =
  let potential = tfPotential field
      coherence = tfFluxCoherence field
      -- Generate color grid
      gridSize = 10
      spacing = 1.0 / fromIntegral gridSize
      positions = [(fromIntegral x * spacing, fromIntegral y * spacing)
                  | x <- [0..gridSize-1], y <- [0..gridSize-1]]
      colorAt (px, py) =
        let dist = sqrt ((px - 0.5)^(2::Int) + (py - 0.5)^(2::Int))
            intensity = potential * (1 - dist)
            r = round (intensity * 255 * coherence)
            g = round (intensity * 200)
            b = round ((1 - intensity) * 150)
        in (px, py, (r `min` 255, g `min` 255, b `min` 255))
  in map colorAt positions

-- =============================================================================
-- Efficiency
-- =============================================================================

-- | Transduction efficiency metrics
data TransductionEfficiency = TransductionEfficiency
  { teInputEnergy   :: !Double      -- ^ Input energy
  , teOutputEnergy  :: !Double      -- ^ Output energy
  , teEfficiency    :: !Double      -- ^ Efficiency ratio [0,1]
  , teLossMode      :: !String      -- ^ Primary loss mechanism
  } deriving (Eq, Show)

-- | Compute transduction efficiency
computeEfficiency :: Double -> Double -> EnergyOutput -> TransductionEfficiency
computeEfficiency inputAlpha inputFlux output =
  let inputEnergy = inputAlpha * inputFlux
      outputEnergy = eoVoltage output / 12.0 * eoCoherence output
      efficiency = if inputEnergy > 0 then outputEnergy / inputEnergy else 0
      lossMode = if eoCoherence output < 0.5 then "coherence_loss"
                 else if eoStable output then "minimal" else "instability"
  in TransductionEfficiency
      { teInputEnergy = inputEnergy
      , teOutputEnergy = clamp01 outputEnergy
      , teEfficiency = clamp01 efficiency
      , teLossMode = lossMode
      }

-- | Optimize transduction parameters
optimizeTransduction :: TransductionField -> Double -> TransductionField
optimizeTransduction field targetEfficiency =
  let currentPotential = tfPotential field
      currentCoherence = tfFluxCoherence field

      -- Adjust toward target
      adjustment = (targetEfficiency - (currentPotential * currentCoherence)) * 0.1
      newPotential = clamp01 (currentPotential + adjustment)
      newCoherence = clamp01 (currentCoherence + adjustment * 0.5)
  in field
      { tfPotential = newPotential
      , tfFluxCoherence = newCoherence
      }

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)

-- | Wrap phase to [0, 2*pi]
wrapPhase :: Double -> Double
wrapPhase p
  | p < 0 = wrapPhase (p + 2 * pi)
  | p >= 2 * pi = wrapPhase (p - 2 * pi)
  | otherwise = p
