{-|
Module      : Ra.Healing.CoilMatrix
Description : Cellular tuning coil matrix for localized healing fields
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Extracts coil arrangement specifications from Hubbard coil generators and
electroculture parameters to optimize localized healing fields. Includes
entangled recovery fields with mirror biometric stabilization.

== Coil Matrix Theory

=== Hubbard Configuration

* Central core with harmonic rods
* Concentric coil windings
* 3:1 step-up resonance ratio
* Ground wire resonance network

=== Electroculture Integration

* Earth antenna coupling
* Atmospheric ion collection
* Plant-growth frequency bands
* Paramagnetic mineral resonance
-}
module Ra.Healing.CoilMatrix
  ( -- * Core Types
    CoilMatrix(..)
  , CoilUnit(..)
  , CoilConfiguration(..)
  , FieldPattern(..)

    -- * Matrix Construction
  , createMatrix
  , addCoil
  , removeCoil
  , configureMatrix

    -- * Field Generation
  , generateField
  , fieldAtPoint
  , fieldStrength

    -- * Resonance Control
  , setResonance
  , harmonicTuning
  , stepUpRatio

    -- * Healing Protocols
  , HealingProtocol(..)
  , selectProtocol
  , applyProtocol

    -- * Biometric Stabilization
  , mirrorStabilize
  , entangledRecovery
  , syncBiometrics
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete coil matrix configuration
data CoilMatrix = CoilMatrix
  { cmCoils       :: ![CoilUnit]           -- ^ Individual coil units
  , cmConfig      :: !CoilConfiguration    -- ^ Overall configuration
  , cmFieldCenter :: !(Double, Double, Double)  -- ^ Field center point
  , cmFieldRadius :: !Double               -- ^ Effective field radius
  , cmResonance   :: !Double               -- ^ Primary resonance (Hz)
  , cmPower       :: !Double               -- ^ Power level [0, 1]
  } deriving (Eq, Show)

-- | Individual coil unit
data CoilUnit = CoilUnit
  { cuPosition    :: !(Double, Double, Double)  -- ^ Coil position
  , cuOrientation :: !(Double, Double, Double)  -- ^ Axis orientation
  , cuWindings    :: !Int                  -- ^ Number of windings
  , cuRadius      :: !Double               -- ^ Coil radius
  , cuWireGauge   :: !Int                  -- ^ Wire gauge (AWG)
  , cuFrequency   :: !Double               -- ^ Operating frequency
  , cuPhase       :: !Double               -- ^ Phase offset [0, 2pi]
  } deriving (Eq, Show)

-- | Matrix configuration type
data CoilConfiguration
  = ConfigHubbard      -- ^ Hubbard transformer style
  | ConfigTesla        -- ^ Tesla coil arrangement
  | ConfigLakhovsky    -- ^ Lakhovsky MWO rings
  | ConfigToroidal     -- ^ Toroidal field pattern
  | ConfigPlanar       -- ^ Flat spiral arrangement
  | ConfigCustom       -- ^ User-defined
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Field pattern output
data FieldPattern = FieldPattern
  { fpShape      :: !PatternShape    -- ^ Pattern geometry
  , fpIntensity  :: !Double          -- ^ Field intensity
  , fpFrequency  :: !Double          -- ^ Dominant frequency
  , fpPhase      :: !Double          -- ^ Phase state
  , fpPolarization :: !Polarization  -- ^ Field polarization
  } deriving (Eq, Show)

-- | Pattern shape types
data PatternShape
  = ShapeSphere
  | ShapeTorus
  | ShapeCone
  | ShapeSpiral
  | ShapePlanar
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Field polarization
data Polarization
  = PolarLinear
  | PolarCircularCW
  | PolarCircularCCW
  | PolarElliptical
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Matrix Construction
-- =============================================================================

-- | Create empty coil matrix
createMatrix :: CoilConfiguration -> CoilMatrix
createMatrix config = CoilMatrix
  { cmCoils = []
  , cmConfig = config
  , cmFieldCenter = (0, 0, 0)
  , cmFieldRadius = 1.0
  , cmResonance = 7.83
  , cmPower = 0.5
  }

-- | Add coil to matrix
addCoil :: CoilMatrix -> CoilUnit -> CoilMatrix
addCoil matrix coil =
  matrix { cmCoils = coil : cmCoils matrix }

-- | Remove coil by position
removeCoil :: CoilMatrix -> (Double, Double, Double) -> CoilMatrix
removeCoil matrix pos =
  matrix { cmCoils = filter ((/= pos) . cuPosition) (cmCoils matrix) }

-- | Configure matrix with preset
configureMatrix :: CoilMatrix -> CoilConfiguration -> CoilMatrix
configureMatrix matrix config =
  let coils = generateCoilsForConfig config
  in matrix { cmCoils = coils, cmConfig = config }

-- =============================================================================
-- Field Generation
-- =============================================================================

-- | Generate field pattern
generateField :: CoilMatrix -> FieldPattern
generateField matrix =
  let shape = configToShape (cmConfig matrix)
      intensity = calculateIntensity matrix
      freq = cmResonance matrix
      phase = averagePhase (cmCoils matrix)
      polar = configToPolarization (cmConfig matrix)
  in FieldPattern shape intensity freq phase polar

-- | Calculate field at specific point
fieldAtPoint :: CoilMatrix -> (Double, Double, Double) -> Double
fieldAtPoint matrix (px, py, pz) =
  let (cx, cy, cz) = cmFieldCenter matrix
      dist = sqrt ((px-cx)^(2::Int) + (py-cy)^(2::Int) + (pz-cz)^(2::Int))
      falloff = if dist > 0 then cmFieldRadius matrix / dist else 1
      coilContributions = sum [coilFieldAt coil (px, py, pz) | coil <- cmCoils matrix]
  in min 1.0 (falloff * coilContributions * cmPower matrix)

-- | Get overall field strength
fieldStrength :: CoilMatrix -> Double
fieldStrength matrix =
  let coilCount = fromIntegral (length (cmCoils matrix)) :: Double
      powerFactor = cmPower matrix
      configFactor = case cmConfig matrix of
        ConfigHubbard -> phi
        ConfigTesla -> phi * phi
        ConfigLakhovsky -> phi * phiInverse
        _ -> 1.0
  in coilCount * powerFactor * configFactor / 10

-- =============================================================================
-- Resonance Control
-- =============================================================================

-- | Set primary resonance frequency
setResonance :: CoilMatrix -> Double -> CoilMatrix
setResonance matrix freq =
  let adjustedCoils = map (\c -> c { cuFrequency = freq }) (cmCoils matrix)
  in matrix { cmCoils = adjustedCoils, cmResonance = freq }

-- | Apply harmonic tuning
harmonicTuning :: CoilMatrix -> Int -> CoilMatrix
harmonicTuning matrix harmonic =
  let baseFreq = cmResonance matrix
      harmonicFreq = baseFreq * fromIntegral harmonic
      tuned = zipWith (tuneCoil harmonicFreq) (cmCoils matrix) [0..]
  in matrix { cmCoils = tuned }

-- | Get step-up ratio for transformer config
stepUpRatio :: CoilMatrix -> Double
stepUpRatio matrix =
  case cmConfig matrix of
    ConfigHubbard -> 3.0  -- Classic Hubbard 3:1
    ConfigTesla -> phi * phi  -- Golden ratio squared
    _ -> 1.0

-- =============================================================================
-- Healing Protocols
-- =============================================================================

-- | Healing protocol specification
data HealingProtocol = HealingProtocol
  { hpName        :: !String          -- ^ Protocol name
  , hpFrequencies :: ![Double]        -- ^ Frequency sequence
  , hpDurations   :: ![Int]           -- ^ Duration per frequency (seconds)
  , hpIntensity   :: !Double          -- ^ Overall intensity
  , hpTarget      :: !String          -- ^ Target area/condition
  } deriving (Eq, Show)

-- | Select appropriate protocol
selectProtocol :: String -> HealingProtocol
selectProtocol target = case target of
  "cellular" -> HealingProtocol "Cellular Repair" [528, 639, 741] [120, 120, 120] 0.6 target
  "nervous" -> HealingProtocol "Neural Balance" [396, 417, 963] [180, 180, 180] 0.5 target
  "immune" -> HealingProtocol "Immune Support" [528, 852, 2128] [150, 150, 150] 0.5 target
  "circulation" -> HealingProtocol "Circulation" [639, 741, 852] [120, 120, 120] 0.6 target
  _ -> HealingProtocol "General Wellness" [528] [300] 0.5 target

-- | Apply protocol to matrix
applyProtocol :: CoilMatrix -> HealingProtocol -> [CoilMatrix]
applyProtocol matrix protocol =
  [setResonance (matrix { cmPower = hpIntensity protocol }) freq
  | freq <- hpFrequencies protocol]

-- =============================================================================
-- Biometric Stabilization
-- =============================================================================

-- | Mirror stabilizer configuration
data MirrorStabilizer = MirrorStabilizer
  { msSourceMatrix :: !CoilMatrix   -- ^ Source field
  , msTargetMatrix :: !CoilMatrix   -- ^ Target (user) field
  , msCoupling     :: !Double       -- ^ Coupling strength
  , msPhaseMatch   :: !Double       -- ^ Phase matching accuracy
  } deriving (Eq, Show)

-- | Create mirror stabilization field
mirrorStabilize :: CoilMatrix -> Double -> MirrorStabilizer
mirrorStabilize sourceMatrix coupling =
  let targetMatrix = invertPhases sourceMatrix
  in MirrorStabilizer
    { msSourceMatrix = sourceMatrix
    , msTargetMatrix = targetMatrix
    , msCoupling = coupling
    , msPhaseMatch = 0
    }

-- | Entangled recovery field generation
entangledRecovery :: CoilMatrix -> CoilMatrix -> Double -> (CoilMatrix, CoilMatrix)
entangledRecovery matrix1 matrix2 entanglementStrength =
  let avgResonance = (cmResonance matrix1 + cmResonance matrix2) / 2
      avgPower = (cmPower matrix1 + cmPower matrix2) / 2 * entanglementStrength
      new1 = matrix1 { cmResonance = avgResonance, cmPower = avgPower }
      new2 = matrix2 { cmResonance = avgResonance, cmPower = avgPower }
  in (new1, new2)

-- | Synchronize with biometric data
syncBiometrics :: CoilMatrix -> Double -> Double -> CoilMatrix
syncBiometrics matrix hrv coherence =
  let syncFactor = hrv * coherence
      newPower = cmPower matrix * (0.8 + syncFactor * 0.4)
      newResonance = cmResonance matrix * (1 + (coherence - 0.5) * 0.1)
  in matrix { cmPower = min 1.0 newPower, cmResonance = newResonance }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Generate coils for configuration
generateCoilsForConfig :: CoilConfiguration -> [CoilUnit]
generateCoilsForConfig ConfigHubbard =
  [CoilUnit (0, 0, 0) (0, 0, 1) 100 0.1 18 528 0
  ,CoilUnit (0.1, 0, 0) (0, 0, 1) 33 0.15 20 528 (pi/3)
  ,CoilUnit (-0.1, 0, 0) (0, 0, 1) 33 0.15 20 528 (2*pi/3)
  ]
generateCoilsForConfig ConfigTesla =
  [CoilUnit (0, 0, 0) (0, 0, 1) 200 0.05 22 1000000 0]
generateCoilsForConfig ConfigLakhovsky =
  [CoilUnit (0, 0, fromIntegral i * 0.01) (0, 0, 1) 1 (0.05 + fromIntegral i * 0.01) 16 (1000 * phi ** fromIntegral i) 0
  | i <- [0..5 :: Int]]
generateCoilsForConfig _ = []

-- | Convert config to shape
configToShape :: CoilConfiguration -> PatternShape
configToShape ConfigHubbard = ShapeTorus
configToShape ConfigTesla = ShapeCone
configToShape ConfigLakhovsky = ShapeSphere
configToShape ConfigToroidal = ShapeTorus
configToShape ConfigPlanar = ShapePlanar
configToShape ConfigCustom = ShapeSphere

-- | Convert config to polarization
configToPolarization :: CoilConfiguration -> Polarization
configToPolarization ConfigTesla = PolarLinear
configToPolarization ConfigLakhovsky = PolarCircularCW
configToPolarization _ = PolarElliptical

-- | Calculate matrix intensity
calculateIntensity :: CoilMatrix -> Double
calculateIntensity matrix =
  let n = fromIntegral (length (cmCoils matrix)) :: Double
  in cmPower matrix * sqrt n / 3

-- | Average phase of all coils
averagePhase :: [CoilUnit] -> Double
averagePhase [] = 0
averagePhase coils =
  sum (map cuPhase coils) / fromIntegral (length coils)

-- | Calculate coil field contribution at point
coilFieldAt :: CoilUnit -> (Double, Double, Double) -> Double
coilFieldAt coil (px, py, pz) =
  let (cx, cy, cz) = cuPosition coil
      dist = sqrt ((px-cx)^(2::Int) + (py-cy)^(2::Int) + (pz-cz)^(2::Int))
      windingFactor = fromIntegral (cuWindings coil) / 100
  in if dist > 0 then windingFactor / dist else windingFactor

-- | Invert phases for mirror field
invertPhases :: CoilMatrix -> CoilMatrix
invertPhases matrix =
  let inverted = map (\c -> c { cuPhase = cuPhase c + pi }) (cmCoils matrix)
  in matrix { cmCoils = inverted }

-- | Tune single coil to harmonic
tuneCoil :: Double -> CoilUnit -> Int -> CoilUnit
tuneCoil baseFreq coil idx =
  coil { cuFrequency = baseFreq, cuPhase = fromIntegral idx * pi / 6 }
