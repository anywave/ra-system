{-|
Module      : Ra.Simulation.AetherLab
Description : Scalar field simulation laboratory environment
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements a simulation laboratory for scalar field experiments.
AetherLab provides a sandboxed environment for testing coherence
dynamics, emergence patterns, and field interactions.

== Simulation Architecture

=== Lab Components

1. Field Generator: Creates test scalar fields
2. Coherence Injector: Injects coherence patterns
3. Observer Grid: Monitors field state
4. Event Recorder: Logs simulation events

=== Experiment Types

* Coherence Dynamics: Study coherence evolution
* Emergence Testing: Test fragment emergence
* Interference Patterns: Study wave interactions
* Stability Analysis: Field stability testing
-}
module Ra.Simulation.AetherLab
  ( -- * Core Types
    AetherLab(..)
  , LabState(..)
  , SimulationConfig(..)
  , Experiment(..)

    -- * Lab Management
  , createLab
  , resetLab
  , configureLab

    -- * Simulation Control
  , runSimulation
  , stepSimulation
  , pauseSimulation
  , resumeSimulation

    -- * Field Generation
  , FieldGenerator(..)
  , generateTestField
  , injectCoherence
  , perturbField

    -- * Observation
  , ObserverGrid(..)
  , createObserverGrid
  , sampleField
  , recordObservation

    -- * Experiments
  , ExperimentResult(..)
  , runExperiment
  , analyzeResult
  , compareResults

    -- * Event Logging
  , SimEvent(..)
  , logEvent
  , getEventLog
  , clearLog

    -- * Metrics
  , LabMetrics(..)
  , computeLabMetrics
  , fieldEnergy
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Aether simulation laboratory
data AetherLab = AetherLab
  { alState      :: !LabState               -- ^ Current lab state
  , alConfig     :: !SimulationConfig       -- ^ Configuration
  , alField      :: !SimField               -- ^ Current field state
  , alObservers  :: !ObserverGrid           -- ^ Observer grid
  , alEvents     :: ![SimEvent]             -- ^ Event log
  , alTick       :: !Int                    -- ^ Simulation tick
  } deriving (Eq, Show)

-- | Lab operational state
data LabState
  = LabIdle       -- ^ Ready for experiment
  | LabRunning    -- ^ Simulation in progress
  | LabPaused     -- ^ Simulation paused
  | LabComplete   -- ^ Experiment complete
  | LabError      -- ^ Error state
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Simulation configuration
data SimulationConfig = SimulationConfig
  { scGridSize    :: !(Int, Int, Int)   -- ^ Simulation grid dimensions
  , scTimeStep    :: !Double            -- ^ Time step (φ^n scale)
  , scMaxTicks    :: !Int               -- ^ Maximum simulation ticks
  , scBoundary    :: !BoundaryType      -- ^ Boundary conditions
  , scPrecision   :: !Double            -- ^ Numerical precision
  } deriving (Eq, Show)

-- | Boundary condition types
data BoundaryType
  = BoundaryPeriodic   -- ^ Wrap-around boundaries
  | BoundaryReflective -- ^ Reflective boundaries
  | BoundaryAbsorbing  -- ^ Absorbing boundaries
  | BoundaryFixed      -- ^ Fixed value boundaries
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Experiment definition
data Experiment = Experiment
  { exName        :: !String            -- ^ Experiment name
  , exType        :: !ExperimentType    -- ^ Experiment type
  , exDuration    :: !Int               -- ^ Duration in ticks
  , exParameters  :: !(Map String Double)  -- ^ Experiment parameters
  , exInitializer :: !FieldInitializer  -- ^ Initial field setup
  } deriving (Eq, Show)

-- | Experiment types
data ExperimentType
  = ExpCoherence     -- ^ Coherence dynamics
  | ExpEmergence     -- ^ Fragment emergence
  | ExpInterference  -- ^ Wave interference
  | ExpStability     -- ^ Stability analysis
  | ExpResonance     -- ^ Resonance testing
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Field initializer
data FieldInitializer
  = InitUniform !Double          -- ^ Uniform coherence
  | InitGaussian !Double !Double -- ^ Gaussian (center, width)
  | InitRandom !Double           -- ^ Random with max amplitude
  | InitVortex !Double !Double   -- ^ Vortex (strength, radius)
  | InitCustom ![(Int, Int, Int, Double)]  -- ^ Custom point values
  deriving (Eq, Show)

-- =============================================================================
-- Internal Field Type
-- =============================================================================

-- | Simulated field state
data SimField = SimField
  { sfValues    :: !(Map (Int, Int, Int) Double)  -- ^ Field values
  , sfEnergy    :: !Double                         -- ^ Total energy
  , sfCoherence :: !Double                         -- ^ Average coherence
  , sfGradient  :: !Double                         -- ^ Max gradient
  } deriving (Eq, Show)

-- =============================================================================
-- Lab Management
-- =============================================================================

-- | Create new laboratory
createLab :: SimulationConfig -> AetherLab
createLab config =
  let emptyField = initializeField config (InitUniform 0)
      observers = createObserverGrid (scGridSize config)
  in AetherLab
    { alState = LabIdle
    , alConfig = config
    , alField = emptyField
    , alObservers = observers
    , alEvents = []
    , alTick = 0
    }

-- | Reset laboratory to initial state
resetLab :: AetherLab -> AetherLab
resetLab lab =
  let emptyField = initializeField (alConfig lab) (InitUniform 0)
  in lab
    { alState = LabIdle
    , alField = emptyField
    , alEvents = []
    , alTick = 0
    }

-- | Update laboratory configuration
configureLab :: AetherLab -> SimulationConfig -> AetherLab
configureLab lab config =
  let newObservers = createObserverGrid (scGridSize config)
  in lab
    { alConfig = config
    , alObservers = newObservers
    }

-- =============================================================================
-- Simulation Control
-- =============================================================================

-- | Run complete simulation
runSimulation :: AetherLab -> Experiment -> (AetherLab, ExperimentResult)
runSimulation lab experiment =
  let initialField = initializeField (alConfig lab) (exInitializer experiment)
      labWithField = lab { alField = initialField, alState = LabRunning }
      finalLab = runTicks labWithField (exDuration experiment)
      result = generateResult experiment finalLab
  in (finalLab { alState = LabComplete }, result)

-- | Step simulation by one tick
stepSimulation :: AetherLab -> AetherLab
stepSimulation lab
  | alState lab /= LabRunning = lab
  | alTick lab >= scMaxTicks (alConfig lab) = lab { alState = LabComplete }
  | otherwise =
      let newField = evolveField (alConfig lab) (alField lab)
          newTick = alTick lab + 1
          event = TickEvent newTick (sfCoherence newField)
      in lab
        { alField = newField
        , alTick = newTick
        , alEvents = event : alEvents lab
        }

-- | Pause running simulation
pauseSimulation :: AetherLab -> AetherLab
pauseSimulation lab
  | alState lab == LabRunning = lab { alState = LabPaused }
  | otherwise = lab

-- | Resume paused simulation
resumeSimulation :: AetherLab -> AetherLab
resumeSimulation lab
  | alState lab == LabPaused = lab { alState = LabRunning }
  | otherwise = lab

-- =============================================================================
-- Field Generation
-- =============================================================================

-- | Field generator configuration
data FieldGenerator = FieldGenerator
  { fgType      :: !GeneratorType   -- ^ Generator type
  , fgAmplitude :: !Double          -- ^ Output amplitude
  , fgFrequency :: !Double          -- ^ Oscillation frequency
  , fgPhase     :: !Double          -- ^ Phase offset
  } deriving (Eq, Show)

-- | Generator types
data GeneratorType
  = GenConstant    -- ^ Constant field
  | GenSinusoidal  -- ^ Sinusoidal pattern
  | GenPulsed      -- ^ Pulsed output
  | GenNoise       -- ^ Random noise
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Generate test field from generator
generateTestField :: FieldGenerator -> SimulationConfig -> SimField
generateTestField gen config =
  let (nx, ny, nz) = scGridSize config
      points = [(x, y, z) | x <- [0..nx-1], y <- [0..ny-1], z <- [0..nz-1]]
      values = Map.fromList [(p, generateValue gen p) | p <- points]
      energy = sum (Map.elems values) * phi
      coherence = averageValue values
  in SimField
    { sfValues = values
    , sfEnergy = energy
    , sfCoherence = coherence
    , sfGradient = computeMaxGradient values
    }

-- | Inject coherence at location
injectCoherence :: AetherLab -> (Int, Int, Int) -> Double -> AetherLab
injectCoherence lab pos coherence =
  let field = alField lab
      newValues = Map.insert pos coherence (sfValues field)
      newEnergy = sfEnergy field + coherence
      newCoherence = averageValue newValues
      newField = field
        { sfValues = newValues
        , sfEnergy = newEnergy
        , sfCoherence = newCoherence
        }
      event = InjectionEvent pos coherence
  in lab
    { alField = newField
    , alEvents = event : alEvents lab
    }

-- | Perturb field with random noise
perturbField :: AetherLab -> Double -> AetherLab
perturbField lab amplitude =
  let field = alField lab
      -- Simple perturbation: add scaled value to each point
      perturbValue v = v + amplitude * (phi - 1) * sin (v * 10)
      newValues = Map.map perturbValue (sfValues field)
      newField = field
        { sfValues = newValues
        , sfCoherence = averageValue newValues
        }
      event = PerturbEvent amplitude
  in lab { alField = newField, alEvents = event : alEvents lab }

-- =============================================================================
-- Observation
-- =============================================================================

-- | Observer grid for field monitoring
data ObserverGrid = ObserverGrid
  { ogPoints     :: ![(Int, Int, Int)]     -- ^ Observer positions
  , ogReadings   :: !(Map (Int, Int, Int) [Double])  -- ^ Historical readings
  , ogResolution :: !Int                    -- ^ Sampling resolution
  } deriving (Eq, Show)

-- | Create observer grid
createObserverGrid :: (Int, Int, Int) -> ObserverGrid
createObserverGrid (nx, ny, nz) =
  let step = max 1 (min nx (min ny nz) `div` 10)
      points = [(x, y, z) | x <- [0, step .. nx-1]
                          , y <- [0, step .. ny-1]
                          , z <- [0, step .. nz-1]]
  in ObserverGrid
    { ogPoints = points
    , ogReadings = Map.empty
    , ogResolution = step
    }

-- | Sample field at observer points
sampleField :: AetherLab -> [(Int, Int, Int, Double)]
sampleField lab =
  let field = alField lab
      observers = alObservers lab
  in [(x, y, z, Map.findWithDefault 0 (x, y, z) (sfValues field))
     | (x, y, z) <- ogPoints observers]

-- | Record observation to history
recordObservation :: AetherLab -> AetherLab
recordObservation lab =
  let samples = sampleField lab
      observers = alObservers lab
      updateReadings ogs =
        foldr (\(x, y, z, v) m ->
                 Map.insertWith (++) (x, y, z) [v] m)
              (ogReadings ogs) samples
      newObservers = observers { ogReadings = updateReadings observers }
  in lab { alObservers = newObservers }

-- =============================================================================
-- Experiments
-- =============================================================================

-- | Experiment result
data ExperimentResult = ExperimentResult
  { erName          :: !String               -- ^ Experiment name
  , erSuccess       :: !Bool                 -- ^ Completion success
  , erFinalEnergy   :: !Double               -- ^ Final field energy
  , erFinalCoherence :: !Double              -- ^ Final coherence
  , erPeakCoherence :: !Double               -- ^ Peak coherence observed
  , erStability     :: !Double               -- ^ Stability metric
  , erDuration      :: !Int                  -- ^ Actual duration
  } deriving (Eq, Show)

-- | Run a defined experiment
runExperiment :: AetherLab -> Experiment -> (AetherLab, ExperimentResult)
runExperiment = runSimulation

-- | Analyze experiment result
analyzeResult :: ExperimentResult -> String
analyzeResult result =
  let efficiency = erFinalCoherence result / max 0.001 (erFinalEnergy result)
      rating = if efficiency > phi then "Excellent"
               else if efficiency > phiInverse then "Good"
               else if efficiency > phiInverse * phiInverse then "Fair"
               else "Poor"
  in "Experiment: " ++ erName result ++
     "\nSuccess: " ++ show (erSuccess result) ++
     "\nRating: " ++ rating ++
     "\nEfficiency: " ++ show efficiency

-- | Compare two experiment results
compareResults :: ExperimentResult -> ExperimentResult -> String
compareResults r1 r2 =
  let energyDiff = erFinalEnergy r2 - erFinalEnergy r1
      cohDiff = erFinalCoherence r2 - erFinalCoherence r1
  in "Comparison: " ++ erName r1 ++ " vs " ++ erName r2 ++
     "\nEnergy difference: " ++ show energyDiff ++
     "\nCoherence difference: " ++ show cohDiff

-- =============================================================================
-- Event Logging
-- =============================================================================

-- | Simulation event
data SimEvent
  = TickEvent !Int !Double           -- ^ Tick and coherence
  | InjectionEvent !(Int, Int, Int) !Double  -- ^ Coherence injection
  | PerturbEvent !Double             -- ^ Perturbation
  | ThresholdEvent !String !Double   -- ^ Threshold crossing
  | EmergenceEvent !(Int, Int, Int)  -- ^ Fragment emergence
  deriving (Eq, Show)

-- | Log custom event
logEvent :: AetherLab -> SimEvent -> AetherLab
logEvent lab event = lab { alEvents = event : alEvents lab }

-- | Get event log
getEventLog :: AetherLab -> [SimEvent]
getEventLog = alEvents

-- | Clear event log
clearLog :: AetherLab -> AetherLab
clearLog lab = lab { alEvents = [] }

-- =============================================================================
-- Metrics
-- =============================================================================

-- | Laboratory metrics
data LabMetrics = LabMetrics
  { lmFieldEnergy    :: !Double    -- ^ Total field energy
  , lmAvgCoherence   :: !Double    -- ^ Average coherence
  , lmMaxGradient    :: !Double    -- ^ Maximum gradient
  , lmEventCount     :: !Int       -- ^ Number of events
  , lmTickRate       :: !Double    -- ^ Simulation tick rate
  } deriving (Eq, Show)

-- | Compute laboratory metrics
computeLabMetrics :: AetherLab -> LabMetrics
computeLabMetrics lab =
  let field = alField lab
      tickRate = if alTick lab > 0
                 then fromIntegral (alTick lab) / phi
                 else 0
  in LabMetrics
    { lmFieldEnergy = sfEnergy field
    , lmAvgCoherence = sfCoherence field
    , lmMaxGradient = sfGradient field
    , lmEventCount = length (alEvents lab)
    , lmTickRate = tickRate
    }

-- | Compute field energy
fieldEnergy :: AetherLab -> Double
fieldEnergy lab = sfEnergy (alField lab)

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Initialize field from initializer
initializeField :: SimulationConfig -> FieldInitializer -> SimField
initializeField config initializer =
  let (nx, ny, nz) = scGridSize config
      points = [(x, y, z) | x <- [0..nx-1], y <- [0..ny-1], z <- [0..nz-1]]
      values = Map.fromList [(p, initValue initializer p (nx, ny, nz)) | p <- points]
      energy = sum (Map.elems values)
      coherence = averageValue values
  in SimField
    { sfValues = values
    , sfEnergy = energy
    , sfCoherence = coherence
    , sfGradient = computeMaxGradient values
    }

-- | Initialize single point value
initValue :: FieldInitializer -> (Int, Int, Int) -> (Int, Int, Int) -> Double
initValue (InitUniform v) _ _ = v
initValue (InitGaussian center width) (x, y, z) (nx, ny, nz) =
  let cx = fromIntegral nx / 2
      cy = fromIntegral ny / 2
      cz = fromIntegral nz / 2
      dist = sqrt ((fromIntegral x - cx)^(2::Int) +
                   (fromIntegral y - cy)^(2::Int) +
                   (fromIntegral z - cz)^(2::Int))
  in center * exp (-(dist / width)^(2::Int))
initValue (InitRandom maxAmp) (x, y, z) _ =
  let pseudo = sin (fromIntegral x * phi + fromIntegral y * phiInverse + fromIntegral z)
  in maxAmp * (pseudo + 1) / 2
initValue (InitVortex strength radius) (x, y, z) (nx, ny, _nz) =
  let cx = fromIntegral nx / 2
      cy = fromIntegral ny / 2
      dist = sqrt ((fromIntegral x - cx)^(2::Int) + (fromIntegral y - cy)^(2::Int))
      angle = atan2 (fromIntegral y - cy) (fromIntegral x - cx)
  in if dist < radius
     then strength * sin (angle + fromIntegral z * 0.1)
     else 0
initValue (InitCustom points) (x, y, z) _ =
  case lookup (x, y, z) [(p, v) | (px, py, pz, v) <- points, let p = (px, py, pz)] of
    Just v  -> v
    Nothing -> 0

-- | Generate value from generator
generateValue :: FieldGenerator -> (Int, Int, Int) -> Double
generateValue gen (x, y, z) =
  let pos = fromIntegral x + fromIntegral y * 0.1 + fromIntegral z * 0.01
  in case fgType gen of
    GenConstant   -> fgAmplitude gen
    GenSinusoidal -> fgAmplitude gen * sin (fgFrequency gen * pos + fgPhase gen)
    GenPulsed     -> if (floor (pos * fgFrequency gen) :: Int) `mod` 2 == 0
                     then fgAmplitude gen else 0
    GenNoise      -> fgAmplitude gen * sin (pos * phi * fgFrequency gen)

-- | Evolve field one time step
evolveField :: SimulationConfig -> SimField -> SimField
evolveField config field =
  let dt = scTimeStep config
      decay = phiInverse * dt
      evolveValue v = v * (1 - decay) + sin (v * phi) * decay * 0.1
      newValues = Map.map evolveValue (sfValues field)
      newEnergy = sum (Map.elems newValues)
      newCoherence = averageValue newValues
  in field
    { sfValues = newValues
    , sfEnergy = newEnergy
    , sfCoherence = newCoherence
    , sfGradient = computeMaxGradient newValues
    }

-- | Run multiple ticks
runTicks :: AetherLab -> Int -> AetherLab
runTicks lab 0 = lab
runTicks lab n = runTicks (stepSimulation lab) (n - 1)

-- | Generate experiment result
generateResult :: Experiment -> AetherLab -> ExperimentResult
generateResult experiment lab =
  let field = alField lab
      events = alEvents lab
      peakCoh = maximum (sfCoherence field : [c | TickEvent _ c <- events])
      stability = computeStability events
  in ExperimentResult
    { erName = exName experiment
    , erSuccess = alState lab == LabComplete
    , erFinalEnergy = sfEnergy field
    , erFinalCoherence = sfCoherence field
    , erPeakCoherence = peakCoh
    , erStability = stability
    , erDuration = alTick lab
    }

-- | Compute stability from events
computeStability :: [SimEvent] -> Double
computeStability events =
  let coherences = [c | TickEvent _ c <- events]
      avg = if null coherences then 0 else sum coherences / fromIntegral (length coherences)
      variance = if null coherences then 0
                 else sum [(c - avg)^(2::Int) | c <- coherences] / fromIntegral (length coherences)
  in 1 - min 1 (sqrt variance)

-- | Average value in map
averageValue :: Map (Int, Int, Int) Double -> Double
averageValue m
  | Map.null m = 0
  | otherwise = sum (Map.elems m) / fromIntegral (Map.size m)

-- | Compute maximum gradient
computeMaxGradient :: Map (Int, Int, Int) Double -> Double
computeMaxGradient m =
  let values = Map.elems m
      maxV = maximum (0 : values)
      minV = minimum (0 : values)
  in maxV - minV
