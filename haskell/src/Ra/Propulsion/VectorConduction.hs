{-|
Module      : Ra.Propulsion.VectorConduction
Description : Scalar field vector conduction for propulsion mechanics
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements vector conduction mechanics for scalar field propulsion systems.
Conduction vectors translate coherence gradients into directional momentum
through torsion field manipulation.

== Vector Conduction Theory

=== Conduction Principle

Scalar field gradients create conduction potential:

* Coherence differential drives vector flow
* Torsion alignment determines direction
* Harmonic coupling modulates intensity

=== Propulsion Modes

1. Gradient Drive: Direct coherence gradient propulsion
2. Vortex Thrust: Rotational torsion propulsion
3. Harmonic Pulse: Pulsed coherence burst propulsion
4. Wave Riding: Scalar wave surfing mode
-}
module Ra.Propulsion.VectorConduction
  ( -- * Core Types
    ConductionVector(..)
  , PropulsionMode(..)
  , VectorField(..)
  , ThrustProfile(..)

    -- * Vector Generation
  , generateConductionVector
  , computeGradientVector
  , alignToTorsion

    -- * Propulsion Modes
  , gradientDrive
  , vortexThrust
  , harmonicPulse
  , waveRiding

    -- * Field Operations
  , conductField
  , vectorSuperposition
  , fieldGradient

    -- * Thrust Computation
  , computeThrust
  , thrustEfficiency
  , sustainedThrust

    -- * Coupling
  , CouplingResonance(..)
  , harmonicCoupling
  , resonantAmplification

    -- * Vector Navigation
  , NavigationVector(..)
  , plotCourse
  , correctTrajectory
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Conduction vector in scalar field
data ConductionVector = ConductionVector
  { cvDirection  :: !(Double, Double, Double)  -- ^ Unit direction vector
  , cvMagnitude  :: !Double                    -- ^ Vector magnitude
  , cvPhase      :: !Double                    -- ^ Phase angle [0, 2π]
  , cvCoherence  :: !Double                    -- ^ Associated coherence [0, 1]
  , cvMode       :: !PropulsionMode            -- ^ Active propulsion mode
  } deriving (Eq, Show)

-- | Propulsion mode types
data PropulsionMode
  = ModeGradient    -- ^ Gradient-driven propulsion
  | ModeVortex      -- ^ Vortex torsion propulsion
  | ModeHarmonic    -- ^ Pulsed harmonic propulsion
  | ModeWaveRide    -- ^ Scalar wave surfing
  | ModeHybrid      -- ^ Combined mode
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Vector field configuration
data VectorField = VectorField
  { vfVectors    :: ![ConductionVector]        -- ^ Field vectors
  , vfCenter     :: !(Double, Double, Double)  -- ^ Field center
  , vfRadius     :: !Double                    -- ^ Effective radius
  , vfStrength   :: !Double                    -- ^ Overall field strength
  , vfRotation   :: !Double                    -- ^ Field rotation rate
  } deriving (Eq, Show)

-- | Thrust profile for propulsion
data ThrustProfile = ThrustProfile
  { tpPeak       :: !Double    -- ^ Peak thrust value
  , tpSustained  :: !Double    -- ^ Sustained thrust level
  , tpRampUp     :: !Double    -- ^ Ramp-up time (φ^n ticks)
  , tpEfficiency :: !Double    -- ^ Energy efficiency [0, 1]
  , tpStability  :: !Double    -- ^ Thrust stability [0, 1]
  } deriving (Eq, Show)

-- =============================================================================
-- Vector Generation
-- =============================================================================

-- | Generate conduction vector from field state
generateConductionVector :: (Double, Double, Double)  -- ^ Position
                         -> Double                     -- ^ Coherence
                         -> Double                     -- ^ Phase
                         -> PropulsionMode             -- ^ Mode
                         -> ConductionVector
generateConductionVector pos coherence phase' mode =
  let (x, y, z) = pos
      len = sqrt (x*x + y*y + z*z)
      direction = if len > 0
                  then (x/len, y/len, z/len)
                  else (0, 0, 1)
      magnitude = coherence * phi
  in ConductionVector
    { cvDirection = direction
    , cvMagnitude = magnitude
    , cvPhase = phase'
    , cvCoherence = coherence
    , cvMode = mode
    }

-- | Compute gradient vector from coherence differential
computeGradientVector :: (Double, Double, Double)  -- ^ Position 1
                      -> Double                     -- ^ Coherence 1
                      -> (Double, Double, Double)  -- ^ Position 2
                      -> Double                     -- ^ Coherence 2
                      -> ConductionVector
computeGradientVector (x1, y1, z1) c1 (x2, y2, z2) c2 =
  let dx = x2 - x1
      dy = y2 - y1
      dz = z2 - z1
      dist = sqrt (dx*dx + dy*dy + dz*dz)
      gradient = if dist > 0 then (c2 - c1) / dist else 0
      direction = if dist > 0
                  then (dx/dist, dy/dist, dz/dist)
                  else (0, 0, 1)
      magnitude = abs gradient * phi
  in ConductionVector
    { cvDirection = direction
    , cvMagnitude = magnitude
    , cvPhase = 0
    , cvCoherence = (c1 + c2) / 2
    , cvMode = ModeGradient
    }

-- | Align vector to torsion field
alignToTorsion :: ConductionVector -> Double -> ConductionVector
alignToTorsion cv torsionAngle =
  let (x, y, z) = cvDirection cv
      cosT = cos torsionAngle
      sinT = sin torsionAngle
      -- Rotate around z-axis by torsion angle
      newX = x * cosT - y * sinT
      newY = x * sinT + y * cosT
  in cv { cvDirection = (newX, newY, z) }

-- =============================================================================
-- Propulsion Modes
-- =============================================================================

-- | Gradient drive propulsion
gradientDrive :: Double  -- ^ Coherence gradient
              -> Double  -- ^ Base thrust
              -> ThrustProfile
gradientDrive gradient baseThrust =
  let peak = baseThrust * gradient * phi
      sustained = peak * phiInverse
      efficiency = min 1.0 (phiInverse * (1 + gradient))
  in ThrustProfile
    { tpPeak = peak
    , tpSustained = sustained
    , tpRampUp = 1.0
    , tpEfficiency = efficiency
    , tpStability = 0.9
    }

-- | Vortex thrust propulsion
vortexThrust :: Double  -- ^ Rotation rate
             -> Double  -- ^ Base thrust
             -> ThrustProfile
vortexThrust rotRate baseThrust =
  let peak = baseThrust * rotRate * phi * phi
      sustained = peak * 0.8
      efficiency = phiInverse * phiInverse  -- Lower efficiency for vortex
  in ThrustProfile
    { tpPeak = peak
    , tpSustained = sustained
    , tpRampUp = 2.0  -- Slower ramp-up for vortex
    , tpEfficiency = efficiency
    , tpStability = 0.7  -- Less stable
    }

-- | Harmonic pulse propulsion
harmonicPulse :: Int     -- ^ Harmonic order
              -> Double  -- ^ Base thrust
              -> ThrustProfile
harmonicPulse harmonic baseThrust =
  let harmonicFactor = 1.0 / fromIntegral (harmonic + 1)
      peak = baseThrust * harmonicFactor * phi * phi * phi
      sustained = 0  -- Pulsed mode has no sustained thrust
      efficiency = harmonicFactor * phi
  in ThrustProfile
    { tpPeak = peak
    , tpSustained = sustained
    , tpRampUp = 0.1  -- Very fast ramp-up
    , tpEfficiency = efficiency
    , tpStability = 0.5  -- Pulsed is inherently unstable
    }

-- | Wave riding propulsion
waveRiding :: Double  -- ^ Wave amplitude
           -> Double  -- ^ Wave frequency
           -> Double  -- ^ Base thrust
           -> ThrustProfile
waveRiding amplitude frequency baseThrust =
  let resonanceFactor = if abs (frequency - phi) < 0.1
                        then phi  -- Resonant boost
                        else 1.0
      peak = baseThrust * amplitude * resonanceFactor
      sustained = peak * phiInverse
      efficiency = phiInverse * resonanceFactor
  in ThrustProfile
    { tpPeak = peak
    , tpSustained = sustained
    , tpRampUp = 0.5
    , tpEfficiency = min 1.0 efficiency
    , tpStability = 0.85
    }

-- =============================================================================
-- Field Operations
-- =============================================================================

-- | Conduct field along vector
conductField :: VectorField -> ConductionVector -> VectorField
conductField field cv =
  let newVectors = cv : vfVectors field
      newStrength = vfStrength field + cvMagnitude cv
  in field { vfVectors = newVectors, vfStrength = newStrength }

-- | Superposition of two vector fields
vectorSuperposition :: VectorField -> VectorField -> VectorField
vectorSuperposition f1 f2 =
  let allVectors = vfVectors f1 ++ vfVectors f2
      (x1, y1, z1) = vfCenter f1
      (x2, y2, z2) = vfCenter f2
      newCenter = ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
      newRadius = max (vfRadius f1) (vfRadius f2)
      newStrength = sqrt (vfStrength f1 ^ (2::Int) + vfStrength f2 ^ (2::Int))
  in VectorField
    { vfVectors = allVectors
    , vfCenter = newCenter
    , vfRadius = newRadius
    , vfStrength = newStrength
    , vfRotation = (vfRotation f1 + vfRotation f2) / 2
    }

-- | Compute field gradient at position
fieldGradient :: VectorField -> (Double, Double, Double) -> Double
fieldGradient field (px, py, pz) =
  let (cx, cy, cz) = vfCenter field
      dist = sqrt ((px-cx)^(2::Int) + (py-cy)^(2::Int) + (pz-cz)^(2::Int))
      -- Gradient decreases with distance
      gradient = if dist > 0
                 then vfStrength field / (dist * dist)
                 else vfStrength field
  in gradient * phiInverse

-- =============================================================================
-- Thrust Computation
-- =============================================================================

-- | Compute thrust from conduction vector
computeThrust :: ConductionVector -> ThrustProfile -> Double
computeThrust cv profile =
  let modeMultiplier = case cvMode cv of
        ModeGradient -> 1.0
        ModeVortex   -> 0.8
        ModeHarmonic -> 1.2
        ModeWaveRide -> 1.1
        ModeHybrid   -> 1.0
      baseThrust = cvMagnitude cv * cvCoherence cv
  in baseThrust * modeMultiplier * tpEfficiency profile

-- | Compute thrust efficiency
thrustEfficiency :: ThrustProfile -> ConductionVector -> Double
thrustEfficiency profile cv =
  let coherenceFactor = cvCoherence cv
      stabilityFactor = tpStability profile
  in tpEfficiency profile * coherenceFactor * stabilityFactor

-- | Compute sustained thrust capability
sustainedThrust :: ThrustProfile -> Int -> Double
sustainedThrust profile ticks =
  let decayFactor = phiInverse ^ ticks
      sustained = tpSustained profile
  in sustained * (1 - decayFactor * 0.1)

-- =============================================================================
-- Coupling
-- =============================================================================

-- | Coupling resonance state
data CouplingResonance = CouplingResonance
  { crFrequency  :: !Double    -- ^ Resonant frequency
  , crAmplitude  :: !Double    -- ^ Coupling amplitude
  , crPhase      :: !Double    -- ^ Phase relationship
  , crStrength   :: !Double    -- ^ Coupling strength [0, 1]
  } deriving (Eq, Show)

-- | Compute harmonic coupling between vectors
harmonicCoupling :: ConductionVector -> ConductionVector -> CouplingResonance
harmonicCoupling v1 v2 =
  let phaseDiff = abs (cvPhase v1 - cvPhase v2)
      frequency = phi / (1 + phaseDiff / (2 * pi))
      amplitude = (cvMagnitude v1 + cvMagnitude v2) / 2
      coupling = cos phaseDiff * cvCoherence v1 * cvCoherence v2
  in CouplingResonance
    { crFrequency = frequency
    , crAmplitude = amplitude
    , crPhase = phaseDiff
    , crStrength = abs coupling
    }

-- | Compute resonant amplification factor
resonantAmplification :: CouplingResonance -> Double -> Double
resonantAmplification coupling targetFreq =
  let freqDiff = abs (crFrequency coupling - targetFreq)
      bandwidth = crFrequency coupling * phiInverse
      resonance = if freqDiff < bandwidth
                  then 1 + (1 - freqDiff / bandwidth) * phi
                  else 1.0
  in crAmplitude coupling * resonance * crStrength coupling

-- =============================================================================
-- Vector Navigation
-- =============================================================================

-- | Navigation vector with trajectory
data NavigationVector = NavigationVector
  { nvCurrent    :: !ConductionVector          -- ^ Current vector
  , nvTarget     :: !(Double, Double, Double)  -- ^ Target position
  , nvDeviation  :: !Double                    -- ^ Course deviation angle
  , nvETA        :: !Double                    -- ^ Estimated time of arrival
  } deriving (Eq, Show)

-- | Plot course to target
plotCourse :: (Double, Double, Double)  -- ^ Current position
           -> (Double, Double, Double)  -- ^ Target position
           -> Double                     -- ^ Available thrust
           -> NavigationVector
plotCourse (cx, cy, cz) (tx, ty, tz) thrust =
  let dx = tx - cx
      dy = ty - cy
      dz = tz - cz
      dist = sqrt (dx*dx + dy*dy + dz*dz)
      direction = if dist > 0
                  then (dx/dist, dy/dist, dz/dist)
                  else (0, 0, 1)
      cv = ConductionVector
        { cvDirection = direction
        , cvMagnitude = thrust
        , cvPhase = 0
        , cvCoherence = 1.0
        , cvMode = ModeGradient
        }
      eta = if thrust > 0 then dist / thrust else 0
  in NavigationVector
    { nvCurrent = cv
    , nvTarget = (tx, ty, tz)
    , nvDeviation = 0
    , nvETA = eta
    }

-- | Correct trajectory deviation
correctTrajectory :: NavigationVector -> Double -> NavigationVector
correctTrajectory nav deviation =
  let cv = nvCurrent nav
      correctedCV = alignToTorsion cv deviation
  in nav
    { nvCurrent = correctedCV
    , nvDeviation = deviation
    }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Empty vector field
_emptyField :: VectorField
_emptyField = VectorField
  { vfVectors = []
  , vfCenter = (0, 0, 0)
  , vfRadius = 0
  , vfStrength = 0
  , vfRotation = 0
  }
