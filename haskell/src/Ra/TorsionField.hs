{-|
Module      : Ra.TorsionField
Description : Scalar-torsion dynamics for avatar modulation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Integrates torsion field mechanics into the Ra scalar field system to enable
memory warping, avatar modulation, and rotational emergence phenomena.

== Torsion Field Theory

=== Spin Dynamics

Torsion fields arise from rotational stress in the scalar medium:

* Clockwise spin → Coherent, constructive emergence
* Counter-clockwise → Disruptive, shadow-access enabling
* Balanced spin → Stable, grounded state

=== Memory Warping

Torsion curl can bend the local scalar topology:

* High curl → Fragment memory becomes accessible
* Curl direction → Determines which timeline branch manifests
* Curl decay → Memories fade back to normal topology

=== Avatar Modulation

Torsion effects on avatar pose and emergence:

* Spin direction modulates avatar orientation
* Curl magnitude affects emergence angle
* Torsion gradient creates motion vectors

== Integration with Ra.Scalar

* TorsionField overlays existing ScalarField
* Spin polarity correlates with inversion state
* Torsion gradient adds to field flux
-}
module Ra.TorsionField
  ( -- * Core Types
    TorsionField(..)
  , SpinDirection(..)
  , TorsionVector(..)
  , mkTorsionField

    -- * Spin Dynamics
  , spinMagnitude
  , spinDirection
  , spinPolarity
  , invertSpin

    -- * Torsion Curl
  , TorsionCurl(..)
  , computeCurl
  , curlMagnitude
  , curlDirection
  , curlDecay

    -- * Memory Warping
  , MemoryWarp(..)
  , WarpState(..)
  , applyWarp
  , warpIntensity
  , unwindWarp

    -- * Avatar Modulation
  , AvatarPose(..)
  , PoseModulation(..)
  , modulatePose
  , emergenceAngle
  , motionVector

    -- * Field Operations
  , TorsionGradient(..)
  , computeGradient
  , applyToScalar
  , combineFields

    -- * Emergence Effects
  , TorsionEmergence(..)
  , emergenceLikelihood
  , spinEffect
  , curlBoost

    -- * Safety and Bounds
  , TorsionLimit(..)
  , checkLimits
  , dampTorsion
  , groundField
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse, coherenceFloorPOR )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Torsion field state
data TorsionField = TorsionField
  { tfSpin          :: !TorsionVector     -- ^ Spin vector
  , tfCurl          :: !TorsionCurl       -- ^ Curl tensor
  , tfMagnitude     :: !Double            -- ^ Overall magnitude [0,1]
  , tfPhase         :: !Double            -- ^ Rotational phase [0, 2*pi]
  , tfDecayRate     :: !Double            -- ^ Field decay rate
  , tfCoherence     :: !Double            -- ^ Field coherence [0,1]
  } deriving (Eq, Show)

-- | Spin direction
data SpinDirection
  = Clockwise       -- ^ Coherent, constructive
  | CounterClock    -- ^ Disruptive, shadow-enabling
  | Neutral         -- ^ Balanced, grounded
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | 3D torsion vector
data TorsionVector = TorsionVector
  { tvX :: !Double
  , tvY :: !Double
  , tvZ :: !Double
  } deriving (Eq, Show)

-- | Create initial torsion field
mkTorsionField :: Double -> SpinDirection -> TorsionField
mkTorsionField magnitude dir = TorsionField
  { tfSpin = mkSpinVector dir magnitude
  , tfCurl = emptyCurl
  , tfMagnitude = clamp01 magnitude
  , tfPhase = 0.0
  , tfDecayRate = 0.05
  , tfCoherence = 0.5
  }

-- Create spin vector from direction and magnitude
mkSpinVector :: SpinDirection -> Double -> TorsionVector
mkSpinVector dir mag = case dir of
  Clockwise    -> TorsionVector 0 0 mag
  CounterClock -> TorsionVector 0 0 (-mag)
  Neutral      -> TorsionVector 0 0 0

-- | Empty curl tensor
emptyCurl :: TorsionCurl
emptyCurl = TorsionCurl 0 0 0 0 0 0

-- =============================================================================
-- Spin Dynamics
-- =============================================================================

-- | Get spin magnitude
spinMagnitude :: TorsionField -> Double
spinMagnitude tf = vectorMagnitude (tfSpin tf)

-- | Get spin direction
spinDirection :: TorsionField -> SpinDirection
spinDirection tf =
  let z = tvZ (tfSpin tf)
  in if z > 0.01 then Clockwise
     else if z < -0.01 then CounterClock
     else Neutral

-- | Get spin polarity (-1, 0, +1)
spinPolarity :: TorsionField -> Double
spinPolarity tf = case spinDirection tf of
  Clockwise    -> 1.0
  CounterClock -> -1.0
  Neutral      -> 0.0

-- | Invert spin direction
invertSpin :: TorsionField -> TorsionField
invertSpin tf =
  let spin = tfSpin tf
      inverted = TorsionVector (-(tvX spin)) (-(tvY spin)) (-(tvZ spin))
  in tf { tfSpin = inverted }

-- =============================================================================
-- Torsion Curl
-- =============================================================================

-- | Curl tensor (antisymmetric part of torsion)
data TorsionCurl = TorsionCurl
  { tcXY :: !Double   -- ^ Curl in XY plane
  , tcXZ :: !Double   -- ^ Curl in XZ plane
  , tcYZ :: !Double   -- ^ Curl in YZ plane
  , tcDX :: !Double   -- ^ Curl gradient X
  , tcDY :: !Double   -- ^ Curl gradient Y
  , tcDZ :: !Double   -- ^ Curl gradient Z
  } deriving (Eq, Show)

-- | Compute curl from torsion field
computeCurl :: TorsionField -> TorsionCurl
computeCurl tf =
  let spin = tfSpin tf
      mag = tfMagnitude tf
      phase = tfPhase tf

      -- Curl in each plane
      xy = mag * sin phase * phi
      xz = mag * cos phase * phi
      yz = mag * sin (phase + pi/4) * phiInverse

      -- Gradients from spin components
      dx = tvX spin * 0.1
      dy = tvY spin * 0.1
      dz = tvZ spin * 0.1
  in TorsionCurl xy xz yz dx dy dz

-- | Get curl magnitude
curlMagnitude :: TorsionCurl -> Double
curlMagnitude tc =
  sqrt (tcXY tc ^ (2::Int) + tcXZ tc ^ (2::Int) + tcYZ tc ^ (2::Int))

-- | Get dominant curl direction
curlDirection :: TorsionCurl -> (Double, Double, Double)
curlDirection tc =
  let mag = curlMagnitude tc
      norm = if mag > 0 then mag else 1.0
  in (tcXY tc / norm, tcXZ tc / norm, tcYZ tc / norm)

-- | Decay curl over time
curlDecay :: Double -> TorsionCurl -> TorsionCurl
curlDecay dt tc =
  let factor = exp (-dt * 0.1)
  in TorsionCurl
      { tcXY = tcXY tc * factor
      , tcXZ = tcXZ tc * factor
      , tcYZ = tcYZ tc * factor
      , tcDX = tcDX tc * factor
      , tcDY = tcDY tc * factor
      , tcDZ = tcDZ tc * factor
      }

-- =============================================================================
-- Memory Warping
-- =============================================================================

-- | Memory warp state
data MemoryWarp = MemoryWarp
  { mwIntensity   :: !Double      -- ^ Warp intensity [0,1]
  , mwDirection   :: !Double      -- ^ Timeline direction (-1 past, +1 future)
  , mwFocus       :: !String      -- ^ Memory focus identifier
  , mwState       :: !WarpState
  , mwDuration    :: !Double      -- ^ Warp duration (seconds)
  } deriving (Eq, Show)

-- | Warp state
data WarpState
  = WarpIdle        -- ^ No warp active
  | WarpBuilding    -- ^ Warp intensity increasing
  | WarpPeak        -- ^ Maximum warp
  | WarpUnwinding   -- ^ Returning to normal
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Apply torsion warp to memory access
applyWarp :: TorsionField -> String -> MemoryWarp
applyWarp tf focus =
  let curl = computeCurl tf
      curlMag = curlMagnitude curl
      (_, _, dz) = curlDirection curl

      -- Intensity from curl magnitude
      intensity = clamp01 (curlMag / phi)

      -- Direction from curl Z component
      direction = clamp dz (-1) 1

      -- State from field coherence
      state = if tfCoherence tf > coherenceFloorPOR
              then WarpBuilding
              else WarpIdle
  in MemoryWarp
      { mwIntensity = intensity
      , mwDirection = direction
      , mwFocus = focus
      , mwState = state
      , mwDuration = 0.0
      }

-- | Get warp intensity
warpIntensity :: MemoryWarp -> Double
warpIntensity = mwIntensity

-- | Unwind warp back to normal
unwindWarp :: Double -> MemoryWarp -> MemoryWarp
unwindWarp dt warp =
  let newIntensity = max 0 (mwIntensity warp - dt * 0.5)
      newState = if newIntensity < 0.1 then WarpIdle else WarpUnwinding
  in warp
      { mwIntensity = newIntensity
      , mwState = newState
      , mwDuration = mwDuration warp + dt
      }

-- =============================================================================
-- Avatar Modulation
-- =============================================================================

-- | Avatar pose in torsion space
data AvatarPose = AvatarPose
  { apOrientation :: !(Double, Double, Double)  -- ^ Euler angles
  , apPosition    :: !(Double, Double, Double)  -- ^ Position offset
  , apScale       :: !Double                    -- ^ Scale factor
  , apCoherence   :: !Double                    -- ^ Pose coherence
  } deriving (Eq, Show)

-- | Pose modulation from torsion
data PoseModulation = PoseModulation
  { pmRotation    :: !(Double, Double, Double)  -- ^ Rotation delta
  , pmTranslation :: !(Double, Double, Double)  -- ^ Translation delta
  , pmScaleFactor :: !Double                    -- ^ Scale multiplier
  , pmPhaseShift  :: !Double                    -- ^ Temporal phase shift
  } deriving (Eq, Show)

-- | Modulate avatar pose from torsion field
modulatePose :: TorsionField -> AvatarPose -> AvatarPose
modulatePose tf pose =
  let modulation = computeModulation tf

      (ox, oy, oz) = apOrientation pose
      (rx, ry, rz) = pmRotation modulation
      newOrientation = (ox + rx, oy + ry, oz + rz)

      (px, py, pz) = apPosition pose
      (tx, ty, tz) = pmTranslation modulation
      newPosition = (px + tx, py + ty, pz + tz)

      newScale = apScale pose * pmScaleFactor modulation
  in pose
      { apOrientation = newOrientation
      , apPosition = newPosition
      , apScale = clamp newScale 0.5 2.0
      , apCoherence = tfCoherence tf
      }

-- Compute modulation from field
computeModulation :: TorsionField -> PoseModulation
computeModulation tf =
  let spin = tfSpin tf
      mag = tfMagnitude tf
      phase = tfPhase tf

      -- Rotation from spin
      rx = tvX spin * 0.1
      ry = tvY spin * 0.1
      rz = tvZ spin * 0.1

      -- Translation from curl
      curl = computeCurl tf
      tx = tcDX curl * mag * 0.5
      ty = tcDY curl * mag * 0.5
      tz = tcDZ curl * mag * 0.5

      -- Scale from coherence
      scaleFactor = 0.9 + tfCoherence tf * 0.2
  in PoseModulation
      { pmRotation = (rx, ry, rz)
      , pmTranslation = (tx, ty, tz)
      , pmScaleFactor = scaleFactor
      , pmPhaseShift = phase
      }

-- | Calculate emergence angle from torsion
emergenceAngle :: TorsionField -> Double
emergenceAngle tf =
  let spin = tfSpin tf
      angle = atan2 (tvY spin) (tvX spin)
      phaseOffset = tfPhase tf * phiInverse
  in angle + phaseOffset

-- | Compute motion vector from torsion
motionVector :: TorsionField -> (Double, Double, Double)
motionVector tf =
  let curl = computeCurl tf
      speed = tfMagnitude tf * phi
      (dx, dy, dz) = curlDirection curl
  in (dx * speed, dy * speed, dz * speed)

-- =============================================================================
-- Field Operations
-- =============================================================================

-- | Torsion gradient
data TorsionGradient = TorsionGradient
  { tgMagnitudeGrad :: !(Double, Double, Double)
  , tgPhaseGrad     :: !(Double, Double, Double)
  , tgCoherenceGrad :: !(Double, Double, Double)
  } deriving (Eq, Show)

-- | Compute field gradient
computeGradient :: TorsionField -> TorsionField -> Double -> TorsionGradient
computeGradient tf1 tf2 distance =
  let d = if distance > 0 then distance else 1.0

      magGrad = (tfMagnitude tf2 - tfMagnitude tf1) / d
      phaseGrad = wrapAngle (tfPhase tf2 - tfPhase tf1) / d
      cohGrad = (tfCoherence tf2 - tfCoherence tf1) / d
  in TorsionGradient
      { tgMagnitudeGrad = (magGrad, 0, 0)
      , tgPhaseGrad = (phaseGrad, 0, 0)
      , tgCoherenceGrad = (cohGrad, 0, 0)
      }

-- | Apply torsion to scalar field value
applyToScalar :: TorsionField -> Double -> Double
applyToScalar tf scalarValue =
  let spinMod = 1.0 + spinPolarity tf * tfMagnitude tf * 0.1
      curlMod = 1.0 + curlMagnitude (tfCurl tf) * 0.05
  in scalarValue * spinMod * curlMod

-- | Combine two torsion fields
combineFields :: TorsionField -> TorsionField -> TorsionField
combineFields tf1 tf2 =
  let -- Vector addition of spins
      newSpin = addVectors (tfSpin tf1) (tfSpin tf2)

      -- Average curls
      c1 = tfCurl tf1
      c2 = tfCurl tf2
      newCurl = TorsionCurl
        { tcXY = (tcXY c1 + tcXY c2) / 2
        , tcXZ = (tcXZ c1 + tcXZ c2) / 2
        , tcYZ = (tcYZ c1 + tcYZ c2) / 2
        , tcDX = (tcDX c1 + tcDX c2) / 2
        , tcDY = (tcDY c1 + tcDY c2) / 2
        , tcDZ = (tcDZ c1 + tcDZ c2) / 2
        }

      -- Magnitude from combined spin
      newMag = vectorMagnitude newSpin

      -- Phase blend
      newPhase = wrapAngle ((tfPhase tf1 + tfPhase tf2) / 2)

      -- Coherence product (destructive if opposite spins)
      coherenceFactor = if spinPolarity tf1 * spinPolarity tf2 < 0
                        then 0.5
                        else 1.0
      newCoherence = (tfCoherence tf1 + tfCoherence tf2) / 2 * coherenceFactor
  in TorsionField
      { tfSpin = newSpin
      , tfCurl = newCurl
      , tfMagnitude = clamp01 newMag
      , tfPhase = newPhase
      , tfDecayRate = (tfDecayRate tf1 + tfDecayRate tf2) / 2
      , tfCoherence = clamp01 newCoherence
      }

-- =============================================================================
-- Emergence Effects
-- =============================================================================

-- | Torsion effect on emergence
data TorsionEmergence = TorsionEmergence
  { teLikelihood   :: !Double     -- ^ Emergence probability modifier
  , teAngle        :: !Double     -- ^ Emergence angle
  , teSpinEffect   :: !Double     -- ^ Spin contribution
  , teCurlBoost    :: !Double     -- ^ Curl contribution
  , teDirection    :: !SpinDirection
  } deriving (Eq, Show)

-- | Calculate emergence likelihood modifier
emergenceLikelihood :: TorsionField -> Double -> TorsionEmergence
emergenceLikelihood tf baseAlpha =
  let -- Spin effect: clockwise increases likelihood
      spinFx = spinPolarity tf * tfMagnitude tf * 0.2

      -- Curl boost: higher curl = more likely
      curlFx = curlMagnitude (tfCurl tf) * 0.15

      -- Coherence gate
      coherenceGate = if tfCoherence tf > coherenceFloorPOR then 1.0 else 0.5

      -- Combined likelihood
      likelihood = clamp01 (baseAlpha + spinFx + curlFx) * coherenceGate

      -- Emergence angle
      angle = emergenceAngle tf
  in TorsionEmergence
      { teLikelihood = likelihood
      , teAngle = angle
      , teSpinEffect = spinFx
      , teCurlBoost = curlFx
      , teDirection = spinDirection tf
      }

-- | Get spin effect on emergence
spinEffect :: TorsionField -> Double
spinEffect tf = spinPolarity tf * tfMagnitude tf * phi

-- | Get curl boost for emergence
curlBoost :: TorsionField -> Double
curlBoost tf = curlMagnitude (tfCurl tf) * phiInverse

-- =============================================================================
-- Safety and Bounds
-- =============================================================================

-- | Torsion limits for safety
data TorsionLimit = TorsionLimit
  { tlMaxMagnitude  :: !Double    -- ^ Maximum allowed magnitude
  , tlMaxCurl       :: !Double    -- ^ Maximum curl magnitude
  , tlMaxPhaseRate  :: !Double    -- ^ Maximum phase change rate
  , tlMinCoherence  :: !Double    -- ^ Minimum required coherence
  } deriving (Eq, Show)

-- | Check if field is within limits
checkLimits :: TorsionLimit -> TorsionField -> Bool
checkLimits limits tf =
  tfMagnitude tf <= tlMaxMagnitude limits &&
  curlMagnitude (tfCurl tf) <= tlMaxCurl limits &&
  tfCoherence tf >= tlMinCoherence limits

-- | Damp excessive torsion
dampTorsion :: TorsionLimit -> TorsionField -> TorsionField
dampTorsion limits tf =
  let maxMag = tlMaxMagnitude limits
      maxCurl = tlMaxCurl limits
      minCoh = tlMinCoherence limits

      -- Damp magnitude
      newMag = min maxMag (tfMagnitude tf)

      -- Damp curl
      curl = tfCurl tf
      curlMag = curlMagnitude curl
      curlFactor = if curlMag > maxCurl then maxCurl / curlMag else 1.0
      newCurl = TorsionCurl
        { tcXY = tcXY curl * curlFactor
        , tcXZ = tcXZ curl * curlFactor
        , tcYZ = tcYZ curl * curlFactor
        , tcDX = tcDX curl * curlFactor
        , tcDY = tcDY curl * curlFactor
        , tcDZ = tcDZ curl * curlFactor
        }

      -- Boost coherence if below minimum
      newCoh = max minCoh (tfCoherence tf)
  in tf
      { tfMagnitude = newMag
      , tfCurl = newCurl
      , tfCoherence = newCoh
      }

-- | Ground field to neutral state
groundField :: TorsionField -> TorsionField
groundField tf = tf
  { tfSpin = TorsionVector 0 0 0
  , tfCurl = emptyCurl
  , tfMagnitude = 0.0
  , tfPhase = 0.0
  , tfCoherence = 1.0
  }

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Vector magnitude
vectorMagnitude :: TorsionVector -> Double
vectorMagnitude v = sqrt (tvX v ^ (2::Int) + tvY v ^ (2::Int) + tvZ v ^ (2::Int))

-- | Add two vectors
addVectors :: TorsionVector -> TorsionVector -> TorsionVector
addVectors v1 v2 = TorsionVector
  { tvX = tvX v1 + tvX v2
  , tvY = tvY v1 + tvY v2
  , tvZ = tvZ v1 + tvZ v2
  }

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)

-- | Clamp value to range
clamp :: Double -> Double -> Double -> Double
clamp x lo hi = max lo (min hi x)

-- | Wrap angle to [0, 2*pi]
wrapAngle :: Double -> Double
wrapAngle a
  | a < 0 = wrapAngle (a + 2 * pi)
  | a >= 2 * pi = wrapAngle (a - 2 * pi)
  | otherwise = a
