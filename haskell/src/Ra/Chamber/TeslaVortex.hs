{-|
Module      : Ra.Chamber.TeslaVortex
Description : Scalar field reactor blueprint with Tesla vortex dynamics
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Models a scalar-torsion vortex chamber inspired by Tesla turbines,
vortex-based propulsion, and zero-point flow coupling. This is a
scalar field model for seeding coherence chambers, modulating torsion,
and triggering nonlocal coupling.

== Tesla Vortex Theory

=== Scalar-Vortex Principles

The vortex chamber operates via:

* Radial inward flow → angular outward conversion
* Torsion bias determines spin handedness
* Layer shells create potential wells
* Harmonic overlays from Keely bands

=== Vortex Geometry

Tesla turbine-inspired geometry:

* Boundary layer disks (harmonic shells)
* Spiral inlet paths (phase channels)
* Central exhaust (coherence focus)
* Radial symmetric field profiles

=== Keely Sympathetic Modes

Optional sympathetic octave coupling:

* Mode 1-7: Physical octaves
* Mode 8-14: Etheric octaves
* Mode 15-21: Spiritual octaves
-}
module Ra.Chamber.TeslaVortex
  ( -- * Core Types
    VortexProfile(..)
  , VortexField(..)
  , SpinHandedness(..)

    -- * Field Generation
  , generateVortexField
  , fieldAtRadius
  , fieldStrength

    -- * Chamber Binding
  , VortexChamber(..)
  , bindVortexChamber
  , chamberResonance

    -- * Tesla Phase Mapping
  , teslaPhaseMap
  , phaseAtLayer
  , phaseVelocity

    -- * Torsion Dynamics
  , TorsionEnvelope(..)
  , computeTorsion
  , torsionGradient

    -- * Keely Integration
  , KeelyMode(..)
  , keelyFrequency
  , keelyHarmonic
  , applyKeelyMode

    -- * Harmonic Shells
  , HarmonicShell(..)
  , generateShells
  , shellResonance

    -- * Multi-Chamber Transfer
  , ChamberLink(..)
  , linkChambers
  , transferCoherence

    -- * Visualization
  , VortexVisual(..)
  , visualizeVortex
  , vortexColors
  ) where

import Ra.Constants.Extended
  ( phi, phiInverse )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Vortex field profile
data VortexProfile = VortexProfile
  { vpBaseSpin     :: !Double         -- ^ Baseline angular velocity (rad/s)
  , vpTorsionBias  :: !Double         -- ^ Field chirality [-1, 1]
  , vpLayerCount   :: !Int            -- ^ Number of harmonic shell layers
  , vpKeelyMode    :: !(Maybe Int)    -- ^ Optional sympathetic octave mode (1-21)
  , vpRadius       :: !Double         -- ^ Chamber radius
  , vpDepth        :: !Double         -- ^ Chamber depth
  } deriving (Eq, Show)

-- | Generated vortex field
data VortexField = VortexField
  { vfProfile      :: !VortexProfile
  , vfRadialFlow   :: ![Double]       -- ^ Radial flow at each layer
  , vfAngularFlow  :: ![Double]       -- ^ Angular flow at each layer
  , vfScalarPot    :: ![Double]       -- ^ Scalar potential at each layer
  , vfPhases       :: ![Double]       -- ^ Phase values
  , vfHandedness   :: !SpinHandedness
  } deriving (Eq, Show)

-- | Spin handedness (chirality)
data SpinHandedness
  = LeftHanded      -- ^ Counter-clockwise (negative torsion)
  | RightHanded     -- ^ Clockwise (positive torsion)
  | Neutral         -- ^ No preferred handedness
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Field Generation
-- =============================================================================

-- | Generate scalar field component from vortex profile
generateVortexField :: VortexProfile -> VortexField
generateVortexField profile =
  let layers = vpLayerCount profile
      radius = vpRadius profile

      -- Radial positions for each layer
      radii = [radius * fromIntegral i / fromIntegral layers
              | i <- [1..layers]]

      -- Tesla flow model: radial inward → angular outward
      -- Radial flow decreases toward center
      radialFlow = [vpBaseSpin profile * (1.0 - r / radius) | r <- radii]

      -- Angular flow increases toward center (conservation of angular momentum)
      angularFlow = [vpBaseSpin profile * radius / max 0.1 r | r <- radii]

      -- Scalar potential wells at each layer
      scalarPot = [computeLayerPotential profile i | i <- [1..layers]]

      -- Phase map
      phases = teslaPhaseMap profile

      -- Handedness from torsion bias
      handedness = if vpTorsionBias profile > 0.1 then RightHanded
                   else if vpTorsionBias profile < -0.1 then LeftHanded
                   else Neutral

  in VortexField
      { vfProfile = profile
      , vfRadialFlow = radialFlow
      , vfAngularFlow = angularFlow
      , vfScalarPot = scalarPot
      , vfPhases = phases
      , vfHandedness = handedness
      }

-- Compute scalar potential at layer
computeLayerPotential :: VortexProfile -> Int -> Double
computeLayerPotential profile layer =
  let n = fromIntegral layer
      total = fromIntegral (vpLayerCount profile)
      -- Potential well depth follows phi^n
      baseWell = phi ** (n / total)
      -- Keely mode modulation
      keelyMod = case vpKeelyMode profile of
        Just mode -> keelyModulator mode layer
        Nothing -> 1.0
      -- Torsion contribution
      torsionMod = 1.0 + abs (vpTorsionBias profile) * 0.2
  in baseWell * keelyMod * torsionMod

-- Keely modulator for layer
keelyModulator :: Int -> Int -> Double
keelyModulator mode layer =
  let octave = (mode - 1) `div` 7 + 1  -- 1-3 for physical/etheric/spiritual
      harmonic = (mode - 1) `mod` 7 + 1
      -- Sympathetic resonance peaks at matching layers
      resonance = if layer `mod` 7 == harmonic then phi else 1.0
  in resonance * (1.0 + fromIntegral octave * 0.1)

-- | Get field value at specific radius
fieldAtRadius :: VortexField -> Double -> (Double, Double, Double)
fieldAtRadius field radius =
  let profile = vfProfile field
      maxR = vpRadius profile
      layers = vpLayerCount profile
      -- Find closest layer
      layerIdx = min (layers - 1) $ max 0 $
                 round (radius / maxR * fromIntegral layers) - 1
      radial = vfRadialFlow field !! layerIdx
      angular = vfAngularFlow field !! layerIdx
      scalar = vfScalarPot field !! layerIdx
  in (radial, angular, scalar)

-- | Get overall field strength
fieldStrength :: VortexField -> Double
fieldStrength field =
  let pots = vfScalarPot field
  in if null pots then 0.0
     else sum pots / fromIntegral (length pots)

-- =============================================================================
-- Chamber Binding
-- =============================================================================

-- | Vortex chamber with bound coherence
data VortexChamber = VortexChamber
  { vcId            :: !String
  , vcField         :: !VortexField
  , vcCoordinate    :: !RaCoordinate
  , vcCoherence     :: !Double          -- ^ Current coherence [0, 1]
  , vcEmergence     :: !EmergenceCondition
  , vcActive        :: !Bool
  } deriving (Eq, Show)

-- | Ra coordinate
data RaCoordinate = RaCoordinate
  { rcRepitan   :: !Int
  , rcPhi       :: !Int
  , rcHarmonic  :: !Int
  } deriving (Eq, Show)

-- | Emergence condition
data EmergenceCondition = EmergenceCondition
  { ecThreshold   :: !Double
  , ecAlpha       :: !Double
  , ecPhase       :: !Double
  , ecInversion   :: !Bool
  } deriving (Eq, Show)

-- | Bind vortex chamber to a coherence condition
bindVortexChamber :: VortexProfile -> RaCoordinate -> EmergenceCondition
bindVortexChamber profile coord =
  let -- Base threshold from spin and torsion
      spinFactor = vpBaseSpin profile / (2 * pi)
      torsionFactor = abs (vpTorsionBias profile)

      -- Threshold influenced by coordinate
      baseThreshold = phiInverse + spinFactor * 0.1

      -- Alpha from layers and Keely mode
      alpha = case vpKeelyMode profile of
        Just mode -> keelyAlpha mode (rcHarmonic coord)
        Nothing -> 0.5 + fromIntegral (vpLayerCount profile) * 0.05

      -- Phase from coordinate mapping
      phase = fromIntegral (rcRepitan coord) * 2 * pi / 27.0

      -- Inversion from handedness and torsion
      inverted = vpTorsionBias profile < -0.5

  in EmergenceCondition
      { ecThreshold = min 0.9 (baseThreshold + torsionFactor * 0.1)
      , ecAlpha = min 1.0 alpha
      , ecPhase = phase
      , ecInversion = inverted
      }

-- Keely alpha calculation
keelyAlpha :: Int -> Int -> Double
keelyAlpha mode harmonic =
  let base = 0.5 + fromIntegral mode * 0.02
      resonance = if mode `mod` 7 == harmonic `mod` 7 then 0.15 else 0.0
  in min 1.0 (base + resonance)

-- | Get chamber resonance frequency
chamberResonance :: VortexChamber -> Double
chamberResonance chamber =
  let profile = vfProfile (vcField chamber)
      baseFreq = 256.0 * vpBaseSpin profile / (2 * pi)
      keelyMod = case vpKeelyMode profile of
        Just mode -> keelyFrequency mode
        Nothing -> 1.0
  in baseFreq * keelyMod

-- =============================================================================
-- Tesla Phase Mapping
-- =============================================================================

-- | Encode Tesla turbine scaling and phase properties
teslaPhaseMap :: VortexProfile -> [Double]
teslaPhaseMap profile =
  let layers = vpLayerCount profile
      spin = vpBaseSpin profile
      -- Phase increases with layer (spiral pattern)
      basePhases = [spin * fromIntegral i * phi / fromIntegral layers
                   | i <- [1..layers]]
      -- Torsion shifts phase
      torsionShift = vpTorsionBias profile * pi / 4
  in map (+ torsionShift) basePhases

-- | Get phase at specific layer
phaseAtLayer :: VortexProfile -> Int -> Double
phaseAtLayer profile layer =
  let phases = teslaPhaseMap profile
  in if layer >= 1 && layer <= length phases
     then phases !! (layer - 1)
     else 0.0

-- | Calculate phase velocity (rate of phase change)
phaseVelocity :: VortexProfile -> Double
phaseVelocity profile =
  vpBaseSpin profile * phi / fromIntegral (max 1 (vpLayerCount profile))

-- =============================================================================
-- Torsion Dynamics
-- =============================================================================

-- | Torsion envelope description
data TorsionEnvelope = TorsionEnvelope
  { teTorsion     :: !Double          -- ^ Torsion magnitude
  , teDirection   :: !SpinHandedness  -- ^ Spin direction
  , teGradient    :: ![Double]        -- ^ Radial gradient
  , teStability   :: !Double          -- ^ Envelope stability [0, 1]
  } deriving (Eq, Show)

-- | Compute torsion envelope from vortex field
computeTorsion :: VortexField -> TorsionEnvelope
computeTorsion field =
  let profile = vfProfile field
      bias = vpTorsionBias profile

      -- Torsion magnitude from bias and spin
      magnitude = abs bias * vpBaseSpin profile

      -- Gradient from angular flow differences
      angular = vfAngularFlow field
      gradient = zipWith (-) (tail angular ++ [0]) angular

      -- Stability from uniformity of gradient
      stability = 1.0 - variance gradient / (max 0.01 (maximum (map abs gradient)))

  in TorsionEnvelope
      { teTorsion = magnitude
      , teDirection = vfHandedness field
      , teGradient = gradient
      , teStability = clamp01 stability
      }

-- | Get torsion gradient at position
torsionGradient :: TorsionEnvelope -> Int -> Double
torsionGradient env idx =
  let grad = teGradient env
  in if idx >= 0 && idx < length grad
     then grad !! idx
     else 0.0

-- =============================================================================
-- Keely Integration
-- =============================================================================

-- | Keely sympathetic mode
data KeelyMode = KeelyMode
  { kmOctave    :: !Int               -- ^ Octave (1=physical, 2=etheric, 3=spiritual)
  , kmHarmonic  :: !Int               -- ^ Harmonic within octave (1-7)
  , kmFrequency :: !Double            -- ^ Base frequency (Hz)
  } deriving (Eq, Show)

-- | Get Keely frequency for mode (Hz)
keelyFrequency :: Int -> Double
keelyFrequency mode =
  let octave = (mode - 1) `div` 7
      harmonic = (mode - 1) `mod` 7 + 1
      -- Base frequencies for each octave
      octaveBase = 256.0 * (2 ** fromIntegral octave)
      -- Harmonic multiplier
      harmMult = fromIntegral harmonic / 7.0 + 1.0
  in octaveBase * harmMult

-- | Get harmonic indices for Keely mode
keelyHarmonic :: Int -> (Int, Int)
keelyHarmonic mode =
  let octave = (mode - 1) `div` 7
      harmonic = (mode - 1) `mod` 7 + 1
  in (octave + 1, harmonic)

-- | Apply Keely mode to vortex profile
applyKeelyMode :: Int -> VortexProfile -> VortexProfile
applyKeelyMode mode profile = profile { vpKeelyMode = Just mode }

-- =============================================================================
-- Harmonic Shells
-- =============================================================================

-- | Harmonic shell layer
data HarmonicShell = HarmonicShell
  { hsLayer       :: !Int             -- ^ Layer index (1-based)
  , hsRadius      :: !Double          -- ^ Shell radius
  , hsPotential   :: !Double          -- ^ Shell potential
  , hsPhase       :: !Double          -- ^ Shell phase
  , hsResonance   :: !Double          -- ^ Resonance strength [0, 1]
  } deriving (Eq, Show)

-- | Generate all harmonic shells from profile
generateShells :: VortexProfile -> [HarmonicShell]
generateShells profile =
  let field = generateVortexField profile
      layers = vpLayerCount profile
      radius = vpRadius profile
  in [ HarmonicShell
        { hsLayer = i
        , hsRadius = radius * fromIntegral i / fromIntegral layers
        , hsPotential = vfScalarPot field !! (i - 1)
        , hsPhase = vfPhases field !! (i - 1)
        , hsResonance = computeShellResonance profile i
        }
     | i <- [1..layers]
     ]

-- Compute shell resonance
computeShellResonance :: VortexProfile -> Int -> Double
computeShellResonance profile layer =
  let n = fromIntegral layer
      total = fromIntegral (vpLayerCount profile)
      -- Resonance peaks at phi-related positions
      phiPosition = n / total
      phiResonance = 1.0 - abs (phiPosition - phiInverse)
      -- Keely boost for matching mode
      keelyBoost = case vpKeelyMode profile of
        Just mode -> if layer `mod` 7 == mode `mod` 7 then 0.3 else 0.0
        Nothing -> 0.0
  in clamp01 (phiResonance + keelyBoost)

-- | Get resonance of specific shell
shellResonance :: HarmonicShell -> Double
shellResonance = hsResonance

-- =============================================================================
-- Multi-Chamber Transfer
-- =============================================================================

-- | Link between two vortex chambers
data ChamberLink = ChamberLink
  { clSource      :: !String          -- ^ Source chamber ID
  , clTarget      :: !String          -- ^ Target chamber ID
  , clCoupling    :: !Double          -- ^ Coupling strength [0, 1]
  , clPhaseMatch  :: !Double          -- ^ Phase alignment [0, 1]
  , clActive      :: !Bool
  } deriving (Eq, Show)

-- | Link two chambers for scalar transfer
linkChambers :: VortexChamber -> VortexChamber -> ChamberLink
linkChambers source target =
  let -- Phase matching
      sourcePhase = ecPhase (vcEmergence source)
      targetPhase = ecPhase (vcEmergence target)
      phaseDelta = abs (sourcePhase - targetPhase)
      phaseMatch = 1.0 - min 1.0 (phaseDelta / pi)

      -- Coupling from coherence product
      coherenceProd = vcCoherence source * vcCoherence target

      -- Handedness compatibility
      sourceHand = vfHandedness (vcField source)
      targetHand = vfHandedness (vcField target)
      handCompatible = sourceHand == targetHand || sourceHand == Neutral || targetHand == Neutral

      -- Overall coupling
      coupling = if handCompatible
                 then coherenceProd * phaseMatch
                 else coherenceProd * phaseMatch * 0.5

  in ChamberLink
      { clSource = vcId source
      , clTarget = vcId target
      , clCoupling = clamp01 coupling
      , clPhaseMatch = phaseMatch
      , clActive = coupling > 0.3
      }

-- | Transfer coherence between linked chambers
transferCoherence :: ChamberLink -> VortexChamber -> VortexChamber -> (VortexChamber, VortexChamber)
transferCoherence link source target =
  let -- Only transfer if link is active
      amount = if clActive link
               then (vcCoherence source - vcCoherence target) * clCoupling link * 0.1
               else 0.0

      -- Update coherence (source loses, target gains)
      newSourceCoh = max 0.0 (vcCoherence source - amount)
      newTargetCoh = min 1.0 (vcCoherence target + amount)

  in (source { vcCoherence = newSourceCoh },
      target { vcCoherence = newTargetCoh })

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Vortex visualization data
data VortexVisual = VortexVisual
  { vvShells      :: ![(Double, (Int, Int, Int))]  -- ^ (radius, color)
  , vvFlowLines   :: ![(Double, Double)]           -- ^ (angle, magnitude)
  , vvCenter      :: !(Int, Int, Int)              -- ^ Center color
  , vvGlow        :: !Double                       -- ^ Glow intensity
  } deriving (Eq, Show)

-- | Generate visualization from vortex chamber
visualizeVortex :: VortexChamber -> VortexVisual
visualizeVortex chamber =
  let field = vcField chamber
      profile = vfProfile field
      shells = generateShells profile

      -- Shell colors based on potential
      shellColors = [(hsRadius s, potentialToColor (hsPotential s)) | s <- shells]

      -- Flow lines from angular flow
      angular = vfAngularFlow field
      flowLines = zip [0, 2*pi / fromIntegral (length angular)..] angular

      -- Center color from handedness
      centerColor = case vfHandedness field of
        LeftHanded -> (100, 100, 255)   -- Blue for CCW
        RightHanded -> (255, 100, 100)  -- Red for CW
        Neutral -> (200, 200, 200)      -- Gray for neutral

      -- Glow from coherence
      glow = vcCoherence chamber

  in VortexVisual
      { vvShells = shellColors
      , vvFlowLines = flowLines
      , vvCenter = centerColor
      , vvGlow = glow
      }

-- Convert potential to RGB color
potentialToColor :: Double -> (Int, Int, Int)
potentialToColor pot =
  let normalized = clamp01 (pot / 3.0)  -- Assuming max ~3.0
      r = round (255 * normalized) :: Int
      g = round (255 * (1.0 - normalized) * 0.5) :: Int
      b = round (255 * (1.0 - normalized)) :: Int
  in (r, g, b)

-- | Get vortex color palette
vortexColors :: SpinHandedness -> [(Double, (Int, Int, Int))]
vortexColors hand = case hand of
  LeftHanded ->
    [ (0.0, (50, 50, 150))
    , (0.5, (100, 100, 255))
    , (1.0, (200, 200, 255))
    ]
  RightHanded ->
    [ (0.0, (150, 50, 50))
    , (0.5, (255, 100, 100))
    , (1.0, (255, 200, 200))
    ]
  Neutral ->
    [ (0.0, (100, 100, 100))
    , (0.5, (180, 180, 180))
    , (1.0, (240, 240, 240))
    ]

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)

-- | Variance of list
variance :: [Double] -> Double
variance [] = 0.0
variance xs =
  let n = fromIntegral (length xs)
      mean = sum xs / n
      sqDiffs = map (\x -> (x - mean) ** 2) xs
  in sum sqDiffs / n
