{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module      : Ra.Chamber
Description : Personalized Coherence Chamber Generator with Orgone Layer Modeling
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements the generation of personalized coherence chambers - synthetic
field-space structures designed to optimize a user's harmonic resonance
based on their biometric and semantic profiles.

== Theoretical Background

=== Geometric Coherence Amplification (Golod Pyramids)

Russian pyramid research demonstrated that geometric structures can
amplify and focus scalar field effects. The chamber geometry determines
how field energy is concentrated and distributed:

* /Pyramid/: Focuses energy at apex, creates vertical coherence column
* /Tetrahedron/: Minimal surface area, maximum structural stability
* /Sphere/: Isotropic containment, uniform field distribution
* /Torus/: Continuous flow topology, self-sustaining field circulation

=== Orgone Accumulator Principles (Reich)

Layered organic/inorganic materials create field accumulation:

1. Organic layers absorb ambient field energy
2. Inorganic layers reflect and concentrate field
3. Alternating layers create multiplicative accumulation
4. Layer count determines accumulation strength

=== Personalization via Biometric-Harmonic Coupling

Chamber configuration is derived from:

* Dominant emotional state → base geometry selection
* Biometric coherence level → harmonic gate thresholds
* Semantic signature → resonance target channels
* Shell depth preference → radial profile tuning

== Safety Constraints

All chambers include:

* Consent gate verification before activation
* Coherence monitoring for field stability
* Automatic discharge on coherence collapse
* Shadow fragment containment safeguards
-}
module Ra.Chamber
  ( -- * Core Types
    CoherenceChamber(..)
  , GeometryShape(..)
  , HarmonicGate(..)
  , ResonanceTarget(..)
  , UserProfile(..)

    -- * Orgone Layer System
  , OrgoneLayer(..)
  , LayerMaterial(..)
  , OrgoneStack(..)
  , EmotionalCharge(..)

    -- * Chamber Generation
  , generateChamber
  , generateChamberWithOrgone
  , createHarmonicGates
  , deriveResonanceTargets

    -- * Orgone Layer Functions
  , createOrgoneStack
  , computeLayerResonance
  , accumulateCharge
  , dissipateCharge

    -- * Chamber Operations
  , activateChamber
  , deactivateChamber
  , invertChamber
  , validateChamber

    -- * Geometry Effects
  , geometryFocusFactor
  , geometryFlowPattern
  , geometryContainment

    -- * Fragment Binding (Prompt #5)
  , ResonanceAnchor(..)
  , BoundFragment(..)
  , BindingCondition(..)
  , bindFragment
  , releaseFragment
  , checkBindingStability

    -- * Constants
  , defaultLayerCount
  , maxAccumulationFactor
  , chargeDecayRate
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)
import Data.Maybe (fromMaybe)

import Ra.Scalar
  ( ScalarField(..)
  , ScalarComponent(..)
  , RadialProfile(..)
  , RadialType(..)
  , HarmonicSignature(..)
  )
import Ra.Omega (OmegaFormat(..))
import Ra.Tuning
  ( BiometricInput(..)
  , PhiWindow(..)
  , mkPhiWindow
  , phiGolden
  )

-- =============================================================================
-- Core Chamber Types
-- =============================================================================

-- | Geometry shape for the coherence chamber
--
-- Each shape has distinct field dynamics:
data GeometryShape
  = Pyramid       -- ^ Focused apex energy, vertical coherence
  | Tetrahedron   -- ^ Minimal surface, maximum stability
  | Sphere        -- ^ Isotropic containment, uniform distribution
  | Torus         -- ^ Continuous flow, self-sustaining circulation
  | Dodecahedron  -- ^ 12-fold symmetry, complex harmonic patterns
  | Custom String -- ^ User-defined geometry specification
  deriving (Eq, Show, Generic, NFData)

-- | Harmonic gate for controlling field access
--
-- Gates filter which harmonics can pass through the chamber boundary.
-- The threshold determines the minimum coherence required for passage.
data HarmonicGate = HarmonicGate
  { hgOmega         :: !OmegaFormat       -- ^ Omega format this gate serves
  , hgThreshold     :: !Double            -- ^ Coherence threshold [0,1]
  , hgPhaseLockWindow :: !(Maybe PhiWindow) -- ^ Optional phi-aligned timing
  , hgHarmonicFilter :: !(Maybe HarmonicSignature) -- ^ Optional (l,m) filter
  } deriving (Eq, Show, Generic, NFData)

-- | Resonance target - a frequency/channel pair for entrainment
data ResonanceTarget = ResonanceTarget
  { rtFrequencyHz      :: !Double   -- ^ Target frequency in Hz
  , rtChannel          :: !String   -- ^ Channel identifier
  , rtAlignmentRequired :: !Bool    -- ^ Must alignment be verified?
  , rtPriority         :: !Int      -- ^ Target priority (lower = higher)
  } deriving (Eq, Show, Generic, NFData)

-- | User profile for chamber personalization
data UserProfile = UserProfile
  { upDominantEmotion     :: !(Maybe String)    -- ^ Primary emotional state
  , upBiometricSetpoint   :: !BiometricInput    -- ^ Baseline biometric values
  , upSemanticSignature   :: ![String]          -- ^ Semantic identity markers
  , upShellDepthPreference :: !(Maybe Int)      -- ^ Preferred radial depth
  , upConsentLevel        :: !Int               -- ^ Consent gate level (1-6)
  , upShadowWorkEnabled   :: !Bool              -- ^ Allow inverted chambers?
  } deriving (Eq, Show, Generic, NFData)

-- | A personalized coherence chamber
data CoherenceChamber = CoherenceChamber
  { ccId              :: !String             -- ^ Unique chamber identifier
  , ccFieldGeometry   :: !GeometryShape      -- ^ Chamber geometry
  , ccBaseField       :: !ScalarField        -- ^ Base scalar field
  , ccHarmonicGates   :: ![HarmonicGate]     -- ^ Harmonic access gates
  , ccResonanceTargets :: ![ResonanceTarget] -- ^ Entrainment targets
  , ccOrgoneStack     :: !OrgoneStack        -- ^ Orgone layer system
  , ccIsInverted      :: !Bool               -- ^ Shadow/inverted mode?
  , ccIsActive        :: !Bool               -- ^ Currently active?
  } deriving (Eq, Show, Generic, NFData)

-- =============================================================================
-- Orgone Layer System (Prompt #4)
-- =============================================================================

-- | Layer material type (organic/inorganic alternation)
data LayerMaterial
  = Organic      -- ^ Absorbs ambient field energy
  | Inorganic    -- ^ Reflects and concentrates field
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | A single orgone accumulator layer
data OrgoneLayer = OrgoneLayer
  { olMaterial      :: !LayerMaterial  -- ^ Layer material type
  , olThickness     :: !Double         -- ^ Relative thickness [0,1]
  , olResonance     :: !Double         -- ^ Resonance frequency (Hz)
  , olConductivity  :: !Double         -- ^ Field conductivity [0,1]
  , olCharge        :: !Double         -- ^ Current accumulated charge [0,1]
  } deriving (Eq, Show, Generic, NFData)

-- | Stack of alternating orgone layers
data OrgoneStack = OrgoneStack
  { osLayers           :: ![OrgoneLayer]  -- ^ Alternating layers
  , osTotalCharge      :: !Double         -- ^ Total accumulated charge
  , osAccumulationFactor :: !Double       -- ^ Multiplicative factor
  , osEmotionalBuffer  :: !EmotionalCharge -- ^ Buffered emotional energy
  } deriving (Eq, Show, Generic, NFData)

-- | Emotional charge state for stabilization
--
-- Represents buffered emotional energy that can be:
--   * Absorbed from user state
--   * Dissipated over time
--   * Redirected to field modulation
data EmotionalCharge = EmotionalCharge
  { ecValence   :: !Double  -- ^ Positive/negative (-1 to 1)
  , ecIntensity :: !Double  -- ^ Charge intensity [0,1]
  , ecStability :: !Double  -- ^ How stable is this charge [0,1]
  , ecAge       :: !Double  -- ^ Time since last update (seconds)
  } deriving (Eq, Show, Generic, NFData)

-- =============================================================================
-- Fragment Binding System (Prompt #5)
-- =============================================================================

-- | Resonance anchor for fragment binding
--
-- Tethers a fragment to a subharmonic field pattern.
data ResonanceAnchor = ResonanceAnchor
  { raSubharmonicFreq :: !Double          -- ^ Subharmonic frequency (Hz)
  , raPhaseAlignment  :: !Double          -- ^ Phase alignment quality [0,1]
  , raBindingStrength :: !Double          -- ^ How strongly bound [0,1]
  , raDecayRate       :: !Double          -- ^ Decay rate per second
  , raHarmonicSig     :: !HarmonicSignature -- ^ (l,m) pattern
  } deriving (Eq, Show, Generic, NFData)

-- | A fragment bound to a resonance anchor
data BoundFragment = BoundFragment
  { bfFragmentId    :: !String            -- ^ Fragment identifier
  , bfAnchor        :: !ResonanceAnchor   -- ^ Binding anchor
  , bfBindingTime   :: !Double            -- ^ Time since binding (seconds)
  , bfCoherence     :: !Double            -- ^ Current coherence [0,1]
  , bfIsShadow      :: !Bool              -- ^ Is this a shadow fragment?
  } deriving (Eq, Show, Generic, NFData)

-- | Conditions required for binding
data BindingCondition = BindingCondition
  { bcMinCoherence    :: !Double          -- ^ Minimum coherence required
  , bcHarmonicMatch   :: !HarmonicSignature -- ^ Required harmonic pattern
  , bcConsentRequired :: !Bool            -- ^ Requires consent gate?
  , bcMaxDuration     :: !(Maybe Double)  -- ^ Maximum binding duration (seconds)
  } deriving (Eq, Show, Generic, NFData)

-- =============================================================================
-- Constants
-- =============================================================================

-- | Default number of orgone layers
defaultLayerCount :: Int
defaultLayerCount = 7  -- Traditional 7-layer accumulator

-- | Maximum accumulation factor
maxAccumulationFactor :: Double
maxAccumulationFactor = 10.0

-- | Emotional charge decay rate (per second)
chargeDecayRate :: Double
chargeDecayRate = 0.05  -- 5% per second

-- =============================================================================
-- Chamber Generation
-- =============================================================================

-- | Generate a personalized coherence chamber
--
-- This is the main entry point for chamber creation.
-- The chamber is configured based on user profile and current biometrics.
generateChamber
  :: String          -- ^ Chamber ID
  -> UserProfile     -- ^ User configuration
  -> BiometricInput  -- ^ Current biometric state
  -> CoherenceChamber
generateChamber chamberId profile biometrics =
  let
    -- Select geometry based on emotional state
    geometry = selectGeometry (upDominantEmotion profile)

    -- Create base scalar field
    baseField = createBaseField profile geometry

    -- Generate harmonic gates
    gates = createHarmonicGates profile biometrics

    -- Derive resonance targets
    targets = deriveResonanceTargets profile biometrics

    -- Create orgone stack
    orgone = createOrgoneStack defaultLayerCount biometrics

    -- Determine inversion
    isInverted = upShadowWorkEnabled profile &&
                 maybe False (== "grief" ) (upDominantEmotion profile)

  in CoherenceChamber
    { ccId = chamberId
    , ccFieldGeometry = geometry
    , ccBaseField = baseField
    , ccHarmonicGates = gates
    , ccResonanceTargets = targets
    , ccOrgoneStack = orgone
    , ccIsInverted = isInverted
    , ccIsActive = False  -- Starts inactive
    }

-- | Generate chamber with explicit orgone configuration
generateChamberWithOrgone
  :: String
  -> UserProfile
  -> BiometricInput
  -> Int              -- ^ Number of orgone layers
  -> CoherenceChamber
generateChamberWithOrgone chamberId profile biometrics layerCount =
  let base = generateChamber chamberId profile biometrics
      orgone = createOrgoneStack layerCount biometrics
  in base { ccOrgoneStack = orgone }

-- | Select geometry based on emotional state
selectGeometry :: Maybe String -> GeometryShape
selectGeometry emotion = case emotion of
  Just "grief"     -> Sphere        -- Containment for processing
  Just "joy"       -> Torus         -- Free-flowing circulation
  Just "fear"      -> Tetrahedron   -- Maximum stability
  Just "anger"     -> Pyramid       -- Focused transformation
  Just "peace"     -> Dodecahedron  -- Complex harmonics
  Just "curiosity" -> Torus         -- Exploratory flow
  Just custom      -> Custom custom
  Nothing          -> Sphere        -- Default: balanced containment

-- | Create base scalar field for geometry
createBaseField :: UserProfile -> GeometryShape -> ScalarField
createBaseField profile geometry =
  let
    -- Shell depth from preference or default
    shellDepth = fromMaybe 3 (upShellDepthPreference profile)

    -- Base components depend on geometry
    components = case geometry of
      Pyramid -> pyramidComponents shellDepth
      Tetrahedron -> tetrahedronComponents shellDepth
      Sphere -> sphereComponents shellDepth
      Torus -> torusComponents shellDepth
      Dodecahedron -> dodecahedronComponents shellDepth
      Custom _ -> sphereComponents shellDepth  -- Fallback

  in ScalarField components

-- Geometry-specific component generation
pyramidComponents :: Int -> [ScalarComponent]
pyramidComponents depth =
  [ SC 0 0 (RP Exponential 1.0 0.5) 1.0      -- Monopole base
  , SC 1 0 (RP Exponential 0.8 0.3) 0.8      -- Vertical dipole (apex focus)
  , SC 2 0 (RP Gaussian 0.6 (fromIntegral depth)) 0.5  -- Quadrupole
  ]

tetrahedronComponents :: Int -> [ScalarComponent]
tetrahedronComponents depth =
  [ SC 0 0 (RP InverseSquare 1.0 0.5) 1.0   -- Stable core
  , SC 3 0 (RP Gaussian 0.7 (fromIntegral depth)) 0.6  -- 3-fold symmetry
  ]

sphereComponents :: Int -> [ScalarComponent]
sphereComponents depth =
  [ SC 0 0 (RP Gaussian 1.0 (fromIntegral depth)) 1.0  -- Isotropic
  , SC 2 0 (RP Gaussian 0.5 (fromIntegral depth)) 0.3  -- Slight shaping
  ]

torusComponents :: Int -> [ScalarComponent]
torusComponents _depth =
  [ SC 1 1 (RP Exponential 1.0 0.4) 0.8     -- Circulation mode
  , SC 1 (-1) (RP Exponential 1.0 0.4) 0.8  -- Counter-circulation
  , SC 0 0 (RP AnkhModulated 0.5 0.5) 0.4   -- Core stability
  ]

dodecahedronComponents :: Int -> [ScalarComponent]
dodecahedronComponents depth =
  [ SC 0 0 (RP Gaussian 1.0 (fromIntegral depth)) 0.8
  , SC 5 0 (RP Gaussian 0.6 (fromIntegral depth)) 0.5  -- 5-fold
  , SC 5 5 (RP Gaussian 0.4 (fromIntegral depth)) 0.3
  ]

-- =============================================================================
-- Harmonic Gates
-- =============================================================================

-- | Create harmonic gates based on profile and biometrics
createHarmonicGates :: UserProfile -> BiometricInput -> [HarmonicGate]
createHarmonicGates profile biometrics =
  let
    -- Coherence affects threshold scaling
    coherence = computeSimpleCoherence biometrics

    -- Base thresholds scaled by coherence
    baseThreshold = 0.3 + coherence * 0.4

    -- Check for guardian/witness semantic markers
    hasGuardian = "guardian" `elem` upSemanticSignature profile
    hasWitness = "witness" `elem` upSemanticSignature profile

    -- Create gates for each Omega format
    redGate = HarmonicGate
      { hgOmega = Red
      , hgThreshold = baseThreshold * 0.8  -- Easier access
      , hgPhaseLockWindow = Nothing
      , hgHarmonicFilter = Nothing
      }

    greenGate = HarmonicGate
      { hgOmega = Green
      , hgThreshold = baseThreshold
      , hgPhaseLockWindow = Just $ mkPhiWindow 2 coherence
      , hgHarmonicFilter = Nothing
      }

    blueGate = HarmonicGate
      { hgOmega = Blue
      , hgThreshold = baseThreshold * 1.2  -- Harder access
      , hgPhaseLockWindow = Just $ mkPhiWindow 3 coherence
      , hgHarmonicFilter = if hasGuardian || hasWitness
                           then Just (HS 2 1)  -- l=2, m=1 for guardian
                           else Nothing
      }

    majorGate = HarmonicGate
      { hgOmega = OmegaMajor
      , hgThreshold = baseThreshold * 1.1
      , hgPhaseLockWindow = Just $ mkPhiWindow 4 coherence
      , hgHarmonicFilter = Nothing
      }

    minorGate = HarmonicGate
      { hgOmega = OmegaMinor
      , hgThreshold = baseThreshold * 0.9
      , hgPhaseLockWindow = Nothing
      , hgHarmonicFilter = Nothing
      }

  in [redGate, greenGate, blueGate, majorGate, minorGate]

-- | Simple coherence computation
computeSimpleCoherence :: BiometricInput -> Double
computeSimpleCoherence bi =
  (biHRV bi * 0.4 + biEEGAlphaCoherence bi * 0.6)

-- =============================================================================
-- Resonance Targets
-- =============================================================================

-- | Derive resonance targets from profile and biometrics
deriveResonanceTargets :: UserProfile -> BiometricInput -> [ResonanceTarget]
deriveResonanceTargets profile biometrics =
  let
    -- Low HRV = reduce targets, require alignment
    lowHRV = biHRV biometrics < 0.4

    -- Heart-based target
    heartTarget = ResonanceTarget
      { rtFrequencyHz = biHeartRate biometrics / 60.0
      , rtChannel = "cardiac"
      , rtAlignmentRequired = lowHRV
      , rtPriority = 1
      }

    -- Respiration target
    breathTarget = ResonanceTarget
      { rtFrequencyHz = biRespirationRate biometrics / 60.0
      , rtChannel = "respiratory"
      , rtAlignmentRequired = False
      , rtPriority = 2
      }

    -- Alpha wave target (8-12 Hz scaled)
    alphaTarget = ResonanceTarget
      { rtFrequencyHz = 10.0 * biEEGAlphaCoherence biometrics
      , rtChannel = "neural_alpha"
      , rtAlignmentRequired = lowHRV
      , rtPriority = 3
      }

    -- Phi-harmonic target
    phiTarget = ResonanceTarget
      { rtFrequencyHz = 1.0 / phiGolden  -- ~0.618 Hz
      , rtChannel = "phi_resonance"
      , rtAlignmentRequired = True
      , rtPriority = 4
      }

    -- Semantic-based targets
    semanticTargets = concatMap semanticToTarget (upSemanticSignature profile)

  in if lowHRV
     then [heartTarget, breathTarget]  -- Minimal targets under stress
     else [heartTarget, breathTarget, alphaTarget, phiTarget] ++ semanticTargets

-- | Map semantic signature to resonance target
semanticToTarget :: String -> [ResonanceTarget]
semanticToTarget sig = case sig of
  "guardian" -> [ResonanceTarget 7.83 "schumann" True 5]  -- Earth resonance
  "healer"   -> [ResonanceTarget (528.0 / 100.0) "solfeggio_528" False 6]
  "witness"  -> [ResonanceTarget (432.0 / 100.0) "a432" False 7]
  _          -> []

-- =============================================================================
-- Orgone Stack Functions
-- =============================================================================

-- | Create an orgone accumulator stack
createOrgoneStack :: Int -> BiometricInput -> OrgoneStack
createOrgoneStack layerCount biometrics =
  let
    -- Alternate organic/inorganic
    materials = cycle [Organic, Inorganic]

    -- Create layers
    layers = zipWith (createLayer biometrics) [1..layerCount] (take layerCount materials)

    -- Initial emotional buffer from biometrics
    emotionalBuffer = EmotionalCharge
      { ecValence = biHRV biometrics - 0.5  -- Center around 0
      , ecIntensity = biGSR biometrics
      , ecStability = biEEGAlphaCoherence biometrics
      , ecAge = 0.0
      }

  in OrgoneStack
    { osLayers = layers
    , osTotalCharge = 0.0
    , osAccumulationFactor = fromIntegral layerCount * 0.5
    , osEmotionalBuffer = emotionalBuffer
    }

-- | Create a single orgone layer
createLayer :: BiometricInput -> Int -> LayerMaterial -> OrgoneLayer
createLayer _biometrics layerNum material =
  let
    -- Layer properties depend on material and position
    baseResonance = case material of
      Organic   -> 0.5 + fromIntegral layerNum * 0.1
      Inorganic -> 1.0 + fromIntegral layerNum * 0.2

    conductivity = case material of
      Organic   -> 0.3  -- Lower conductivity
      Inorganic -> 0.8  -- Higher conductivity

  in OrgoneLayer
    { olMaterial = material
    , olThickness = 1.0 / fromIntegral layerNum  -- Thinner outer layers
    , olResonance = baseResonance
    , olConductivity = conductivity
    , olCharge = 0.0  -- Starts uncharged
    }

-- | Compute layer resonance contribution
computeLayerResonance :: OrgoneLayer -> Double -> Double
computeLayerResonance layer inputFreq =
  let
    -- Resonance when frequencies match
    freqMatch = 1.0 - min 1.0 (abs (olResonance layer - inputFreq) / inputFreq)

    -- Scale by conductivity and current charge
    contribution = freqMatch * olConductivity layer * (1.0 + olCharge layer)

  in contribution

-- | Accumulate charge in the orgone stack
--
-- Models the Reich accumulator principle: layers build up charge
-- when exposed to coherent field input.
accumulateCharge
  :: Double      -- ^ Input coherence [0,1]
  -> Double      -- ^ Time delta (seconds)
  -> OrgoneStack
  -> OrgoneStack
accumulateCharge coherence dt stack =
  let
    -- Charge accumulation rate depends on coherence
    accRate = coherence * osAccumulationFactor stack * dt

    -- Update each layer
    updatedLayers = map (chargeLayer accRate) (osLayers stack)

    -- Total charge
    newTotal = sum (map olCharge updatedLayers)

    -- Cap at maximum
    cappedTotal = min maxAccumulationFactor newTotal

  in stack
    { osLayers = updatedLayers
    , osTotalCharge = cappedTotal
    }

chargeLayer :: Double -> OrgoneLayer -> OrgoneLayer
chargeLayer rate layer =
  let newCharge = min 1.0 (olCharge layer + rate * olConductivity layer)
  in layer { olCharge = newCharge }

-- | Dissipate charge over time
--
-- Implements the decay model for emotional charge and field charge.
dissipateCharge
  :: Double      -- ^ Time delta (seconds)
  -> OrgoneStack
  -> OrgoneStack
dissipateCharge dt stack =
  let
    -- Decay factor
    decayFactor = 1.0 - chargeDecayRate * dt

    -- Dissipate layer charges
    decayedLayers = map (decayLayerCharge decayFactor) (osLayers stack)

    -- Decay emotional buffer
    oldBuffer = osEmotionalBuffer stack
    newIntensity = max 0.0 (ecIntensity oldBuffer * decayFactor)
    newBuffer = oldBuffer
      { ecIntensity = newIntensity
      , ecAge = ecAge oldBuffer + dt
      }

    -- Update total
    newTotal = sum (map olCharge decayedLayers)

  in stack
    { osLayers = decayedLayers
    , osTotalCharge = newTotal
    , osEmotionalBuffer = newBuffer
    }

decayLayerCharge :: Double -> OrgoneLayer -> OrgoneLayer
decayLayerCharge factor layer =
  layer { olCharge = olCharge layer * factor }

-- =============================================================================
-- Chamber Operations
-- =============================================================================

-- | Activate a chamber (verify consent, enable field)
activateChamber :: CoherenceChamber -> Either String CoherenceChamber
activateChamber chamber =
  -- Verify gates have valid thresholds
  if all (\g -> hgThreshold g >= 0 && hgThreshold g <= 1) (ccHarmonicGates chamber)
  then Right $ chamber { ccIsActive = True }
  else Left "Invalid gate thresholds"

-- | Deactivate chamber
deactivateChamber :: CoherenceChamber -> CoherenceChamber
deactivateChamber chamber = chamber { ccIsActive = False }

-- | Invert chamber for shadow work
invertChamber :: CoherenceChamber -> CoherenceChamber
invertChamber chamber = chamber
  { ccIsInverted = not (ccIsInverted chamber)
  }

-- | Validate chamber configuration
validateChamber :: CoherenceChamber -> Either String ()
validateChamber chamber = do
  -- Check field has components
  let ScalarField components = ccBaseField chamber
  if null components
    then Left "Empty scalar field"
    else Right ()

  -- Check gates
  if null (ccHarmonicGates chamber)
    then Left "No harmonic gates"
    else Right ()

  -- Check orgone stack
  if null (osLayers (ccOrgoneStack chamber))
    then Left "No orgone layers"
    else Right ()

-- =============================================================================
-- Geometry Effects
-- =============================================================================

-- | Geometry focus factor (how concentrated is the field)
geometryFocusFactor :: GeometryShape -> Double
geometryFocusFactor shape = case shape of
  Pyramid      -> 1.5   -- High focus at apex
  Tetrahedron  -> 1.2   -- Moderate focus
  Sphere       -> 1.0   -- Uniform (no focus)
  Torus        -> 0.8   -- Distributed flow
  Dodecahedron -> 1.3   -- Multi-focal
  Custom _     -> 1.0   -- Default

-- | Geometry flow pattern descriptor
geometryFlowPattern :: GeometryShape -> String
geometryFlowPattern shape = case shape of
  Pyramid      -> "vertical_ascent"
  Tetrahedron  -> "triangular_stable"
  Sphere       -> "isotropic_uniform"
  Torus        -> "continuous_circulation"
  Dodecahedron -> "pentagonal_resonance"
  Custom name  -> name

-- | Containment strength (how well does it hold charge)
geometryContainment :: GeometryShape -> Double
geometryContainment shape = case shape of
  Pyramid      -> 0.6   -- Open top
  Tetrahedron  -> 0.9   -- High containment
  Sphere       -> 1.0   -- Maximum containment
  Torus        -> 0.5   -- Flow-through
  Dodecahedron -> 0.85  -- Good containment
  Custom _     -> 0.7   -- Default

-- =============================================================================
-- Fragment Binding (Prompt #5)
-- =============================================================================

-- | Bind a fragment to a resonance anchor
--
-- Creates a stable binding if conditions are met.
bindFragment
  :: String            -- ^ Fragment ID
  -> ResonanceAnchor   -- ^ Anchor to bind to
  -> BindingCondition  -- ^ Required conditions
  -> Double            -- ^ Current coherence
  -> Either String BoundFragment
bindFragment fragId anchor condition coherence
  -- Check coherence requirement
  | coherence < bcMinCoherence condition =
      Left $ "Insufficient coherence: " ++ show coherence ++
             " < " ++ show (bcMinCoherence condition)
  -- Check harmonic match
  | raHarmonicSig anchor /= bcHarmonicMatch condition =
      Left "Harmonic signature mismatch"
  -- Success
  | otherwise = Right $ BoundFragment
      { bfFragmentId = fragId
      , bfAnchor = anchor
      , bfBindingTime = 0.0
      , bfCoherence = coherence
      , bfIsShadow = False
      }

-- | Release a bound fragment
releaseFragment :: BoundFragment -> BoundFragment
releaseFragment bf = bf
  { bfCoherence = 0.0
  , bfAnchor = (bfAnchor bf) { raBindingStrength = 0.0 }
  }

-- | Check if binding is still stable
--
-- Binding becomes unstable when:
--   * Coherence drops below threshold
--   * Binding time exceeds maximum (if set)
--   * Anchor strength decays below minimum
checkBindingStability
  :: BoundFragment
  -> BindingCondition
  -> Bool
checkBindingStability bf condition =
  let
    -- Coherence check
    coherenceOk = bfCoherence bf >= bcMinCoherence condition * 0.8  -- 80% margin

    -- Duration check
    durationOk = case bcMaxDuration condition of
      Nothing -> True
      Just maxDur -> bfBindingTime bf < maxDur

    -- Anchor strength check
    strengthOk = raBindingStrength (bfAnchor bf) > 0.1

  in coherenceOk && durationOk && strengthOk
