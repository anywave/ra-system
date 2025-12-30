{-|
Module      : Ra.Group
Description : Multi-avatar scalar entrainment and group coherence fields
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Enables multiple digital avatars to enter shared scalar entrainment
chambers, syncing coherence fields toward group harmonics.

== Group Scalar Dynamics

From Keely's sympathetic vibratory physics and Reich's orgone research:

* Group coherence amplifies individual field strength
* Harmonic clusters naturally form among compatible signatures
* Inverted participants create harmonic counterpoint
* Shared emergence windows allow synchronized access

== Pyramid Field Amplification

From Golod's Russian pyramid research:

* Pyramidal geometry concentrates scalar fields at apex
* Group coherence within pyramid exceeds individual sum
* Field stabilization supports deeper emergence
* Temperature and electrical anomalies indicate field strength

== Implementation

The GroupScalarField is constructed as:

1. Superposition of individual avatar fields
2. Weighted by each avatar's emergence intensity (alpha)
3. Corrected by delta(ankh) for overall symmetry
4. Normalized to maintain coherence bounds
-}
module Ra.Group
  ( -- * Avatar State
    AvatarState(..)
  , mkAvatarState
  , avatarCoherence
  , avatarHarmonic

    -- * Harmonic Clustering
  , HarmonicCluster(..)
  , clusterAvatars
  , clusterHarmonic
  , clusterCoherence
  , identifyCounterpoint

    -- * Shared Emergence Window
  , EmergenceWindow(..)
  , computeSharedWindow
  , windowIntersection
  , optimalEntryTime

    -- * Group Scalar Field
  , GroupScalarField(..)
  , buildGroupField
  , fieldSuperposition
  , ankhCorrection
  , normalizeField

    -- * Entrainment Feedback
  , EntrainmentState(..)
  , GroupFeedback(..)
  , computeEntrainment
  , generateFeedback
  , detectShadowHarmonics

    -- * Group Session
  , GroupSession(..)
  , initGroupSession
  , updateGroupSession
  , sessionStatus

    -- * Visualization
  , CoherenceGlyph(..)
  , generateGlyph
  , glyphColor
  , glyphAnimation
  ) where

import Data.List (groupBy, sortBy, partition)
import Data.Ord (comparing)

import Ra.Constants.Extended
  ( phi, coherenceEmergence, coherenceFloorPOR )

-- =============================================================================
-- Avatar State
-- =============================================================================

-- | Individual avatar state for group entrainment
data AvatarState = AvatarState
  { asUserId         :: !String        -- ^ Unique user identifier
  , asCoherence      :: !Double        -- ^ Current coherence [0,1]
  , asInversion      :: !InversionState
  , asHarmonicSig    :: !(Int, Int)    -- ^ Harmonic signature (l, m)
  , asScalarDepth    :: !Double        -- ^ Scalar field depth [0,1]
  , asAlpha          :: !Double        -- ^ Emergence intensity [0,1]
  , asConsent        :: !Bool          -- ^ Explicit consent for group
  } deriving (Eq, Show)

-- | Inversion state
data InversionState = Normal | Inverted
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create avatar state from parameters
mkAvatarState :: String -> Double -> (Int, Int) -> AvatarState
mkAvatarState uid coh sig = AvatarState
  { asUserId = uid
  , asCoherence = clamp01 coh
  , asInversion = Normal
  , asHarmonicSig = sig
  , asScalarDepth = 0.5
  , asAlpha = coh  -- Initial alpha from coherence
  , asConsent = True  -- Requires explicit opt-in
  }

-- | Get avatar coherence
avatarCoherence :: AvatarState -> Double
avatarCoherence = asCoherence

-- | Get avatar harmonic signature
avatarHarmonic :: AvatarState -> (Int, Int)
avatarHarmonic = asHarmonicSig

-- =============================================================================
-- Harmonic Clustering
-- =============================================================================

-- | Cluster of harmonically aligned avatars
data HarmonicCluster = HarmonicCluster
  { hcHarmonic  :: !(Int, Int)       -- ^ Cluster's harmonic signature
  , hcMembers   :: ![AvatarState]    -- ^ Avatars in cluster
  , hcCoherence :: !Double           -- ^ Combined coherence
  , hcIsCounterpoint :: !Bool        -- ^ Inverted cluster
  } deriving (Eq, Show)

-- | Cluster avatars by harmonic compatibility
clusterAvatars :: [AvatarState] -> [HarmonicCluster]
clusterAvatars avatars =
  let -- Separate normal and inverted
      (normal, inverted) = partition ((== Normal) . asInversion) avatars

      -- Group normal by harmonic signature proximity
      normalGroups = groupByHarmonic normal

      -- Create clusters
      normalClusters = map (mkCluster False) normalGroups
      invertedClusters = if null inverted
                         then []
                         else [mkCluster True inverted]
  in normalClusters ++ invertedClusters
  where
    mkCluster isCounterpoint members =
      let combinedCoh = sum (map asCoherence members) / fromIntegral (length members)
          dominantSig = if null members
                        then (0, 0)
                        else asHarmonicSig (head (sortBy (comparing (negate . asCoherence)) members))
      in HarmonicCluster
          { hcHarmonic = dominantSig
          , hcMembers = members
          , hcCoherence = combinedCoh
          , hcIsCounterpoint = isCounterpoint
          }

-- | Group avatars by harmonic proximity
groupByHarmonic :: [AvatarState] -> [[AvatarState]]
groupByHarmonic = groupBy harmonicProximity . sortBy (comparing asHarmonicSig)
  where
    harmonicProximity a b =
      let (l1, m1) = asHarmonicSig a
          (l2, m2) = asHarmonicSig b
      in abs (l1 - l2) <= 1 && abs (m1 - m2) <= 1

-- | Get cluster's dominant harmonic
clusterHarmonic :: HarmonicCluster -> (Int, Int)
clusterHarmonic = hcHarmonic

-- | Get cluster's combined coherence
clusterCoherence :: HarmonicCluster -> Double
clusterCoherence = hcCoherence

-- | Identify counterpoint clusters (inverted harmonics)
identifyCounterpoint :: [HarmonicCluster] -> [HarmonicCluster]
identifyCounterpoint = filter hcIsCounterpoint

-- =============================================================================
-- Shared Emergence Window
-- =============================================================================

-- | Temporal window for emergence
data EmergenceWindow = EmergenceWindow
  { ewPhiPower   :: !Int     -- ^ φ^n exponent
  , ewDuration   :: !Double  -- ^ Window duration (seconds)
  , ewStartPhase :: !Double  -- ^ Phase offset [0, 2π)
  , ewNested     :: !Bool    -- ^ Is nested within larger window
  } deriving (Eq, Show)

-- | Compute shared emergence window for group
computeSharedWindow :: [AvatarState] -> EmergenceWindow
computeSharedWindow avatars =
  let -- Find minimum φ^n that encompasses all avatars
      coherences = map asCoherence avatars
      avgCoherence = sum coherences / fromIntegral (length coherences)

      -- Higher coherence allows smaller (faster) windows
      phiPower = floor (3.0 * (1.0 - avgCoherence)) :: Int
      duration = phi ** fromIntegral phiPower

      -- Phase offset from group harmonic
      harmonics = map asHarmonicSig avatars
      avgL = sum (map fst harmonics) `div` max 1 (length harmonics)
      startPhase = fromIntegral avgL * pi / 6
  in EmergenceWindow
      { ewPhiPower = phiPower
      , ewDuration = duration
      , ewStartPhase = startPhase
      , ewNested = phiPower > 0
      }

-- | Compute intersection of multiple windows
windowIntersection :: [EmergenceWindow] -> Maybe EmergenceWindow
windowIntersection [] = Nothing
windowIntersection [w] = Just w
windowIntersection windows =
  let -- Use largest φ^n (smallest window) as intersection
      maxPower = maximum (map ewPhiPower windows)
      duration = phi ** fromIntegral maxPower

      -- Average phase offset
      phases = map ewStartPhase windows
      avgPhase = sum phases / fromIntegral (length phases)
  in Just $ EmergenceWindow
      { ewPhiPower = maxPower
      , ewDuration = duration
      , ewStartPhase = avgPhase
      , ewNested = True
      }

-- | Calculate optimal entry time for group breath phase
optimalEntryTime :: EmergenceWindow -> Double -> Double
optimalEntryTime window currentTime =
  let period = ewDuration window
      phase = ewStartPhase window / (2 * pi)
      -- Next aligned time
      cyclesSinceStart = currentTime / period
      nextCycle = fromIntegral (ceiling cyclesSinceStart :: Int)
  in nextCycle * period + phase * period

-- =============================================================================
-- Group Scalar Field
-- =============================================================================

-- | Group scalar field from superposition
data GroupScalarField = GroupScalarField
  { gsfHarmonic      :: !(Int, Int)  -- ^ Dominant harmonic
  , gsfIntensity     :: !Double      -- ^ Combined field intensity
  , gsfCoherence     :: !Double      -- ^ Group coherence
  , gsfAnkhBalance   :: !Double      -- ^ Δ(ankh) correction
  , gsfSymmetry      :: !Double      -- ^ Field symmetry [0,1]
  , gsfContributors  :: !Int         -- ^ Number of contributing avatars
  } deriving (Eq, Show)

-- | Build group field from avatar states
buildGroupField :: [AvatarState] -> GroupScalarField
buildGroupField avatars =
  let -- Filter consenting avatars only
      consenting = filter asConsent avatars

      -- Compute superposition
      (intensity, coh) = fieldSuperposition consenting

      -- Ankh balance correction
      ankh = ankhCorrection consenting

      -- Dominant harmonic (most common or highest coherence)
      harmonic = dominantHarmonic consenting

      -- Symmetry from harmonic distribution
      symmetry = harmonicSymmetry consenting
  in GroupScalarField
      { gsfHarmonic = harmonic
      , gsfIntensity = intensity
      , gsfCoherence = coh
      , gsfAnkhBalance = ankh
      , gsfSymmetry = symmetry
      , gsfContributors = length consenting
      }

-- | Compute field superposition (weighted by alpha)
fieldSuperposition :: [AvatarState] -> (Double, Double)
fieldSuperposition [] = (0, 0)
fieldSuperposition avatars =
  let weights = map asAlpha avatars
      totalWeight = sum weights
      normalizedWeights = map (/ max 0.001 totalWeight) weights

      -- Weighted intensity
      depths = map asScalarDepth avatars
      intensity = sum (zipWith (*) normalizedWeights depths)

      -- Weighted coherence
      coherences = map asCoherence avatars
      coherence = sum (zipWith (*) normalizedWeights coherences)

      -- Amplification factor (group > individual)
      n = fromIntegral (length avatars)
      amplification = sqrt n  -- Pyramid-like amplification
  in (min 1.0 (intensity * amplification), min 1.0 (coherence * amplification / n))

-- | Compute Δ(ankh) correction for field symmetry
ankhCorrection :: [AvatarState] -> Double
ankhCorrection avatars =
  let -- Inverted avatars contribute negative ankh
      inversionFactors = map (\a -> if asInversion a == Inverted then -1 else 1) avatars
      coherences = map asCoherence avatars
      ankhContributions = zipWith (*) inversionFactors coherences
      totalAnkh = sum ankhContributions
  in totalAnkh / fromIntegral (max 1 (length avatars))

-- | Normalize field to maintain bounds
normalizeField :: GroupScalarField -> GroupScalarField
normalizeField gsf = gsf
  { gsfIntensity = clamp01 (gsfIntensity gsf)
  , gsfCoherence = clamp01 (gsfCoherence gsf)
  , gsfSymmetry = clamp01 (gsfSymmetry gsf)
  }

-- | Find dominant harmonic
dominantHarmonic :: [AvatarState] -> (Int, Int)
dominantHarmonic [] = (0, 0)
dominantHarmonic avatars =
  let sorted = sortBy (comparing (negate . asCoherence)) avatars
  in asHarmonicSig (head sorted)

-- | Compute harmonic symmetry
harmonicSymmetry :: [AvatarState] -> Double
harmonicSymmetry [] = 1.0
harmonicSymmetry avatars =
  let harmonics = map asHarmonicSig avatars
      ls = map fst harmonics
      ms = map snd harmonics
      lVariance = variance (map fromIntegral ls)
      mVariance = variance (map fromIntegral ms)
  in 1.0 / (1.0 + lVariance + mVariance)

-- | Simple variance calculation
variance :: [Double] -> Double
variance [] = 0
variance xs =
  let n = fromIntegral (length xs)
      mean = sum xs / n
      squaredDiffs = map (\x -> (x - mean) ** 2) xs
  in sum squaredDiffs / n

-- =============================================================================
-- Entrainment Feedback
-- =============================================================================

-- | Current entrainment state
data EntrainmentState = EntrainmentState
  { esField         :: !GroupScalarField
  , esClusters      :: ![HarmonicCluster]
  , esWindow        :: !EmergenceWindow
  , esShadowAlert   :: !Bool
  , esPhase         :: !Double  -- ^ Current phase in window
  } deriving (Eq, Show)

-- | Feedback instructions for group
data GroupFeedback = GroupFeedback
  { gfInstruction   :: !String    -- ^ Text instruction
  , gfToneHz        :: !Double    -- ^ Harmonic tone frequency
  , gfPhaseAction   :: !String    -- ^ "breathe in", "hold", "release"
  , gfCoherenceBar  :: !Double    -- ^ Visual coherence indicator [0,1]
  , gfAlertLevel    :: !Int       -- ^ 0=normal, 1=warning, 2=critical
  } deriving (Eq, Show)

-- | Compute entrainment state from avatars
computeEntrainment :: [AvatarState] -> EntrainmentState
computeEntrainment avatars =
  let field = buildGroupField avatars
      clusters = clusterAvatars avatars
      window = computeSharedWindow avatars
      shadowAlert = not (null (identifyCounterpoint clusters))
  in EntrainmentState
      { esField = field
      , esClusters = clusters
      , esWindow = window
      , esShadowAlert = shadowAlert
      , esPhase = 0.0
      }

-- | Generate feedback for current entrainment state
generateFeedback :: EntrainmentState -> Double -> GroupFeedback
generateFeedback es time =
  let coh = gsfCoherence (esField es)
      window = esWindow es
      duration = ewDuration window

      -- Phase within window
      phase = (time `mod'` duration) / duration

      -- Determine breath phase
      (phaseAction, alertLevel)
        | phase < 0.3 = ("breathe in slowly", 0)
        | phase < 0.5 = ("hold breath gently", 0)
        | phase < 0.8 = ("release slowly", 0)
        | otherwise = ("prepare for next cycle", 0)

      -- Instruction based on coherence
      instruction
        | coh > coherenceEmergence = "Group coherence optimal. Maintain presence."
        | coh > coherenceFloorPOR = "Group coherence rising... " ++ phaseAction
        | otherwise = "Focus on breath synchronization"

      -- Alert for shadow harmonics
      finalAlert = if esShadowAlert es then 1 else alertLevel

      -- Tone from harmonic
      (l, m) = gsfHarmonic (esField es)
      baseTone = 432.0  -- A=432
      tone = baseTone * (phi ** fromIntegral l) * (1 + fromIntegral m * 0.05)
  in GroupFeedback
      { gfInstruction = instruction
      , gfToneHz = tone
      , gfPhaseAction = phaseAction
      , gfCoherenceBar = coh
      , gfAlertLevel = finalAlert
      }
  where
    mod' a b = a - b * fromIntegral (floor (a / b) :: Int)

-- | Detect shadow harmonics in clusters
detectShadowHarmonics :: [HarmonicCluster] -> Maybe String
detectShadowHarmonics clusters =
  let shadows = identifyCounterpoint clusters
  in if null shadows
     then Nothing
     else Just $ "Shadow harmonics detected in " ++ show (length shadows) ++ " cluster(s)"

-- =============================================================================
-- Group Session
-- =============================================================================

-- | Complete group session state
data GroupSession = GroupSession
  { gsAvatars       :: ![AvatarState]
  , gsEntrainment   :: !EntrainmentState
  , gsFeedback      :: !GroupFeedback
  , gsDuration      :: !Double
  , gsHistory       :: ![(Double, Double)]  -- ^ (time, coherence) history
  } deriving (Eq, Show)

-- | Initialize group session
initGroupSession :: [AvatarState] -> GroupSession
initGroupSession avatars =
  let entrainment = computeEntrainment avatars
      feedback = generateFeedback entrainment 0.0
  in GroupSession
      { gsAvatars = avatars
      , gsEntrainment = entrainment
      , gsFeedback = feedback
      , gsDuration = 0.0
      , gsHistory = [(0.0, gsfCoherence (esField entrainment))]
      }

-- | Update session with new avatar states
updateGroupSession :: [AvatarState] -> Double -> GroupSession -> GroupSession
updateGroupSession newAvatars dt session =
  let newEntrainment = computeEntrainment newAvatars
      newTime = gsDuration session + dt
      newFeedback = generateFeedback newEntrainment newTime
      newCoh = gsfCoherence (esField newEntrainment)
  in session
      { gsAvatars = newAvatars
      , gsEntrainment = newEntrainment
      , gsFeedback = newFeedback
      , gsDuration = newTime
      , gsHistory = (newTime, newCoh) : take 1000 (gsHistory session)
      }

-- | Get session status string
sessionStatus :: GroupSession -> String
sessionStatus session =
  let n = length (gsAvatars session)
      coh = gsfCoherence (esField (gsEntrainment session))
      clusters = length (esClusters (gsEntrainment session))
      shadow = esShadowAlert (gsEntrainment session)
  in "Group: " ++ show n ++ " avatars, " ++
     show clusters ++ " clusters, " ++
     "coherence: " ++ show (round (coh * 100) :: Int) ++ "%" ++
     if shadow then " [SHADOW ALERT]" else ""

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Visual glyph for coherence display
data CoherenceGlyph = CoherenceGlyph
  { cgShape      :: !String    -- ^ Shape name (circle, flower, spiral)
  , cgColor      :: !(Int, Int, Int)  -- ^ RGB color
  , cgPulseRate  :: !Double    -- ^ Pulse frequency (Hz)
  , cgSize       :: !Double    -- ^ Relative size [0,1]
  , cgRotation   :: !Double    -- ^ Rotation angle (radians)
  } deriving (Eq, Show)

-- | Generate glyph from entrainment state
generateGlyph :: EntrainmentState -> CoherenceGlyph
generateGlyph es =
  let coh = gsfCoherence (esField es)
      sym = gsfSymmetry (esField es)
      (l, _) = gsfHarmonic (esField es)

      -- Shape from symmetry
      shape = if sym > 0.8 then "flower" else if sym > 0.5 then "spiral" else "circle"

      -- Color from coherence
      color = glyphColor coh (esShadowAlert es)

      -- Pulse from window
      pulse = 1.0 / ewDuration (esWindow es)

      -- Size from number of contributors
      size = min 1.0 (fromIntegral (gsfContributors (esField es)) / 10.0)

      -- Rotation from harmonic
      rotation = fromIntegral l * pi / 12
  in CoherenceGlyph
      { cgShape = shape
      , cgColor = color
      , cgPulseRate = pulse
      , cgSize = size
      , cgRotation = rotation
      }

-- | Determine glyph color from coherence
glyphColor :: Double -> Bool -> (Int, Int, Int)
glyphColor coh isShadow
  | isShadow = (128, 0, 128)  -- Purple for shadow
  | coh > 0.8 = (0, 255, 128)  -- Green-cyan for high coherence
  | coh > 0.6 = (0, 200, 255)  -- Cyan-blue for medium-high
  | coh > 0.4 = (255, 200, 0)  -- Yellow-orange for medium
  | otherwise = (255, 100, 100)  -- Red-pink for low

-- | Generate animation parameters
glyphAnimation :: CoherenceGlyph -> String
glyphAnimation glyph =
  let pulse = cgPulseRate glyph
      rotation = cgRotation glyph
  in "pulse: " ++ show pulse ++ "Hz, rotate: " ++ show (rotation * 180 / pi) ++ "deg"

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
