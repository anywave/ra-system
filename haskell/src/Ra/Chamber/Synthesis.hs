{-|
Module      : Ra.Chamber.Synthesis
Description : Multi-layer scalar chamber generator
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Synthesizes multi-layer scalar coherence chambers based on biometric input,
fragment requirements, torsion tuning, and environmental resonance modeling.

== Chamber Synthesis Theory

Chambers are personalized field environments where fragments, avatars, and
experiences can safely and fully emerge. Synthesis balances:

* Biometric input (coherence, HRV phase)
* Scalar harmonics (l, m, φ^n)
* Ankh balance (symmetry delta)
* Torsion field state (normal, inverted, null)
* Fragment needs (emergence conditions)

== Layered Architecture

Chambers consist of nested ScalarLayers, each with:

* Specific harmonic signature (l, m)
* Radial depth profile
* Coherence band requirements
* Torsion tuning parameters
-}
module Ra.Chamber.Synthesis
  ( -- * Core Types
    ScalarChamber(..)
  , ScalarLayer(..)
  , CoherenceBand(..)
  , TorsionSignature(..)
  , FragmentRequirement(..)
  , BiometricField(..)
  , ChamberID
  , FragmentID

    -- * Chamber Synthesis
  , synthesizeChamber
  , synthesizeFromProfile
  , adjustChamber

    -- * Layer Operations
  , addLayer
  , removeLayer
  , modifyLayer
  , getLayer

    -- * Coherence Zones
  , computeCoherenceZone
  , isInCoherenceZone
  , zoneOverlap

    -- * Fragment Matching
  , matchFragmentToChamber
  , findCompatibleFragments
  , checkEmergenceThreshold

    -- * Resonance Analysis
  , computeResonancePeak
  , harmonicAlignment
  , torsionCompatibility

    -- * Visualization
  , ChamberVisualization(..)
  , exportChamberVisualization
  , chamberToShells

    -- * Warnings
  , ChamberWarning(..)
  , validateChamber
  ) where

import Data.List (sortBy)
import Data.Ord (comparing)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import qualified Data.Set as Set
import Data.Set (Set)

import Ra.Constants.Extended (phi)
import Ra.Omega (OmegaFormat(..))

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Chamber identifier
type ChamberID = String

-- | Fragment identifier
type FragmentID = String

-- | User identifier
type UserID = String

-- | Coherence band specification
data CoherenceBand = CoherenceBand
  { cbLow     :: !Double    -- ^ Lower bound [0, 1]
  , cbHigh    :: !Double    -- ^ Upper bound [0, 1]
  , cbOptimal :: !Double    -- ^ Optimal coherence level
  } deriving (Eq, Show)

-- | Torsion signature for chamber tuning
data TorsionSignature = TorsionSignature
  { tsPolarity  :: !TorsionPolarity
  , tsMagnitude :: !Double           -- ^ [0, 1]
  , tsPhase     :: !Double           -- ^ [0, 2π]
  } deriving (Eq, Show)

-- | Torsion polarity states
data TorsionPolarity
  = TorsionPositive   -- ^ Right-hand spin
  | TorsionNegative   -- ^ Left-hand spin
  | TorsionNeutral    -- ^ Balanced/null torsion
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Single layer in scalar chamber
data ScalarLayer = ScalarLayer
  { slHarmonicL    :: !Int            -- ^ Spherical harmonic degree
  , slHarmonicM    :: !Int            -- ^ Spherical harmonic order
  , slDepth        :: !Int            -- ^ φ^n depth index
  , slRadius       :: !Double         -- ^ Layer radius
  , slCoherence    :: !Double         -- ^ Required coherence [0, 1]
  , slTorsion      :: !TorsionSignature
  , slWeight       :: !Double         -- ^ Layer contribution weight
  } deriving (Eq, Show)

-- | Complete scalar chamber
data ScalarChamber = ScalarChamber
  { chamberId       :: !ChamberID
  , depthProfile    :: ![ScalarLayer]
  , coherenceZone   :: !CoherenceBand
  , resonancePeak   :: !Double
  , torsionTune     :: !TorsionSignature
  , harmonicRoots   :: ![OmegaFormat]
  , safeFor         :: !(Set FragmentID)
  , chamberMetadata :: !(Map String String)
  } deriving (Eq, Show)

-- | Fragment emergence requirements
data FragmentRequirement = FragmentRequirement
  { frFragmentId     :: !FragmentID
  , frMinCoherence   :: !Double
  , frMaxCoherence   :: !Double
  , frRequiredL      :: !(Maybe Int)
  , frRequiredM      :: !(Maybe Int)
  , frTorsionPref    :: !(Maybe TorsionPolarity)
  , frDepthRange     :: !(Int, Int)  -- ^ (minDepth, maxDepth)
  } deriving (Eq, Show)

-- | Biometric field input
data BiometricField = BiometricField
  { bfCoherence   :: !Double     -- ^ Current coherence [0, 1]
  , bfHRVPhase    :: !Double     -- ^ Heart rate variability phase
  , bfBreathPhase :: !Double     -- ^ Breath cycle phase
  , bfAnkhDelta   :: !Double     -- ^ Symmetry deviation
  , bfTorsion     :: !TorsionSignature
  } deriving (Eq, Show)

-- =============================================================================
-- Chamber Synthesis
-- =============================================================================

-- | Synthesize chamber from user biometrics and fragment requirements
synthesizeChamber :: UserID
                  -> BiometricField
                  -> [FragmentRequirement]
                  -> ScalarChamber
synthesizeChamber userId bioField reqs =
  let -- Compute optimal coherence zone
      cohZone = computeOptimalZone bioField reqs

      -- Generate layers based on requirements
      layers = generateLayers bioField reqs

      -- Compute resonance peak
      resPeak = computeResonancePeak layers

      -- Torsion from biometrics
      torsion = bfTorsion bioField

      -- Collect harmonic formats
      formats = selectHarmonicFormats reqs

      -- Fragments that can safely emerge
      safe = Set.fromList [frFragmentId r | r <- reqs, canEmerge cohZone r]

  in ScalarChamber
    { chamberId = "chamber-" ++ userId
    , depthProfile = layers
    , coherenceZone = cohZone
    , resonancePeak = resPeak
    , torsionTune = torsion
    , harmonicRoots = formats
    , safeFor = safe
    , chamberMetadata = Map.empty
    }

-- | Synthesize from predefined profile
synthesizeFromProfile :: ChamberProfile -> BiometricField -> ScalarChamber
synthesizeFromProfile profile bioField =
  let baseChamberId = cpName profile
      layers = cpLayers profile
      cohZone = adjustZoneForBio (cpCoherenceZone profile) bioField
      resPeak = computeResonancePeak layers
  in ScalarChamber
    { chamberId = baseChamberId
    , depthProfile = layers
    , coherenceZone = cohZone
    , resonancePeak = resPeak
    , torsionTune = bfTorsion bioField
    , harmonicRoots = cpFormats profile
    , safeFor = Set.empty
    , chamberMetadata = cpMetadata profile
    }

-- | Chamber profile for predefined configurations
data ChamberProfile = ChamberProfile
  { cpName          :: !String
  , cpLayers        :: ![ScalarLayer]
  , cpCoherenceZone :: !CoherenceBand
  , cpFormats       :: ![OmegaFormat]
  , cpMetadata      :: !(Map String String)
  } deriving (Eq, Show)

-- | Adjust existing chamber based on new biometrics
adjustChamber :: ScalarChamber -> BiometricField -> ScalarChamber
adjustChamber chamber bioField =
  let newZone = adjustZoneForBio (coherenceZone chamber) bioField
      newTorsion = blendTorsion (torsionTune chamber) (bfTorsion bioField)
      adjustedLayers = map (adjustLayerForBio bioField) (depthProfile chamber)
  in chamber
    { coherenceZone = newZone
    , torsionTune = newTorsion
    , depthProfile = adjustedLayers
    , resonancePeak = computeResonancePeak adjustedLayers
    }

-- =============================================================================
-- Layer Operations
-- =============================================================================

-- | Add layer to chamber
addLayer :: ScalarLayer -> ScalarChamber -> ScalarChamber
addLayer layer chamber =
  let newLayers = sortBy (comparing slDepth) (layer : depthProfile chamber)
  in chamber
    { depthProfile = newLayers
    , resonancePeak = computeResonancePeak newLayers
    }

-- | Remove layer at depth
removeLayer :: Int -> ScalarChamber -> ScalarChamber
removeLayer depth chamber =
  let newLayers = filter (\l -> slDepth l /= depth) (depthProfile chamber)
  in chamber
    { depthProfile = newLayers
    , resonancePeak = computeResonancePeak newLayers
    }

-- | Modify layer at depth
modifyLayer :: Int -> (ScalarLayer -> ScalarLayer) -> ScalarChamber -> ScalarChamber
modifyLayer depth f chamber =
  let newLayers = map (\l -> if slDepth l == depth then f l else l) (depthProfile chamber)
  in chamber
    { depthProfile = newLayers
    , resonancePeak = computeResonancePeak newLayers
    }

-- | Get layer at depth
getLayer :: Int -> ScalarChamber -> Maybe ScalarLayer
getLayer depth chamber =
  case filter (\l -> slDepth l == depth) (depthProfile chamber) of
    [layer] -> Just layer
    _ -> Nothing

-- =============================================================================
-- Coherence Zones
-- =============================================================================

-- | Compute optimal coherence zone from biometrics and requirements
computeCoherenceZone :: [ScalarLayer] -> CoherenceBand
computeCoherenceZone [] = CoherenceBand 0.0 1.0 0.618
computeCoherenceZone layers =
  let coherences = map slCoherence layers
      minC = minimum coherences
      maxC = maximum coherences
      optC = sum coherences / fromIntegral (length coherences)
  in CoherenceBand minC maxC optC

-- | Check if coherence is in zone
isInCoherenceZone :: Double -> CoherenceBand -> Bool
isInCoherenceZone coh band =
  coh >= cbLow band && coh <= cbHigh band

-- | Compute overlap between two zones
zoneOverlap :: CoherenceBand -> CoherenceBand -> Double
zoneOverlap b1 b2 =
  let overlapLow = max (cbLow b1) (cbLow b2)
      overlapHigh = min (cbHigh b1) (cbHigh b2)
  in if overlapLow <= overlapHigh
     then (overlapHigh - overlapLow) / max 0.001 (min (cbHigh b1 - cbLow b1) (cbHigh b2 - cbLow b2))
     else 0.0

-- =============================================================================
-- Fragment Matching
-- =============================================================================

-- | Match fragment to chamber
matchFragmentToChamber :: FragmentRequirement -> ScalarChamber -> Double
matchFragmentToChamber req chamber =
  let -- Coherence alignment
      cohScore = coherenceAlignment req (coherenceZone chamber)

      -- Harmonic alignment
      harmScore = harmonicAlignmentScore req (depthProfile chamber)

      -- Torsion compatibility
      torsScore = torsionCompatibilityScore (frTorsionPref req) (torsionTune chamber)

      -- Depth match
      depthScore = depthMatchScore req (depthProfile chamber)

  in (cohScore * 0.4 + harmScore * 0.3 + torsScore * 0.2 + depthScore * 0.1)

-- | Find all compatible fragments
findCompatibleFragments :: ScalarChamber -> [FragmentRequirement] -> [FragmentRequirement]
findCompatibleFragments chamber = filter (\r -> matchFragmentToChamber r chamber >= 0.5)

-- | Check if fragment meets emergence threshold
checkEmergenceThreshold :: FragmentRequirement -> ScalarChamber -> Bool
checkEmergenceThreshold req chamber =
  matchFragmentToChamber req chamber >= 0.618  -- φ⁻¹ threshold

-- =============================================================================
-- Resonance Analysis
-- =============================================================================

-- | Compute resonance peak from layers
computeResonancePeak :: [ScalarLayer] -> Double
computeResonancePeak [] = 0.0
computeResonancePeak layers =
  let -- Weighted sum of layer contributions
      weightedSum = sum [slCoherence l * slWeight l | l <- layers]
      totalWeight = sum [slWeight l | l <- layers]
  in if totalWeight > 0
     then weightedSum / totalWeight
     else 0.0

-- | Check harmonic alignment between chamber and target
harmonicAlignment :: ScalarChamber -> (Int, Int) -> Double
harmonicAlignment chamber (targetL, targetM) =
  let layers = depthProfile chamber
      alignments = [harmonicDistance (slHarmonicL l, slHarmonicM l) (targetL, targetM) | l <- layers]
  in if null alignments then 0.0 else maximum alignments

-- | Harmonic distance (inverse of alignment)
harmonicDistance :: (Int, Int) -> (Int, Int) -> Double
harmonicDistance (l1, m1) (l2, m2) =
  let lDist = abs (l1 - l2)
      mDist = abs (m1 - m2)
  in 1.0 / (1.0 + fromIntegral (lDist + mDist))

-- | Check torsion compatibility
torsionCompatibility :: TorsionSignature -> TorsionSignature -> Double
torsionCompatibility t1 t2 =
  let polarityMatch = if tsPolarity t1 == tsPolarity t2 then 1.0 else 0.5
      magDiff = abs (tsMagnitude t1 - tsMagnitude t2)
      phaseDiff = min (abs (tsPhase t1 - tsPhase t2)) (2 * pi - abs (tsPhase t1 - tsPhase t2))
  in polarityMatch * (1.0 - magDiff) * (1.0 - phaseDiff / pi)

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Chamber visualization data
data ChamberVisualization = ChamberVisualization
  { cvShells       :: ![ShellVisualization]
  , cvCoherenceMap :: ![[Double]]          -- ^ 2D coherence grid
  , cvTorsionFlow  :: ![(Double, Double)]  -- ^ Vector field
  , cvPulsePhase   :: !Double              -- ^ Current pulse phase
  } deriving (Eq, Show)

-- | Single shell visualization
data ShellVisualization = ShellVisualization
  { svRadius     :: !Double
  , svHarmonic   :: !(Int, Int)
  , svIntensity  :: !Double
  , svColor      :: !ShellColor
  } deriving (Eq, Show)

-- | Shell color based on harmonic
data ShellColor
  = ShellRed
  | ShellOrange
  | ShellYellow
  | ShellGreen
  | ShellBlue
  | ShellIndigo
  | ShellViolet
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Export chamber visualization
exportChamberVisualization :: ScalarChamber -> ChamberVisualization
exportChamberVisualization chamber =
  let shells = map layerToShell (depthProfile chamber)
      cohMap = generateCoherenceMap chamber
      torsFlow = generateTorsionFlow chamber
      pulse = resonancePeak chamber
  in ChamberVisualization shells cohMap torsFlow pulse

-- | Convert layer to shell visualization
layerToShell :: ScalarLayer -> ShellVisualization
layerToShell layer = ShellVisualization
  { svRadius = slRadius layer
  , svHarmonic = (slHarmonicL layer, slHarmonicM layer)
  , svIntensity = slCoherence layer * slWeight layer
  , svColor = harmonicToColor (slHarmonicL layer)
  }

-- | Map harmonic degree to color
harmonicToColor :: Int -> ShellColor
harmonicToColor l = case l `mod` 7 of
  0 -> ShellRed
  1 -> ShellOrange
  2 -> ShellYellow
  3 -> ShellGreen
  4 -> ShellBlue
  5 -> ShellIndigo
  _ -> ShellViolet

-- | Convert chamber to shell list
chamberToShells :: ScalarChamber -> [ShellVisualization]
chamberToShells = cvShells . exportChamberVisualization

-- =============================================================================
-- Warnings
-- =============================================================================

-- | Chamber validation warnings
data ChamberWarning
  = FragmentUnplaceable !FragmentID !String
  | CoherenceOutOfRange !Double !CoherenceBand
  | TorsionMismatch !TorsionSignature !TorsionSignature
  | LayerConflict !Int !Int
  | ResonanceUnstable !Double
  deriving (Eq, Show)

-- | Validate chamber configuration
validateChamber :: ScalarChamber -> [FragmentRequirement] -> [ChamberWarning]
validateChamber chamber reqs =
  let fragWarnings = concatMap (checkFragment chamber) reqs
      cohWarnings = checkCoherence chamber
      resWarnings = checkResonance chamber
  in fragWarnings ++ cohWarnings ++ resWarnings

-- | Check fragment placement
checkFragment :: ScalarChamber -> FragmentRequirement -> [ChamberWarning]
checkFragment chamber req =
  if matchFragmentToChamber req chamber < 0.3
  then [FragmentUnplaceable (frFragmentId req) "Coherence or harmonic mismatch"]
  else []

-- | Check coherence bounds
checkCoherence :: ScalarChamber -> [ChamberWarning]
checkCoherence chamber =
  let zone = coherenceZone chamber
  in if cbLow zone > cbHigh zone
     then [CoherenceOutOfRange (cbOptimal zone) zone]
     else []

-- | Check resonance stability
checkResonance :: ScalarChamber -> [ChamberWarning]
checkResonance chamber =
  if resonancePeak chamber < 0.1
  then [ResonanceUnstable (resonancePeak chamber)]
  else []

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Compute optimal zone from biometrics and requirements
computeOptimalZone :: BiometricField -> [FragmentRequirement] -> CoherenceBand
computeOptimalZone bioField reqs =
  let baseLow = bfCoherence bioField * 0.8
      baseHigh = min 1.0 (bfCoherence bioField * 1.2)
      reqLows = map frMinCoherence reqs
      reqHighs = map frMaxCoherence reqs
      finalLow = if null reqLows then baseLow else max baseLow (minimum reqLows)
      finalHigh = if null reqHighs then baseHigh else min baseHigh (maximum reqHighs)
  in CoherenceBand finalLow finalHigh (bfCoherence bioField)

-- | Generate layers from requirements
generateLayers :: BiometricField -> [FragmentRequirement] -> [ScalarLayer]
generateLayers bioField reqs =
  let baseTorsion = bfTorsion bioField
      baseCoherence = bfCoherence bioField
      layers = zipWith (makeLayer baseTorsion baseCoherence) [0..] reqs
  in sortBy (comparing slDepth) layers

-- | Make a layer from requirement
makeLayer :: TorsionSignature -> Double -> Int -> FragmentRequirement -> ScalarLayer
makeLayer torsion baseCoh idx req = ScalarLayer
  { slHarmonicL = maybe 0 id (frRequiredL req)
  , slHarmonicM = maybe 0 id (frRequiredM req)
  , slDepth = fst (frDepthRange req) + idx
  , slRadius = phi ** fromIntegral idx
  , slCoherence = min baseCoh (frMaxCoherence req)
  , slTorsion = adjustTorsionForPref torsion (frTorsionPref req)
  , slWeight = 1.0 / fromIntegral (idx + 1)
  }

-- | Adjust torsion for preference
adjustTorsionForPref :: TorsionSignature -> Maybe TorsionPolarity -> TorsionSignature
adjustTorsionForPref torsion Nothing = torsion
adjustTorsionForPref torsion (Just pref) = torsion { tsPolarity = pref }

-- | Check if fragment can emerge in zone
canEmerge :: CoherenceBand -> FragmentRequirement -> Bool
canEmerge zone req =
  frMinCoherence req <= cbHigh zone && frMaxCoherence req >= cbLow zone

-- | Select harmonic formats from requirements
selectHarmonicFormats :: [FragmentRequirement] -> [OmegaFormat]
selectHarmonicFormats _ = [Green, OmegaMajor, OmegaMinor]  -- Default

-- | Adjust zone for biometrics
adjustZoneForBio :: CoherenceBand -> BiometricField -> CoherenceBand
adjustZoneForBio zone bioField =
  let cohFactor = bfCoherence bioField
      newLow = max 0 (cbLow zone * cohFactor)
      newHigh = min 1 (cbHigh zone * (1 + (1 - cohFactor) * 0.2))
  in zone { cbLow = newLow, cbHigh = newHigh }

-- | Blend two torsion signatures
blendTorsion :: TorsionSignature -> TorsionSignature -> TorsionSignature
blendTorsion t1 t2 = TorsionSignature
  { tsPolarity = if tsMagnitude t1 >= tsMagnitude t2 then tsPolarity t1 else tsPolarity t2
  , tsMagnitude = (tsMagnitude t1 + tsMagnitude t2) / 2
  , tsPhase = (tsPhase t1 + tsPhase t2) / 2
  }

-- | Adjust layer for biometrics
adjustLayerForBio :: BiometricField -> ScalarLayer -> ScalarLayer
adjustLayerForBio bioField layer =
  layer { slCoherence = slCoherence layer * bfCoherence bioField }

-- | Coherence alignment score
coherenceAlignment :: FragmentRequirement -> CoherenceBand -> Double
coherenceAlignment req zone =
  let reqMid = (frMinCoherence req + frMaxCoherence req) / 2
      zoneMid = cbOptimal zone
  in 1.0 - min 1.0 (abs (reqMid - zoneMid))

-- | Harmonic alignment score
harmonicAlignmentScore :: FragmentRequirement -> [ScalarLayer] -> Double
harmonicAlignmentScore req layers =
  case (frRequiredL req, frRequiredM req) of
    (Just l, Just m) ->
      let matches = [harmonicDistance (slHarmonicL lay, slHarmonicM lay) (l, m) | lay <- layers]
      in if null matches then 0.0 else maximum matches
    _ -> 0.8  -- No requirement means good compatibility

-- | Torsion compatibility score
torsionCompatibilityScore :: Maybe TorsionPolarity -> TorsionSignature -> Double
torsionCompatibilityScore Nothing _ = 1.0
torsionCompatibilityScore (Just pref) torsion =
  if tsPolarity torsion == pref then 1.0 else 0.5

-- | Depth match score
depthMatchScore :: FragmentRequirement -> [ScalarLayer] -> Double
depthMatchScore req layers =
  let (minD, maxD) = frDepthRange req
      layerDepths = map slDepth layers
      inRange = filter (\d -> d >= minD && d <= maxD) layerDepths
  in fromIntegral (length inRange) / fromIntegral (max 1 (length layers))

-- | Generate coherence map (simplified grid)
generateCoherenceMap :: ScalarChamber -> [[Double]]
generateCoherenceMap chamber =
  let layers = depthProfile chamber
      gridSize = 10 :: Int
      grid = [[layerCoherenceAt i j layers | j <- [0..gridSize-1]] | i <- [0..gridSize-1]]
  in grid

-- | Get coherence at grid position
layerCoherenceAt :: Int -> Int -> [ScalarLayer] -> Double
layerCoherenceAt i j layers =
  let depth = (i + j) `mod` max 1 (length layers)
  in case drop depth layers of
    (l:_) -> slCoherence l
    [] -> 0.0

-- | Generate torsion flow vectors
generateTorsionFlow :: ScalarChamber -> [(Double, Double)]
generateTorsionFlow chamber =
  let torsion = torsionTune chamber
      mag = tsMagnitude torsion
      phase = tsPhase torsion
      -- Generate flow vectors based on torsion
      vectors = [(mag * cos (phase + fromIntegral i * 0.5),
                  mag * sin (phase + fromIntegral i * 0.5)) | i <- [0..7 :: Int]]
  in vectors
