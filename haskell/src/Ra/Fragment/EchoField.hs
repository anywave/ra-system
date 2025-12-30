{-|
Module      : Ra.Fragment.EchoField
Description : Fragment resonance echo and persistence mechanics
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements fragment echo fields - the resonance patterns that persist after
fragment emergence and can influence future emergences. Echo fields create
"memory" in the scalar field that affects coherence patterns.

== Echo Field Theory

=== Persistence Mechanics

When a fragment emerges, it leaves an echo in the scalar field:

* Echo intensity decays by φ^(-n) over time
* Echoes can reinforce or interfere with new emergences
* Multiple echoes create complex interference patterns

=== Echo Types

1. Constructive Echo: Reinforces similar fragments
2. Destructive Echo: Inhibits opposing fragments
3. Neutral Echo: Passive field imprint
4. Resonant Echo: Creates standing wave patterns
-}
module Ra.Fragment.EchoField
  ( -- * Core Types
    EchoField(..)
  , EchoPoint(..)
  , EchoType(..)
  , EchoIntensity(..)

    -- * Echo Generation
  , generateEcho
  , fragmentToEcho
  , combineEchoes

    -- * Echo Decay
  , decayEcho
  , echoHalfLife
  , persistenceTime

    -- * Field Interactions
  , echoInterference
  , constructiveSum
  , destructiveSum

    -- * Echo Detection
  , detectEchoes
  , echoSignature
  , matchEcho

    -- * Standing Waves
  , StandingWave(..)
  , formStandingWave
  , waveNodes

    -- * Echo Memory
  , EchoMemory(..)
  , recordEcho
  , recallEcho
  , pruneMemory

    -- * Field Integration
  , applyEchoField
  , fieldWithEchoes
  , clearEchoes
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.List (sortBy)
import Data.Ord (comparing)

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Echo field containing multiple echo points
data EchoField = EchoField
  { efPoints     :: ![EchoPoint]
  , efCenter     :: !(Double, Double, Double)
  , efRadius     :: !Double
  , efTotalPower :: !Double
  , efAge        :: !Int              -- ^ φ^n ticks since creation
  } deriving (Eq, Show)

-- | Single echo point in the field
data EchoPoint = EchoPoint
  { epPosition   :: !(Double, Double, Double)
  , epIntensity  :: !EchoIntensity
  , epType       :: !EchoType
  , epHarmonic   :: !(Int, Int)       -- ^ (l, m) signature
  , epPhase      :: !Double           -- ^ Phase at creation
  , epFragmentId :: !(Maybe String)   -- ^ Source fragment
  } deriving (Eq, Show)

-- | Echo type classification
data EchoType
  = EchoConstructive   -- ^ Reinforces similar patterns
  | EchoDestructive    -- ^ Inhibits opposing patterns
  | EchoNeutral        -- ^ Passive imprint
  | EchoResonant       -- ^ Creates standing waves
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Echo intensity with decay tracking
data EchoIntensity = EchoIntensity
  { eiInitial  :: !Double    -- ^ Initial intensity [0, 1]
  , eiCurrent  :: !Double    -- ^ Current intensity
  , eiDecayRate :: !Double   -- ^ Decay rate per tick
  } deriving (Eq, Show)

-- =============================================================================
-- Echo Generation
-- =============================================================================

-- | Generate echo from emergence event
generateEcho :: (Double, Double, Double)  -- ^ Position
             -> (Int, Int)                 -- ^ Harmonic (l, m)
             -> Double                     -- ^ Intensity
             -> Double                     -- ^ Phase
             -> EchoType                   -- ^ Echo type
             -> EchoPoint
generateEcho pos (l, m) intensity phase' echoType =
  let decayRate = phiInverse / fromIntegral (l + 1)
      echoIntensity = EchoIntensity intensity intensity decayRate
  in EchoPoint
    { epPosition = pos
    , epIntensity = echoIntensity
    , epType = echoType
    , epHarmonic = (l, m)
    , epPhase = phase'
    , epFragmentId = Nothing
    }

-- | Convert fragment emergence to echo
fragmentToEcho :: String                   -- ^ Fragment ID
               -> (Double, Double, Double) -- ^ Position
               -> Double                   -- ^ Coherence
               -> (Int, Int)               -- ^ Harmonic
               -> EchoPoint
fragmentToEcho fragId pos coherence (l, m) =
  let echoType = if coherence > phiInverse
                 then EchoConstructive
                 else EchoNeutral
      intensity = coherence * phi
      phase' = fromIntegral m * pi / fromIntegral (l + 1)
      echo = generateEcho pos (l, m) intensity phase' echoType
  in echo { epFragmentId = Just fragId }

-- | Combine multiple echoes into field
combineEchoes :: [EchoPoint] -> EchoField
combineEchoes [] = emptyEchoField
combineEchoes points =
  let positions = map epPosition points
      center = averagePosition positions
      maxDist = maximum $ map (distance center . epPosition) points
      totalPower = sum $ map (eiCurrent . epIntensity) points
  in EchoField
    { efPoints = points
    , efCenter = center
    , efRadius = maxDist
    , efTotalPower = totalPower
    , efAge = 0
    }

-- | Empty echo field
emptyEchoField :: EchoField
emptyEchoField = EchoField [] (0, 0, 0) 0 0 0

-- =============================================================================
-- Echo Decay
-- =============================================================================

-- | Apply decay to echo
decayEcho :: Int -> EchoPoint -> EchoPoint
decayEcho ticks ep =
  let intensity = epIntensity ep
      decayFactor = (1 - eiDecayRate intensity) ^ ticks
      newCurrent = eiInitial intensity * decayFactor
  in ep { epIntensity = intensity { eiCurrent = max 0 newCurrent } }

-- | Half-life of echo in φ^n ticks
echoHalfLife :: EchoPoint -> Int
echoHalfLife ep =
  let rate = eiDecayRate (epIntensity ep)
  in if rate > 0
     then ceiling (log 0.5 / log (1 - rate))
     else maxBound

-- | Time until echo fades below threshold
persistenceTime :: EchoPoint -> Double -> Int
persistenceTime ep threshold =
  let intensity = epIntensity ep
      rate = eiDecayRate intensity
      initial = eiInitial intensity
  in if rate > 0 && initial > threshold
     then ceiling (log (threshold / initial) / log (1 - rate))
     else 0

-- =============================================================================
-- Field Interactions
-- =============================================================================

-- | Compute interference between two echoes
echoInterference :: EchoPoint -> EchoPoint -> Double
echoInterference e1 e2 =
  let (l1, m1) = epHarmonic e1
      (l2, m2) = epHarmonic e2
      harmonicMatch = if l1 == l2 && m1 == m2 then 1.0 else 0.5
      phaseDiff = abs (epPhase e1 - epPhase e2)
      phaseMatch = cos phaseDiff
      i1 = eiCurrent (epIntensity e1)
      i2 = eiCurrent (epIntensity e2)
  in i1 * i2 * harmonicMatch * phaseMatch

-- | Constructive sum of echoes
constructiveSum :: [EchoPoint] -> Double
constructiveSum echoes =
  let intensities = map (eiCurrent . epIntensity) echoes
      phases = map epPhase echoes
      -- Sum with phase consideration
      realParts = zipWith (\i p -> i * cos p) intensities phases
      imagParts = zipWith (\i p -> i * sin p) intensities phases
      totalReal = sum realParts
      totalImag = sum imagParts
  in sqrt (totalReal ^ (2 :: Int) + totalImag ^ (2 :: Int))

-- | Destructive sum (cancellation)
destructiveSum :: [EchoPoint] -> Double
destructiveSum echoes =
  let intensities = map (eiCurrent . epIntensity) echoes
      totalIntensity = sum intensities
      constructive = constructiveSum echoes
  in totalIntensity - constructive

-- =============================================================================
-- Echo Detection
-- =============================================================================

-- | Detect echoes at location
detectEchoes :: (Double, Double, Double) -> EchoField -> [EchoPoint]
detectEchoes pos field =
  let searchRadius = efRadius field * 0.1
  in filter (\ep -> distance pos (epPosition ep) <= searchRadius) (efPoints field)

-- | Get echo signature (dominant harmonic)
echoSignature :: EchoField -> Maybe (Int, Int)
echoSignature field
  | null (efPoints field) = Nothing
  | otherwise =
      let sorted = sortBy (comparing (negate . eiCurrent . epIntensity)) (efPoints field)
          strongest = head sorted
      in Just (epHarmonic strongest)

-- | Match echo to fragment pattern
matchEcho :: EchoPoint -> (Int, Int) -> Double -> Bool
matchEcho ep (l, m) threshold =
  let (el, em) = epHarmonic ep
      harmonicMatch = el == l && em == m
      intensityOk = eiCurrent (epIntensity ep) >= threshold
  in harmonicMatch && intensityOk

-- =============================================================================
-- Standing Waves
-- =============================================================================

-- | Standing wave pattern
data StandingWave = StandingWave
  { swNodes      :: ![(Double, Double, Double)]  -- ^ Node positions
  , swAntinodes  :: ![(Double, Double, Double)]  -- ^ Antinode positions
  , swWavelength :: !Double
  , swAmplitude  :: !Double
  , swHarmonic   :: !(Int, Int)
  } deriving (Eq, Show)

-- | Form standing wave from resonant echoes
formStandingWave :: [EchoPoint] -> Maybe StandingWave
formStandingWave echoes =
  let resonant = filter ((== EchoResonant) . epType) echoes
  in if length resonant < 2
     then Nothing
     else
       let positions = map epPosition resonant
           harmonics = map epHarmonic resonant
           (l, m) = head harmonics
           wavelength = 2 * pi / fromIntegral (l + 1)
           amplitude = constructiveSum resonant
           nodes = computeNodes positions wavelength
           antinodes = computeAntinodes positions wavelength
       in Just StandingWave
         { swNodes = nodes
         , swAntinodes = antinodes
         , swWavelength = wavelength
         , swAmplitude = amplitude
         , swHarmonic = (l, m)
         }

-- | Get wave nodes (zero amplitude points)
waveNodes :: StandingWave -> [(Double, Double, Double)]
waveNodes = swNodes

-- | Compute node positions
computeNodes :: [(Double, Double, Double)] -> Double -> [(Double, Double, Double)]
computeNodes positions wavelength =
  let center = averagePosition positions
      (cx, cy, cz) = center
      halfWave = wavelength / 2
  in [(cx + halfWave * cos (fromIntegral i * pi / 4),
       cy + halfWave * sin (fromIntegral i * pi / 4),
       cz)
     | i <- [0..7 :: Int]]

-- | Compute antinode positions
computeAntinodes :: [(Double, Double, Double)] -> Double -> [(Double, Double, Double)]
computeAntinodes positions wavelength =
  let center = averagePosition positions
      (cx, cy, cz) = center
      quarterWave = wavelength / 4
  in [(cx + quarterWave * cos (fromIntegral i * pi / 4 + pi / 8),
       cy + quarterWave * sin (fromIntegral i * pi / 4 + pi / 8),
       cz)
     | i <- [0..7 :: Int]]

-- =============================================================================
-- Echo Memory
-- =============================================================================

-- | Echo memory storage
data EchoMemory = EchoMemory
  { emFields    :: !(Map String EchoField)  -- ^ Named echo fields
  , emHistory   :: ![EchoField]             -- ^ Historical fields
  , emMaxAge    :: !Int                     -- ^ Max age before pruning
  , emCapacity  :: !Int                     -- ^ Max fields in history
  } deriving (Eq, Show)

-- | Record echo field to memory
recordEcho :: String -> EchoField -> EchoMemory -> EchoMemory
recordEcho name field memory =
  let newFields = Map.insert name field (emFields memory)
      newHistory = field : take (emCapacity memory - 1) (emHistory memory)
  in memory { emFields = newFields, emHistory = newHistory }

-- | Recall echo field by name
recallEcho :: String -> EchoMemory -> Maybe EchoField
recallEcho name memory = Map.lookup name (emFields memory)

-- | Prune old echoes from memory
pruneMemory :: Int -> EchoMemory -> EchoMemory
pruneMemory currentAge memory =
  let maxAge = emMaxAge memory
      prunedFields = Map.filter (\f -> currentAge - efAge f <= maxAge) (emFields memory)
      prunedHistory = filter (\f -> currentAge - efAge f <= maxAge) (emHistory memory)
  in memory { emFields = prunedFields, emHistory = prunedHistory }

-- =============================================================================
-- Field Integration
-- =============================================================================

-- | Apply echo field effects to coherence
applyEchoField :: EchoField -> Double -> (Double, Double, Double) -> Double
applyEchoField field baseCoherence pos =
  let nearbyEchoes = detectEchoes pos field
      constructiveBoost = constructiveSum nearbyEchoes
      destructiveLoss = destructiveSum nearbyEchoes
      modifier = constructiveBoost * 0.1 - destructiveLoss * 0.05
  in max 0 (min 1 (baseCoherence + modifier))

-- | Create field with echo effects
fieldWithEchoes :: [(Double, Double, Double)] -> [EchoPoint] -> EchoField
fieldWithEchoes positions echoes =
  let field = combineEchoes echoes
      center = if null positions then efCenter field else averagePosition positions
      radius = if null positions
               then efRadius field
               else maximum $ map (distance center) positions
  in field { efCenter = center, efRadius = radius }

-- | Clear all echoes from field
clearEchoes :: EchoField -> EchoField
clearEchoes field = field { efPoints = [], efTotalPower = 0 }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Average position of points
averagePosition :: [(Double, Double, Double)] -> (Double, Double, Double)
averagePosition [] = (0, 0, 0)
averagePosition positions =
  let n = fromIntegral (length positions)
      (xs, ys, zs) = unzip3 positions
  in (sum xs / n, sum ys / n, sum zs / n)

-- | Distance between two points
distance :: (Double, Double, Double) -> (Double, Double, Double) -> Double
distance (x1, y1, z1) (x2, y2, z2) =
  sqrt ((x2 - x1) ^ (2 :: Int) + (y2 - y1) ^ (2 :: Int) + (z2 - z1) ^ (2 :: Int))
