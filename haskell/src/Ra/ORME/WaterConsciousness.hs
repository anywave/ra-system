{-|
Module      : Ra.ORME.WaterConsciousness
Description : ORME-Water consciousness field loopback integration
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Integrates orbitally-rearranged monoatomic elements (ORME) with water memory
and consciousness field effects. Creates loopback systems for amplifying
coherence through ORME-infused water structures.

== ORME-Water Theory

=== Water Memory Integration

* Structured water holds frequency imprints
* ORME elements stabilize water coherence
* Scalar field coupling through H-bond networks
* Consciousness intention amplification

=== Loopback Mechanisms

1. ORME resonance initialization
2. Water structure harmonization
3. Consciousness field imprint
4. Scalar loopback amplification
5. Coherence stabilization cycle
-}
module Ra.ORME.WaterConsciousness
  ( -- * Core Types
    WaterConsciousnessField(..)
  , ORMEInfusion(..)
  , WaterStructure(..)
  , LoopbackState(..)

    -- * Field Creation
  , createConsciousnessField
  , initializeORMEWater
  , activateField

    -- * ORME Configuration
  , ORMEElement(..)
  , configureORME
  , setORMEConcentration
  , ormeResonance

    -- * Water Structure
  , structureWater
  , waterCoherence
  , imprint
  , clearImprint

    -- * Loopback Control
  , LoopbackMode(..)
  , startLoopback
  , stopLoopback
  , loopbackCycle
  , amplificationFactor

    -- * Consciousness Integration
  , ConsciousnessImprint(..)
  , imprintIntention
  , readImprint
  , mergeImprints

    -- * Field Monitoring
  , fieldStrength
  , structureIntegrity
  , resonanceQuality
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete ORME-Water consciousness field
data WaterConsciousnessField = WaterConsciousnessField
  { wcfORME         :: !ORMEInfusion           -- ^ ORME configuration
  , wcfStructure    :: !WaterStructure         -- ^ Water structure state
  , wcfLoopback     :: !LoopbackState          -- ^ Loopback system state
  , wcfImprints     :: ![ConsciousnessImprint] -- ^ Active imprints
  , wcfCoherence    :: !Double                 -- ^ Overall coherence
  , wcfActive       :: !Bool                   -- ^ Field active flag
  } deriving (Eq, Show)

-- | ORME infusion configuration
data ORMEInfusion = ORMEInfusion
  { oiElements      :: ![(ORMEElement, Double)]  -- ^ Elements and concentrations
  , oiTotalConc     :: !Double                   -- ^ Total concentration (ppm)
  , oiResonance     :: !Double                   -- ^ Primary resonance (Hz)
  , oiStability     :: !Double                   -- ^ Stability factor [0, 1]
  , oiActivated     :: !Bool                     -- ^ Activation state
  } deriving (Eq, Show)

-- | ORME element types
data ORMEElement
  = ORMEGold        -- ^ Monoatomic gold (most common)
  | ORMEPlatinum    -- ^ Monoatomic platinum
  | ORMERhodium     -- ^ Monoatomic rhodium
  | ORMEIridium     -- ^ Monoatomic iridium
  | ORMEPalladium   -- ^ Monoatomic palladium
  | ORMERuthenium   -- ^ Monoatomic ruthenium
  | ORMEOsmium      -- ^ Monoatomic osmium
  | ORMEMixed       -- ^ Mixed ORME blend
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Water structure state
data WaterStructure = WaterStructure
  { wsHexagonalRatio :: !Double    -- ^ Hexagonal cluster ratio [0, 1]
  , wsCoherence      :: !Double    -- ^ Structural coherence
  , wsBondStrength   :: !Double    -- ^ Average H-bond strength
  , wsMemoryDepth    :: !Int       -- ^ Memory imprint depth (layers)
  , wsTemperature    :: !Double    -- ^ Water temperature (C)
  , wsPH             :: !Double    -- ^ pH level
  } deriving (Eq, Show)

-- | Loopback system state
data LoopbackState = LoopbackState
  { lsMode          :: !LoopbackMode   -- ^ Current mode
  , lsCycles        :: !Int            -- ^ Completed cycles
  , lsAmplification :: !Double         -- ^ Current amplification
  , lsFrequency     :: !Double         -- ^ Loop frequency (Hz)
  , lsPhase         :: !Double         -- ^ Phase offset [0, 2pi]
  , lsStable        :: !Bool           -- ^ Stability indicator
  } deriving (Eq, Show)

-- | Loopback operating modes
data LoopbackMode
  = LoopbackOff       -- ^ System inactive
  | LoopbackSlow      -- ^ Low frequency cycling
  | LoopbackFast      -- ^ High frequency cycling
  | LoopbackResonant  -- ^ Auto-resonant mode
  | LoopbackBurst     -- ^ Burst amplification
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Field Creation
-- =============================================================================

-- | Create new consciousness field
createConsciousnessField :: WaterConsciousnessField
createConsciousnessField = WaterConsciousnessField
  { wcfORME = defaultORME
  , wcfStructure = defaultWaterStructure
  , wcfLoopback = defaultLoopback
  , wcfImprints = []
  , wcfCoherence = 0.5
  , wcfActive = False
  }

-- | Initialize ORME-infused water
initializeORMEWater :: WaterConsciousnessField -> [(ORMEElement, Double)] -> WaterConsciousnessField
initializeORMEWater field elements =
  let totalConc = sum (map snd elements)
      resonance = calculateORMEResonance elements
      newORME = (wcfORME field)
        { oiElements = elements
        , oiTotalConc = totalConc
        , oiResonance = resonance
        , oiActivated = True
        }
      newStructure = structureWaterForORME (wcfStructure field) totalConc
  in field { wcfORME = newORME, wcfStructure = newStructure }

-- | Activate the consciousness field
activateField :: WaterConsciousnessField -> WaterConsciousnessField
activateField field =
  let ormeActivated = oiActivated (wcfORME field)
      structureReady = wsCoherence (wcfStructure field) > phiInverse
      newCoherence = if ormeActivated && structureReady
                     then calculateFieldCoherence field
                     else wcfCoherence field * 0.8
  in field { wcfActive = ormeActivated && structureReady
           , wcfCoherence = newCoherence
           }

-- =============================================================================
-- ORME Configuration
-- =============================================================================

-- | Configure ORME parameters
configureORME :: WaterConsciousnessField -> ORMEInfusion -> WaterConsciousnessField
configureORME field ormeConfig =
  let newStructure = structureWaterForORME (wcfStructure field) (oiTotalConc ormeConfig)
  in field { wcfORME = ormeConfig, wcfStructure = newStructure }

-- | Set ORME concentration for specific element
setORMEConcentration :: WaterConsciousnessField -> ORMEElement -> Double -> WaterConsciousnessField
setORMEConcentration field element conc =
  let ormeInf = wcfORME field
      newElements = updateElement (oiElements ormeInf) element conc
      newTotal = sum (map snd newElements)
      newResonance = calculateORMEResonance newElements
  in field { wcfORME = ormeInf
    { oiElements = newElements
    , oiTotalConc = newTotal
    , oiResonance = newResonance
    }}

-- | Get ORME resonance frequency
ormeResonance :: WaterConsciousnessField -> Double
ormeResonance = oiResonance . wcfORME

-- =============================================================================
-- Water Structure
-- =============================================================================

-- | Structure water with specific coherence target
structureWater :: WaterConsciousnessField -> Double -> WaterConsciousnessField
structureWater field targetCoherence =
  let ws = wcfStructure field
      hexRatio = targetCoherence * phi / 2  -- Max ~0.809 hexagonal
      newStructure = ws
        { wsHexagonalRatio = min 1.0 hexRatio
        , wsCoherence = targetCoherence
        , wsBondStrength = 1 + targetCoherence * 0.5
        }
  in field { wcfStructure = newStructure }

-- | Get water coherence level
waterCoherence :: WaterConsciousnessField -> Double
waterCoherence = wsCoherence . wcfStructure

-- | Imprint pattern onto water structure
imprint :: WaterConsciousnessField -> ConsciousnessImprint -> WaterConsciousnessField
imprint field imprintData =
  let ws = wcfStructure field
      newDepth = min 7 (wsMemoryDepth ws + 1)  -- Max 7 layers
      newStructure = ws { wsMemoryDepth = newDepth }
  in field
    { wcfStructure = newStructure
    , wcfImprints = imprintData : take 6 (wcfImprints field)
    }

-- | Clear all imprints
clearImprint :: WaterConsciousnessField -> WaterConsciousnessField
clearImprint field =
  let ws = wcfStructure field
      clearedStructure = ws { wsMemoryDepth = 0 }
  in field { wcfStructure = clearedStructure, wcfImprints = [] }

-- =============================================================================
-- Loopback Control
-- =============================================================================

-- | Start loopback system
startLoopback :: WaterConsciousnessField -> LoopbackMode -> WaterConsciousnessField
startLoopback field mode =
  let ls = wcfLoopback field
      freq = loopbackFrequency mode (oiResonance (wcfORME field))
      newLoopback = ls
        { lsMode = mode
        , lsCycles = 0
        , lsFrequency = freq
        , lsAmplification = 1.0
        , lsStable = True
        }
  in field { wcfLoopback = newLoopback }

-- | Stop loopback system
stopLoopback :: WaterConsciousnessField -> WaterConsciousnessField
stopLoopback field =
  let ls = wcfLoopback field
      newLoopback = ls { lsMode = LoopbackOff, lsStable = True }
  in field { wcfLoopback = newLoopback }

-- | Execute one loopback cycle
loopbackCycle :: WaterConsciousnessField -> WaterConsciousnessField
loopbackCycle field =
  let ls = wcfLoopback field
      ormeRes = oiResonance (wcfORME field)
      waterCoh = wsCoherence (wcfStructure field)

      -- Calculate new amplification
      resonanceMatch = 1 - abs (lsFrequency ls - ormeRes) / max 1 ormeRes
      newAmp = lsAmplification ls * (1 + resonanceMatch * waterCoh * 0.1)

      -- Update phase
      newPhase = (lsPhase ls + 2 * pi / 100) `mod'` (2 * pi)

      -- Stability check
      stable = newAmp < 10 && newAmp > 0.1

      newLoopback = ls
        { lsCycles = lsCycles ls + 1
        , lsAmplification = min 10 (max 0.1 newAmp)
        , lsPhase = newPhase
        , lsStable = stable
        }

      -- Update field coherence based on loopback
      cohBoost = if lsStable newLoopback then lsAmplification newLoopback * 0.01 else 0
      newCoherence = min 1.0 (wcfCoherence field + cohBoost)

  in field { wcfLoopback = newLoopback, wcfCoherence = newCoherence }

-- | Get current amplification factor
amplificationFactor :: WaterConsciousnessField -> Double
amplificationFactor = lsAmplification . wcfLoopback

-- =============================================================================
-- Consciousness Integration
-- =============================================================================

-- | Consciousness imprint data
data ConsciousnessImprint = ConsciousnessImprint
  { ciIntention    :: !String          -- ^ Intention description
  , ciFrequency    :: !Double          -- ^ Carrier frequency
  , ciIntensity    :: !Double          -- ^ Imprint intensity [0, 1]
  , ciTimestamp    :: !Int             -- ^ Creation time
  , ciSignature    :: !(Double, Double, Double)  -- ^ Frequency signature (l, m, n)
  } deriving (Eq, Show)

-- | Imprint intention onto field
imprintIntention :: WaterConsciousnessField -> String -> Double -> WaterConsciousnessField
imprintIntention field intention intensity =
  let ormeRes = oiResonance (wcfORME field)
      waterCoh = wsCoherence (wcfStructure field)
      imprintData = ConsciousnessImprint
        { ciIntention = intention
        , ciFrequency = ormeRes * phi
        , ciIntensity = intensity * waterCoh
        , ciTimestamp = lsCycles (wcfLoopback field)
        , ciSignature = (phi, phiInverse, intensity)
        }
  in imprint field imprintData

-- | Read imprint at specific depth
readImprint :: WaterConsciousnessField -> Int -> Maybe ConsciousnessImprint
readImprint field depth =
  if depth >= 0 && depth < length (wcfImprints field)
  then Just (wcfImprints field !! depth)
  else Nothing

-- | Merge multiple imprints
mergeImprints :: [ConsciousnessImprint] -> Maybe ConsciousnessImprint
mergeImprints [] = Nothing
mergeImprints [x] = Just x
mergeImprints imprints =
  let avgFreq = sum (map ciFrequency imprints) / fromIntegral (length imprints)
      avgIntensity = sum (map ciIntensity imprints) / fromIntegral (length imprints)
      (ls, ms, ns) = unzip3 (map ciSignature imprints)
      avgSig = (sum ls / fromIntegral (length ls),
                sum ms / fromIntegral (length ms),
                sum ns / fromIntegral (length ns))
  in Just ConsciousnessImprint
    { ciIntention = "merged(" ++ show (length imprints) ++ ")"
    , ciFrequency = avgFreq
    , ciIntensity = avgIntensity
    , ciTimestamp = maximum (map ciTimestamp imprints)
    , ciSignature = avgSig
    }

-- =============================================================================
-- Field Monitoring
-- =============================================================================

-- | Get overall field strength
fieldStrength :: WaterConsciousnessField -> Double
fieldStrength field =
  let ormeFactor = oiStability (wcfORME field) * (if oiActivated (wcfORME field) then 1 else 0.2)
      waterFactor = wsCoherence (wcfStructure field) * wsHexagonalRatio (wcfStructure field)
      loopFactor = if lsStable (wcfLoopback field) then lsAmplification (wcfLoopback field) else 0.5
      imprintFactor = fromIntegral (length (wcfImprints field)) / 7
  in wcfCoherence field * ormeFactor * waterFactor * loopFactor * (1 + imprintFactor * 0.2)

-- | Get structure integrity
structureIntegrity :: WaterConsciousnessField -> Double
structureIntegrity field =
  let ws = wcfStructure field
      tempFactor = if wsTemperature ws >= 0 && wsTemperature ws <= 25 then 1.0 else 0.8
      phFactor = if wsPH ws >= 6.5 && wsPH ws <= 8.5 then 1.0 else 0.7
  in wsCoherence ws * wsBondStrength ws * tempFactor * phFactor / 2

-- | Get resonance quality
resonanceQuality :: WaterConsciousnessField -> Double
resonanceQuality field =
  let ormeRes = oiResonance (wcfORME field)
      loopFreq = lsFrequency (wcfLoopback field)
      match = 1 - abs (ormeRes - loopFreq) / max 1 (max ormeRes loopFreq)
  in match * wcfCoherence field

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Default ORME configuration
defaultORME :: ORMEInfusion
defaultORME = ORMEInfusion
  { oiElements = [(ORMEGold, 5.0)]  -- 5 ppm gold
  , oiTotalConc = 5.0
  , oiResonance = 528  -- Love frequency
  , oiStability = 0.8
  , oiActivated = False
  }

-- | Default water structure
defaultWaterStructure :: WaterStructure
defaultWaterStructure = WaterStructure
  { wsHexagonalRatio = 0.3
  , wsCoherence = 0.5
  , wsBondStrength = 1.0
  , wsMemoryDepth = 0
  , wsTemperature = 20.0
  , wsPH = 7.0
  }

-- | Default loopback state
defaultLoopback :: LoopbackState
defaultLoopback = LoopbackState
  { lsMode = LoopbackOff
  , lsCycles = 0
  , lsAmplification = 1.0
  , lsFrequency = 7.83  -- Schumann
  , lsPhase = 0
  , lsStable = True
  }

-- | Calculate ORME resonance from elements
calculateORMEResonance :: [(ORMEElement, Double)] -> Double
calculateORMEResonance [] = 528
calculateORMEResonance elements =
  let weighted = [elementResonance e * c | (e, c) <- elements]
      totalConc = sum (map snd elements)
  in if totalConc > 0 then sum weighted / totalConc else 528

-- | Get resonance frequency for ORME element
elementResonance :: ORMEElement -> Double
elementResonance ORMEGold = 528        -- DNA repair frequency
elementResonance ORMEPlatinum = 639    -- Connecting relationships
elementResonance ORMERhodium = 741     -- Awakening intuition
elementResonance ORMEIridium = 852     -- Spiritual order
elementResonance ORMEPalladium = 417   -- Facilitating change
elementResonance ORMERuthenium = 396   -- Liberation from fear
elementResonance ORMEOsmium = 963      -- Divine consciousness
elementResonance ORMEMixed = 528 * phi -- Golden frequency

-- | Structure water for ORME concentration
structureWaterForORME :: WaterStructure -> Double -> WaterStructure
structureWaterForORME ws conc =
  let concFactor = min 1.0 (conc / 20)  -- Normalize to 20 ppm
      newHex = wsHexagonalRatio ws + concFactor * (1 - wsHexagonalRatio ws) * 0.3
      newCoh = wsCoherence ws + concFactor * 0.1
  in ws { wsHexagonalRatio = min 1.0 newHex, wsCoherence = min 1.0 newCoh }

-- | Update element in list
updateElement :: [(ORMEElement, Double)] -> ORMEElement -> Double -> [(ORMEElement, Double)]
updateElement [] element conc = [(element, conc)]
updateElement ((e, c):rest) element conc
  | e == element = (e, conc) : rest
  | otherwise = (e, c) : updateElement rest element conc

-- | Get loopback frequency for mode
loopbackFrequency :: LoopbackMode -> Double -> Double
loopbackFrequency LoopbackOff base = base
loopbackFrequency LoopbackSlow base = base / phi
loopbackFrequency LoopbackFast base = base * phi
loopbackFrequency LoopbackResonant base = base
loopbackFrequency LoopbackBurst base = base * phi * phi

-- | Calculate field coherence
calculateFieldCoherence :: WaterConsciousnessField -> Double
calculateFieldCoherence field =
  let ormeCoh = oiStability (wcfORME field)
      waterCoh = wsCoherence (wcfStructure field)
  in (ormeCoh + waterCoh) / 2 * phi

-- | Modulo for Double
mod' :: Double -> Double -> Double
mod' x m = x - m * fromIntegral (floor (x / m) :: Int)
