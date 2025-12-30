{-|
Module      : Ra.Radionics
Description : Quantum radionics overlay for scalar pathway tuning
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Overlays radionics rate fields onto Ra coordinates to modulate
emergence outcomes using intention encoding, rate harmonics,
and field interaction principles.

== Radionics Principles

From dowsing and radionics research:

* Rates are numeric encodings of resonant frequencies
* Intention modulates rate effectiveness
* Projection radius defines field of influence
* Modulation modes: constructive, diffusive, directive

== Overlay Mechanics

Each overlay consists of:

* Base rate (numeric string, e.g., "568.12")
* Intention string (natural language)
* Projection radius (scalar radius units)
* Modulation mode

Claude maps each overlay to Ra coordinates using harmonic correlation.

== Intention Encoding

Semantic vectorization of human-entered intentions:

* "insight" -> boosts H_{2,2} nodes
* "protection" -> restricts emergence in shells > 3
* "clarity" -> enhances flux coherence
-}
module Ra.Radionics
  ( -- * Rate Types
    RadionicsRate(..)
  , mkRate
  , rateToFrequency
  , rateHarmonic

    -- * Overlay
  , RadionicsOverlay(..)
  , ModulationMode(..)
  , mkOverlay
  , overlayActive
  , overlayStrength

    -- * Intention Encoding
  , IntentionVector(..)
  , encodeIntention
  , intentionToModifier
  , knownIntentions

    -- * Scalar Pathway Modulation
  , PathwayModifier(..)
  , applyOverlay
  , potentialModifier
  , fluxStabilizer
  , ankhShift

    -- * Overlay Intersection
  , OverlayIntersection(..)
  , checkIntersection
  , resolveOverlaps
  , constructiveInterference
  , destructiveInterference

    -- * Consent Gating
  , OverlayConsent(..)
  , gateOverlay
  , overlayVetoed

    -- * Visualization
  , OverlayVisual(..)
  , overlayToVisual
  , visualColor
  , visualRadius
  ) where

import Data.Char (isDigit)
import Data.List (find)

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Rate Types
-- =============================================================================

-- | Radionics rate (numeric encoding)
data RadionicsRate = RadionicsRate
  { rrValue     :: !String    -- ^ Rate string (e.g., "568.12")
  , rrNumeric   :: !Double    -- ^ Parsed numeric value
  , rrHarmonic  :: !(Int, Int) -- ^ Mapped harmonic (l, m)
  } deriving (Eq, Show)

-- | Create rate from string
mkRate :: String -> Maybe RadionicsRate
mkRate s =
  case parseRate s of
    Nothing -> Nothing
    Just val -> Just $ RadionicsRate
      { rrValue = s
      , rrNumeric = val
      , rrHarmonic = rateToHarmonic val
      }

-- | Parse rate string to numeric
parseRate :: String -> Maybe Double
parseRate s =
  let cleaned = filter (\c -> isDigit c || c == '.') s
  in if null cleaned
     then Nothing
     else Just (read cleaned :: Double)

-- | Convert rate to frequency (Hz)
rateToFrequency :: RadionicsRate -> Double
rateToFrequency rr = rrNumeric rr * phi  -- Golden scaling

-- | Get harmonic coupling from rate
rateHarmonic :: RadionicsRate -> (Int, Int)
rateHarmonic = rrHarmonic

-- | Map numeric rate to harmonic indices
rateToHarmonic :: Double -> (Int, Int)
rateToHarmonic val =
  let l = floor (val / 100) `mod` 10
      m = floor (val / 10) `mod` 10 - 5  -- Center around 0
  in (l, m)

-- =============================================================================
-- Overlay
-- =============================================================================

-- | Complete radionics overlay
data RadionicsOverlay = RadionicsOverlay
  { roRate        :: !RadionicsRate
  , roIntention   :: !IntentionVector
  , roRadius      :: !Double         -- ^ Projection radius [0,1]
  , roMode        :: !ModulationMode
  , roActive      :: !Bool
  , roStrength    :: !Double         -- ^ Current strength [0,1]
  } deriving (Eq, Show)

-- | Modulation mode for overlay
data ModulationMode
  = Constructive  -- ^ Amplify emergence
  | Diffusive     -- ^ Spread/distribute field
  | Directive     -- ^ Focus/direct field
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create overlay from parameters
mkOverlay :: RadionicsRate -> String -> Double -> ModulationMode -> RadionicsOverlay
mkOverlay rate intentStr radius mode = RadionicsOverlay
  { roRate = rate
  , roIntention = encodeIntention intentStr
  , roRadius = clamp01 radius
  , roMode = mode
  , roActive = True
  , roStrength = 1.0
  }

-- | Check if overlay is active
overlayActive :: RadionicsOverlay -> Bool
overlayActive = roActive

-- | Get overlay strength
overlayStrength :: RadionicsOverlay -> Double
overlayStrength = roStrength

-- =============================================================================
-- Intention Encoding
-- =============================================================================

-- | Encoded intention vector
data IntentionVector = IntentionVector
  { ivRaw         :: !String        -- ^ Original text
  , ivCategory    :: !String        -- ^ Mapped category
  , ivHarmonicMod :: !(Int, Int)    -- ^ Harmonic modifier
  , ivFluxMod     :: !Double        -- ^ Flux modifier
  , ivShellLimit  :: !(Maybe Int)   -- ^ Shell restriction
  } deriving (Eq, Show)

-- | Encode intention string to vector
encodeIntention :: String -> IntentionVector
encodeIntention raw =
  case find (\(kw, _, _, _, _) -> kw `elem` words (map toLower' raw)) knownIntentions of
    Just (_, cat, harm, flux, shell) -> IntentionVector
      { ivRaw = raw
      , ivCategory = cat
      , ivHarmonicMod = harm
      , ivFluxMod = flux
      , ivShellLimit = shell
      }
    Nothing -> defaultIntention raw
  where
    toLower' c = if c >= 'A' && c <= 'Z' then toEnum (fromEnum c + 32) else c

-- | Default intention for unrecognized input
defaultIntention :: String -> IntentionVector
defaultIntention raw = IntentionVector
  { ivRaw = raw
  , ivCategory = "general"
  , ivHarmonicMod = (0, 0)
  , ivFluxMod = 1.0
  , ivShellLimit = Nothing
  }

-- | Convert intention to field modifier
intentionToModifier :: IntentionVector -> Double
intentionToModifier iv = ivFluxMod iv

-- | Known intention patterns
-- (keyword, category, harmonic_mod, flux_mod, shell_limit)
knownIntentions :: [(String, String, (Int, Int), Double, Maybe Int)]
knownIntentions =
  [ ("insight", "mental", (2, 2), 1.3, Nothing)
  , ("clarity", "mental", (1, 1), 1.2, Nothing)
  , ("protection", "safety", (0, 0), 0.8, Just 3)
  , ("healing", "physical", (3, 0), 1.4, Nothing)
  , ("release", "emotional", (2, -1), 1.1, Nothing)
  , ("grounding", "physical", (0, 0), 0.9, Just 2)
  , ("love", "emotional", (4, 0), 1.5, Nothing)
  , ("peace", "mental", (1, 0), 1.0, Nothing)
  ]

-- =============================================================================
-- Scalar Pathway Modulation
-- =============================================================================

-- | Pathway modifier from overlay
data PathwayModifier = PathwayModifier
  { pmPotential   :: !Double  -- ^ Potential modifier (multiplicative)
  , pmFlux        :: !Double  -- ^ Flux coherence stabilizer
  , pmAnkh        :: !Double  -- ^ Ankh balance shift
  , pmHarmonic    :: !(Int, Int) -- ^ Harmonic boost
  } deriving (Eq, Show)

-- | Apply overlay to create pathway modifier
applyOverlay :: RadionicsOverlay -> PathwayModifier
applyOverlay overlay =
  let rate = roRate overlay
      intent = roIntention overlay
      mode = roMode overlay
      strength = roStrength overlay

      -- Potential modifier based on mode
      potential = case mode of
        Constructive -> 1.0 + strength * phi * 0.1
        Diffusive -> 1.0
        Directive -> 1.0 + strength * 0.2

      -- Flux from intention
      flux = ivFluxMod intent * strength

      -- Ankh shift toward intention vector
      ankh = intentionToModifier intent * strength * 0.1

      -- Combined harmonic
      (rl, rm) = rrHarmonic rate
      (il, im) = ivHarmonicMod intent
  in PathwayModifier
      { pmPotential = potential
      , pmFlux = flux
      , pmAnkh = ankh
      , pmHarmonic = (rl + il, rm + im)
      }

-- | Get potential modifier from overlay
potentialModifier :: RadionicsOverlay -> Double
potentialModifier = pmPotential . applyOverlay

-- | Get flux stabilizer from overlay
fluxStabilizer :: RadionicsOverlay -> Double
fluxStabilizer = pmFlux . applyOverlay

-- | Get ankh shift from overlay
ankhShift :: RadionicsOverlay -> Double
ankhShift = pmAnkh . applyOverlay

-- =============================================================================
-- Overlay Intersection
-- =============================================================================

-- | Result of overlay intersection check
data OverlayIntersection = OverlayIntersection
  { oiOverlays    :: ![RadionicsOverlay]
  , oiCenter      :: !(Double, Double)  -- ^ Intersection center
  , oiMode        :: !InterferenceMode
  , oiStrength    :: !Double
  } deriving (Eq, Show)

-- | Interference mode
data InterferenceMode
  = Constructive'  -- ^ Overlays reinforce
  | Destructive'   -- ^ Overlays cancel
  | Mixed          -- ^ Partial interference
  deriving (Eq, Show)

-- | Check for intersection between overlays
checkIntersection :: RadionicsOverlay -> RadionicsOverlay -> Maybe OverlayIntersection
checkIntersection o1 o2 =
  let r1 = roRadius o1
      r2 = roRadius o2
      -- Simplified: assume overlays at origin, check radius overlap
      overlap = (r1 + r2) > 0.5
  in if overlap && overlayActive o1 && overlayActive o2
     then Just $ OverlayIntersection
          { oiOverlays = [o1, o2]
          , oiCenter = (0.5, 0.5)
          , oiMode = determineInterference o1 o2
          , oiStrength = (roStrength o1 + roStrength o2) / 2
          }
     else Nothing

-- | Determine interference mode
determineInterference :: RadionicsOverlay -> RadionicsOverlay -> InterferenceMode
determineInterference o1 o2 =
  let (l1, m1) = rrHarmonic (roRate o1)
      (l2, m2) = rrHarmonic (roRate o2)
      harmDiff = abs (l1 - l2) + abs (m1 - m2)
  in if harmDiff == 0
     then Constructive'
     else if harmDiff > 5
     then Destructive'
     else Mixed

-- | Resolve multiple overlapping overlays
resolveOverlaps :: [RadionicsOverlay] -> PathwayModifier
resolveOverlaps overlays =
  let mods = map applyOverlay overlays
      potential = product (map pmPotential mods)
      flux = sum (map pmFlux mods) / fromIntegral (max 1 (length mods))
      ankh = sum (map pmAnkh mods)
      harmonics = map pmHarmonic mods
      (avgL, avgM) = if null harmonics
                     then (0, 0)
                     else (sum (map fst harmonics) `div` length harmonics,
                           sum (map snd harmonics) `div` length harmonics)
  in PathwayModifier
      { pmPotential = potential
      , pmFlux = flux
      , pmAnkh = ankh
      , pmHarmonic = (avgL, avgM)
      }

-- | Calculate constructive interference strength
constructiveInterference :: [RadionicsOverlay] -> Double
constructiveInterference overlays =
  let strengths = map roStrength overlays
  in sum strengths / fromIntegral (max 1 (length overlays)) * phi

-- | Calculate destructive interference (cancellation)
destructiveInterference :: [RadionicsOverlay] -> Double
destructiveInterference overlays =
  let strengths = map roStrength overlays
      maxS = maximum (1 : strengths)
      minS = minimum (1 : strengths)
  in 1.0 - (maxS - minS)

-- =============================================================================
-- Consent Gating
-- =============================================================================

-- | Consent state for overlay
data OverlayConsent = OverlayConsent
  { ocOverlay   :: !RadionicsOverlay
  , ocConsented :: !Bool
  , ocVetoReason :: !(Maybe String)
  } deriving (Eq, Show)

-- | Gate overlay by consent
gateOverlay :: RadionicsOverlay -> Bool -> OverlayConsent
gateOverlay overlay consented = OverlayConsent
  { ocOverlay = overlay
  , ocConsented = consented
  , ocVetoReason = if consented then Nothing else Just "User did not consent"
  }

-- | Check if overlay was vetoed
overlayVetoed :: OverlayConsent -> Bool
overlayVetoed oc = not (ocConsented oc)

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Visual representation of overlay
data OverlayVisual = OverlayVisual
  { ovColor    :: !(Int, Int, Int)  -- ^ RGB color
  , ovRadius   :: !Double           -- ^ Visual radius
  , ovOpacity  :: !Double           -- ^ Opacity [0,1]
  , ovPulse    :: !Double           -- ^ Pulse rate (Hz)
  } deriving (Eq, Show)

-- | Convert overlay to visual representation
overlayToVisual :: RadionicsOverlay -> OverlayVisual
overlayToVisual overlay =
  let color = visualColor (roMode overlay) (roStrength overlay)
      radius = roRadius overlay
      opacity = roStrength overlay * 0.7
      pulse = rateToFrequency (roRate overlay) / 100  -- Scale to visible pulse
  in OverlayVisual
      { ovColor = color
      , ovRadius = radius
      , ovOpacity = opacity
      , ovPulse = pulse
      }

-- | Get visual color from mode and strength
visualColor :: ModulationMode -> Double -> (Int, Int, Int)
visualColor mode strength =
  let intensity = round (255 * strength) :: Int
  in case mode of
      Constructive -> (0, intensity, intensity `div` 2)      -- Cyan-green
      Diffusive -> (intensity `div` 2, intensity `div` 2, intensity)  -- Blue-white
      Directive -> (intensity, intensity `div` 3, 0)         -- Orange-gold

-- | Get visual radius
visualRadius :: RadionicsOverlay -> Double
visualRadius = roRadius

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
