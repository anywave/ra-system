{-|
Module      : Ra.Contact.ArtifactResonator
Description : Artifact resonator for ancient technology activation
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements artifact resonator systems for interfacing with ancient
technology artifacts. Provides frequency scanning, harmonic activation,
and consciousness coupling for artifact exploration.

== Artifact Theory

=== Resonance Activation

* Ancient artifacts often respond to specific frequencies
* Phi-ratio harmonics unlock dormant functions
* Consciousness coupling enables intuitive operation
* Coherence levels determine access depth

=== Artifact Classification

1. Communication artifacts (crystals, rods)
2. Power artifacts (pyramids, obelisks)
3. Navigation artifacts (star maps, compasses)
4. Healing artifacts (wands, chambers)
5. Unknown artifacts (unclassified)
-}
module Ra.Contact.ArtifactResonator
  ( -- * Core Types
    ArtifactResonator(..)
  , Artifact(..)
  , ArtifactType(..)
  , ResonanceState(..)

    -- * Resonator Control
  , createResonator
  , linkArtifact
  , unlinkArtifact
  , resonatorStatus

    -- * Frequency Scanning
  , scanArtifact
  , findResonance
  , harmonicScan
  , resonanceMap

    -- * Activation
  , activateArtifact
  , deactivateArtifact
  , activationLevel
  , pulseActivation

    -- * Artifact Registry
  , ArtifactEntry(..)
  , registerArtifact
  , lookupArtifact
  , catalogArtifacts

    -- * Consciousness Interface
  , ConsciousnessLink(..)
  , establishLink
  , transmitIntent
  , receiveImpression

    -- * Safety Protocols
  , SafetyState(..)
  , checkArtifactSafety
  , safetyOverride
  , emergencyShutdown
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Artifact resonator system
data ArtifactResonator = ArtifactResonator
  { arState        :: !ResonanceState      -- ^ Current state
  , arLinkedArtifact :: !(Maybe Artifact)  -- ^ Currently linked artifact
  , arFrequencies  :: ![Double]            -- ^ Scanned frequencies
  , arResonance    :: !Double              -- ^ Resonance strength [0, 1]
  , arCoherence    :: !Double              -- ^ Operator coherence [0, 1]
  , arConsciousness :: !(Maybe ConsciousnessLink)  -- ^ Consciousness link
  , arSafety       :: !SafetyState         -- ^ Safety status
  , arHistory      :: ![ArtifactEntry]     -- ^ Artifact history
  } deriving (Eq, Show)

-- | Artifact definition
data Artifact = Artifact
  { artId          :: !String              -- ^ Artifact identifier
  , artName        :: !String              -- ^ Human name
  , artType        :: !ArtifactType        -- ^ Classification
  , artOrigin      :: !ArtifactOrigin      -- ^ Origin classification
  , artPrimaryFreq :: !Double              -- ^ Primary resonance frequency
  , artHarmonics   :: ![Double]            -- ^ Harmonic frequencies
  , artAge         :: !Int                 -- ^ Estimated age (years)
  , artMaterial    :: !String              -- ^ Primary material
  , artPower       :: !Double              -- ^ Power potential [0, 1]
  , artCoherenceReq :: !Double             -- ^ Required coherence
  } deriving (Eq, Show)

-- | Artifact type classification
data ArtifactType
  = TypeCommunication  -- ^ Communication devices
  | TypePower          -- ^ Power generation/storage
  | TypeNavigation     -- ^ Navigation/mapping
  | TypeHealing        -- ^ Healing/therapeutic
  | TypeInitiation     -- ^ Initiation/consciousness
  | TypeStorage        -- ^ Information storage
  | TypeUnknown        -- ^ Unclassified
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Artifact origin classification
data ArtifactOrigin
  = OriginTerrestrial   -- ^ Earth-made (ancient)
  | OriginAtlantean     -- ^ Atlantean
  | OriginLemuruan      -- ^ Lemurian
  | OriginEgyptian      -- ^ Egyptian
  | OriginExtraterrestrial  -- ^ Non-Earth origin
  | OriginInterdimensional  -- ^ Interdimensional
  | OriginUnknown       -- ^ Unknown origin
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Resonator state
data ResonanceState
  = StateInactive      -- ^ Not operating
  | StateScanning      -- ^ Scanning artifact
  | StateResonating    -- ^ Actively resonating
  | StateLinked        -- ^ Consciousness linked
  | StateActive        -- ^ Artifact activated
  | StateCooldown      -- ^ Post-activation cooldown
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Resonator Control
-- =============================================================================

-- | Create new artifact resonator
createResonator :: ArtifactResonator
createResonator = ArtifactResonator
  { arState = StateInactive
  , arLinkedArtifact = Nothing
  , arFrequencies = []
  , arResonance = 0
  , arCoherence = 0.5
  , arConsciousness = Nothing
  , arSafety = SafetyOk
  , arHistory = []
  }

-- | Link artifact to resonator
linkArtifact :: ArtifactResonator -> Artifact -> ArtifactResonator
linkArtifact res artifact =
  if arCoherence res >= artCoherenceReq artifact
  then res
    { arLinkedArtifact = Just artifact
    , arState = StateScanning
    , arFrequencies = artHarmonics artifact
    }
  else res  -- Insufficient coherence

-- | Unlink artifact
unlinkArtifact :: ArtifactResonator -> ArtifactResonator
unlinkArtifact res =
  res
    { arLinkedArtifact = Nothing
    , arState = StateInactive
    , arResonance = 0
    , arConsciousness = Nothing
    }

-- | Get resonator status summary
resonatorStatus :: ArtifactResonator -> String
resonatorStatus res =
  let artifactInfo = case arLinkedArtifact res of
        Nothing -> "No artifact linked"
        Just a -> "Linked: " ++ artName a
  in "State: " ++ show (arState res) ++
     ", " ++ artifactInfo ++
     ", Resonance: " ++ show (arResonance res) ++
     ", Coherence: " ++ show (arCoherence res)

-- =============================================================================
-- Frequency Scanning
-- =============================================================================

-- | Scan artifact for resonant frequencies
scanArtifact :: ArtifactResonator -> (ArtifactResonator, [Double])
scanArtifact res =
  case arLinkedArtifact res of
    Nothing -> (res, [])
    Just artifact ->
      let baseFreq = artPrimaryFreq artifact
          scannedFreqs = generateScanFrequencies baseFreq
          resonantFreqs = filter (isResonant artifact) scannedFreqs
      in (res { arFrequencies = resonantFreqs, arState = StateScanning }, resonantFreqs)

-- | Find primary resonance frequency
findResonance :: ArtifactResonator -> Maybe Double
findResonance res =
  case arLinkedArtifact res of
    Nothing -> Nothing
    Just artifact -> Just (artPrimaryFreq artifact)

-- | Perform harmonic scan
harmonicScan :: ArtifactResonator -> Int -> [Double]
harmonicScan res harmonicCount =
  case arLinkedArtifact res of
    Nothing -> []
    Just artifact ->
      let base = artPrimaryFreq artifact
      in [ base * (phi ^ i) | i <- [0..harmonicCount-1] ] ++
         [ base * fromIntegral i | i <- [1..harmonicCount] ]

-- | Generate resonance map
resonanceMap :: ArtifactResonator -> [(Double, Double)]
resonanceMap res =
  case arLinkedArtifact res of
    Nothing -> []
    Just artifact ->
      let freqs = artHarmonics artifact
          strengths = map (resonanceStrength artifact) freqs
      in zip freqs strengths

-- =============================================================================
-- Activation
-- =============================================================================

-- | Activate linked artifact
activateArtifact :: ArtifactResonator -> ArtifactResonator
activateArtifact res =
  case arLinkedArtifact res of
    Nothing -> res
    Just artifact ->
      if arCoherence res >= artCoherenceReq artifact && arSafety res == SafetyOk
      then let newResonance = artPower artifact * arCoherence res
           in res
             { arState = StateActive
             , arResonance = newResonance
             }
      else res

-- | Deactivate artifact
deactivateArtifact :: ArtifactResonator -> ArtifactResonator
deactivateArtifact res =
  res
    { arState = StateCooldown
    , arResonance = arResonance res * phiInverse
    }

-- | Get activation level
activationLevel :: ArtifactResonator -> Double
activationLevel res =
  if arState res == StateActive
  then arResonance res * arCoherence res
  else 0

-- | Send activation pulse
pulseActivation :: ArtifactResonator -> Double -> ArtifactResonator
pulseActivation res intensity =
  case arLinkedArtifact res of
    Nothing -> res
    Just artifact ->
      let pulsedResonance = min 1.0 (arResonance res + intensity * artPower artifact)
          coherenceCost = intensity * 0.1
      in res
        { arResonance = pulsedResonance
        , arCoherence = max 0 (arCoherence res - coherenceCost)
        , arState = StateResonating
        }

-- =============================================================================
-- Artifact Registry
-- =============================================================================

-- | Artifact registry entry
data ArtifactEntry = ArtifactEntry
  { aeArtifact     :: !Artifact            -- ^ Artifact data
  , aeLastScanned  :: !Int                 -- ^ Last scan timestamp
  , aeActivations  :: !Int                 -- ^ Activation count
  , aeMaxResonance :: !Double              -- ^ Maximum achieved resonance
  , aeNotes        :: !String              -- ^ Operator notes
  } deriving (Eq, Show)

-- | Register artifact in history
registerArtifact :: ArtifactResonator -> Artifact -> ArtifactResonator
registerArtifact res artifact =
  let entry = ArtifactEntry
        { aeArtifact = artifact
        , aeLastScanned = 0
        , aeActivations = 0
        , aeMaxResonance = 0
        , aeNotes = ""
        }
      existing = filter ((/= artId artifact) . artId . aeArtifact) (arHistory res)
  in res { arHistory = entry : existing }

-- | Look up artifact by ID
lookupArtifact :: ArtifactResonator -> String -> Maybe ArtifactEntry
lookupArtifact res artIdSearch =
  case filter ((== artIdSearch) . artId . aeArtifact) (arHistory res) of
    (e:_) -> Just e
    [] -> Nothing

-- | Catalog all known artifacts
catalogArtifacts :: ArtifactResonator -> [ArtifactEntry]
catalogArtifacts = arHistory

-- =============================================================================
-- Consciousness Interface
-- =============================================================================

-- | Consciousness link state
data ConsciousnessLink = ConsciousnessLink
  { clEstablished  :: !Bool                -- ^ Link active
  , clDepth        :: !Double              -- ^ Link depth [0, 1]
  , clClarity      :: !Double              -- ^ Communication clarity
  , clImpressions  :: ![Impression]        -- ^ Received impressions
  , clIntentsSent  :: !Int                 -- ^ Intents transmitted
  } deriving (Eq, Show)

-- | Impression received from artifact
data Impression = Impression
  { impType        :: !ImpressionType      -- ^ Type of impression
  , impContent     :: !String              -- ^ Content description
  , impIntensity   :: !Double              -- ^ Intensity [0, 1]
  , impTimestamp   :: !Int                 -- ^ Receipt time
  } deriving (Eq, Show)

-- | Impression types
data ImpressionType
  = ImpVisual         -- ^ Visual image
  | ImpAuditory       -- ^ Sound/voice
  | ImpKinesthetic    -- ^ Physical sensation
  | ImpEmotional      -- ^ Emotional state
  | ImpCognitive      -- ^ Thought/knowledge
  | ImpSymbolic       -- ^ Symbol/archetype
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Establish consciousness link with artifact
establishLink :: ArtifactResonator -> ArtifactResonator
establishLink res =
  case arLinkedArtifact res of
    Nothing -> res
    Just artifact ->
      if arCoherence res >= artCoherenceReq artifact * phi
      then let link = ConsciousnessLink
                 { clEstablished = True
                 , clDepth = arCoherence res * phiInverse
                 , clClarity = arResonance res
                 , clImpressions = []
                 , clIntentsSent = 0
                 }
           in res { arConsciousness = Just link, arState = StateLinked }
      else res

-- | Transmit intent to artifact
transmitIntent :: ArtifactResonator -> String -> ArtifactResonator
transmitIntent res _intent =
  case arConsciousness res of
    Nothing -> res
    Just link ->
      let newLink = link
            { clIntentsSent = clIntentsSent link + 1
            , clClarity = clClarity link * 0.98  -- Slight degradation
            }
      in res { arConsciousness = Just newLink }

-- | Receive impression from artifact
receiveImpression :: ArtifactResonator -> (ArtifactResonator, Maybe Impression)
receiveImpression res =
  case arConsciousness res of
    Nothing -> (res, Nothing)
    Just link ->
      if clEstablished link && clClarity link > 0.3
      then let imp = generateImpression link (arResonance res)
               newLink = link { clImpressions = imp : take 9 (clImpressions link) }
           in (res { arConsciousness = Just newLink }, Just imp)
      else (res, Nothing)

-- =============================================================================
-- Safety Protocols
-- =============================================================================

-- | Safety state enumeration
data SafetyState
  = SafetyOk          -- ^ All systems nominal
  | SafetyWarning     -- ^ Minor concerns
  | SafetyAlert       -- ^ Significant risk
  | SafetyCritical    -- ^ Immediate shutdown required
  | SafetyLocked      -- ^ System locked out
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Check artifact safety
checkArtifactSafety :: ArtifactResonator -> SafetyState
checkArtifactSafety res =
  let resonanceRisk = arResonance res > 0.9
      coherenceRisk = arCoherence res < 0.2
      artifactRisk = case arLinkedArtifact res of
        Nothing -> False
        Just a -> artPower a > 0.8 && artOrigin a == OriginUnknown
  in if resonanceRisk && coherenceRisk then SafetyCritical
     else if resonanceRisk || artifactRisk then SafetyAlert
     else if coherenceRisk then SafetyWarning
     else SafetyOk

-- | Safety override (requires high coherence)
safetyOverride :: ArtifactResonator -> ArtifactResonator
safetyOverride res =
  if arCoherence res > phi * 0.5  -- Need 0.809+ coherence
  then res { arSafety = SafetyOk }
  else res

-- | Emergency shutdown
emergencyShutdown :: ArtifactResonator -> ArtifactResonator
emergencyShutdown res =
  res
    { arState = StateInactive
    , arResonance = 0
    , arConsciousness = Nothing
    , arSafety = SafetyLocked
    }

-- =============================================================================
-- Sample Artifacts
-- =============================================================================

-- | Sample artifact database
_sampleArtifacts :: [Artifact]
_sampleArtifacts =
  [ Artifact "ark_covenant" "Ark of the Covenant" TypePower OriginEgyptian
      528 (phiEncoding 528 5) 3500 "Gold-plated Acacia" 0.95 0.8
  , Artifact "crystal_skull" "Crystal Skull" TypeCommunication OriginAtlantean
      963 (phiEncoding 963 7) 12000 "Quartz Crystal" 0.8 0.7
  , Artifact "rod_amon_ra" "Rod of Amon Ra" TypeInitiation OriginEgyptian
      432 (phiEncoding 432 9) 4500 "Gold-Copper Alloy" 0.9 0.75
  , Artifact "emerald_tablet" "Emerald Tablet" TypeStorage OriginAtlantean
      639 (phiEncoding 639 6) 15000 "Emerald" 0.7 0.6
  ]

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Generate scan frequencies around base
generateScanFrequencies :: Double -> [Double]
generateScanFrequencies base =
  [ base * (1 + offset * 0.01) | offset <- [-10..10] ] ++
  phiEncoding base 5

-- | Check if frequency is resonant with artifact
isResonant :: Artifact -> Double -> Bool
isResonant artifact freq =
  let primary = artPrimaryFreq artifact
      tolerance = primary * 0.02  -- 2% tolerance
  in abs (freq - primary) < tolerance ||
     any (\h -> abs (freq - h) < tolerance) (artHarmonics artifact)

-- | Calculate resonance strength at frequency
resonanceStrength :: Artifact -> Double -> Double
resonanceStrength artifact freq =
  let primary = artPrimaryFreq artifact
      distance = abs (freq - primary) / primary
  in max 0 (1 - distance * 5)

-- | Phi-ratio encoded frequencies
phiEncoding :: Double -> Int -> [Double]
phiEncoding base levels =
  [ base * (phi ^ i) | i <- [0..levels-1] ]

-- | Generate impression based on link state
generateImpression :: ConsciousnessLink -> Double -> Impression
generateImpression link resonance =
  let selectedImpType = selectImpressionType (clDepth link)
      intensity = resonance * clClarity link
  in Impression
    { impType = selectedImpType
    , impContent = "Artifact communication"
    , impIntensity = intensity
    , impTimestamp = clIntentsSent link
    }

-- | Select impression type based on link depth
selectImpressionType :: Double -> ImpressionType
selectImpressionType depth
  | depth > 0.8 = ImpCognitive
  | depth > 0.6 = ImpSymbolic
  | depth > 0.4 = ImpEmotional
  | depth > 0.2 = ImpVisual
  | otherwise = ImpKinesthetic
