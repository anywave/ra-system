{-|
Module      : Ra.Dream.Surface
Description : Dream surface API for lucid dreaming and dream interaction
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Provides a surface API for interacting with dream states, including
lucid dreaming induction, dream recording, symbol extraction, and
dream-reality bridging functionality.

== Dream Surface Theory

=== Consciousness Layers

* Waking surface - Normal conscious awareness
* Hypnagogic surface - Sleep onset transitions
* Dream surface - Full dream immersion
* Lucid surface - Aware dreaming state
* Void surface - Deep dreamless sleep

=== Dream Interaction Model

1. Entry protocols for state transitions
2. Symbol extraction and interpretation
3. Dream navigation and manipulation
4. Reality anchoring for safe return
-}
module Ra.Dream.Surface
  ( -- * Core Types
    DreamSurface(..)
  , DreamState(..)
  , DreamSymbol(..)
  , DreamNarrative(..)

    -- * Surface Access
  , initializeSurface
  , enterSurface
  , exitSurface
  , currentDepth

    -- * State Transitions
  , transitionTo
  , isLucid
  , induceLucidity
  , anchorReality

    -- * Symbol Operations
  , extractSymbols
  , interpretSymbol
  , symbolResonance
  , archetype

    -- * Dream Recording
  , DreamRecord(..)
  , startRecording
  , stopRecording
  , saveRecord

    -- * Navigation
  , DreamLocation(..)
  , navigate
  , bookmark
  , returnToBookmark

    -- * Coherence Monitoring
  , surfaceCoherence
  , stabilize
  , coherenceWarning
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Dream surface state container
data DreamSurface = DreamSurface
  { dsState        :: !DreamState         -- ^ Current dream state
  , dsDepth        :: !Double             -- ^ Depth level [0, 1]
  , dsLucidity     :: !Double             -- ^ Lucidity level [0, 1]
  , dsCoherence    :: !Double             -- ^ Surface coherence [0, 1]
  , dsSymbols      :: ![DreamSymbol]      -- ^ Extracted symbols
  , dsNarrative    :: !(Maybe DreamNarrative)  -- ^ Active narrative
  , dsAnchors      :: ![String]           -- ^ Reality anchors
  , dsRecording    :: !Bool               -- ^ Recording active
  } deriving (Eq, Show)

-- | Dream state enumeration
data DreamState
  = StateWaking        -- ^ Normal waking consciousness
  | StateHypnagogic    -- ^ Falling asleep transition
  | StateDream         -- ^ Full dream state
  | StateLucid         -- ^ Lucid dreaming
  | StateHypnopompic   -- ^ Waking up transition
  | StateVoid          -- ^ Deep dreamless
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Dream symbol representation
data DreamSymbol = DreamSymbol
  { symId          :: !String             -- ^ Symbol identifier
  , symName        :: !String             -- ^ Symbol name
  , symArchetype   :: !Archetype          -- ^ Archetypal classification
  , symIntensity   :: !Double             -- ^ Symbol intensity [0, 1]
  , symFrequency   :: !Double             -- ^ Resonance frequency
  , symContext     :: !String             -- ^ Context description
  } deriving (Eq, Show)

-- | Archetypal classifications
data Archetype
  = ArchSelf           -- ^ Self/ego
  | ArchShadow         -- ^ Shadow aspects
  | ArchAnima          -- ^ Feminine principle
  | ArchAnimus         -- ^ Masculine principle
  | ArchTrickster      -- ^ Trickster/chaos
  | ArchWiseFigure     -- ^ Wise old person
  | ArchHero           -- ^ Hero journey
  | ArchMother         -- ^ Great mother
  | ArchFather         -- ^ Great father
  | ArchChild          -- ^ Divine child
  | ArchThreshold      -- ^ Threshold/doorway
  | ArchTransformation -- ^ Death/rebirth
  | ArchUnknown        -- ^ Unclassified
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Dream narrative structure
data DreamNarrative = DreamNarrative
  { dnTitle        :: !String             -- ^ Narrative title
  , dnPhase        :: !NarrativePhase     -- ^ Current phase
  , dnSymbols      :: ![DreamSymbol]      -- ^ Narrative symbols
  , dnLocations    :: ![DreamLocation]    -- ^ Visited locations
  , dnCoherence    :: !Double             -- ^ Narrative coherence
  , dnTimestamp    :: !Int                -- ^ Start timestamp
  } deriving (Eq, Show)

-- | Narrative phases (Hero's Journey structure)
data NarrativePhase
  = PhaseOrdinary      -- ^ Ordinary world
  | PhaseCall          -- ^ Call to adventure
  | PhaseThreshold     -- ^ Crossing threshold
  | PhaseTrials        -- ^ Tests and trials
  | PhaseAbyss         -- ^ Ordeal/abyss
  | PhaseTransform     -- ^ Transformation
  | PhaseReturn        -- ^ Return journey
  | PhaseIntegration   -- ^ Integration
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Surface Access
-- =============================================================================

-- | Initialize dream surface
initializeSurface :: DreamSurface
initializeSurface = DreamSurface
  { dsState = StateWaking
  , dsDepth = 0
  , dsLucidity = 1.0  -- Fully awake = full lucidity
  , dsCoherence = 1.0
  , dsSymbols = []
  , dsNarrative = Nothing
  , dsAnchors = ["breath", "body", "name"]
  , dsRecording = False
  }

-- | Enter dream surface (begin sleep)
enterSurface :: DreamSurface -> DreamSurface
enterSurface surface =
  surface
    { dsState = StateHypnagogic
    , dsDepth = 0.1
    , dsLucidity = dsLucidity surface * phiInverse
    }

-- | Exit dream surface (wake up)
exitSurface :: DreamSurface -> DreamSurface
exitSurface surface =
  surface
    { dsState = StateHypnopompic
    , dsDepth = dsDepth surface * phiInverse
    , dsLucidity = min 1.0 (dsLucidity surface + 0.3)
    }

-- | Get current depth level
currentDepth :: DreamSurface -> Double
currentDepth = dsDepth

-- =============================================================================
-- State Transitions
-- =============================================================================

-- | Transition to specific dream state
transitionTo :: DreamSurface -> DreamState -> DreamSurface
transitionTo surface targetState =
  let newDepth = stateDepth targetState
      newLucidity = case targetState of
        StateLucid -> min 1.0 (dsLucidity surface + 0.3)
        StateVoid -> 0
        StateWaking -> 1.0
        _ -> dsLucidity surface * 0.9
      coherenceLoss = abs (newDepth - dsDepth surface) * 0.1
  in surface
    { dsState = targetState
    , dsDepth = newDepth
    , dsLucidity = newLucidity
    , dsCoherence = max 0 (dsCoherence surface - coherenceLoss)
    }

-- | Check if currently lucid
isLucid :: DreamSurface -> Bool
isLucid surface =
  dsState surface == StateLucid || dsLucidity surface > phiInverse

-- | Induce lucidity in current dream
induceLucidity :: DreamSurface -> DreamSurface
induceLucidity surface =
  if dsState surface == StateDream
  then surface
    { dsState = StateLucid
    , dsLucidity = min 1.0 (dsLucidity surface + phi * 0.3)
    , dsCoherence = dsCoherence surface * 0.95
    }
  else surface

-- | Anchor to reality (prevent getting lost)
anchorReality :: DreamSurface -> String -> DreamSurface
anchorReality surface anchor =
  let newAnchors = if anchor `elem` dsAnchors surface
                   then dsAnchors surface
                   else anchor : take 9 (dsAnchors surface)
      stabilization = 0.05 * fromIntegral (length newAnchors)
  in surface
    { dsAnchors = newAnchors
    , dsCoherence = min 1.0 (dsCoherence surface + stabilization)
    }

-- =============================================================================
-- Symbol Operations
-- =============================================================================

-- | Extract symbols from current state
extractSymbols :: DreamSurface -> [DreamSymbol]
extractSymbols surface =
  let depthFactor = dsDepth surface
      lucidFactor = dsLucidity surface
      count = floor (depthFactor * 5 + lucidFactor * 3) :: Int
  in take count (dsSymbols surface)

-- | Interpret symbol meaning
interpretSymbol :: DreamSymbol -> String
interpretSymbol sym = case symArchetype sym of
  ArchSelf -> "Represents core identity and ego consciousness"
  ArchShadow -> "Hidden or repressed aspects seeking integration"
  ArchAnima -> "Feminine psychological aspects"
  ArchAnimus -> "Masculine psychological aspects"
  ArchTrickster -> "Chaos and transformation catalyst"
  ArchWiseFigure -> "Inner wisdom and guidance"
  ArchHero -> "Journey of growth and achievement"
  ArchMother -> "Nurturing and protective forces"
  ArchFather -> "Authority and structure"
  ArchChild -> "New beginnings and potential"
  ArchThreshold -> "Transition and opportunity"
  ArchTransformation -> "Death of old, birth of new"
  ArchUnknown -> "Symbol requires further exploration"

-- | Calculate symbol resonance with dreamer
symbolResonance :: DreamSurface -> DreamSymbol -> Double
symbolResonance surface sym =
  let baseResonance = symIntensity sym
      lucidBonus = if isLucid surface then phi * 0.2 else 0
      depthFactor = dsDepth surface * phiInverse
  in min 1.0 (baseResonance + lucidBonus + depthFactor)

-- | Get archetype for symbol
archetype :: DreamSymbol -> Archetype
archetype = symArchetype

-- =============================================================================
-- Dream Recording
-- =============================================================================

-- | Dream record structure
data DreamRecord = DreamRecord
  { drId           :: !String             -- ^ Record identifier
  , drTimestamp    :: !Int                -- ^ Recording start time
  , drDuration     :: !Int                -- ^ Duration in ticks
  , drMaxDepth     :: !Double             -- ^ Maximum depth reached
  , drSymbols      :: ![DreamSymbol]      -- ^ All extracted symbols
  , drNarratives   :: ![DreamNarrative]   -- ^ Narrative fragments
  , drLucidPeriods :: !Int                -- ^ Count of lucid periods
  , drCoherence    :: !Double             -- ^ Average coherence
  } deriving (Eq, Show)

-- | Start recording dream
startRecording :: DreamSurface -> Int -> DreamSurface
startRecording surface _timestamp =
  surface { dsRecording = True }

-- | Stop recording dream
stopRecording :: DreamSurface -> DreamSurface
stopRecording surface =
  surface { dsRecording = False }

-- | Save current state to record
saveRecord :: DreamSurface -> Int -> Int -> DreamRecord
saveRecord surface startTime currentTime = DreamRecord
  { drId = "dream_" ++ show startTime
  , drTimestamp = startTime
  , drDuration = currentTime - startTime
  , drMaxDepth = dsDepth surface
  , drSymbols = dsSymbols surface
  , drNarratives = case dsNarrative surface of
      Just n -> [n]
      Nothing -> []
  , drLucidPeriods = if isLucid surface then 1 else 0
  , drCoherence = dsCoherence surface
  }

-- =============================================================================
-- Navigation
-- =============================================================================

-- | Dream location representation
data DreamLocation = DreamLocation
  { dlId           :: !String             -- ^ Location identifier
  , dlName         :: !String             -- ^ Location name
  , dlType         :: !LocationType       -- ^ Location classification
  , dlCoherence    :: !Double             -- ^ Location stability
  , dlSymbols      :: ![String]           -- ^ Associated symbol IDs
  , dlBookmarked   :: !Bool               -- ^ Is bookmarked
  } deriving (Eq, Show)

-- | Location types
data LocationType
  = LocFamiliar      -- ^ Known/familiar place
  | LocArchetypal    -- ^ Archetypal location (forest, ocean, etc.)
  | LocAbstract      -- ^ Abstract/non-physical
  | LocMemory        -- ^ Memory-based
  | LocConstructed   -- ^ Consciously constructed
  | LocUnknown       -- ^ Unknown territory
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Navigate to location
navigate :: DreamSurface -> DreamLocation -> DreamSurface
navigate surface location =
  let coherenceCost = 0.1 * (1 - dlCoherence location)
      newNarrative = case dsNarrative surface of
        Just n -> Just (addLocation n location)
        Nothing -> Just (createNarrative location)
  in surface
    { dsCoherence = max 0 (dsCoherence surface - coherenceCost)
    , dsNarrative = newNarrative
    }

-- | Bookmark current location
bookmark :: DreamSurface -> String -> DreamLocation
bookmark surface locId = DreamLocation
  { dlId = locId
  , dlName = "Bookmark_" ++ locId
  , dlType = LocConstructed
  , dlCoherence = dsCoherence surface
  , dlSymbols = map symId (dsSymbols surface)
  , dlBookmarked = True
  }

-- | Return to bookmarked location
returnToBookmark :: DreamSurface -> DreamLocation -> Maybe DreamSurface
returnToBookmark surface location =
  if dlBookmarked location && isLucid surface
  then Just (navigate surface location)
  else Nothing

-- =============================================================================
-- Coherence Monitoring
-- =============================================================================

-- | Get surface coherence level
surfaceCoherence :: DreamSurface -> Double
surfaceCoherence = dsCoherence

-- | Stabilize dream surface
stabilize :: DreamSurface -> DreamSurface
stabilize surface =
  let anchorBonus = fromIntegral (length (dsAnchors surface)) * 0.02
      lucidBonus = if isLucid surface then 0.1 else 0
      newCoherence = min 1.0 (dsCoherence surface + anchorBonus + lucidBonus)
  in surface { dsCoherence = newCoherence }

-- | Check for coherence warning
coherenceWarning :: DreamSurface -> Maybe String
coherenceWarning surface
  | dsCoherence surface < 0.2 = Just "Critical: Dream collapse imminent"
  | dsCoherence surface < 0.4 = Just "Warning: Surface unstable"
  | dsCoherence surface < phiInverse = Just "Caution: Coherence dropping"
  | otherwise = Nothing

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Get depth for state
stateDepth :: DreamState -> Double
stateDepth StateWaking = 0
stateDepth StateHypnagogic = 0.2
stateDepth StateDream = 0.6
stateDepth StateLucid = 0.7
stateDepth StateHypnopompic = 0.3
stateDepth StateVoid = 1.0

-- | Add location to narrative
addLocation :: DreamNarrative -> DreamLocation -> DreamNarrative
addLocation narrative loc =
  narrative { dnLocations = loc : take 9 (dnLocations narrative) }

-- | Create new narrative from location
createNarrative :: DreamLocation -> DreamNarrative
createNarrative loc = DreamNarrative
  { dnTitle = "Journey through " ++ dlName loc
  , dnPhase = PhaseOrdinary
  , dnSymbols = []
  , dnLocations = [loc]
  , dnCoherence = dlCoherence loc
  , dnTimestamp = 0
  }
