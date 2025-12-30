{-|
Module      : Ra.Initiation
Description : Avatar-alchemical initiation path with symbolic chambers
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements an interactive, metaphysically-aligned initiation path using
symbolic chambers (Most Holy Trinosophia) and inner energetic stages
(The Secret of the Golden Flower). This system guides users along an
avatar-alchemical journey using biometric coherence to unlock progression.

== Initiation Architecture

=== Outer Symbolism (Chambers)

12 symbolic rooms from Trinosophia, each with distinct symbols:

1. Chamber of Entry - Threshold (Door)
2. Chamber of Earth - Foundation (Stone)
3. Chamber of Fire - Purification (Torch)
4. Chamber of Water - Dissolution (Vessel)
5. Chamber of Air - Sublimation (Feather)
6. Chamber of the Sword - Discrimination (Sword)
7. Chamber of the Mirror - Reflection (Mirror)
8. Chamber of the Star - Aspiration (Star)
9. Chamber of the Moon - Intuition (Crescent)
10. Chamber of the Sun - Illumination (Sun)
11. Chamber of the Phoenix - Transformation (Phoenix)
12. Chamber of Unity - Integration (Ankh)

=== Inner Energetics (Alchemy Stages)

Six-stage progression based on The Secret of the Golden Flower:

1. Darkness - Initial unconscious state
2. Gathering Light - Concentration begins
3. Circulation - Energy moves through body
4. Embryo Formation - New consciousness gestates
5. Crystallization - Spiritual body solidifies
6. Return to Source - Unity achieved
-}
module Ra.Initiation
  ( -- * Core Types
    Chamber(..)
  , AlchemicalStage(..)
  , ChamberSymbol(..)
  , Element(..)
  , CoherenceThreshold(..)
  , InitiationState(..)

    -- * Progression Functions
  , unlockChamber
  , nextAlchemyStage
  , isReadyForAscension

    -- * State Management
  , initializeInitiation
  , updateCoherence
  , getChamberSymbol
  , getElementAlignment

    -- * HUD Integration
  , initiationHUD
  , initiationHUDCompact
  , stageDescription

    -- * Chamber Settings
  , ChamberSettings(..)
  , chamberSettings
  , recommendedFrequency

    -- * Simulation
  , simulateInitiationPath
  , progressionTrace

    -- * Symbolic Mapping
  , deriveSymbolFromElement
  , elementFromChamber
  , chamberColor
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Ra.Constants.Extended (phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | The 12 symbolic chambers from Trinosophia
data Chamber
  = Chamber1Entry     -- ^ Threshold crossing
  | Chamber2Earth     -- ^ Grounding foundation
  | Chamber3Fire      -- ^ Purification trial
  | Chamber4Water     -- ^ Emotional dissolution
  | Chamber5Air       -- ^ Mental sublimation
  | Chamber6Sword     -- ^ Discrimination
  | Chamber7Mirror    -- ^ Self-reflection
  | Chamber8Star      -- ^ Higher aspiration
  | Chamber9Moon      -- ^ Intuitive knowledge
  | Chamber10Sun      -- ^ Full illumination
  | Chamber11Phoenix  -- ^ Death and rebirth
  | Chamber12Unity    -- ^ Final integration
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Six alchemical stages from The Secret of the Golden Flower
data AlchemicalStage
  = StageDarkness         -- ^ Initial unconscious state
  | StageGatheringLight   -- ^ Concentration begins
  | StageCirculation      -- ^ Energy circulation
  | StageEmbryoFormation  -- ^ New consciousness gestates
  | StageCrystallization  -- ^ Spiritual body solidifies
  | StageReturnToSource   -- ^ Unity achieved
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Iconographic symbols for each chamber
data ChamberSymbol
  = SymbolDoor      -- ^ Entry threshold
  | SymbolStone     -- ^ Earth foundation
  | SymbolTorch     -- ^ Fire purification
  | SymbolVessel    -- ^ Water container
  | SymbolFeather   -- ^ Air lightness
  | SymbolSword     -- ^ Discrimination blade
  | SymbolMirror    -- ^ Reflection surface
  | SymbolStar      -- ^ Celestial aspiration
  | SymbolCrescent  -- ^ Moon intuition
  | SymbolSun       -- ^ Solar illumination
  | SymbolPhoenix   -- ^ Rebirth bird
  | SymbolAnkh      -- ^ Life unity
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Classical elements
data Element
  = ElementEarth
  | ElementFire
  | ElementWater
  | ElementAir
  | ElementAether    -- ^ Fifth element, quintessence
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Coherence threshold levels for advancement
data CoherenceThreshold
  = ThresholdLow      -- ^ 0.0 - 0.382
  | ThresholdMid      -- ^ 0.382 - 0.618
  | ThresholdHigh     -- ^ 0.618 - 0.854
  | ThresholdAscend   -- ^ 0.854 - 1.0
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Full initiation state for a user
data InitiationState = InitiationState
  { isChamber         :: !Chamber           -- ^ Current chamber
  , isStage           :: !AlchemicalStage   -- ^ Current alchemy stage
  , isCoherence       :: !Double            -- ^ Current coherence [0, 1]
  , isThreshold       :: !CoherenceThreshold -- ^ Current threshold level
  , isElement         :: !(Maybe Element)   -- ^ Elemental alignment
  , isChambersVisited :: ![Chamber]         -- ^ Chambers completed
  , isStagesCompleted :: ![AlchemicalStage] -- ^ Stages completed
  , isSessionTicks    :: !Int               -- ^ Session duration
  } deriving (Eq, Show)

-- =============================================================================
-- Progression Functions
-- =============================================================================

-- | Attempt to unlock the next chamber based on coherence
unlockChamber :: InitiationState -> Maybe Chamber
unlockChamber state
  | isChamber state == maxBound = Nothing  -- Already at final chamber
  | coherenceMeetsThreshold (isCoherence state) (chamberThreshold nextChamber) =
      Just nextChamber
  | otherwise = Nothing
  where
    nextChamber = succ (isChamber state)

-- | Attempt to advance to next alchemical stage
nextAlchemyStage :: InitiationState -> Maybe AlchemicalStage
nextAlchemyStage state
  | isStage state == maxBound = Nothing  -- Already at final stage
  | meetsStageConditions state = Just $ succ (isStage state)
  | otherwise = Nothing

-- | Check if ready for final ascension
isReadyForAscension :: InitiationState -> Bool
isReadyForAscension state =
  isChamber state == Chamber12Unity &&
  isStage state == StageReturnToSource &&
  isCoherence state >= ascensionThreshold

-- | Coherence required for ascension
ascensionThreshold :: Double
ascensionThreshold = 1 - phiInverse * phiInverse  -- ~0.854

-- =============================================================================
-- State Management
-- =============================================================================

-- | Initialize new initiation state
initializeInitiation :: InitiationState
initializeInitiation = InitiationState
  { isChamber = Chamber1Entry
  , isStage = StageDarkness
  , isCoherence = 0
  , isThreshold = ThresholdLow
  , isElement = Nothing
  , isChambersVisited = []
  , isStagesCompleted = []
  , isSessionTicks = 0
  }

-- | Update coherence and derive threshold
updateCoherence :: InitiationState -> Double -> InitiationState
updateCoherence state newCoherence =
  let clamped = max 0 (min 1 newCoherence)
      newThreshold = coherenceToThreshold clamped
      newElement = deriveElement clamped
  in state
    { isCoherence = clamped
    , isThreshold = newThreshold
    , isElement = Just newElement
    , isSessionTicks = isSessionTicks state + 1
    }

-- | Get symbol for current chamber
getChamberSymbol :: InitiationState -> ChamberSymbol
getChamberSymbol = chamberToSymbol . isChamber

-- | Get current elemental alignment
getElementAlignment :: InitiationState -> Maybe Element
getElementAlignment = isElement

-- =============================================================================
-- HUD Integration
-- =============================================================================

-- | Generate formatted HUD display for initiation state
initiationHUD :: InitiationState -> Text
initiationHUD state = T.unlines
  [ T.pack "╔═══════════════════════════════════════╗"
  , T.pack "║        INITIATION STATE               ║"
  , T.pack "╠═══════════════════════════════════════╣"
  , T.concat [T.pack "║ Chamber: ", chamberName (isChamber state), T.pack (replicate (20 - T.length (chamberName (isChamber state))) ' '), T.pack "║"]
  , T.concat [T.pack "║ Symbol:  ", symbolName (getChamberSymbol state), T.pack (replicate (20 - T.length (symbolName (getChamberSymbol state))) ' '), T.pack "║"]
  , T.pack "╠═══════════════════════════════════════╣"
  , T.concat [T.pack "║ Stage:   ", stageName (isStage state), T.pack (replicate (20 - T.length (stageName (isStage state))) ' '), T.pack "║"]
  , T.concat [T.pack "║ Element: ", elementName (isElement state), T.pack (replicate (20 - T.length (elementName (isElement state))) ' '), T.pack "║"]
  , T.pack "╠═══════════════════════════════════════╣"
  , T.concat [T.pack "║ Coherence: ", coherenceBar (isCoherence state), T.pack " ║"]
  , T.concat [T.pack "║ Level:     ", thresholdName (isThreshold state), T.pack (replicate (18 - T.length (thresholdName (isThreshold state))) ' '), T.pack "║"]
  , T.pack "╠═══════════════════════════════════════╣"
  , T.concat [T.pack "║ Ready for Ascension: ", readyText state, T.pack (replicate (9 - T.length (readyText state)) ' '), T.pack "║"]
  , T.pack "╚═══════════════════════════════════════╝"
  ]

-- | Compact single-line HUD
initiationHUDCompact :: InitiationState -> Text
initiationHUDCompact state = T.concat
  [ T.pack "["
  , chamberShort (isChamber state)
  , T.pack "|"
  , stageShort (isStage state)
  , T.pack "|"
  , coherencePercent (isCoherence state)
  , T.pack "|"
  , elementShort (isElement state)
  , T.pack "]"
  ]

-- | Get description for current stage
stageDescription :: AlchemicalStage -> Text
stageDescription StageDarkness = T.pack "In the initial state of unconsciousness, light has not yet gathered."
stageDescription StageGatheringLight = T.pack "Concentration begins, light gathers at the center."
stageDescription StageCirculation = T.pack "Energy circulates through the meridians, establishing flow."
stageDescription StageEmbryoFormation = T.pack "The immortal embryo gestates in the golden flower."
stageDescription StageCrystallization = T.pack "The spiritual body crystallizes into permanent form."
stageDescription StageReturnToSource = T.pack "Unity with the Source achieved, the great work complete."

-- =============================================================================
-- Chamber Settings
-- =============================================================================

-- | Chamber configuration for Ra.Chamber integration
data ChamberSettings = ChamberSettings
  { csFrequency :: !Double       -- ^ Recommended frequency (Hz)
  , csGeometry  :: !Text         -- ^ Geometric configuration
  , csFilters   :: ![Text]       -- ^ Active filters
  , csElement   :: !Element      -- ^ Associated element
  , csColor     :: !(Int, Int, Int)  -- ^ RGB color signature
  } deriving (Eq, Show)

-- | Get recommended settings for a chamber
chamberSettings :: Chamber -> ChamberSettings
chamberSettings Chamber1Entry = ChamberSettings 396 (T.pack "threshold") [T.pack "ground"] ElementEarth (128, 128, 128)
chamberSettings Chamber2Earth = ChamberSettings 417 (T.pack "cube") [T.pack "stabilize"] ElementEarth (139, 90, 43)
chamberSettings Chamber3Fire = ChamberSettings 528 (T.pack "tetrahedron") [T.pack "purify"] ElementFire (255, 69, 0)
chamberSettings Chamber4Water = ChamberSettings 639 (T.pack "icosahedron") [T.pack "dissolve"] ElementWater (0, 105, 148)
chamberSettings Chamber5Air = ChamberSettings 741 (T.pack "octahedron") [T.pack "sublimate"] ElementAir (135, 206, 235)
chamberSettings Chamber6Sword = ChamberSettings 852 (T.pack "dodecahedron") [T.pack "discriminate"] ElementFire (192, 192, 192)
chamberSettings Chamber7Mirror = ChamberSettings 963 (T.pack "sphere") [T.pack "reflect"] ElementWater (220, 220, 250)
chamberSettings Chamber8Star = ChamberSettings 528 (T.pack "stellated") [T.pack "aspire"] ElementAether (255, 215, 0)
chamberSettings Chamber9Moon = ChamberSettings 396 (T.pack "crescent") [T.pack "intuit"] ElementWater (200, 200, 220)
chamberSettings Chamber10Sun = ChamberSettings 528 (T.pack "torus") [T.pack "illuminate"] ElementFire (255, 223, 0)
chamberSettings Chamber11Phoenix = ChamberSettings 963 (T.pack "spiral") [T.pack "transform"] ElementFire (255, 100, 50)
chamberSettings Chamber12Unity = ChamberSettings 432 (T.pack "merkaba") [T.pack "integrate", T.pack "unify"] ElementAether (255, 255, 255)

-- | Get recommended frequency for chamber
recommendedFrequency :: Chamber -> Double
recommendedFrequency = csFrequency . chamberSettings

-- =============================================================================
-- Simulation
-- =============================================================================

-- | Simulate initiation path from coherence trace
simulateInitiationPath :: [Double] -> [InitiationState]
simulateInitiationPath = scanl step initializeInitiation
  where
    step state coherence =
      let updated = updateCoherence state coherence
          -- Try to advance chamber
          withChamber = case unlockChamber updated of
            Just nextChamber -> updated
              { isChamber = nextChamber
              , isChambersVisited = isChamber updated : isChambersVisited updated
              }
            Nothing -> updated
          -- Try to advance stage
          withStage = case nextAlchemyStage withChamber of
            Just nextStage -> withChamber
              { isStage = nextStage
              , isStagesCompleted = isStage withChamber : isStagesCompleted withChamber
              }
            Nothing -> withChamber
      in withStage

-- | Get progression trace summary
progressionTrace :: [InitiationState] -> Text
progressionTrace states = T.unlines
  [ T.pack "=== Initiation Progression Trace ==="
  , T.concat [T.pack "Total steps: ", T.pack (show (length states))]
  , T.concat [T.pack "Final chamber: ", chamberName (isChamber final)]
  , T.concat [T.pack "Final stage: ", stageName (isStage final)]
  , T.concat [T.pack "Chambers visited: ", T.pack (show (length (isChambersVisited final)))]
  , T.concat [T.pack "Stages completed: ", T.pack (show (length (isStagesCompleted final)))]
  , T.concat [T.pack "Ascension ready: ", T.pack (show (isReadyForAscension final))]
  ]
  where
    final = if null states then initializeInitiation else last states

-- =============================================================================
-- Symbolic Mapping
-- =============================================================================

-- | Derive chamber symbol from element
deriveSymbolFromElement :: Element -> ChamberSymbol
deriveSymbolFromElement ElementEarth = SymbolStone
deriveSymbolFromElement ElementFire = SymbolTorch
deriveSymbolFromElement ElementWater = SymbolVessel
deriveSymbolFromElement ElementAir = SymbolFeather
deriveSymbolFromElement ElementAether = SymbolAnkh

-- | Get element associated with chamber
elementFromChamber :: Chamber -> Element
elementFromChamber = csElement . chamberSettings

-- | Get RGB color for chamber
chamberColor :: Chamber -> (Int, Int, Int)
chamberColor = csColor . chamberSettings

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Map chamber to its symbol
chamberToSymbol :: Chamber -> ChamberSymbol
chamberToSymbol Chamber1Entry = SymbolDoor
chamberToSymbol Chamber2Earth = SymbolStone
chamberToSymbol Chamber3Fire = SymbolTorch
chamberToSymbol Chamber4Water = SymbolVessel
chamberToSymbol Chamber5Air = SymbolFeather
chamberToSymbol Chamber6Sword = SymbolSword
chamberToSymbol Chamber7Mirror = SymbolMirror
chamberToSymbol Chamber8Star = SymbolStar
chamberToSymbol Chamber9Moon = SymbolCrescent
chamberToSymbol Chamber10Sun = SymbolSun
chamberToSymbol Chamber11Phoenix = SymbolPhoenix
chamberToSymbol Chamber12Unity = SymbolAnkh

-- | Check if coherence meets threshold
coherenceMeetsThreshold :: Double -> CoherenceThreshold -> Bool
coherenceMeetsThreshold coh thresh =
  coh >= thresholdValue thresh

-- | Get minimum value for threshold
thresholdValue :: CoherenceThreshold -> Double
thresholdValue ThresholdLow = 0
thresholdValue ThresholdMid = phiInverse * phiInverse  -- ~0.382
thresholdValue ThresholdHigh = phiInverse              -- ~0.618
thresholdValue ThresholdAscend = 1 - phiInverse * phiInverse  -- ~0.854

-- | Get required threshold for chamber
chamberThreshold :: Chamber -> CoherenceThreshold
chamberThreshold Chamber1Entry = ThresholdLow
chamberThreshold Chamber2Earth = ThresholdLow
chamberThreshold Chamber3Fire = ThresholdLow
chamberThreshold Chamber4Water = ThresholdMid
chamberThreshold Chamber5Air = ThresholdMid
chamberThreshold Chamber6Sword = ThresholdMid
chamberThreshold Chamber7Mirror = ThresholdHigh
chamberThreshold Chamber8Star = ThresholdHigh
chamberThreshold Chamber9Moon = ThresholdHigh
chamberThreshold Chamber10Sun = ThresholdHigh
chamberThreshold Chamber11Phoenix = ThresholdAscend
chamberThreshold Chamber12Unity = ThresholdAscend

-- | Check stage advancement conditions
meetsStageConditions :: InitiationState -> Bool
meetsStageConditions state =
  let coh = isCoherence state
      stage = isStage state
      chambersRequired = stageRequiredChambers stage
      currentChamberNum = fromEnum (isChamber state)
  in coh >= stageCoherenceReq stage && currentChamberNum >= chambersRequired

-- | Minimum coherence for stage
stageCoherenceReq :: AlchemicalStage -> Double
stageCoherenceReq StageDarkness = 0
stageCoherenceReq StageGatheringLight = 0.2
stageCoherenceReq StageCirculation = phiInverse * phiInverse
stageCoherenceReq StageEmbryoFormation = phiInverse
stageCoherenceReq StageCrystallization = phiInverse + 0.1
stageCoherenceReq StageReturnToSource = 1 - phiInverse * phiInverse

-- | Required chambers for stage
stageRequiredChambers :: AlchemicalStage -> Int
stageRequiredChambers StageDarkness = 0
stageRequiredChambers StageGatheringLight = 2
stageRequiredChambers StageCirculation = 4
stageRequiredChambers StageEmbryoFormation = 6
stageRequiredChambers StageCrystallization = 9
stageRequiredChambers StageReturnToSource = 11

-- | Convert coherence to threshold
coherenceToThreshold :: Double -> CoherenceThreshold
coherenceToThreshold coh
  | coh >= 1 - phiInverse * phiInverse = ThresholdAscend
  | coh >= phiInverse = ThresholdHigh
  | coh >= phiInverse * phiInverse = ThresholdMid
  | otherwise = ThresholdLow

-- | Derive element from coherence
deriveElement :: Double -> Element
deriveElement coh
  | coh < 0.2 = ElementEarth
  | coh < 0.4 = ElementWater
  | coh < 0.6 = ElementFire
  | coh < 0.8 = ElementAir
  | otherwise = ElementAether

-- | Chamber display name
chamberName :: Chamber -> Text
chamberName Chamber1Entry = T.pack "Entry"
chamberName Chamber2Earth = T.pack "Earth"
chamberName Chamber3Fire = T.pack "Fire"
chamberName Chamber4Water = T.pack "Water"
chamberName Chamber5Air = T.pack "Air"
chamberName Chamber6Sword = T.pack "Sword"
chamberName Chamber7Mirror = T.pack "Mirror"
chamberName Chamber8Star = T.pack "Star"
chamberName Chamber9Moon = T.pack "Moon"
chamberName Chamber10Sun = T.pack "Sun"
chamberName Chamber11Phoenix = T.pack "Phoenix"
chamberName Chamber12Unity = T.pack "Unity"

-- | Chamber short code
chamberShort :: Chamber -> Text
chamberShort c = T.pack $ show (fromEnum c + 1)

-- | Symbol display name
symbolName :: ChamberSymbol -> Text
symbolName SymbolDoor = T.pack "Door"
symbolName SymbolStone = T.pack "Stone"
symbolName SymbolTorch = T.pack "Torch"
symbolName SymbolVessel = T.pack "Vessel"
symbolName SymbolFeather = T.pack "Feather"
symbolName SymbolSword = T.pack "Sword"
symbolName SymbolMirror = T.pack "Mirror"
symbolName SymbolStar = T.pack "Star"
symbolName SymbolCrescent = T.pack "Crescent"
symbolName SymbolSun = T.pack "Sun"
symbolName SymbolPhoenix = T.pack "Phoenix"
symbolName SymbolAnkh = T.pack "Ankh"

-- | Stage display name
stageName :: AlchemicalStage -> Text
stageName StageDarkness = T.pack "Darkness"
stageName StageGatheringLight = T.pack "Gathering Light"
stageName StageCirculation = T.pack "Circulation"
stageName StageEmbryoFormation = T.pack "Embryo Formation"
stageName StageCrystallization = T.pack "Crystallization"
stageName StageReturnToSource = T.pack "Return to Source"

-- | Stage short code
stageShort :: AlchemicalStage -> Text
stageShort s = T.pack $ show (fromEnum s + 1) ++ "/6"

-- | Element display name
elementName :: Maybe Element -> Text
elementName Nothing = T.pack "None"
elementName (Just ElementEarth) = T.pack "Earth"
elementName (Just ElementFire) = T.pack "Fire"
elementName (Just ElementWater) = T.pack "Water"
elementName (Just ElementAir) = T.pack "Air"
elementName (Just ElementAether) = T.pack "Aether"

-- | Element short code
elementShort :: Maybe Element -> Text
elementShort Nothing = T.pack "-"
elementShort (Just ElementEarth) = T.pack "E"
elementShort (Just ElementFire) = T.pack "F"
elementShort (Just ElementWater) = T.pack "W"
elementShort (Just ElementAir) = T.pack "A"
elementShort (Just ElementAether) = T.pack "Æ"

-- | Threshold display name
thresholdName :: CoherenceThreshold -> Text
thresholdName ThresholdLow = T.pack "Low"
thresholdName ThresholdMid = T.pack "Mid"
thresholdName ThresholdHigh = T.pack "High"
thresholdName ThresholdAscend = T.pack "Ascend"

-- | Coherence percentage
coherencePercent :: Double -> Text
coherencePercent coh = T.pack $ show (round (coh * 100) :: Int) ++ "%"

-- | Coherence visual bar
coherenceBar :: Double -> Text
coherenceBar coh =
  let filled = round (coh * 20) :: Int
      empty = 20 - filled
  in T.pack $ replicate filled '█' ++ replicate empty '░'

-- | Ready for ascension text
readyText :: InitiationState -> Text
readyText state = if isReadyForAscension state then T.pack "YES" else T.pack "NO"
