{-|
Module      : Ra.Alchemy.Stages
Description : Classical alchemical stages for avatar state progression
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements classical alchemical stages from Flamel's Hieroglyphics and
the Rosarium Philosophorum as avatar coherence states. Each stage maps
to specific coherence requirements, color signatures, and chamber settings.

== Alchemical Tradition

=== The Great Work Stages

The opus magnum progresses through distinct phases:

1. Nigredo (Blackening) - Dissolution and putrefaction
2. Albedo (Whitening) - Purification and illumination
3. Citrinitas (Yellowing) - Solar awakening
4. Rubedo (Reddening) - Final integration

=== Symbolic Imagery

Each stage has associated symbolic imagery:

* Dragons - Primal forces in opposition
* Crow/Raven - Death of the old self
* White Queen - Lunar consciousness
* Red King - Solar consciousness
* Rebis - Unified being
-}
module Ra.Alchemy.Stages
  ( -- * Core Types
    AlchemicalStage(..)
  , StagePhase(..)
  , ColorSignature(..)
  , StageRequirements(..)

    -- * Stage Properties
  , stageDescription
  , stageColor
  , stagePhase
  , stageSymbol

    -- * Progression
  , advanceStage
  , canAdvance
  , progressionPath

    -- * Coherence Requirements
  , stageCoherenceReq
  , stageConsentCondition
  , meetsRequirements

    -- * Chamber Settings
  , StageChamberSettings(..)
  , chamberSettings
  , stageFrequency
  , stageGeometry
  , stageFilters

    -- * Visualization
  , StageVisual(..)
  , stageVisual
  , stageMood
  , stageMusic

    -- * State Management
  , AlchemyState(..)
  , initAlchemyState
  , updateAlchemyState
  , alchemyHUD
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Ra.Constants.Extended (phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | The 16 classical alchemical stages
data AlchemicalStage
  = StagePrimaMaterias     -- ^ Raw material, unconscious state
  | StageTwoDragons        -- ^ Opposition of forces (Flamel)
  | StageCalcinatio        -- ^ Burning away impurities
  | StageSolutio           -- ^ Dissolution in water
  | StagePutrefactio       -- ^ Death and decay (Black Crow)
  | StageNigredo           -- ^ Complete blackening
  | StageSeparatio         -- ^ Separation of elements
  | StageConjunctio        -- ^ First union of opposites
  | StageFermentatio       -- ^ Fermentation begins
  | StageAlbedo            -- ^ Whitening (White Queen)
  | StageDistillatio       -- ^ Purification through distillation
  | StageSublimatio        -- ^ Elevation of spirit
  | StageCitrinitas        -- ^ Yellowing, solar dawn
  | StageMultiplicatio     -- ^ Multiplication of the stone
  | StageRubedo            -- ^ Reddening (Red King/Lion)
  | StageProjectio         -- ^ Final projection, Rebis achieved
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Major phases of the work
data StagePhase
  = PhaseNigredo     -- ^ Black phase - death/dissolution
  | PhaseAlbedo      -- ^ White phase - purification
  | PhaseCitrinitas  -- ^ Yellow phase - awakening
  | PhaseRubedo      -- ^ Red phase - completion
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Color signature for stage
data ColorSignature = ColorSignature
  { csRed     :: !Int    -- ^ Red component 0-255
  , csGreen   :: !Int    -- ^ Green component 0-255
  , csBlue    :: !Int    -- ^ Blue component 0-255
  , csName    :: !Text   -- ^ Color name
  } deriving (Eq, Show)

-- | Requirements for stage advancement
data StageRequirements = StageRequirements
  { srMinCoherence    :: !Double    -- ^ Minimum coherence [0, 1]
  , srConsentLevel    :: !Int       -- ^ Consent level required (1-4)
  , srPreviousStages  :: ![AlchemicalStage]  -- ^ Prerequisites
  , srMinDuration     :: !Int       -- ^ Minimum ticks at previous stage
  } deriving (Eq, Show)

-- =============================================================================
-- Stage Properties
-- =============================================================================

-- | Symbolic description of each stage
stageDescription :: AlchemicalStage -> Text
stageDescription StagePrimaMaterias = T.pack "The raw material of the soul awaits transformation. All potential lies dormant."
stageDescription StageTwoDragons = T.pack "Two primal forces engage in sacred combat. The fixed and volatile must be reconciled."
stageDescription StageCalcinatio = T.pack "The fire burns away attachments. What cannot withstand the flame is released."
stageDescription StageSolutio = T.pack "Dissolution in the waters of the unconscious. Rigid structures melt into possibility."
stageDescription StagePutrefactio = T.pack "The Black Crow appears. The old self dies that the new may be born."
stageDescription StageNigredo = T.pack "Complete darkness before dawn. The prima materia is reduced to its essence."
stageDescription StageSeparatio = T.pack "Elements are distinguished and purified. The wheat is separated from the chaff."
stageDescription StageConjunctio = T.pack "The sacred marriage begins. King and Queen meet in the alchemical bath."
stageDescription StageFermentatio = T.pack "New life stirs within the vessel. The seed of gold begins to grow."
stageDescription StageAlbedo = T.pack "The White Queen emerges. Lunar consciousness illuminates the darkness."
stageDescription StageDistillatio = T.pack "Essence rises and purifies. The subtle is extracted from the gross."
stageDescription StageSublimatio = T.pack "Spirit ascends to heaven. Matter is elevated to its highest expression."
stageDescription StageCitrinitas = T.pack "The yellowing of dawn. Solar consciousness awakens in the vessel."
stageDescription StageMultiplicatio = T.pack "The Stone multiplies its power. What was one becomes many."
stageDescription StageRubedo = T.pack "The Red Lion roars. Solar and lunar unite in the crimson dawn."
stageDescription StageProjectio = T.pack "The Rebis is achieved. The Philosopher's Stone transforms all it touches."

-- | Get color signature for stage
stageColor :: AlchemicalStage -> ColorSignature
stageColor StagePrimaMaterias = ColorSignature 64 64 64 (T.pack "Lead Gray")
stageColor StageTwoDragons = ColorSignature 128 0 128 (T.pack "Dragon Purple")
stageColor StageCalcinatio = ColorSignature 255 100 0 (T.pack "Flame Orange")
stageColor StageSolutio = ColorSignature 0 100 150 (T.pack "Deep Blue")
stageColor StagePutrefactio = ColorSignature 30 30 30 (T.pack "Crow Black")
stageColor StageNigredo = ColorSignature 0 0 0 (T.pack "Absolute Black")
stageColor StageSeparatio = ColorSignature 100 100 100 (T.pack "Ash Gray")
stageColor StageConjunctio = ColorSignature 150 100 150 (T.pack "Union Violet")
stageColor StageFermentatio = ColorSignature 100 80 40 (T.pack "Earth Brown")
stageColor StageAlbedo = ColorSignature 255 255 255 (T.pack "Pure White")
stageColor StageDistillatio = ColorSignature 200 220 255 (T.pack "Crystal Blue")
stageColor StageSublimatio = ColorSignature 230 230 250 (T.pack "Spirit Lavender")
stageColor StageCitrinitas = ColorSignature 255 223 0 (T.pack "Solar Yellow")
stageColor StageMultiplicatio = ColorSignature 255 200 50 (T.pack "Gold")
stageColor StageRubedo = ColorSignature 200 0 0 (T.pack "Crimson Red")
stageColor StageProjectio = ColorSignature 255 215 0 (T.pack "Philosopher's Gold")

-- | Get major phase for stage
stagePhase :: AlchemicalStage -> StagePhase
stagePhase StagePrimaMaterias = PhaseNigredo
stagePhase StageTwoDragons = PhaseNigredo
stagePhase StageCalcinatio = PhaseNigredo
stagePhase StageSolutio = PhaseNigredo
stagePhase StagePutrefactio = PhaseNigredo
stagePhase StageNigredo = PhaseNigredo
stagePhase StageSeparatio = PhaseAlbedo
stagePhase StageConjunctio = PhaseAlbedo
stagePhase StageFermentatio = PhaseAlbedo
stagePhase StageAlbedo = PhaseAlbedo
stagePhase StageDistillatio = PhaseCitrinitas
stagePhase StageSublimatio = PhaseCitrinitas
stagePhase StageCitrinitas = PhaseCitrinitas
stagePhase StageMultiplicatio = PhaseRubedo
stagePhase StageRubedo = PhaseRubedo
stagePhase StageProjectio = PhaseRubedo

-- | Get symbolic image for stage
stageSymbol :: AlchemicalStage -> Text
stageSymbol StagePrimaMaterias = T.pack "[Clay] Unformed Clay"
stageSymbol StageTwoDragons = T.pack "[Dragons] Two Dragons"
stageSymbol StageCalcinatio = T.pack "[Fire] Sacred Fire"
stageSymbol StageSolutio = T.pack "[Water] Waters of Life"
stageSymbol StagePutrefactio = T.pack "[Crow] Black Crow"
stageSymbol StageNigredo = T.pack "[Black] Black Sun"
stageSymbol StageSeparatio = T.pack "[Scales] Balance Scales"
stageSymbol StageConjunctio = T.pack "[Crown] King & Queen"
stageSymbol StageFermentatio = T.pack "[Seed] Golden Seed"
stageSymbol StageAlbedo = T.pack "[Moon] White Queen"
stageSymbol StageDistillatio = T.pack "[Alembic] Alembic"
stageSymbol StageSublimatio = T.pack "[Dove] Rising Dove"
stageSymbol StageCitrinitas = T.pack "[Sun] Solar Dawn"
stageSymbol StageMultiplicatio = T.pack "[Star] Multiplying Light"
stageSymbol StageRubedo = T.pack "[Lion] Red Lion"
stageSymbol StageProjectio = T.pack "[Stone] Philosopher's Stone"

-- =============================================================================
-- Progression
-- =============================================================================

-- | Attempt to advance to next stage
advanceStage :: AlchemicalStage -> Double -> Int -> Maybe AlchemicalStage
advanceStage currentStage coherence consent
  | currentStage == maxBound = Nothing
  | canAdvance currentStage coherence consent = Just (succ currentStage)
  | otherwise = Nothing

-- | Check if advancement conditions are met
canAdvance :: AlchemicalStage -> Double -> Int -> Bool
canAdvance stage coherence consent =
  let reqs = stageRequirements (succ' stage)
  in coherence >= srMinCoherence reqs &&
     consent >= srConsentLevel reqs
  where
    succ' s = if s == maxBound then s else succ s

-- | Get full progression path
progressionPath :: [AlchemicalStage]
progressionPath = [minBound .. maxBound]

-- =============================================================================
-- Coherence Requirements
-- =============================================================================

-- | Get requirements for a stage
stageRequirements :: AlchemicalStage -> StageRequirements
stageRequirements StagePrimaMaterias = StageRequirements 0 1 [] 0
stageRequirements StageTwoDragons = StageRequirements 0.1 1 [StagePrimaMaterias] 10
stageRequirements StageCalcinatio = StageRequirements 0.15 1 [StageTwoDragons] 20
stageRequirements StageSolutio = StageRequirements 0.2 1 [StageCalcinatio] 20
stageRequirements StagePutrefactio = StageRequirements 0.25 2 [StageSolutio] 30
stageRequirements StageNigredo = StageRequirements 0.3 2 [StagePutrefactio] 40
stageRequirements StageSeparatio = StageRequirements phiInverseSquared 2 [StageNigredo] 30
stageRequirements StageConjunctio = StageRequirements 0.45 2 [StageSeparatio] 30
stageRequirements StageFermentatio = StageRequirements 0.5 2 [StageConjunctio] 40
stageRequirements StageAlbedo = StageRequirements phiInverse 3 [StageFermentatio] 50
stageRequirements StageDistillatio = StageRequirements 0.65 3 [StageAlbedo] 40
stageRequirements StageSublimatio = StageRequirements 0.7 3 [StageDistillatio] 40
stageRequirements StageCitrinitas = StageRequirements 0.75 3 [StageSublimatio] 50
stageRequirements StageMultiplicatio = StageRequirements 0.8 3 [StageCitrinitas] 50
stageRequirements StageRubedo = StageRequirements ascendThreshold 4 [StageMultiplicatio] 60
stageRequirements StageProjectio = StageRequirements 0.95 4 [StageRubedo] 100

-- | Get minimum coherence for stage
stageCoherenceReq :: AlchemicalStage -> Double
stageCoherenceReq = srMinCoherence . stageRequirements

-- | Get consent condition description
stageConsentCondition :: AlchemicalStage -> Text
stageConsentCondition stage =
  let level = srConsentLevel (stageRequirements stage)
  in case level of
    1 -> T.pack "Basic consent - awareness of the process"
    2 -> T.pack "Informed consent - understanding of transformation"
    3 -> T.pack "Deep consent - commitment to completion"
    4 -> T.pack "Full consent - surrender to the Great Work"
    _ -> T.pack "Unknown consent level"

-- | Check if state meets stage requirements
meetsRequirements :: AlchemyState -> AlchemicalStage -> Bool
meetsRequirements state targetStage =
  let reqs = stageRequirements targetStage
  in asCoherence state >= srMinCoherence reqs &&
     asConsentLevel state >= srConsentLevel reqs &&
     all (`elem` asCompletedStages state) (srPreviousStages reqs)

-- =============================================================================
-- Chamber Settings
-- =============================================================================

-- | Chamber settings for stage
data StageChamberSettings = StageChamberSettings
  { scsFrequency  :: !Double     -- ^ Primary frequency (Hz)
  , scsGeometry   :: !Text       -- ^ Sacred geometry
  , scsFilters    :: ![Text]     -- ^ Active filters
  , scsElement    :: !Text       -- ^ Associated element
  , scsTemperature :: !Text      -- ^ Alchemical temperature
  } deriving (Eq, Show)

-- | Get chamber settings for stage
chamberSettings :: AlchemicalStage -> StageChamberSettings
chamberSettings StagePrimaMaterias = StageChamberSettings 396 (T.pack "sphere") [T.pack "ground"] (T.pack "Earth") (T.pack "Cold")
chamberSettings StageTwoDragons = StageChamberSettings 417 (T.pack "vesica") [T.pack "polarize"] (T.pack "Fire/Water") (T.pack "Moderate")
chamberSettings StageCalcinatio = StageChamberSettings 528 (T.pack "tetrahedron") [T.pack "burn", T.pack "purify"] (T.pack "Fire") (T.pack "Hot")
chamberSettings StageSolutio = StageChamberSettings 639 (T.pack "icosahedron") [T.pack "dissolve", T.pack "flow"] (T.pack "Water") (T.pack "Cool")
chamberSettings StagePutrefactio = StageChamberSettings 396 (T.pack "void") [T.pack "decay", T.pack "release"] (T.pack "Earth") (T.pack "Cold")
chamberSettings StageNigredo = StageChamberSettings 174 (T.pack "black-cube") [T.pack "absorb", T.pack "still"] (T.pack "Saturn") (T.pack "Frozen")
chamberSettings StageSeparatio = StageChamberSettings 741 (T.pack "octahedron") [T.pack "separate", T.pack "discern"] (T.pack "Air") (T.pack "Variable")
chamberSettings StageConjunctio = StageChamberSettings 528 (T.pack "merkaba") [T.pack "unite", T.pack "merge"] (T.pack "Aether") (T.pack "Warm")
chamberSettings StageFermentatio = StageChamberSettings 639 (T.pack "torus") [T.pack "grow", T.pack "nurture"] (T.pack "Earth/Water") (T.pack "Warm")
chamberSettings StageAlbedo = StageChamberSettings 852 (T.pack "silver-sphere") [T.pack "illuminate", T.pack "reflect"] (T.pack "Moon/Silver") (T.pack "Cool")
chamberSettings StageDistillatio = StageChamberSettings 963 (T.pack "alembic") [T.pack "extract", T.pack "refine"] (T.pack "Air/Fire") (T.pack "Hot")
chamberSettings StageSublimatio = StageChamberSettings 963 (T.pack "ascending-spiral") [T.pack "elevate", T.pack "transcend"] (T.pack "Air") (T.pack "Light")
chamberSettings StageCitrinitas = StageChamberSettings 528 (T.pack "solar-disc") [T.pack "awaken", T.pack "shine"] (T.pack "Sun/Gold") (T.pack "Radiant")
chamberSettings StageMultiplicatio = StageChamberSettings 528 (T.pack "fractal") [T.pack "multiply", T.pack "expand"] (T.pack "Light") (T.pack "Intense")
chamberSettings StageRubedo = StageChamberSettings 432 (T.pack "ruby-sphere") [T.pack "complete", T.pack "embody"] (T.pack "Sun/Mars") (T.pack "Hot")
chamberSettings StageProjectio = StageChamberSettings 432 (T.pack "philosopher-stone") [T.pack "project", T.pack "transform"] (T.pack "Quintessence") (T.pack "Perfect")

-- | Get frequency for stage
stageFrequency :: AlchemicalStage -> Double
stageFrequency = scsFrequency . chamberSettings

-- | Get geometry for stage
stageGeometry :: AlchemicalStage -> Text
stageGeometry = scsGeometry . chamberSettings

-- | Get filters for stage
stageFilters :: AlchemicalStage -> [Text]
stageFilters = scsFilters . chamberSettings

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Visual rendering specification
data StageVisual = StageVisual
  { svPrimaryColor   :: !ColorSignature   -- ^ Primary color
  , svSecondaryColor :: !ColorSignature   -- ^ Secondary color
  , svPattern        :: !Text             -- ^ Visual pattern
  , svAnimation      :: !Text             -- ^ Animation type
  , svIntensity      :: !Double           -- ^ Visual intensity [0, 1]
  } deriving (Eq, Show)

-- | Get visual specification for stage
stageVisual :: AlchemicalStage -> StageVisual
stageVisual stage =
  let primary = stageColor stage
      secondary = phaseSecondaryColor (stagePhase stage)
      pattern' = visualPattern stage
      animation = animationType stage
      intensity = stageIntensity stage
  in StageVisual primary secondary pattern' animation intensity

-- | Get mood/atmosphere for stage
stageMood :: AlchemicalStage -> Text
stageMood StagePrimaMaterias = T.pack "Heavy, dormant, potential"
stageMood StageTwoDragons = T.pack "Tense, dynamic, conflicted"
stageMood StageCalcinatio = T.pack "Intense, purifying, releasing"
stageMood StageSolutio = T.pack "Flowing, dissolving, surrendering"
stageMood StagePutrefactio = T.pack "Dark, decomposing, transforming"
stageMood StageNigredo = T.pack "Absolute stillness, void, death"
stageMood StageSeparatio = T.pack "Clear, discerning, analytical"
stageMood StageConjunctio = T.pack "Loving, unifying, sacred"
stageMood StageFermentatio = T.pack "Growing, nurturing, alive"
stageMood StageAlbedo = T.pack "Pure, peaceful, illuminated"
stageMood StageDistillatio = T.pack "Refined, elevated, essential"
stageMood StageSublimatio = T.pack "Transcendent, ascending, ethereal"
stageMood StageCitrinitas = T.pack "Awakening, radiant, joyful"
stageMood StageMultiplicatio = T.pack "Abundant, expansive, multiplying"
stageMood StageRubedo = T.pack "Powerful, complete, embodied"
stageMood StageProjectio = T.pack "Perfect, transformative, golden"

-- | Get musical mode for stage
stageMusic :: AlchemicalStage -> Text
stageMusic StagePrimaMaterias = T.pack "Drone, deep bass, earthen"
stageMusic StageTwoDragons = T.pack "Dissonant, building tension"
stageMusic StageCalcinatio = T.pack "Intense percussion, fire sounds"
stageMusic StageSolutio = T.pack "Flowing water, gentle melody"
stageMusic StagePutrefactio = T.pack "Silence with subtle decay sounds"
stageMusic StageNigredo = T.pack "Complete silence, void tones"
stageMusic StageSeparatio = T.pack "Clear bell tones, separation"
stageMusic StageConjunctio = T.pack "Harmonious duet, merging themes"
stageMusic StageFermentatio = T.pack "Growing complexity, organic"
stageMusic StageAlbedo = T.pack "Silver bells, lunar melody"
stageMusic StageDistillatio = T.pack "Pure tones, ascending scales"
stageMusic StageSublimatio = T.pack "Ethereal choir, heavenly"
stageMusic StageCitrinitas = T.pack "Bright brass, solar fanfare"
stageMusic StageMultiplicatio = T.pack "Full orchestra, multiplying themes"
stageMusic StageRubedo = T.pack "Triumphant, royal, complete"
stageMusic StageProjectio = T.pack "Perfect harmony, golden ratio"

-- =============================================================================
-- State Management
-- =============================================================================

-- | Full alchemy state for tracking
data AlchemyState = AlchemyState
  { asCurrentStage    :: !AlchemicalStage   -- ^ Current stage
  , asCoherence       :: !Double            -- ^ Current coherence
  , asConsentLevel    :: !Int               -- ^ Current consent (1-4)
  , asCompletedStages :: ![AlchemicalStage] -- ^ Completed stages
  , asStageDuration   :: !Int               -- ^ Ticks at current stage
  , asTotalDuration   :: !Int               -- ^ Total ticks
  } deriving (Eq, Show)

-- | Initialize alchemy state
initAlchemyState :: AlchemyState
initAlchemyState = AlchemyState
  { asCurrentStage = StagePrimaMaterias
  , asCoherence = 0
  , asConsentLevel = 1
  , asCompletedStages = []
  , asStageDuration = 0
  , asTotalDuration = 0
  }

-- | Update alchemy state with new coherence
updateAlchemyState :: AlchemyState -> Double -> Int -> AlchemyState
updateAlchemyState state coherence consent =
  let updated = state
        { asCoherence = max 0 (min 1 coherence)
        , asConsentLevel = max 1 (min 4 consent)
        , asStageDuration = asStageDuration state + 1
        , asTotalDuration = asTotalDuration state + 1
        }
      -- Try to advance
      maybeNext = advanceStage (asCurrentStage updated) (asCoherence updated) (asConsentLevel updated)
  in case maybeNext of
    Just nextStage -> updated
      { asCurrentStage = nextStage
      , asCompletedStages = asCurrentStage updated : asCompletedStages updated
      , asStageDuration = 0
      }
    Nothing -> updated

-- | Generate HUD display for alchemy state
alchemyHUD :: AlchemyState -> Text
alchemyHUD state = T.unlines
  [ T.pack "╔════════════════════════════════════════════════╗"
  , T.pack "║            ALCHEMICAL PROGRESSION              ║"
  , T.pack "╠════════════════════════════════════════════════╣"
  , T.concat [T.pack "║ Stage: ", padRight 38 (stageName' (asCurrentStage state)), T.pack " ║"]
  , T.concat [T.pack "║ Phase: ", padRight 38 (phaseName (stagePhase (asCurrentStage state))), T.pack " ║"]
  , T.concat [T.pack "║ Symbol: ", padRight 37 (stageSymbol (asCurrentStage state)), T.pack " ║"]
  , T.pack "╠════════════════════════════════════════════════╣"
  , T.concat [T.pack "║ ", stageDescription (asCurrentStage state)]
  , T.pack "╠════════════════════════════════════════════════╣"
  , T.concat [T.pack "║ Coherence: ", coherenceBar (asCoherence state), T.pack "  ║"]
  , T.concat [T.pack "║ Consent:   ", consentBar (asConsentLevel state), T.pack "        ║"]
  , T.pack "╠════════════════════════════════════════════════╣"
  , T.concat [T.pack "║ Mood: ", padRight 39 (stageMood (asCurrentStage state)), T.pack " ║"]
  , T.pack "╚════════════════════════════════════════════════╝"
  ]

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | φ^(-2) threshold
phiInverseSquared :: Double
phiInverseSquared = phiInverse * phiInverse

-- | Ascension threshold
ascendThreshold :: Double
ascendThreshold = 1 - phiInverseSquared

-- | Secondary color for phase
phaseSecondaryColor :: StagePhase -> ColorSignature
phaseSecondaryColor PhaseNigredo = ColorSignature 50 50 50 (T.pack "Shadow")
phaseSecondaryColor PhaseAlbedo = ColorSignature 200 200 220 (T.pack "Silver")
phaseSecondaryColor PhaseCitrinitas = ColorSignature 255 240 200 (T.pack "Pale Gold")
phaseSecondaryColor PhaseRubedo = ColorSignature 255 100 50 (T.pack "Orange")

-- | Visual pattern for stage
visualPattern :: AlchemicalStage -> Text
visualPattern StagePrimaMaterias = T.pack "static-noise"
visualPattern StageTwoDragons = T.pack "intertwined-spirals"
visualPattern StageCalcinatio = T.pack "flame-particles"
visualPattern StageSolutio = T.pack "rippling-water"
visualPattern StagePutrefactio = T.pack "dissolving-form"
visualPattern StageNigredo = T.pack "void-pulse"
visualPattern StageSeparatio = T.pack "splitting-rays"
visualPattern StageConjunctio = T.pack "merging-forms"
visualPattern StageFermentatio = T.pack "growing-crystal"
visualPattern StageAlbedo = T.pack "lunar-glow"
visualPattern StageDistillatio = T.pack "rising-vapor"
visualPattern StageSublimatio = T.pack "ascending-light"
visualPattern StageCitrinitas = T.pack "solar-rays"
visualPattern StageMultiplicatio = T.pack "fractal-bloom"
visualPattern StageRubedo = T.pack "crimson-pulse"
visualPattern StageProjectio = T.pack "golden-radiance"

-- | Animation type for stage
animationType :: AlchemicalStage -> Text
animationType stage = case stagePhase stage of
  PhaseNigredo -> T.pack "slow-fade"
  PhaseAlbedo -> T.pack "gentle-pulse"
  PhaseCitrinitas -> T.pack "radiant-expand"
  PhaseRubedo -> T.pack "triumphant-glow"

-- | Intensity for stage
stageIntensity :: AlchemicalStage -> Double
stageIntensity stage =
  let idx = fromIntegral (fromEnum stage) :: Double
      total = fromIntegral (fromEnum (maxBound :: AlchemicalStage)) :: Double
  in 0.3 + (idx / total) * 0.7

-- | Stage display name
stageName' :: AlchemicalStage -> Text
stageName' StagePrimaMaterias = T.pack "Prima Materia"
stageName' StageTwoDragons = T.pack "Two Dragons"
stageName' StageCalcinatio = T.pack "Calcinatio"
stageName' StageSolutio = T.pack "Solutio"
stageName' StagePutrefactio = T.pack "Putrefactio"
stageName' StageNigredo = T.pack "Nigredo"
stageName' StageSeparatio = T.pack "Separatio"
stageName' StageConjunctio = T.pack "Conjunctio"
stageName' StageFermentatio = T.pack "Fermentatio"
stageName' StageAlbedo = T.pack "Albedo"
stageName' StageDistillatio = T.pack "Distillatio"
stageName' StageSublimatio = T.pack "Sublimatio"
stageName' StageCitrinitas = T.pack "Citrinitas"
stageName' StageMultiplicatio = T.pack "Multiplicatio"
stageName' StageRubedo = T.pack "Rubedo"
stageName' StageProjectio = T.pack "Projectio"

-- | Phase display name
phaseName :: StagePhase -> Text
phaseName PhaseNigredo = T.pack "Nigredo (Blackening)"
phaseName PhaseAlbedo = T.pack "Albedo (Whitening)"
phaseName PhaseCitrinitas = T.pack "Citrinitas (Yellowing)"
phaseName PhaseRubedo = T.pack "Rubedo (Reddening)"

-- | Coherence visual bar
coherenceBar :: Double -> Text
coherenceBar coh =
  let filled = round (coh * 25) :: Int
      empty = 25 - filled
  in T.pack $ replicate filled '█' ++ replicate empty '░' ++ " " ++ show (round (coh * 100) :: Int) ++ "%"

-- | Consent visual bar
consentBar :: Int -> Text
consentBar level =
  T.pack $ replicate level '◆' ++ replicate (4 - level) '◇' ++ " Level " ++ show level

-- | Pad text to width
padRight :: Int -> Text -> Text
padRight width txt =
  let len = T.length txt
  in if len >= width
     then T.take width txt
     else T.append txt (T.pack (replicate (width - len) ' '))
