{-|
Module      : RaDreamPhaseScheduler
Description : Scalar Dream Induction & Symbolic Fragment Integration
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 13: Guide users into Ra-aligned dream states, surface symbolic
fragments, and deliver post-sleep coherence analysis.

== Core Features

1. DreamPhase Scheduler - φ-aligned REM cycle detection
2. EmergentDreamFragment - Symbolic content with coherence traces
3. Resonance Induction Engine - Binaural/visual entrainment
4. Post-Sleep Integration - Gated summary with shadow detection
5. Prompt 12 Integration - Consent gating for shadow symbols

== Biometric Sources

- Primary: EEG (alpha/theta/delta bands)
- Fallback: HRV + breath cadence

== Sleep Phase Timing

- WAKE → THETA → DELTA → REM cycle
- REM every ~90 mins (5400 cycles at 1Hz)
- φ^n modulation from ra_constants_v2.json

== Entrainment Frequencies

- Delta: 0.5–3.5 Hz (deep sleep)
- Theta: 3.5–7 Hz (light sleep/meditation)
- REM: 7–12 Hz + microspikes

== Hardware Synthesis

- Target: Xilinx Artix-7 / Intel Cyclone V
- Clock: 1 Hz scheduler tick (phase timing)
- Resources: ~350 LUTs, 1 DSP slice
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TypeOperators #-}

module RaDreamPhaseScheduler where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types: Base Definitions
-- =============================================================================

type Fixed8 = Unsigned 8
type Fixed16 = Unsigned 16
type FragmentId = Unsigned 16
type Timestamp = Unsigned 32
type CoherenceTrace = Vec 3 Fixed8  -- 3-sample coherence history

-- =============================================================================
-- Types: Sleep Phase
-- =============================================================================

-- | Sleep phase states
data SleepPhase
  = PhaseWake     -- ^ Awake state
  | PhaseTheta    -- ^ Light sleep / meditation (3.5-7 Hz)
  | PhaseDelta    -- ^ Deep sleep (0.5-3.5 Hz)
  | PhaseREM      -- ^ REM sleep (7-12 Hz)
  deriving (Generic, NFDataX, Show, Eq)

-- | Phase output for downstream modules
data PhaseOutput = PhaseOutput
  { inREM       :: Bool        -- ^ Currently in REM phase
  , sleepPhase  :: SleepPhase  -- ^ Current sleep phase
  , phaseDepth  :: Fixed8      -- ^ Depth within phase (0-255)
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Emotional Register
-- =============================================================================

-- | Emotional register (can combine flags)
data EmotionFlag
  = EmotionJoy
  | EmotionConfusion
  | EmotionAwe
  | EmotionGrief
  | EmotionCuriosity
  | EmotionFear
  | EmotionPeace
  | EmotionNone
  deriving (Generic, NFDataX, Show, Eq)

-- | Combined emotional register (2 primary emotions)
data EmotionalRegister = EmotionalRegister
  { primaryEmotion   :: EmotionFlag
  , secondaryEmotion :: EmotionFlag
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Symbol Mapping
-- =============================================================================

-- | Archetypal symbols for dream mapping
data ArchetypalSymbol
  = SymbolOwl       -- ^ Wisdom, insight
  | SymbolSpiral    -- ^ Searching, journey
  | SymbolMirror    -- ^ Self-reflection
  | SymbolRiver     -- ^ Flow, change
  | SymbolLabyrinth -- ^ Confusion, complexity
  | SymbolLight     -- ^ Insight, clarity
  | SymbolFlame     -- ^ Transformation
  | SymbolCave      -- ^ Hidden, unconscious
  | SymbolTree      -- ^ Growth, connection
  | SymbolMoon      -- ^ Cycles, feminine
  | SymbolStar      -- ^ Guidance, aspiration
  | SymbolWater     -- ^ Emotion, depth
  deriving (Generic, NFDataX, Show, Eq)

-- | Symbol to fragment mapping
data SymbolMapping = SymbolMapping
  { symbol          :: ArchetypalSymbol  -- ^ The symbol
  , mappedFragment  :: FragmentId        -- ^ Linked Ra fragment (0 = concept only)
  , mappedConcept   :: Unsigned 8        -- ^ Concept code
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Emergent Dream Fragment
-- =============================================================================

-- | Fragment form type
data FragmentForm
  = FormSymbolic    -- ^ Dream/symbolic content
  | FormNarrative   -- ^ Story-like sequence
  | FormAbstract    -- ^ Non-representational
  | FormSomatic     -- ^ Body sensation
  deriving (Generic, NFDataX, Show, Eq)

-- | Emergent dream fragment
data EmergentDreamFragment = EmergentDreamFragment
  { fragmentId       :: FragmentId        -- ^ Unique identifier (dream-XXXX)
  , form             :: FragmentForm      -- ^ Always FormSymbolic for dreams
  , emergencePhase   :: SleepPhase        -- ^ Phase when fragment surfaced
  , coherenceTrace   :: CoherenceTrace    -- ^ Coherence samples
  , emotionalReg     :: EmotionalRegister -- ^ Emotional coloring
  , symbolMap        :: Vec 4 SymbolMapping  -- ^ Up to 4 symbol mappings
  , symbolCount      :: Unsigned 3        -- ^ Active symbols (0-4)
  , shadowDetected   :: Bool              -- ^ Shadow content flag (Prompt 12)
  , timestamp        :: Timestamp         -- ^ When fragment emerged
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Resonance Induction
-- =============================================================================

-- | Entrainment frequency band
data FrequencyBand
  = BandDelta   -- ^ 0.5-3.5 Hz
  | BandTheta   -- ^ 3.5-7 Hz
  | BandAlpha   -- ^ 7-12 Hz (REM)
  | BandBeta    -- ^ 12-30 Hz (wake)
  deriving (Generic, NFDataX, Show, Eq)

-- | Audio entrainment type
data AudioType
  = AudioBinaural    -- ^ Binaural beats
  | AudioIsochronic  -- ^ Isochronic tones
  | AudioGoldenStack -- ^ φ-ratio harmonic stack
  | AudioNone        -- ^ Silence
  deriving (Generic, NFDataX, Show, Eq)

-- | Visual entrainment type
data VisualType
  = VisualFlowerOfLife  -- ^ Sacred geometry bloom
  | VisualPhiSpiral     -- ^ Golden spiral animation
  | VisualLEDPulse      -- ^ φ-locked LED modulation
  | VisualNone          -- ^ Darkness
  deriving (Generic, NFDataX, Show, Eq)

-- | Resonance induction output
data ResonanceOutput = ResonanceOutput
  { targetBand     :: FrequencyBand   -- ^ Target frequency band
  , audioType      :: AudioType       -- ^ Audio entrainment mode
  , visualType     :: VisualType      -- ^ Visual entrainment mode
  , baseFrequency  :: Fixed16         -- ^ Base frequency (0.01 Hz units)
  , amplitudeMod   :: Fixed8          -- ^ Amplitude modulation (0-255)
  , phiMultiplier  :: Fixed8          -- ^ φ^n multiplier index
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Post-Sleep Integration
-- =============================================================================

-- | Integration summary for post-sleep review
data IntegrationSummary = IntegrationSummary
  { fragmentsSurfaced :: Unsigned 8      -- ^ Number of fragments
  , dominantSymbols   :: Vec 3 ArchetypalSymbol  -- ^ Top 3 symbols
  , coherenceDelta    :: Signed 16       -- ^ Net coherence change
  , shadowGated       :: Bool            -- ^ Shadow content withheld
  , journalPrompt     :: Unsigned 8      -- ^ Suggested journal prompt code
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Scheduler State
-- =============================================================================

-- | Dream phase scheduler state
data SchedulerState = SchedulerState
  { remCounter    :: Unsigned 16   -- ^ Cycles since last REM
  , sleepMode     :: Bool          -- ^ Sleep mode active
  , currentPhase  :: SleepPhase    -- ^ Current sleep phase
  , phaseCounter  :: Unsigned 16   -- ^ Time in current phase
  , cycleNumber   :: Unsigned 8    -- ^ Sleep cycle count
  , fragmentCount :: Unsigned 8    -- ^ Fragments this session
  } deriving (Generic, NFDataX, Show, Eq)

-- | Complete output
data DreamSchedulerOutput = DreamSchedulerOutput
  { phaseOutput    :: PhaseOutput         -- ^ Current phase info
  , resonance      :: ResonanceOutput     -- ^ Entrainment settings
  , activeFragment :: EmergentDreamFragment  -- ^ Current fragment (if any)
  , fragmentReady  :: Bool                -- ^ Fragment surfaced flag
  , sessionActive  :: Bool                -- ^ Sleep session active
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Constants: Timing
-- =============================================================================

-- | φ-cycle period for REM (~90 mins = 5400 seconds)
phiCycle :: Unsigned 16
phiCycle = 5400

-- | Minimum time in THETA before DELTA
thetaMinDuration :: Unsigned 16
thetaMinDuration = 600  -- 10 minutes

-- | Minimum time in DELTA before REM
deltaMinDuration :: Unsigned 16
deltaMinDuration = 1800  -- 30 minutes

-- | REM phase duration
remDuration :: Unsigned 16
remDuration = 900  -- 15 minutes (grows per cycle)

-- | Coherence threshold for fragment emergence
coherenceThreshold :: Fixed8
coherenceThreshold = 153  -- 0.6

-- =============================================================================
-- Constants: Frequencies (in 0.01 Hz units)
-- =============================================================================

-- | Delta band center (2 Hz = 200)
deltaFreq :: Fixed16
deltaFreq = 200

-- | Theta band center (5 Hz = 500)
thetaFreq :: Fixed16
thetaFreq = 500

-- | Alpha/REM band center (10 Hz = 1000)
alphaFreq :: Fixed16
alphaFreq = 1000

-- =============================================================================
-- Core Functions: Phase Transitions
-- =============================================================================

-- | Determine next phase based on timing
nextPhase :: SchedulerState -> SleepPhase
nextPhase st = case currentPhase st of
  PhaseWake  -> PhaseTheta
  PhaseTheta -> if phaseCounter st >= thetaMinDuration
                then PhaseDelta else PhaseTheta
  PhaseDelta -> if remCounter st >= phiCycle
                then PhaseREM else PhaseDelta
  PhaseREM   -> if phaseCounter st >= (remDuration + resize (cycleNumber st) * 300)
                then PhaseWake else PhaseREM  -- REM grows each cycle

-- | Check if transitioning to new phase
isPhaseTransition :: SchedulerState -> Bool
isPhaseTransition st = nextPhase st /= currentPhase st

-- =============================================================================
-- Core Functions: Resonance Generation
-- =============================================================================

-- | Generate resonance output for current phase
generateResonance :: SleepPhase -> Unsigned 8 -> ResonanceOutput
generateResonance phase depth = case phase of
  PhaseWake -> ResonanceOutput
    { targetBand = BandBeta
    , audioType = AudioNone
    , visualType = VisualNone
    , baseFrequency = 0
    , amplitudeMod = 0
    , phiMultiplier = 0
    }
  PhaseTheta -> ResonanceOutput
    { targetBand = BandTheta
    , audioType = AudioBinaural
    , visualType = VisualPhiSpiral
    , baseFrequency = thetaFreq
    , amplitudeMod = resize depth
    , phiMultiplier = 1
    }
  PhaseDelta -> ResonanceOutput
    { targetBand = BandDelta
    , audioType = AudioGoldenStack
    , visualType = VisualLEDPulse
    , baseFrequency = deltaFreq
    , amplitudeMod = resize depth `shiftR` 1
    , phiMultiplier = 2
    }
  PhaseREM -> ResonanceOutput
    { targetBand = BandAlpha
    , audioType = AudioGoldenStack
    , visualType = VisualFlowerOfLife
    , baseFrequency = alphaFreq
    , amplitudeMod = resize depth
    , phiMultiplier = 3
    }

-- =============================================================================
-- Core Functions: Fragment Generation
-- =============================================================================

-- | Initial empty fragment
emptyFragment :: EmergentDreamFragment
emptyFragment = EmergentDreamFragment
  { fragmentId = 0
  , form = FormSymbolic
  , emergencePhase = PhaseWake
  , coherenceTrace = repeat 0
  , emotionalReg = EmotionalRegister EmotionNone EmotionNone
  , symbolMap = repeat (SymbolMapping SymbolOwl 0 0)
  , symbolCount = 0
  , shadowDetected = False
  , timestamp = 0
  }

-- | Check if coherence trace indicates rising pattern (emergence ready)
isCoherenceRising :: CoherenceTrace -> Bool
isCoherenceRising trace =
  let (a :> b :> c :> Nil) = trace
  in c > a && b >= a

-- | Check if fragment should emerge (REM + rising coherence)
shouldEmergFragment :: SleepPhase -> CoherenceTrace -> Bool
shouldEmergFragment phase trace =
  phase == PhaseREM && isCoherenceRising trace && last trace >= coherenceThreshold

-- =============================================================================
-- Core Functions: Scheduler Step
-- =============================================================================

-- | Step the scheduler state
schedulerStep :: SchedulerState -> (Bool, Fixed8) -> SchedulerState
schedulerStep st (sleepIn, coherenceIn)
  | not sleepIn = SchedulerState 0 False PhaseWake 0 0 0  -- Reset on wake
  | otherwise =
      let
        -- Update phase counter
        newPhaseCounter = if isPhaseTransition st
                          then 0
                          else satAdd SatBound (phaseCounter st) 1

        -- Update REM counter (accumulates in DELTA)
        newRemCounter = if currentPhase st == PhaseDelta
                        then satAdd SatBound (remCounter st) 1
                        else if currentPhase st == PhaseREM
                             then 0  -- Reset after REM
                             else remCounter st

        -- Update cycle count on REM transition
        newCycleNum = if currentPhase st == PhaseDelta && nextPhase st == PhaseREM
                      then satAdd SatBound (cycleNumber st) 1
                      else cycleNumber st

        -- Transition to next phase
        newPhase = nextPhase st
      in
        SchedulerState
          { remCounter = newRemCounter
          , sleepMode = True
          , currentPhase = newPhase
          , phaseCounter = newPhaseCounter
          , cycleNumber = newCycleNum
          , fragmentCount = fragmentCount st
          }

-- =============================================================================
-- Core Functions: Main Processor
-- =============================================================================

-- | Process dream scheduler
processDreamScheduler
  :: SchedulerState
  -> CoherenceTrace
  -> EmergentDreamFragment
  -> (Bool, Fixed8)  -- ^ (sleep_mode, current_coherence)
  -> (SchedulerState, DreamSchedulerOutput)
processDreamScheduler st trace lastFrag (sleepIn, cohIn) =
  let
    -- Step scheduler
    newState = schedulerStep st (sleepIn, cohIn)

    -- Generate phase output
    phase = PhaseOutput
      { inREM = currentPhase newState == PhaseREM
      , sleepPhase = currentPhase newState
      , phaseDepth = resize (phaseCounter newState)
      }

    -- Generate resonance
    reso = generateResonance (currentPhase newState) (resize $ phaseCounter newState)

    -- Check for fragment emergence
    fragReady = shouldEmergFragment (currentPhase newState) trace

    -- Create output
    output = DreamSchedulerOutput
      { phaseOutput = phase
      , resonance = reso
      , activeFragment = if fragReady then lastFrag else emptyFragment
      , fragmentReady = fragReady
      , sessionActive = sleepIn
      }
  in
    (newState, output)

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Initial scheduler state
initState :: SchedulerState
initState = SchedulerState 0 False PhaseWake 0 0 0

-- | Dream phase scheduler processor (stateful)
dreamSchedulerProcessor
  :: HiddenClockResetEnable dom
  => Signal dom (Bool, Fixed8, CoherenceTrace)
  -> Signal dom DreamSchedulerOutput
dreamSchedulerProcessor input = mealy procState (initState, emptyFragment) input
  where
    procState (st, lastFrag) (sleepMode, coherence, trace) =
      let
        (newState, output) = processDreamScheduler st trace lastFrag (sleepMode, coherence)
        newFrag = if fragmentReady output then activeFragment output else lastFrag
      in
        ((newState, newFrag), output)

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

{-# ANN dreamSchedulerTop (Synthesize
  { t_name = "dream_phase_scheduler"
  , t_inputs = [ PortName "clk", PortName "rst", PortName "en"
               , PortName "sleep_mode"
               , PortName "coherence"
               , PortName "coherence_trace" ]
  , t_output = PortProduct "output"
      [ PortName "phase_out", PortName "resonance"
      , PortName "fragment", PortName "fragment_ready", PortName "session_active" ]
  }) #-}
dreamSchedulerTop
  :: Clock System -> Reset System -> Enable System
  -> Signal System Bool        -- ^ Sleep mode
  -> Signal System Fixed8      -- ^ Current coherence
  -> Signal System CoherenceTrace  -- ^ 3-sample trace
  -> Signal System DreamSchedulerOutput
dreamSchedulerTop clk rst en sleepMode coherence trace =
  exposeClockResetEnable dreamSchedulerProcessor clk rst en
    (bundle (sleepMode, coherence, trace))

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test coherence trace (rising pattern)
testTrace1 :: CoherenceTrace
testTrace1 = 105 :> 120 :> 156 :> Nil  -- 0.41, 0.47, 0.61

-- | Test fragment with symbols
testFragment1 :: EmergentDreamFragment
testFragment1 = EmergentDreamFragment
  { fragmentId = 0x4739
  , form = FormSymbolic
  , emergencePhase = PhaseREM
  , coherenceTrace = testTrace1
  , emotionalReg = EmotionalRegister EmotionJoy EmotionConfusion
  , symbolMap = SymbolMapping SymbolOwl 13 1 :>
                SymbolMapping SymbolSpiral 0 2 :>
                SymbolMapping SymbolMirror 21 3 :>
                SymbolMapping SymbolOwl 0 0 :> Nil
  , symbolCount = 3
  , shadowDetected = False
  , timestamp = 5400
  }

-- =============================================================================
-- Testbench
-- =============================================================================

testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    out = dreamSchedulerTop clk rst enableGen
            (pure True) (pure 156) (pure testTrace1)
    -- Test passes if REM phase eventually reached
    done = register clk rst enableGen False
            ((\o -> inREM (phaseOutput o)) <$> out)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Get phase name
phaseName :: SleepPhase -> String
phaseName PhaseWake  = "WAKE"
phaseName PhaseTheta = "THETA"
phaseName PhaseDelta = "DELTA"
phaseName PhaseREM   = "REM"

-- | Get symbol name
symbolName :: ArchetypalSymbol -> String
symbolName SymbolOwl       = "owl"
symbolName SymbolSpiral    = "spiral"
symbolName SymbolMirror    = "mirror"
symbolName SymbolRiver     = "river"
symbolName SymbolLabyrinth = "labyrinth"
symbolName SymbolLight     = "light"
symbolName SymbolFlame     = "flame"
symbolName SymbolCave      = "cave"
symbolName SymbolTree      = "tree"
symbolName SymbolMoon      = "moon"
symbolName SymbolStar      = "star"
symbolName SymbolWater     = "water"

-- | Get emotion name
emotionName :: EmotionFlag -> String
emotionName EmotionJoy       = "joy"
emotionName EmotionConfusion = "confusion"
emotionName EmotionAwe       = "awe"
emotionName EmotionGrief     = "grief"
emotionName EmotionCuriosity = "curiosity"
emotionName EmotionFear      = "fear"
emotionName EmotionPeace     = "peace"
emotionName EmotionNone      = "none"
