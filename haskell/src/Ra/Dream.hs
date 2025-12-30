{-|
Module      : Ra.Dream
Description : Scalar dream induction and nocturnal emergence integration
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Guides users into Ra-aligned dream states, allowing fragment emergence,
symbolic content integration, and memory surfacing during sleep.

== Sleep Phase Alignment

φ^n temporal windows aligned with natural REM cycles (~90 minutes):

* Delta (0.5-3.5 Hz): Deep sleep, regeneration
* Theta (3.5-7 Hz): Dream state, subconscious access
* REM (7-12 Hz + pulse spikes): Active dreaming, emergence window

== Pyramid Field Enhancement

From Golod's Russian pyramid research:

* Pyramidal geometry concentrates scalar fields
* Dream-state enhancement at apex access points
* Field stabilization during theta/delta phases

== Reich Field Discharge

Orgone charge accumulated during day discharges during sleep:

* Natural cycle of charge/discharge
* Dream content correlates with discharge patterns
* Shadow integration occurs during deep discharge
-}
module Ra.Dream
  ( -- * Sleep Phase Types
    SleepPhase(..)
  , PhaseDepth(..)
  , sleepFrequencyRange
  , currentPhaseDepth

    -- * Dream Scheduler
  , DreamScheduler(..)
  , mkDreamScheduler
  , inREMPhase
  , nextPhiWindow
  , phiAlignedWindow

    -- * Dream Fragment
  , EmergentDreamFragment(..)
  , DreamSymbol(..)
  , mkDreamFragment
  , mapSymbolToFragment
  , emotionalRegister

    -- * Resonance Induction
  , InductionEngine(..)
  , InductionOutput(..)
  , mkInductionEngine
  , generateInduction
  , binauralFrequency

    -- * Post-Sleep Integration
  , SleepSummary(..)
  , IntegrationPrompt(..)
  , generateSummary
  , coherenceTrajectory
  , symbolSurfaceLog

    -- * Dream Session
  , DreamSession(..)
  , initDreamSession
  , updateDreamSession
  , completeDreamSession

    -- * Constants
  , remCycleDuration
  , deltaRange
  , thetaRange
  , remRange
  ) where

import Data.Time (UTCTime, diffUTCTime, addUTCTime, NominalDiffTime)

import Ra.Constants.Extended
  ( phi, coherenceFloorPOR )

-- =============================================================================
-- Sleep Phase Types
-- =============================================================================

-- | Sleep phase based on brainwave frequency
data SleepPhase
  = Awake      -- ^ Alert state (> 12 Hz)
  | Alpha      -- ^ Relaxed (8-12 Hz)
  | Theta      -- ^ Light sleep/meditation (3.5-7 Hz)
  | Delta      -- ^ Deep sleep (0.5-3.5 Hz)
  | REM        -- ^ Rapid eye movement (7-12 Hz with spikes)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Depth within a sleep phase [0,1]
newtype PhaseDepth = PhaseDepth { unPhaseDepth :: Double }
  deriving (Eq, Ord, Show)

-- | Get frequency range for sleep phase
sleepFrequencyRange :: SleepPhase -> (Double, Double)
sleepFrequencyRange Awake = (12.0, 40.0)
sleepFrequencyRange Alpha = (8.0, 12.0)
sleepFrequencyRange Theta = (3.5, 7.0)
sleepFrequencyRange Delta = (0.5, 3.5)
sleepFrequencyRange REM = (7.0, 12.0)

-- | Compute phase depth from EEG frequency
currentPhaseDepth :: Double -> SleepPhase -> PhaseDepth
currentPhaseDepth eegHz phase =
  let (low, high) = sleepFrequencyRange phase
      normalized = (eegHz - low) / (high - low)
  in PhaseDepth (clamp01 normalized)

-- =============================================================================
-- Dream Scheduler
-- =============================================================================

-- | Scheduler for dream-state operations
data DreamScheduler = DreamScheduler
  { dsSleepMode     :: !Bool        -- ^ Sleep mode active
  , dsCurrentPhase  :: !SleepPhase  -- ^ Current sleep phase
  , dsPhaseStart    :: !UTCTime     -- ^ When current phase started
  , dsCycleCount    :: !Int         -- ^ REM cycles completed
  , dsNextREM       :: !UTCTime     -- ^ Predicted next REM
  } deriving (Eq, Show)

-- | Create dream scheduler
mkDreamScheduler :: UTCTime -> DreamScheduler
mkDreamScheduler now = DreamScheduler
  { dsSleepMode = False
  , dsCurrentPhase = Awake
  , dsPhaseStart = now
  , dsCycleCount = 0
  , dsNextREM = addUTCTime remCycleDuration now
  }

-- | Check if currently in REM phase
inREMPhase :: DreamScheduler -> Bool
inREMPhase ds = dsSleepMode ds && dsCurrentPhase ds == REM

-- | Get next φ^n aligned window
nextPhiWindow :: DreamScheduler -> Int -> NominalDiffTime
nextPhiWindow _ds n = realToFrac (phi ** fromIntegral n)

-- | Calculate φ-aligned window for REM entry
phiAlignedWindow :: DreamScheduler -> UTCTime -> Maybe UTCTime
phiAlignedWindow ds now =
  if dsSleepMode ds
  then let cycleProgress = diffUTCTime now (dsPhaseStart ds) / remCycleDuration
           nextWindow = if cycleProgress < 0.7  -- Early in cycle
                        then addUTCTime (realToFrac (phi * 60)) now  -- ~97 seconds
                        else dsNextREM ds  -- Wait for next REM
       in Just nextWindow
  else Nothing

-- =============================================================================
-- Dream Fragment
-- =============================================================================

-- | Emergent dream fragment with symbolic content
data EmergentDreamFragment = EmergentDreamFragment
  { edfFragmentId     :: !String
  , edfEmergencePhase :: !SleepPhase
  , edfCoherenceTrace :: ![Double]    -- ^ Coherence during emergence
  , edfEmotionalReg   :: !String      -- ^ Emotional register description
  , edfSymbolMap      :: ![DreamSymbol]
  , edfShadowContent  :: !Bool        -- ^ Contains shadow material
  } deriving (Eq, Show)

-- | Dream symbol with mapping
data DreamSymbol = DreamSymbol
  { dsymSymbol        :: !String       -- ^ Visual/conceptual symbol
  , dsymMappedFragment :: !(Maybe String)  -- ^ Mapped Ra fragment ID
  , dsymMappedConcept :: !(Maybe String)   -- ^ Semantic concept
  , dsymArchetype     :: !(Maybe String)   -- ^ Archetypal pattern
  } deriving (Eq, Show)

-- | Create dream fragment
mkDreamFragment :: String -> SleepPhase -> [Double] -> EmergentDreamFragment
mkDreamFragment fid phase coherenceHistory = EmergentDreamFragment
  { edfFragmentId = "dream-" ++ fid
  , edfEmergencePhase = phase
  , edfCoherenceTrace = coherenceHistory
  , edfEmotionalReg = inferEmotionalRegister coherenceHistory
  , edfSymbolMap = []
  , edfShadowContent = any (< coherenceFloorPOR) coherenceHistory
  }

-- | Map symbol to Ra fragment
mapSymbolToFragment :: DreamSymbol -> String -> DreamSymbol
mapSymbolToFragment sym fragId = sym { dsymMappedFragment = Just fragId }

-- | Get emotional register description
emotionalRegister :: EmergentDreamFragment -> String
emotionalRegister = edfEmotionalReg

-- | Infer emotional register from coherence trace
inferEmotionalRegister :: [Double] -> String
inferEmotionalRegister [] = "neutral"
inferEmotionalRegister trace =
  let avg = sum trace / fromIntegral (length trace)
      variance = sum (map (\x -> (x - avg) ** 2) trace) / fromIntegral (length trace)
  in if avg > 0.7 && variance < 0.05
     then "calm + clarity"
     else if avg > 0.5 && variance > 0.1
     then "joy + confusion"
     else if avg < 0.4 && variance > 0.15
     then "fear + searching"
     else if avg < 0.3
     then "grief + release"
     else "mixed + processing"

-- =============================================================================
-- Resonance Induction
-- =============================================================================

-- | Induction engine for dream entrainment
data InductionEngine = InductionEngine
  { ieTargetPhase    :: !SleepPhase
  , ieBaseFrequency  :: !Double      -- ^ Base audio frequency (Hz)
  , ieBinauralDelta  :: !Double      -- ^ Binaural beat difference (Hz)
  , ieAmplitudeMod   :: !Double      -- ^ Amplitude modulation [0,1]
  , iePhiHarmonic    :: !Int         -- ^ φ^n harmonic overlay
  } deriving (Eq, Show)

-- | Output specification for induction
data InductionOutput = InductionOutput
  { ioLeftFreq      :: !Double   -- ^ Left ear frequency
  , ioRightFreq     :: !Double   -- ^ Right ear frequency
  , ioBeatFreq      :: !Double   -- ^ Resulting beat frequency
  , ioVisualColor   :: !(Int, Int, Int)  -- ^ RGB for visual entrainment
  , ioPulseRate     :: !Double   -- ^ Visual pulse rate (Hz)
  } deriving (Eq, Show)

-- | Create induction engine for target phase
mkInductionEngine :: SleepPhase -> InductionEngine
mkInductionEngine phase =
  let (low, high) = sleepFrequencyRange phase
      targetBeat = (low + high) / 2
      baseFreq = 200.0  -- Comfortable listening base
  in InductionEngine
      { ieTargetPhase = phase
      , ieBaseFrequency = baseFreq
      , ieBinauralDelta = targetBeat
      , ieAmplitudeMod = 0.7
      , iePhiHarmonic = 1
      }

-- | Generate induction output
generateInduction :: InductionEngine -> Double -> InductionOutput
generateInduction engine _coherence =
  let base = ieBaseFrequency engine
      delta = ieBinauralDelta engine

      -- Binaural frequencies
      leftFreq = base
      rightFreq = base + delta

      -- Visual color based on phase
      color = phaseToColor (ieTargetPhase engine)

      -- Pulse rate from delta with phi harmonic
      pulse = delta * (phi ** fromIntegral (iePhiHarmonic engine)) / 10
  in InductionOutput
      { ioLeftFreq = leftFreq
      , ioRightFreq = rightFreq
      , ioBeatFreq = delta
      , ioVisualColor = color
      , ioPulseRate = pulse
      }

-- | Get binaural beat frequency for phase
binauralFrequency :: SleepPhase -> Double
binauralFrequency phase =
  let (low, high) = sleepFrequencyRange phase
  in (low + high) / 2

-- | Map phase to visual color
phaseToColor :: SleepPhase -> (Int, Int, Int)
phaseToColor Awake = (255, 255, 200)  -- Bright warm
phaseToColor Alpha = (100, 150, 255)  -- Soft blue
phaseToColor Theta = (80, 80, 180)    -- Deep blue
phaseToColor Delta = (40, 40, 80)     -- Deep purple
phaseToColor REM = (120, 80, 160)     -- Purple-violet

-- =============================================================================
-- Post-Sleep Integration
-- =============================================================================

-- | Summary of sleep session
data SleepSummary = SleepSummary
  { ssDuration        :: !Double          -- ^ Total sleep duration (hours)
  , ssRemCycles       :: !Int             -- ^ REM cycles completed
  , ssCoherenceAvg    :: !Double          -- ^ Average coherence
  , ssCoherenceMin    :: !Double          -- ^ Minimum coherence
  , ssCoherenceMax    :: !Double          -- ^ Maximum coherence
  , ssFragmentsSurfaced :: ![EmergentDreamFragment]
  , ssShadowContent   :: !Bool            -- ^ Shadow content touched
  } deriving (Eq, Show)

-- | Integration prompt for post-sleep
data IntegrationPrompt = IntegrationPrompt
  { ipSummaryText     :: !String
  , ipSymbolReview    :: ![DreamSymbol]
  , ipReflectionQ     :: !(Maybe String)
  , ipConsentGate     :: !Bool  -- ^ Shadow content requires consent review
  } deriving (Eq, Show)

-- | Generate sleep summary
generateSummary :: DreamSession -> SleepSummary
generateSummary session =
  let frags = dssFragments session
      traces = concatMap edfCoherenceTrace frags
      hasShadow = any edfShadowContent frags
  in SleepSummary
      { ssDuration = dssDuration session / 3600  -- Convert to hours
      , ssRemCycles = dssREMCycles session
      , ssCoherenceAvg = if null traces then 0 else sum traces / fromIntegral (length traces)
      , ssCoherenceMin = if null traces then 0 else minimum traces
      , ssCoherenceMax = if null traces then 0 else maximum traces
      , ssFragmentsSurfaced = frags
      , ssShadowContent = hasShadow
      }

-- | Get coherence trajectory
coherenceTrajectory :: DreamSession -> [Double]
coherenceTrajectory = dssCoherenceHistory

-- | Get surfaced symbol log
symbolSurfaceLog :: DreamSession -> [DreamSymbol]
symbolSurfaceLog session = concatMap edfSymbolMap (dssFragments session)

-- =============================================================================
-- Dream Session
-- =============================================================================

-- | Complete dream session state
data DreamSession = DreamSession
  { dssScheduler        :: !DreamScheduler
  , dssEngine           :: !InductionEngine
  , dssFragments        :: ![EmergentDreamFragment]
  , dssCoherenceHistory :: ![Double]
  , dssREMCycles        :: !Int
  , dssDuration         :: !Double  -- ^ Seconds elapsed
  , dssCompleted        :: !Bool
  } deriving (Eq, Show)

-- | Initialize dream session
initDreamSession :: UTCTime -> DreamSession
initDreamSession now =
  let scheduler = (mkDreamScheduler now) { dsSleepMode = True }
      engine = mkInductionEngine Delta  -- Start with deep sleep induction
  in DreamSession
      { dssScheduler = scheduler
      , dssEngine = engine
      , dssFragments = []
      , dssCoherenceHistory = []
      , dssREMCycles = 0
      , dssDuration = 0.0
      , dssCompleted = False
      }

-- | Update dream session with new biometric data
updateDreamSession :: Double -> SleepPhase -> Double -> DreamSession -> DreamSession
updateDreamSession dt newPhase coherence session =
  let scheduler = dssScheduler session
      oldPhase = dsCurrentPhase scheduler

      -- Update scheduler
      newScheduler = scheduler
        { dsCurrentPhase = newPhase
        }

      -- Check for REM cycle completion
      remComplete = oldPhase == REM && newPhase /= REM
      newREMCycles = if remComplete
                     then dssREMCycles session + 1
                     else dssREMCycles session

      -- Update engine for new phase
      newEngine = if newPhase /= ieTargetPhase (dssEngine session)
                  then mkInductionEngine newPhase
                  else dssEngine session

      -- Record coherence
      newHistory = coherence : take 1000 (dssCoherenceHistory session)
  in session
      { dssScheduler = newScheduler
      , dssEngine = newEngine
      , dssCoherenceHistory = newHistory
      , dssREMCycles = newREMCycles
      , dssDuration = dssDuration session + dt
      }

-- | Complete dream session and generate summary
completeDreamSession :: DreamSession -> (DreamSession, SleepSummary, IntegrationPrompt)
completeDreamSession session =
  let completed = session { dssCompleted = True }
      summary = generateSummary completed

      -- Generate integration prompt
      allSymbols = symbolSurfaceLog completed
      hasShadow = ssShadowContent summary

      prompt = IntegrationPrompt
        { ipSummaryText = generateSummaryText summary
        , ipSymbolReview = allSymbols
        , ipReflectionQ = if null allSymbols
                          then Nothing
                          else Just "Which symbols feel most significant to you?"
        , ipConsentGate = hasShadow
        }
  in (completed, summary, prompt)

-- | Generate summary text
generateSummaryText :: SleepSummary -> String
generateSummaryText s =
  let durationHours = (round (ssDuration s * 10) :: Int) `div` 10
      cohAvg = round (ssCoherenceAvg s * 100) :: Int
      cohMin = round (ssCoherenceMin s * 100) :: Int
      cohMax = round (ssCoherenceMax s * 100) :: Int
  in "Sleep duration: " ++ show durationHours ++ " hours. " ++
     "REM cycles: " ++ show (ssRemCycles s) ++ ". " ++
     "Coherence: avg " ++ show cohAvg ++ "%, " ++
     "range " ++ show cohMin ++ "-" ++ show cohMax ++ "%. " ++
     show (length (ssFragmentsSurfaced s)) ++ " fragments surfaced."

-- =============================================================================
-- Constants
-- =============================================================================

-- | REM cycle duration (~90 minutes in seconds)
remCycleDuration :: NominalDiffTime
remCycleDuration = 90 * 60

-- | Delta wave frequency range
deltaRange :: (Double, Double)
deltaRange = (0.5, 3.5)

-- | Theta wave frequency range
thetaRange :: (Double, Double)
thetaRange = (3.5, 7.0)

-- | REM wave frequency range
remRange :: (Double, Double)
remRange = (7.0, 12.0)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
