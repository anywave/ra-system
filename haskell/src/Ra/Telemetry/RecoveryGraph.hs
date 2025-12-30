{-|
Module      : Ra.Telemetry.RecoveryGraph
Description : Topological trail of coherence recovery
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Tracks the scalar field's recovery path after coherence disruption.
Visualizes emergence state transitions, field realignments, and
harmonic restorations as a topological trail—a "memory map" of the
chamber's field healing dynamics.

== Recovery Theory

=== Coherence Disruption

Disruptions occur from:

* External interference
* Internal phase conflicts
* Consent state changes
* Biometric desync

=== Recovery Patterns

Field recovery follows:

* φ^n timing for recharge cycles
* Harmonic axis realignment
* Coherence gradient restoration
* Inversion normalization
-}
module Ra.Telemetry.RecoveryGraph
  ( -- * Core Types
    RecoveryEvent(..)
  , RecoveryGraph(..)
  , HarmonicAxis(..)

    -- * Event Recording
  , recordEvent
  , eventFromSnapshot
  , timestampEvent

    -- * Graph Building
  , buildRecoveryGraph
  , addEvent
  , pruneOldEvents

    -- * Trail Analysis
  , coherenceMinimum
  , coherenceDelta
  , recoveryDuration

    -- * Axis Tracking
  , dominantAxisChange
  , axisStability
  , axisTrend

    -- * Loop Detection
  , detectLoop
  , loopClosed
  , loopDuration

    -- * Visualization
  , graphToPoints
  , trailToPath
  , renderRecovery

    -- * Diagnostics
  , DiagnosticReport(..)
  , generateDiagnostic
  , healthScore
  ) where

import Data.Time (UTCTime, getCurrentTime, diffUTCTime)
import Data.Time.Clock (NominalDiffTime)

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Recovery event in the trail
data RecoveryEvent = RecoveryEvent
  { reTimestamp     :: !UTCTime
  , reLocation      :: !RaCoordinate
  , reCoherenceValue :: !Double        -- ^ Coherence at event
  , reFluxValue     :: !Double         -- ^ Flux at event
  , reStateBefore   :: !EmergenceResult
  , reStateAfter    :: !EmergenceResult
  , reInversionShift :: !(Maybe Inversion)
  } deriving (Eq, Show)

-- | Recovery graph (trail of events)
data RecoveryGraph = RecoveryGraph
  { rgEventTrail    :: ![RecoveryEvent]
  , rgLoopClosed    :: !Bool           -- ^ Full recovery cycle complete
  , rgDominantAxis  :: !HarmonicAxis
  , rgCoherenceDelta :: !Double        -- ^ Total Δ coherence
  , rgStartTime     :: !UTCTime
  } deriving (Eq, Show)

-- | Harmonic axis
data HarmonicAxis
  = AxisTheta      -- ^ θ-axis (radial)
  | AxisPhi        -- ^ φ-axis (azimuthal)
  | AxisHarmonic   -- ^ h-axis (harmonic depth)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Ra coordinate
data RaCoordinate = RaCoordinate
  { rcRepitan   :: !Int
  , rcPhi       :: !Int
  , rcHarmonic  :: !Int
  } deriving (Eq, Show)

-- | Emergence result
data EmergenceResult
  = NoEmergence
  | PartialEmergence !Double
  | FullEmergence !String
  | ShadowEmergence
  deriving (Eq, Show)

-- | Inversion state
data Inversion
  = Normal
  | Inverted
  deriving (Eq, Show)

-- =============================================================================
-- Event Recording
-- =============================================================================

-- | Record a recovery event
recordEvent :: RaCoordinate -> Double -> Double -> EmergenceResult -> EmergenceResult -> Maybe Inversion -> IO RecoveryEvent
recordEvent loc coherence flux before after inv = do
  now <- getCurrentTime
  pure RecoveryEvent
    { reTimestamp = now
    , reLocation = loc
    , reCoherenceValue = coherence
    , reFluxValue = flux
    , reStateBefore = before
    , reStateAfter = after
    , reInversionShift = inv
    }

-- | Create event from field snapshot
eventFromSnapshot :: ScalarFieldSnapshot -> ScalarFieldSnapshot -> IO RecoveryEvent
eventFromSnapshot before after = do
  now <- getCurrentTime
  let coherence = sfsCoherence after
      flux = sfsFlux after
      locBefore = sfsLocation before
      stateBefore = sfsEmergence before
      stateAfter = sfsEmergence after
      inv = if sfsInverted before /= sfsInverted after
            then Just (if sfsInverted after then Inverted else Normal)
            else Nothing
  pure RecoveryEvent
    { reTimestamp = now
    , reLocation = locBefore
    , reCoherenceValue = coherence
    , reFluxValue = flux
    , reStateBefore = stateBefore
    , reStateAfter = stateAfter
    , reInversionShift = inv
    }

-- | Add timestamp to event
timestampEvent :: RecoveryEvent -> IO RecoveryEvent
timestampEvent event = do
  now <- getCurrentTime
  pure event { reTimestamp = now }

-- =============================================================================
-- Graph Building
-- =============================================================================

-- | Build recovery graph from data
buildRecoveryGraph :: [EmergenceResult] -> [ScalarFieldSnapshot] -> [InversionShift] -> IO RecoveryGraph
buildRecoveryGraph results snapshots shifts = do
  now <- getCurrentTime

  let -- Create events from snapshots
      eventPairs = zip snapshots (tail snapshots)
      events = zipWith3 makeEvent eventPairs results shifts

      -- Detect loop closure
      closed = detectLoopFromEvents events

      -- Find dominant axis
      axis = findDominantAxis events

      -- Calculate coherence delta
      cMin = minimum (map reCoherenceValue events ++ [1.0])
      cMax = maximum (map reCoherenceValue events ++ [0.0])
      delta = cMax - cMin

  pure RecoveryGraph
    { rgEventTrail = events
    , rgLoopClosed = closed
    , rgDominantAxis = axis
    , rgCoherenceDelta = delta
    , rgStartTime = now
    }
  where
    makeEvent (before, after) result shift =
      RecoveryEvent
        { reTimestamp = sfsTime before  -- Using snapshot time
        , reLocation = sfsLocation before
        , reCoherenceValue = sfsCoherence after
        , reFluxValue = sfsFlux after
        , reStateBefore = sfsEmergence before
        , reStateAfter = result
        , reInversionShift = Just (isInversion shift)
        }

    isInversion (InvToNormal) = Normal
    isInversion (NormalToInv) = Inverted
    isInversion (NoShift) = Normal

-- | Add event to graph
addEvent :: RecoveryEvent -> RecoveryGraph -> RecoveryGraph
addEvent event graph =
  let events = rgEventTrail graph ++ [event]
      newDelta = rgCoherenceDelta graph +
                 abs (reCoherenceValue event -
                      maybe 0.5 reCoherenceValue (lastEvent graph))
      closed = detectLoopFromEvents events
  in graph
      { rgEventTrail = events
      , rgLoopClosed = closed
      , rgCoherenceDelta = newDelta
      }
  where
    lastEvent g = case rgEventTrail g of
      [] -> Nothing
      xs -> Just (last xs)

-- | Prune old events (keep last N)
pruneOldEvents :: Int -> RecoveryGraph -> RecoveryGraph
pruneOldEvents n graph =
  let events = rgEventTrail graph
      pruned = drop (max 0 (length events - n)) events
  in graph { rgEventTrail = pruned }

-- =============================================================================
-- Trail Analysis
-- =============================================================================

-- | Find minimum coherence in trail
coherenceMinimum :: RecoveryGraph -> Double
coherenceMinimum graph =
  case rgEventTrail graph of
    [] -> 0.5
    events -> minimum (map reCoherenceValue events)

-- | Get total coherence delta
coherenceDelta :: RecoveryGraph -> Double
coherenceDelta = rgCoherenceDelta

-- | Calculate recovery duration
recoveryDuration :: RecoveryGraph -> NominalDiffTime
recoveryDuration graph =
  case rgEventTrail graph of
    [] -> 0
    events ->
      let first = reTimestamp (head events)
          lastEv = reTimestamp (last events)
      in diffUTCTime lastEv first

-- =============================================================================
-- Axis Tracking
-- =============================================================================

-- | Detect dominant axis changes
dominantAxisChange :: RecoveryGraph -> [(UTCTime, HarmonicAxis)]
dominantAxisChange graph =
  let events = rgEventTrail graph
      axes = map eventAxis events
      changes = detectChanges axes (map reTimestamp events)
  in changes
  where
    eventAxis event =
      let loc = reLocation event
          r = rcRepitan loc
          p = rcPhi loc
          h = rcHarmonic loc
      in if r > p && r > h then AxisTheta
         else if p > h then AxisPhi
         else AxisHarmonic

    detectChanges [] _ = []
    detectChanges _ [] = []
    detectChanges [a] [t] = [(t, a)]
    detectChanges [a] _ = [(rgStartTime graph, a)]  -- Handle single axis, multiple times
    detectChanges _ [t] = [(t, AxisTheta)]  -- Handle multiple axes, single time
    detectChanges (a1:a2:as) (t1:t2:ts)
      | a1 /= a2 = (t1, a1) : (t2, a2) : detectChanges (a2:as) (t2:ts)
      | otherwise = (t1, a1) : detectChanges (a2:as) (t2:ts)

-- | Calculate axis stability
axisStability :: RecoveryGraph -> Double
axisStability graph =
  let changes = dominantAxisChange graph
      numChanges = length changes - 1
      duration = recoveryDuration graph
  in if duration == 0 then 1.0
     else 1.0 / (1.0 + fromIntegral numChanges / realToFrac duration)

-- | Get axis trend direction
axisTrend :: RecoveryGraph -> HarmonicAxis
axisTrend = rgDominantAxis

-- =============================================================================
-- Loop Detection
-- =============================================================================

-- | Detect φ^n aligned recovery loop
detectLoop :: RecoveryGraph -> Bool
detectLoop = rgLoopClosed

-- | Check if loop is closed
loopClosed :: RecoveryGraph -> Bool
loopClosed = rgLoopClosed

-- | Get loop duration (φ^n ticks)
loopDuration :: RecoveryGraph -> Int
loopDuration graph =
  let duration = realToFrac (recoveryDuration graph) :: Double
  in round (logBase phi duration)

-- Detect loop from events
detectLoopFromEvents :: [RecoveryEvent] -> Bool
detectLoopFromEvents events =
  case events of
    [] -> False
    [_] -> False
    _ ->
      let first = reCoherenceValue (head events)
          lastC = reCoherenceValue (last events)
          -- Loop closed if we return to near starting coherence
      in abs (first - lastC) < 0.1 && length events >= 3

-- Find dominant axis from events
findDominantAxis :: [RecoveryEvent] -> HarmonicAxis
findDominantAxis events =
  let locs = map reLocation events
      thetaVar = variance (map (fromIntegral . rcRepitan) locs)
      phiVar = variance (map (fromIntegral . rcPhi) locs)
      harmVar = variance (map (fromIntegral . rcHarmonic) locs)
  in if thetaVar >= phiVar && thetaVar >= harmVar then AxisTheta
     else if phiVar >= harmVar then AxisPhi
     else AxisHarmonic
  where
    variance xs =
      let n = fromIntegral (length xs) :: Double
          mean = sum xs / n
          sqDiffs = map (\x -> (x - mean) ** 2) xs
      in if n == 0 then 0 else sum sqDiffs / n

-- =============================================================================
-- Visualization
-- =============================================================================

-- | Convert graph to plot points
graphToPoints :: RecoveryGraph -> [(Double, Double)]
graphToPoints graph =
  [ (realToFrac (diffUTCTime (reTimestamp e) (rgStartTime graph)),
     reCoherenceValue e)
  | e <- rgEventTrail graph
  ]

-- | Convert trail to path string
trailToPath :: RecoveryGraph -> String
trailToPath graph =
  let points = graphToPoints graph
      pathParts = [ "(" ++ show t ++ "," ++ show c ++ ")"
                  | (t, c) <- points
                  ]
  in unwords pathParts

-- | Render recovery as text
renderRecovery :: RecoveryGraph -> String
renderRecovery graph =
  unlines
    [ "Recovery Graph"
    , "=============="
    , "Events: " ++ show (length (rgEventTrail graph))
    , "Loop Closed: " ++ show (rgLoopClosed graph)
    , "Dominant Axis: " ++ show (rgDominantAxis graph)
    , "Coherence Δ: " ++ show (rgCoherenceDelta graph)
    , "Duration: " ++ show (recoveryDuration graph)
    , ""
    , "Trail:"
    , trailToPath graph
    ]

-- =============================================================================
-- Diagnostics
-- =============================================================================

-- | Diagnostic report
data DiagnosticReport = DiagnosticReport
  { drHealthScore    :: !Double        -- ^ Overall health [0, 1]
  , drRecoveryRate   :: !Double        -- ^ Recovery speed
  , drStability      :: !Double        -- ^ Axis stability
  , drIssues         :: ![String]      -- ^ Detected issues
  , drRecommendations :: ![String]     -- ^ Suggested actions
  } deriving (Eq, Show)

-- | Generate diagnostic report
generateDiagnostic :: RecoveryGraph -> DiagnosticReport
generateDiagnostic graph =
  let -- Calculate metrics
      minCoh = coherenceMinimum graph
      delta' = coherenceDelta graph
      stability = axisStability graph
      duration = realToFrac (recoveryDuration graph) :: Double
      rate = if duration > 0 then delta' / duration else 0

      -- Health score
      health = (minCoh + stability + rate) / 3.0

      -- Detect issues
      issues = concat
        [ ["Low coherence minimum" | minCoh < 0.3]
        , ["High flux variability" | delta' > 0.5]
        , ["Axis instability" | stability < 0.5]
        , ["Slow recovery" | rate < 0.1 && duration > 10]
        ]

      -- Generate recommendations
      recs = concat
        [ ["Increase grounding time" | minCoh < 0.3]
        , ["Reduce external stimuli" | delta' > 0.5]
        , ["Focus on breath coherence" | stability < 0.5]
        , ["Consider session break" | length issues > 2]
        ]

  in DiagnosticReport
      { drHealthScore = clamp01 health
      , drRecoveryRate = rate
      , drStability = stability
      , drIssues = issues
      , drRecommendations = if null recs then ["Continue current practice"] else recs
      }

-- | Get health score
healthScore :: DiagnosticReport -> Double
healthScore = drHealthScore

-- =============================================================================
-- Internal Types
-- =============================================================================

-- | Scalar field snapshot
data ScalarFieldSnapshot = ScalarFieldSnapshot
  { sfsCoherence :: !Double
  , sfsFlux      :: !Double
  , sfsLocation  :: !RaCoordinate
  , sfsEmergence :: !EmergenceResult
  , sfsInverted  :: !Bool
  , sfsTime      :: !UTCTime
  } deriving (Eq, Show)

-- | Inversion shift
data InversionShift
  = InvToNormal
  | NormalToInv
  | NoShift
  deriving (Eq, Show)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
