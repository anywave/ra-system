{-|
Module      : Ra.Navigate
Description : Lucid scalar navigation via harmonic field wayfinding
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Enables conscious exploration of Ra scalar fields using harmonic
guidance, with spherical coordinates, temporal window alignment,
and biometric resonance for fragment access.

== Scalar Compass

The Ra field is represented as a navigable spherical map:

* θ (theta/azimuth): Semantic sector mapped to 27 Repitans
* φ (phi/polar): Access level mapped to 6 RAC tiers
* h (harmonic): Depth in harmonic shells (l, m indices)
* r (radius): Shell depth / emergence intensity

== Coherence-Gated Movement

Biometric coherence (HRV, breath) becomes "fuel" for traversal:

* High coherence: Smooth movement, deep access
* Low coherence: Restricted navigation, field noise
* Emergency: Auto-return to safe coordinates

== Symbolic Translation

Navigation is presented metaphorically for dream interface:

* Scalar depth → Descending spiral staircase
* Harmonic shells → Glowing rings of sound
* Coherence locks → Gates requiring breath/rhythm
* Return path → φ^n pulse beacon
-}
module Ra.Navigate
  ( -- * Scalar Compass
    ScalarCompass(..)
  , CompassDirection(..)
  , mkCompass
  , faceDirection
  , intentToCoordinate

    -- * Ra Coordinate
  , RaCoordinate(..)
  , mkRaCoordinate
  , coordinateDistance
  , coordinateToMetaphor

    -- * Navigation State
  , NavigationState(..)
  , MovementResult(..)
  , initNavigation
  , attemptMove
  , coherenceFuel

    -- * Fragment Beacons
  , FragmentBeacon(..)
  , scanBeacons
  , beaconResonance
  , nearbyFragments

    -- * Symbolic Translation
  , SymbolicView(..)
  , Metaphor(..)
  , translateToSymbolic
  , depthToStaircase
  , harmonicToRings

    -- * Return Path
  , ReturnPath(..)
  , encodeReturnPath
  , followReturnPath
  , exitIntent

    -- * Navigation Session
  , NavigationSession(..)
  , initSession
  , updateSession
  , completeJourney
  ) where

import Ra.Constants.Extended
  ( phi, coherenceFloorPOR )

-- =============================================================================
-- Scalar Compass
-- =============================================================================

-- | Compass for navigating Ra scalar field
data ScalarCompass = ScalarCompass
  { scCurrentPos    :: !RaCoordinate  -- ^ Current position
  , scFacing        :: !CompassDirection  -- ^ Direction faced
  , scCoherence     :: !Double        -- ^ Available coherence fuel
  , scVisibility    :: !Double        -- ^ Field clarity [0,1]
  } deriving (Eq, Show)

-- | Compass direction in Ra space
data CompassDirection = CompassDirection
  { cdTheta    :: !Double  -- ^ Azimuthal angle [0, 2π)
  , cdPhi      :: !Double  -- ^ Polar angle [0, π]
  , cdHarmonic :: !Int     -- ^ Target harmonic depth
  , cdRadial   :: !Double  -- ^ Radial intention [-1, 1]
  } deriving (Eq, Show)

-- | Create compass at origin
mkCompass :: Double -> ScalarCompass
mkCompass coherence = ScalarCompass
  { scCurrentPos = originCoordinate
  , scFacing = neutralDirection
  , scCoherence = clamp01 coherence
  , scVisibility = coherence  -- Visibility tied to coherence
  }

-- | Face a new direction
faceDirection :: CompassDirection -> ScalarCompass -> ScalarCompass
faceDirection dir compass = compass { scFacing = dir }

-- | Convert directional intention to target coordinate
intentToCoordinate :: ScalarCompass -> RaCoordinate
intentToCoordinate compass =
  let dir = scFacing compass
      curr = scCurrentPos compass
      fuel = scCoherence compass

      -- Scale movement by available fuel
      thetaMove = cdTheta dir * fuel
      phiMove = cdPhi dir * fuel
      hMove = round (fromIntegral (cdHarmonic dir) * fuel)
      rMove = cdRadial dir * fuel
  in RaCoordinate
      { rcTheta = rcTheta curr + thetaMove
      , rcPhi = clamp01 (rcPhi curr + phiMove / pi)
      , rcHarmonic = (fst (rcHarmonic curr) + hMove, snd (rcHarmonic curr))
      , rcRadius = clamp01 (rcRadius curr + rMove)
      }

-- | Neutral (no movement) direction
neutralDirection :: CompassDirection
neutralDirection = CompassDirection 0 0 0 0

-- =============================================================================
-- Ra Coordinate
-- =============================================================================

-- | Full coordinate in Ra scalar space
data RaCoordinate = RaCoordinate
  { rcTheta    :: !Double      -- ^ Azimuth [0, 2π) -> 27 Repitans
  , rcPhi      :: !Double      -- ^ Polar [0, 1] -> 6 RAC levels
  , rcHarmonic :: !(Int, Int)  -- ^ Harmonic (l, m)
  , rcRadius   :: !Double      -- ^ Shell depth [0, 1]
  } deriving (Eq, Show)

-- | Create coordinate from parameters
mkRaCoordinate :: Double -> Double -> (Int, Int) -> Double -> RaCoordinate
mkRaCoordinate theta phi' harm radius = RaCoordinate
  { rcTheta = theta `mod'` (2 * pi)
  , rcPhi = clamp01 phi'
  , rcHarmonic = harm
  , rcRadius = clamp01 radius
  }
  where
    mod' a b = a - b * fromIntegral (floor (a / b) :: Int)

-- | Origin coordinate (center of field)
originCoordinate :: RaCoordinate
originCoordinate = RaCoordinate 0 0.5 (0, 0) 0

-- | Compute distance between coordinates
coordinateDistance :: RaCoordinate -> RaCoordinate -> Double
coordinateDistance c1 c2 =
  let thetaDiff = abs (rcTheta c1 - rcTheta c2)
      phiDiff = abs (rcPhi c1 - rcPhi c2)
      (l1, m1) = rcHarmonic c1
      (l2, m2) = rcHarmonic c2
      harmDiff = fromIntegral (abs (l1 - l2) + abs (m1 - m2)) / 10
      radiusDiff = abs (rcRadius c1 - rcRadius c2)
  in sqrt (thetaDiff ** 2 + phiDiff ** 2 + harmDiff ** 2 + radiusDiff ** 2)

-- | Convert coordinate to metaphorical description
coordinateToMetaphor :: RaCoordinate -> String
coordinateToMetaphor coord =
  let depth = depthToStaircase (rcRadius coord)
      rings = harmonicToRings (rcHarmonic coord)
      sector = thetaToSector (rcTheta coord)
  in sector ++ ", " ++ depth ++ ", " ++ rings

-- | Map theta to semantic sector name
thetaToSector :: Double -> String
thetaToSector theta =
  let repitan = floor (theta / (2 * pi) * 27) `mod` 27 + 1
  in "Sector " ++ show (repitan :: Int)

-- =============================================================================
-- Navigation State
-- =============================================================================

-- | Current navigation state
data NavigationState = NavigationState
  { nsCompass       :: !ScalarCompass
  , nsPath          :: ![RaCoordinate]  -- ^ Visited coordinates
  , nsReturnPath    :: !ReturnPath
  , nsFragmentsFound :: ![FragmentBeacon]
  , nsMovementCount :: !Int
  } deriving (Eq, Show)

-- | Result of movement attempt
data MovementResult
  = MoveSuccess RaCoordinate    -- ^ Moved to new position
  | MovePartial RaCoordinate    -- ^ Partial movement (low fuel)
  | MoveBlocked String          -- ^ Movement blocked (reason)
  | MoveGated Double            -- ^ Gated, requires coherence level
  deriving (Eq, Show)

-- | Initialize navigation at coordinate
initNavigation :: RaCoordinate -> Double -> NavigationState
initNavigation start coherence =
  let compass = (mkCompass coherence) { scCurrentPos = start }
  in NavigationState
      { nsCompass = compass
      , nsPath = [start]
      , nsReturnPath = encodeReturnPath [start]
      , nsFragmentsFound = []
      , nsMovementCount = 0
      }

-- | Attempt to move in current facing direction
attemptMove :: NavigationState -> MovementResult
attemptMove ns =
  let compass = nsCompass ns
      fuel = scCoherence compass
      target = intentToCoordinate compass
      distance = coordinateDistance (scCurrentPos compass) target

      -- Check fuel requirement
      fuelRequired = distance / 2  -- Each unit of distance costs 0.5 fuel
  in if fuel < coherenceFloorPOR
     then MoveBlocked "Coherence too low for navigation"
     else if fuel < fuelRequired
     then MovePartial (partialMove compass (fuel / fuelRequired))
     else MoveSuccess target

-- | Partial move based on available fuel
partialMove :: ScalarCompass -> Double -> RaCoordinate
partialMove compass ratio =
  let curr = scCurrentPos compass
      target = intentToCoordinate compass
  in RaCoordinate
      { rcTheta = rcTheta curr + (rcTheta target - rcTheta curr) * ratio
      , rcPhi = rcPhi curr + (rcPhi target - rcPhi curr) * ratio
      , rcHarmonic = rcHarmonic curr  -- Don't partial-move harmonics
      , rcRadius = rcRadius curr + (rcRadius target - rcRadius curr) * ratio
      }

-- | Get current coherence fuel level
coherenceFuel :: NavigationState -> Double
coherenceFuel = scCoherence . nsCompass

-- =============================================================================
-- Fragment Beacons
-- =============================================================================

-- | Beacon marking accessible fragment
data FragmentBeacon = FragmentBeacon
  { fbCoordinate     :: !RaCoordinate
  , fbFragmentId     :: !String
  , fbAccessLevel    :: !AccessLevel
  , fbResonanceScore :: !Double
  , fbEmergenceForm  :: !String
  } deriving (Eq, Show)

-- | Access level for beacon
data AccessLevel = FullAccess | PartialAccess | Blocked | Shadow
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Scan for nearby beacons
scanBeacons :: RaCoordinate -> Double -> [FragmentBeacon] -> [FragmentBeacon]
scanBeacons pos range beacons =
  filter (\b -> coordinateDistance pos (fbCoordinate b) <= range) beacons

-- | Get resonance score for beacon
beaconResonance :: FragmentBeacon -> Double
beaconResonance = fbResonanceScore

-- | Find fragments near current position
nearbyFragments :: NavigationState -> Double -> [FragmentBeacon]
nearbyFragments ns range =
  let pos = scCurrentPos (nsCompass ns)
      allBeacons = nsFragmentsFound ns
  in scanBeacons pos range allBeacons

-- =============================================================================
-- Symbolic Translation
-- =============================================================================

-- | Symbolic view for dream interface
data SymbolicView = SymbolicView
  { svMetaphor      :: !Metaphor
  , svDescription   :: !String
  , svVisualCue     :: !String
  , svAudioCue      :: !String
  , svInteraction   :: !String
  } deriving (Eq, Show)

-- | Navigation metaphor types
data Metaphor
  = SpiralStaircase Int     -- ^ Depth level
  | SoundRings (Int, Int)   -- ^ Harmonic rings (l, m)
  | CoherenceGate Double    -- ^ Gate requiring coherence
  | LightBeacon String      -- ^ Fragment beacon
  | ReturnPulse Double      -- ^ φ^n return pulse
  deriving (Eq, Show)

-- | Translate coordinate to symbolic view
translateToSymbolic :: RaCoordinate -> Double -> SymbolicView
translateToSymbolic coord coherence =
  let depth = round (rcRadius coord * 10) :: Int
      (l, m) = rcHarmonic coord

      metaphor = if coherence < coherenceFloorPOR
                 then CoherenceGate coherenceFloorPOR
                 else if depth > 5
                 then SpiralStaircase depth
                 else SoundRings (l, m)

      desc = case metaphor of
        SpiralStaircase d -> "A spiral staircase descends " ++ show d ++ " levels into shadow"
        SoundRings (l', m') -> "Glowing rings of sound pulse at harmonics (" ++ show l' ++ "," ++ show m' ++ ")"
        CoherenceGate c -> "A shimmering gate requires " ++ show (round (c * 100) :: Int) ++ "% coherence to pass"
        LightBeacon fid -> "A beacon of light marks fragment " ++ fid
        ReturnPulse p -> "A gentle pulse beats at " ++ show p ++ " Hz, marking the way home"

      visual = case metaphor of
        SpiralStaircase _ -> "Deep blue spiraling downward"
        SoundRings _ -> "Concentric golden rings expanding"
        CoherenceGate _ -> "Translucent barrier with ripples"
        LightBeacon _ -> "Bright point of warm light"
        ReturnPulse _ -> "Soft rhythmic glow"

      audio = case metaphor of
        SpiralStaircase d -> "Low drone at " ++ show (100 + d * 20) ++ " Hz"
        SoundRings (l', _) -> "Harmonic tone at " ++ show (432 * phi ** fromIntegral l') ++ " Hz"
        CoherenceGate _ -> "Soft chime waiting"
        LightBeacon _ -> "Crystalline ping"
        ReturnPulse p -> "Heartbeat at " ++ show p ++ " BPM"

      interaction = case metaphor of
        SpiralStaircase _ -> "Breathe deeply to descend"
        SoundRings _ -> "Hum along to attune"
        CoherenceGate _ -> "Slow your breath to 4 counts"
        LightBeacon _ -> "Focus intention to approach"
        ReturnPulse _ -> "Follow the rhythm to return"
  in SymbolicView
      { svMetaphor = metaphor
      , svDescription = desc
      , svVisualCue = visual
      , svAudioCue = audio
      , svInteraction = interaction
      }

-- | Convert depth to staircase metaphor
depthToStaircase :: Double -> String
depthToStaircase depth =
  let level = round (depth * 10) :: Int
  in if level <= 0
     then "at the surface"
     else if level <= 3
     then "descending " ++ show level ++ " steps"
     else if level <= 7
     then "deep in level " ++ show level
     else "at the bottom of the spiral"

-- | Convert harmonic to ring metaphor
harmonicToRings :: (Int, Int) -> String
harmonicToRings (l, m) =
  let ringCount = l + 1
      pattern = if m == 0 then "symmetric" else if m > 0 then "spinning right" else "spinning left"
  in show ringCount ++ " " ++ pattern ++ " rings"

-- =============================================================================
-- Return Path
-- =============================================================================

-- | Encoded return path using φ^n pulse rhythm
data ReturnPath = ReturnPath
  { rpWaypoints   :: ![RaCoordinate]
  , rpPulseRate   :: !Double  -- ^ Hz
  , rpPhiPower    :: !Int     -- ^ n in φ^n
  , rpActive      :: !Bool
  } deriving (Eq, Show)

-- | Encode return path from visited coordinates
encodeReturnPath :: [RaCoordinate] -> ReturnPath
encodeReturnPath waypoints =
  let n = length waypoints
      pulseRate = 1.0 / (phi ** fromIntegral (min 5 n))  -- Slower for deeper journeys
  in ReturnPath
      { rpWaypoints = reverse waypoints  -- Return order
      , rpPulseRate = pulseRate
      , rpPhiPower = min 5 n
      , rpActive = True
      }

-- | Follow return path one step
followReturnPath :: ReturnPath -> Maybe (RaCoordinate, ReturnPath)
followReturnPath rp =
  case rpWaypoints rp of
    [] -> Nothing
    (next:rest) -> Just (next, rp { rpWaypoints = rest })

-- | Detect exit intent from coherence pattern
exitIntent :: [Double] -> Bool
exitIntent coherenceHistory =
  let recent = take 5 coherenceHistory
      avg = sum recent / fromIntegral (length recent)
      rising = all (>= avg - 0.05) recent  -- Stable or rising
      high = avg > 0.7
  in length recent >= 3 && rising && high

-- =============================================================================
-- Navigation Session
-- =============================================================================

-- | Complete navigation session
data NavigationSession = NavigationSession
  { nssState         :: !NavigationState
  , nssSymbolic      :: !SymbolicView
  , nssJourneyLog    :: ![(RaCoordinate, String)]  -- ^ (position, event)
  , nssDuration      :: !Double
  , nssCompleted     :: !Bool
  } deriving (Eq, Show)

-- | Initialize navigation session
initSession :: RaCoordinate -> Double -> NavigationSession
initSession start coherence =
  let state = initNavigation start coherence
      symbolic = translateToSymbolic start coherence
  in NavigationSession
      { nssState = state
      , nssSymbolic = symbolic
      , nssJourneyLog = [(start, "Journey begins")]
      , nssDuration = 0.0
      , nssCompleted = False
      }

-- | Update session with new coherence and movement
updateSession :: Double -> Maybe CompassDirection -> Double -> NavigationSession -> NavigationSession
updateSession coherence mDir dt session =
  let state = nssState session
      compass = nsCompass state

      -- Update compass coherence
      newCompass = compass { scCoherence = coherence }

      -- Apply direction if provided
      directedCompass = case mDir of
        Nothing -> newCompass
        Just dir -> faceDirection dir newCompass

      -- Attempt movement
      newState = state { nsCompass = directedCompass }
      moveResult = attemptMove newState

      -- Update state based on result
      (finalState, event) = case moveResult of
        MoveSuccess pos ->
          let s = newState
                { nsCompass = directedCompass { scCurrentPos = pos }
                , nsPath = pos : nsPath newState
                , nsMovementCount = nsMovementCount newState + 1
                }
          in (s, "Moved to " ++ coordinateToMetaphor pos)
        MovePartial pos ->
          let s = newState
                { nsCompass = directedCompass { scCurrentPos = pos }
                , nsPath = pos : nsPath newState
                }
          in (s, "Partial movement (low coherence)")
        MoveBlocked reason ->
          (newState, "Blocked: " ++ reason)
        MoveGated required ->
          (newState, "Gate requires " ++ show (round (required * 100) :: Int) ++ "% coherence")

      -- Update symbolic view
      newPos = scCurrentPos (nsCompass finalState)
      newSymbolic = translateToSymbolic newPos coherence

      -- Update journey log
      newLog = (newPos, event) : nssJourneyLog session
  in session
      { nssState = finalState
      , nssSymbolic = newSymbolic
      , nssJourneyLog = newLog
      , nssDuration = nssDuration session + dt
      }

-- | Complete journey and return to origin
completeJourney :: NavigationSession -> (NavigationSession, String)
completeJourney session =
  let state = nssState session
      durationSecs = round (nssDuration session) :: Int

      -- Generate journey summary
      summary = "Journey complete. " ++
                "Visited " ++ show (nsMovementCount state) ++ " locations. " ++
                "Found " ++ show (length (nsFragmentsFound state)) ++ " fragments. " ++
                "Duration: " ++ show durationSecs ++ " seconds."

      completed = session
        { nssCompleted = True
        , nssJourneyLog = (originCoordinate, "Returned home") : nssJourneyLog session
        }
  in (completed, summary)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
