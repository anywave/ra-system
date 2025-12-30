{-|
Module      : Ra.PlanetaryGrid.PylonNode
Description : Scalar pylon nodes for planetary grid amplification
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements scalar pylon nodes that serve as amplification and relay points
within the planetary grid network. Pylons can be physical structures or
virtual installations that enhance energy flow and coherence.

== Pylon Theory

=== Scalar Transmission

* Longitudinal wave propagation through pylons
* Standing wave formation between pylon pairs
* Resonant frequency locking
* Phase-conjugate beam generation

=== Network Topology

1. Hub pylons - Major distribution centers
2. Relay pylons - Signal boosting nodes
3. Terminal pylons - End-user connection points
4. Shadow pylons - Inactive/dormant installations
-}
module Ra.PlanetaryGrid.PylonNode
  ( -- * Core Types
    Pylon(..)
  , PylonType(..)
  , PylonState(..)
  , PylonNetwork(..)

    -- * Pylon Creation
  , createPylon
  , installPylon
  , removePylon

    -- * Network Management
  , createNetwork
  , addToNetwork
  , connectPylons
  , disconnectPylons

    -- * Pylon Operations
  , activatePylon
  , deactivatePylon
  , tunePylon
  , setPylonPower

    -- * Scalar Field Functions
  , ScalarBeam(..)
  , generateBeam
  , beamStrength
  , standingWavePattern

    -- * Network Analysis
  , networkCoverage
  , signalPathQuality
  , findOptimalRoute

    -- * Synchronization
  , syncPylonPair
  , networkSync
  , phaseAlignment
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Single pylon node
data Pylon = Pylon
  { pylonId        :: !String              -- ^ Unique identifier
  , pylonName      :: !String              -- ^ Human-readable name
  , pylonCoord     :: !(Double, Double, Double)  -- ^ (lat, lon, alt)
  , pylonType      :: !PylonType           -- ^ Pylon classification
  , pylonState     :: !PylonState          -- ^ Current operational state
  , pylonPower     :: !Double              -- ^ Power level [0, 1]
  , pylonFrequency :: !Double              -- ^ Operating frequency (Hz)
  , pylonPhase     :: !Double              -- ^ Phase offset [0, 2pi]
  , pylonRange     :: !Double              -- ^ Effective range (km)
  , pylonConnections :: ![String]          -- ^ Connected pylon IDs
  } deriving (Eq, Show)

-- | Pylon type classification
data PylonType
  = PylonHub          -- ^ Central distribution hub
  | PylonRelay        -- ^ Signal relay node
  | PylonTerminal     -- ^ End-user access point
  | PylonShadow       -- ^ Dormant/inactive
  | PylonMobile       -- ^ Movable installation
  | PylonVirtual      -- ^ Non-physical (consciousness-based)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Pylon operational state
data PylonState
  = StateOffline       -- ^ Not operational
  | StateStartup       -- ^ Powering up
  | StateOnline        -- ^ Fully operational
  | StateDegraded      -- ^ Reduced capacity
  | StateMaintenance   -- ^ Under maintenance
  | StateEmergency     -- ^ Emergency mode
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Network of interconnected pylons
data PylonNetwork = PylonNetwork
  { pnPylons       :: ![Pylon]             -- ^ All pylons in network
  , pnConnections  :: ![(String, String)]  -- ^ Pylon connection pairs
  , pnBaseFrequency :: !Double             -- ^ Network base frequency
  , pnSyncQuality  :: !Double              -- ^ Network synchronization [0, 1]
  , pnCoverage     :: !Double              -- ^ Coverage percentage
  , pnActive       :: !Bool                -- ^ Network active flag
  } deriving (Eq, Show)

-- =============================================================================
-- Pylon Creation
-- =============================================================================

-- | Create new pylon
createPylon :: String -> (Double, Double, Double) -> PylonType -> Pylon
createPylon pid coord ptype = Pylon
  { pylonId = pid
  , pylonName = pid
  , pylonCoord = coord
  , pylonType = ptype
  , pylonState = StateOffline
  , pylonPower = 0
  , pylonFrequency = 7.83
  , pylonPhase = 0
  , pylonRange = typeRange ptype
  , pylonConnections = []
  }

-- | Install pylon in network
installPylon :: PylonNetwork -> Pylon -> PylonNetwork
installPylon network pylon =
  network { pnPylons = pylon : pnPylons network }

-- | Remove pylon from network
removePylon :: PylonNetwork -> String -> PylonNetwork
removePylon network pid =
  let newPylons = filter ((/= pid) . pylonId) (pnPylons network)
      newConns = filter (\(a, b) -> a /= pid && b /= pid) (pnConnections network)
  in network { pnPylons = newPylons, pnConnections = newConns }

-- =============================================================================
-- Network Management
-- =============================================================================

-- | Create empty network
createNetwork :: Double -> PylonNetwork
createNetwork baseFreq = PylonNetwork
  { pnPylons = []
  , pnConnections = []
  , pnBaseFrequency = baseFreq
  , pnSyncQuality = 0
  , pnCoverage = 0
  , pnActive = False
  }

-- | Add pylon to network with auto-connect
addToNetwork :: PylonNetwork -> Pylon -> PylonNetwork
addToNetwork network pylon =
  let installed = installPylon network pylon
      -- Auto-connect to nearby pylons
      nearbyIds = findNearbyPylons installed pylon (pylonRange pylon)
      withConnections = foldr (connectPylons (pylonId pylon)) installed nearbyIds
  in updateNetworkMetrics withConnections

-- | Connect two pylons
connectPylons :: String -> String -> PylonNetwork -> PylonNetwork
connectPylons pid1 pid2 network =
  let conn = if pid1 < pid2 then (pid1, pid2) else (pid2, pid1)
      existing = (pid1, pid2) `elem` pnConnections network ||
                 (pid2, pid1) `elem` pnConnections network
  in if existing then network
     else let newConns = conn : pnConnections network
              -- Update pylon connection lists
              updatedPylons = map (updatePylonConns pid1 pid2) (pnPylons network)
          in network { pnConnections = newConns, pnPylons = updatedPylons }

-- | Disconnect two pylons
disconnectPylons :: String -> String -> PylonNetwork -> PylonNetwork
disconnectPylons pid1 pid2 network =
  let newConns = filter (\(a, b) -> not ((a == pid1 && b == pid2) ||
                                          (a == pid2 && b == pid1)))
                        (pnConnections network)
      updatedPylons = map (removePylonConn pid1 pid2) (pnPylons network)
  in network { pnConnections = newConns, pnPylons = updatedPylons }

-- =============================================================================
-- Pylon Operations
-- =============================================================================

-- | Activate pylon
activatePylon :: PylonNetwork -> String -> PylonNetwork
activatePylon network pid =
  updatePylon network pid (\p -> p { pylonState = StateStartup, pylonPower = 0.5 })

-- | Deactivate pylon
deactivatePylon :: PylonNetwork -> String -> PylonNetwork
deactivatePylon network pid =
  updatePylon network pid (\p -> p { pylonState = StateOffline, pylonPower = 0 })

-- | Tune pylon to frequency
tunePylon :: PylonNetwork -> String -> Double -> PylonNetwork
tunePylon network pid freq =
  updatePylon network pid (\p -> p { pylonFrequency = freq })

-- | Set pylon power level
setPylonPower :: PylonNetwork -> String -> Double -> PylonNetwork
setPylonPower network pid power =
  let clampedPower = max 0 (min 1 power)
      newState = if clampedPower > 0.8 then StateOnline
                 else if clampedPower > 0.3 then StateDegraded
                 else StateOffline
  in updatePylon network pid (\p -> p { pylonPower = clampedPower, pylonState = newState })

-- =============================================================================
-- Scalar Field Functions
-- =============================================================================

-- | Scalar beam between pylons
data ScalarBeam = ScalarBeam
  { sbSource       :: !String          -- ^ Source pylon ID
  , sbTarget       :: !String          -- ^ Target pylon ID
  , sbFrequency    :: !Double          -- ^ Beam frequency
  , sbPower        :: !Double          -- ^ Beam power
  , sbPhase        :: !Double          -- ^ Phase relationship
  , sbCoherence    :: !Double          -- ^ Beam coherence [0, 1]
  } deriving (Eq, Show)

-- | Generate scalar beam between two pylons
generateBeam :: PylonNetwork -> String -> String -> Maybe ScalarBeam
generateBeam network src tgt =
  case (findPylon network src, findPylon network tgt) of
    (Just p1, Just p2) ->
      if pylonState p1 == StateOnline && pylonState p2 == StateOnline
      then Just ScalarBeam
        { sbSource = src
        , sbTarget = tgt
        , sbFrequency = (pylonFrequency p1 + pylonFrequency p2) / 2
        , sbPower = pylonPower p1 * pylonPower p2
        , sbPhase = abs (pylonPhase p1 - pylonPhase p2)
        , sbCoherence = calculateBeamCoherence p1 p2
        }
      else Nothing
    _ -> Nothing

-- | Calculate beam strength at distance
beamStrength :: ScalarBeam -> Double -> Double -> Double
beamStrength beam totalDist pointDist =
  let normalizedPos = pointDist / totalDist
      -- Standing wave pattern
      wavePattern = cos (normalizedPos * 2 * pi * phi)^(2::Int)
      powerFalloff = sbPower beam * (1 - abs (normalizedPos - 0.5) * 0.5)
  in powerFalloff * wavePattern * sbCoherence beam

-- | Generate standing wave pattern along beam path
standingWavePattern :: ScalarBeam -> Int -> [Double]
standingWavePattern beam samples =
  [ beamStrength beam 1.0 (fromIntegral i / fromIntegral (samples - 1))
  | i <- [0..samples-1]
  ]

-- =============================================================================
-- Network Analysis
-- =============================================================================

-- | Calculate network coverage
networkCoverage :: PylonNetwork -> Double
networkCoverage network =
  let activePylons = filter ((== StateOnline) . pylonState) (pnPylons network)
      totalRange = sum [pylonRange p * pylonPower p | p <- activePylons]
      -- Simplified coverage model (actual would use proper spatial analysis)
      maxCoverage = fromIntegral (length (pnPylons network)) * 500
  in if maxCoverage > 0 then min 1.0 (totalRange / maxCoverage) else 0

-- | Calculate signal path quality between two pylons
signalPathQuality :: PylonNetwork -> String -> String -> Double
signalPathQuality network src tgt =
  case findOptimalRoute network src tgt of
    Nothing -> 0
    Just path ->
      let hops = length path - 1
          hopPenalty = phiInverse^hops
          pylonQuality = product [pylonPower p | pid <- path, Just p <- [findPylon network pid]]
      in hopPenalty * pylonQuality

-- | Find optimal route between pylons
findOptimalRoute :: PylonNetwork -> String -> String -> Maybe [String]
findOptimalRoute network src tgt
  | src == tgt = Just [src]
  | otherwise = bfsRoute network src tgt

-- =============================================================================
-- Synchronization
-- =============================================================================

-- | Synchronize pylon pair
syncPylonPair :: PylonNetwork -> String -> String -> PylonNetwork
syncPylonPair network pid1 pid2 =
  case (findPylon network pid1, findPylon network pid2) of
    (Just p1, Just p2) ->
      let avgFreq = (pylonFrequency p1 + pylonFrequency p2) / 2
          avgPhase = (pylonPhase p1 + pylonPhase p2) / 2
          net1 = updatePylon network pid1 (\p -> p { pylonFrequency = avgFreq, pylonPhase = avgPhase })
      in updatePylon net1 pid2 (\p -> p { pylonFrequency = avgFreq, pylonPhase = avgPhase + pi })
    _ -> network

-- | Synchronize entire network
networkSync :: PylonNetwork -> PylonNetwork
networkSync network =
  let baseFreq = pnBaseFrequency network
      synced = map (\p -> p { pylonFrequency = baseFreq }) (pnPylons network)
      newQuality = calculateSyncQuality synced
  in network { pnPylons = synced, pnSyncQuality = newQuality }

-- | Calculate phase alignment between pylons
phaseAlignment :: Pylon -> Pylon -> Double
phaseAlignment p1 p2 =
  let phaseDiff = abs (pylonPhase p1 - pylonPhase p2)
      normalized = if phaseDiff > pi then 2 * pi - phaseDiff else phaseDiff
  in 1 - normalized / pi

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Get range for pylon type
typeRange :: PylonType -> Double
typeRange PylonHub = 1000      -- km
typeRange PylonRelay = 500
typeRange PylonTerminal = 100
typeRange PylonShadow = 0
typeRange PylonMobile = 200
typeRange PylonVirtual = 50

-- | Find pylon by ID
findPylon :: PylonNetwork -> String -> Maybe Pylon
findPylon network pid =
  case filter ((== pid) . pylonId) (pnPylons network) of
    (p:_) -> Just p
    [] -> Nothing

-- | Find nearby pylons within range
findNearbyPylons :: PylonNetwork -> Pylon -> Double -> [String]
findNearbyPylons network pylon range' =
  [ pylonId p
  | p <- pnPylons network
  , pylonId p /= pylonId pylon
  , pylonDistance pylon p <= range'
  ]

-- | Calculate distance between pylons
pylonDistance :: Pylon -> Pylon -> Double
pylonDistance p1 p2 =
  let (lat1, lon1, _) = pylonCoord p1
      (lat2, lon2, _) = pylonCoord p2
      dlat = (lat2 - lat1) * pi / 180
      dlon = (lon2 - lon1) * pi / 180
      lat1r = lat1 * pi / 180
      lat2r = lat2 * pi / 180
      a = sin (dlat/2)^(2::Int) + cos lat1r * cos lat2r * sin (dlon/2)^(2::Int)
      c = 2 * atan2 (sqrt a) (sqrt (1-a))
  in 6371 * c

-- | Update pylon in network
updatePylon :: PylonNetwork -> String -> (Pylon -> Pylon) -> PylonNetwork
updatePylon network pid f =
  let updated = map (\p -> if pylonId p == pid then f p else p) (pnPylons network)
  in network { pnPylons = updated }

-- | Update pylon connection lists
updatePylonConns :: String -> String -> Pylon -> Pylon
updatePylonConns pid1 pid2 pylon
  | pylonId pylon == pid1 = pylon { pylonConnections = pid2 : pylonConnections pylon }
  | pylonId pylon == pid2 = pylon { pylonConnections = pid1 : pylonConnections pylon }
  | otherwise = pylon

-- | Remove pylon connection
removePylonConn :: String -> String -> Pylon -> Pylon
removePylonConn pid1 pid2 pylon
  | pylonId pylon == pid1 = pylon { pylonConnections = filter (/= pid2) (pylonConnections pylon) }
  | pylonId pylon == pid2 = pylon { pylonConnections = filter (/= pid1) (pylonConnections pylon) }
  | otherwise = pylon

-- | Update network metrics
updateNetworkMetrics :: PylonNetwork -> PylonNetwork
updateNetworkMetrics network =
  let cov = networkCoverage network
      sync = calculateSyncQuality (pnPylons network)
  in network { pnCoverage = cov, pnSyncQuality = sync }

-- | Calculate beam coherence
calculateBeamCoherence :: Pylon -> Pylon -> Double
calculateBeamCoherence p1 p2 =
  let freqMatch = 1 - abs (pylonFrequency p1 - pylonFrequency p2) / max 1 (max (pylonFrequency p1) (pylonFrequency p2))
      phase = phaseAlignment p1 p2
  in freqMatch * phase * phi / 2

-- | Calculate network sync quality
calculateSyncQuality :: [Pylon] -> Double
calculateSyncQuality [] = 0
calculateSyncQuality [_] = 1
calculateSyncQuality pylons =
  let freqs = map pylonFrequency pylons
      avgFreq = sum freqs / fromIntegral (length freqs)
      variance = sum [(f - avgFreq)^(2::Int) | f <- freqs] / fromIntegral (length freqs)
  in max 0 (1 - variance / 100)

-- | BFS route finding
bfsRoute :: PylonNetwork -> String -> String -> Maybe [String]
bfsRoute network src tgt = go [[src]] []
  where
    go [] _ = Nothing
    go (path:queue) visited
      | head path == tgt = Just (reverse path)
      | head path `elem` visited = go queue visited
      | otherwise =
          let current = head path
              neighbors = case findPylon network current of
                Just p -> pylonConnections p
                Nothing -> []
              newPaths = [n : path | n <- neighbors, n `notElem` visited]
          in go (queue ++ newPaths) (current : visited)
