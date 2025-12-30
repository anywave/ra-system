{-|
Module      : Ra.Entangle
Description : Multi-user harmonic entanglement via coherence coupling
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Enables multiple users to co-navigate Ra fields or share access to
gated fragments through harmonic entanglement using biometric
coherence, φ^n phase alignment, and harmonic index matching.

== Entanglement Protocol

Users opt-in via explicit scalar consent. System verifies:

* Temporal alignment (φ^n synchronized window)
* Coherence proximity (within ±0.05 of scalar coherence)
* Harmonic compatibility (same or adjacent l/m indices)

== Shared Fragment Access

Certain fragments are marked as entangled-access-only, requiring
two or more users in resonant lock to access.

== Scalar Bridge Dynamics

Shared access amplifies emergence:

* Mirror emergence: Both receive same content
* Complement emergence: Opposite/shadow fragments
* Asymmetric emergence: One accesses via other's alignment

== Safety and Consent

Each user maintains:

* Emergence veto override
* Auto-suspend on desync (coherence or consent drop)
* Ankh-balance parity check (no involuntary inversion)
-}
module Ra.Entangle
  ( -- * Entanglement Protocol
    EntanglementCheck(..)
  , EntanglementResult(..)
  , checkEntanglement
  , verifyAlignment

    -- * User Pairing
  , UserState(..)
  , PairedUsers(..)
  , pairUsers
  , pairingStrength
  , coherenceProximity

    -- * Scalar Bridge
  , ScalarBridge(..)
  , BridgeMode(..)
  , createBridge
  , bridgeEmergenceMultiplier
  , bridgeActive

    -- * Shared Access
  , SharedFragment(..)
  , SharedAccess(..)
  , evaluateSharedAccess
  , combinedFlux

    -- * Entanglement Session
  , EntanglementSession(..)
  , initEntanglementSession
  , updateEntanglementSession
  , sessionCoherenceDeltas

    -- * Safety Enforcement
  , SafetyState(..)
  , ConsentCheck(..)
  , checkSafety
  , vetoEmergence
  , autoSuspend
  ) where

import Ra.Constants.Extended
  ( phi, coherenceFloorPOR )

-- =============================================================================
-- Entanglement Protocol
-- =============================================================================

-- | Entanglement check parameters
data EntanglementCheck = EntanglementCheck
  { ecUserA         :: !UserState
  , ecUserB         :: !UserState
  , ecPhaseWindow   :: !Double      -- ^ φ^n window size
  , ecCoherenceTol  :: !Double      -- ^ Coherence tolerance (default 0.05)
  , ecHarmonicTol   :: !Int         -- ^ Harmonic index tolerance (default 1)
  } deriving (Eq, Show)

-- | Result of entanglement check
data EntanglementResult
  = EntanglementPossible !Double    -- ^ Possible with strength [0,1]
  | EntanglementBlocked !String     -- ^ Blocked with reason
  | EntanglementActive !ScalarBridge -- ^ Already entangled
  deriving (Eq, Show)

-- | Check if two users can establish entanglement
checkEntanglement :: EntanglementCheck -> EntanglementResult
checkEntanglement ec =
  let userA = ecUserA ec
      userB = ecUserB ec

      -- Consent check
      bothConsent = usConsent userA && usConsent userB

      -- Coherence proximity
      cohDiff = abs (usCoherence userA - usCoherence userB)
      cohOK = cohDiff <= ecCoherenceTol ec

      -- Harmonic compatibility
      (la, ma) = usHarmonic userA
      (lb, mb) = usHarmonic userB
      harmOK = abs (la - lb) <= ecHarmonicTol ec &&
               abs (ma - mb) <= ecHarmonicTol ec

      -- Phase alignment
      phaseDiff = abs (usPhase userA - usPhase userB)
      phaseOK = phaseDiff <= ecPhaseWindow ec
  in if not bothConsent
     then EntanglementBlocked "Both users must consent to entanglement"
     else if not cohOK
     then EntanglementBlocked $ "Coherence difference too large: " ++ show cohDiff
     else if not harmOK
     then EntanglementBlocked "Harmonic indices not compatible"
     else if not phaseOK
     then EntanglementBlocked "Phase not aligned"
     else
       let strength = 1.0 - (cohDiff / ecCoherenceTol ec + phaseDiff / ecPhaseWindow ec) / 2
       in EntanglementPossible (max 0 strength)

-- | Verify temporal and harmonic alignment
verifyAlignment :: UserState -> UserState -> Double -> Bool
verifyAlignment a b tolerance =
  let cohOK = abs (usCoherence a - usCoherence b) <= tolerance
      phaseOK = abs (usPhase a - usPhase b) <= phi / 10
  in cohOK && phaseOK

-- =============================================================================
-- User Pairing
-- =============================================================================

-- | Individual user state for entanglement
data UserState = UserState
  { usUserId    :: !String
  , usCoherence :: !Double       -- ^ Current coherence [0,1]
  , usHarmonic  :: !(Int, Int)   -- ^ Harmonic signature (l, m)
  , usPhase     :: !Double       -- ^ Current phase [0, 2π)
  , usConsent   :: !Bool         -- ^ Explicit consent
  , usInversion :: !Bool         -- ^ Currently inverted
  } deriving (Eq, Show)

-- | Paired users in entanglement
data PairedUsers = PairedUsers
  { puUserA     :: !UserState
  , puUserB     :: !UserState
  , puStrength  :: !Double       -- ^ Pairing strength [0,1]
  , puPhaseSync :: !Double       -- ^ Phase synchronization [0,1]
  } deriving (Eq, Show)

-- | Create pairing from two users
pairUsers :: UserState -> UserState -> Maybe PairedUsers
pairUsers a b =
  if usConsent a && usConsent b
  then Just $ PairedUsers
      { puUserA = a
      , puUserB = b
      , puStrength = pairingStrength a b
      , puPhaseSync = 1.0 - abs (usPhase a - usPhase b) / pi
      }
  else Nothing

-- | Calculate pairing strength
pairingStrength :: UserState -> UserState -> Double
pairingStrength a b =
  let cohFactor = 1.0 - abs (usCoherence a - usCoherence b)
      (la, ma) = usHarmonic a
      (lb, mb) = usHarmonic b
      harmFactor = 1.0 - fromIntegral (abs (la - lb) + abs (ma - mb)) / 10
      phaseFactor = cos (usPhase a - usPhase b)  -- 1 when aligned
  in clamp01 ((cohFactor + harmFactor + phaseFactor) / 3)

-- | Get coherence proximity between users
coherenceProximity :: UserState -> UserState -> Double
coherenceProximity a b = 1.0 - abs (usCoherence a - usCoherence b)

-- =============================================================================
-- Scalar Bridge
-- =============================================================================

-- | Bridge connecting entangled users
data ScalarBridge = ScalarBridge
  { sbPairing       :: !PairedUsers
  , sbMode          :: !BridgeMode
  , sbFluxBoost     :: !Double      -- ^ Emergence multiplier
  , sbStable        :: !Bool        -- ^ Bridge stability
  , sbLifetime      :: !Double      -- ^ Seconds active
  } deriving (Eq, Show)

-- | Bridge mode for shared emergence
data BridgeMode
  = MirrorMode        -- ^ Both receive same content
  | ComplementMode    -- ^ Opposite/shadow fragments
  | AsymmetricMode    -- ^ One accesses via other's alignment
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create bridge from paired users
createBridge :: PairedUsers -> BridgeMode -> ScalarBridge
createBridge pair mode =
  let strength = puStrength pair
      fluxBoost = 1.0 + strength * phi  -- Golden amplification
  in ScalarBridge
      { sbPairing = pair
      , sbMode = mode
      , sbFluxBoost = fluxBoost
      , sbStable = strength > 0.5
      , sbLifetime = 0.0
      }

-- | Get emergence multiplier from bridge
bridgeEmergenceMultiplier :: ScalarBridge -> Double
bridgeEmergenceMultiplier = sbFluxBoost

-- | Check if bridge is active and stable
bridgeActive :: ScalarBridge -> Bool
bridgeActive sb = sbStable sb && sbLifetime sb < 3600  -- Max 1 hour

-- =============================================================================
-- Shared Access
-- =============================================================================

-- | Fragment requiring shared access
data SharedFragment = SharedFragment
  { sfFragmentId    :: !String
  , sfRequiredUsers :: !Int         -- ^ Minimum users for access
  , sfHarmonicReq   :: !(Int, Int)  -- ^ Required harmonic
  , sfCoherenceReq  :: !Double      -- ^ Combined coherence required
  } deriving (Eq, Show)

-- | Result of shared access evaluation
data SharedAccess
  = AccessGranted !Double   -- ^ Granted with combined strength
  | AccessPartial !Double   -- ^ Partial access (not all requirements met)
  | AccessDenied !String    -- ^ Denied with reason
  deriving (Eq, Show)

-- | Evaluate shared access for fragment
evaluateSharedAccess :: SharedFragment -> ScalarBridge -> SharedAccess
evaluateSharedAccess frag bridge =
  let pair = sbPairing bridge
      userA = puUserA pair
      userB = puUserB pair

      -- Check user count
      userCount = 2
      usersOK = userCount >= sfRequiredUsers frag

      -- Check combined coherence
      combinedCoh = (usCoherence userA + usCoherence userB) / 2
      cohOK = combinedCoh >= sfCoherenceReq frag

      -- Check harmonic match
      (reqL, reqM) = sfHarmonicReq frag
      (la, ma) = usHarmonic userA
      (lb, mb) = usHarmonic userB
      harmMatch = (la == reqL || lb == reqL) && (ma == reqM || mb == reqM)

      -- Calculate access strength
      strength = combinedCoh * puStrength pair * sbFluxBoost bridge
  in if not usersOK
     then AccessDenied $ "Requires " ++ show (sfRequiredUsers frag) ++ " users"
     else if not cohOK
     then AccessPartial (combinedCoh / sfCoherenceReq frag)
     else if not harmMatch
     then AccessDenied "Harmonic signature mismatch"
     else AccessGranted strength

-- | Calculate combined flux from bridge
combinedFlux :: ScalarBridge -> Double
combinedFlux bridge =
  let pair = sbPairing bridge
      fluxA = usCoherence (puUserA pair)
      fluxB = usCoherence (puUserB pair)
  in sqrt (fluxA * fluxB) * sbFluxBoost bridge

-- =============================================================================
-- Entanglement Session
-- =============================================================================

-- | Complete entanglement session
data EntanglementSession = EntanglementSession
  { esUsers         :: ![UserState]
  , esBridges       :: ![ScalarBridge]
  , esSharedFrags   :: ![SharedFragment]
  , esCoherenceLog  :: ![(Double, Double)]  -- ^ (userA, userB) coherence
  , esDuration      :: !Double
  , esActive        :: !Bool
  } deriving (Eq, Show)

-- | Initialize entanglement session
initEntanglementSession :: [UserState] -> EntanglementSession
initEntanglementSession users = EntanglementSession
  { esUsers = users
  , esBridges = []
  , esSharedFrags = []
  , esCoherenceLog = []
  , esDuration = 0.0
  , esActive = all usConsent users
  }

-- | Update session with new user states
updateEntanglementSession :: [UserState] -> Double -> EntanglementSession -> EntanglementSession
updateEntanglementSession newUsers dt session =
  let -- Check ongoing consent
      allConsent = all usConsent newUsers

      -- Update bridges
      updatedBridges = map (updateBridge dt) (esBridges session)
      stableBridges = filter bridgeActive updatedBridges

      -- Log coherence (first two users)
      cohLog = case newUsers of
        (a:b:_) -> (usCoherence a, usCoherence b) : take 1000 (esCoherenceLog session)
        _ -> esCoherenceLog session
  in session
      { esUsers = newUsers
      , esBridges = stableBridges
      , esCoherenceLog = cohLog
      , esDuration = esDuration session + dt
      , esActive = allConsent && not (null stableBridges)
      }

-- | Update bridge lifetime
updateBridge :: Double -> ScalarBridge -> ScalarBridge
updateBridge dt bridge = bridge { sbLifetime = sbLifetime bridge + dt }

-- | Get coherence deltas between users
sessionCoherenceDeltas :: EntanglementSession -> [Double]
sessionCoherenceDeltas session =
  map (\(a, b) -> abs (a - b)) (esCoherenceLog session)

-- =============================================================================
-- Safety Enforcement
-- =============================================================================

-- | Safety state for entanglement
data SafetyState = SafetyState
  { ssAllConsent     :: !Bool       -- ^ All users consenting
  , ssCoherenceOK    :: !Bool       -- ^ Coherence within bounds
  , ssNoInversion    :: !Bool       -- ^ No involuntary inversion
  , ssAnkhBalance    :: !Double     -- ^ Ankh parity [0,1]
  } deriving (Eq, Show)

-- | Consent check result
data ConsentCheck
  = ConsentOK
  | ConsentVetoed !String   -- ^ User who vetoed
  | ConsentExpired
  deriving (Eq, Show)

-- | Check safety for entanglement session
checkSafety :: EntanglementSession -> SafetyState
checkSafety session =
  let users = esUsers session
      allConsent = all usConsent users
      coherences = map usCoherence users
      avgCoh = sum coherences / fromIntegral (length coherences)
      cohOK = avgCoh >= coherenceFloorPOR
      noInversion = not (any usInversion users)

      -- Ankh balance: check for forced inversions
      inversionCount = length (filter usInversion users)
      ankhBalance = 1.0 - fromIntegral inversionCount / fromIntegral (max 1 (length users))
  in SafetyState
      { ssAllConsent = allConsent
      , ssCoherenceOK = cohOK
      , ssNoInversion = noInversion
      , ssAnkhBalance = ankhBalance
      }

-- | Check if any user has vetoed
vetoEmergence :: [UserState] -> ConsentCheck
vetoEmergence users =
  case filter (not . usConsent) users of
    [] -> ConsentOK
    (u:_) -> ConsentVetoed (usUserId u)

-- | Auto-suspend on safety violation
autoSuspend :: EntanglementSession -> Maybe String
autoSuspend session =
  let safety = checkSafety session
  in if not (ssAllConsent safety)
     then Just "Consent withdrawn - session suspended"
     else if not (ssCoherenceOK safety)
     then Just "Coherence dropped below threshold"
     else if not (ssNoInversion safety) && ssAnkhBalance safety < 0.5
     then Just "Involuntary inversion detected"
     else Nothing

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
