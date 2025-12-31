{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

-- | Prompt 15: Multi-User Harmonic Entanglement via Scalar Bridges
-- FPGA module for real-time bridge synchronization, consent validation,
-- and harmonic compatibility in multi-user scalar coupling.
--
-- Codex References:
-- - Ra.Gates: Consent gating protocols
-- - Ra.Emergence: Shared fragment emergence
-- - Ra.Identity: User identity validation
--
-- Integration:
-- - Prompt 11: Group coherence (RaCohereNet)
-- - Prompt 14: Shared lucid navigation (leader_follow mode)
-- - Prompt 12: Shadow consent for shared shadow fragments

module RaBridgeSync where

import Clash.Prelude
import GHC.Generics (Generic)

-- ============================================================================
-- Constants (Codex-aligned, per Architect clarifications)
-- ============================================================================

-- | Phase alignment threshold (PHI_THRESHOLD = 0.03 * 255)
phiThreshold :: Unsigned 8
phiThreshold = 8  -- 0.03 * 255

-- | Coherence tolerance (COHERENCE_TOLERANCE = 0.08 * 255)
coherenceTolerance :: Unsigned 8
coherenceTolerance = 20  -- 0.08 * 255

-- | Group coherence minimum (GROUP_COHERENCE_MIN = 0.72 * 255)
groupCoherenceMin :: Unsigned 8
groupCoherenceMin = 184  -- 0.72 * 255

-- | Minimum entanglement score for stability (0.6 * 255)
entanglementScoreMin :: Unsigned 8
entanglementScoreMin = 153  -- 0.6 * 255

-- | Shadow coherence minimum (0.75 * 255)
shadowCoherenceMin :: Unsigned 8
shadowCoherenceMin = 191  -- 0.75 * 255

-- | Maximum bridges per user
maxBridgesPerUser :: Unsigned 4
maxBridgesPerUser = 5

-- | Safety latency cycles (250ms at 100MHz)
safetyLatencyCycles :: Unsigned 24
safetyLatencyCycles = 25000000  -- 250ms

-- ============================================================================
-- Types
-- ============================================================================

-- | Unified consent states (Prompt 12 + 15)
data ConsentState
  = ConsentNone
  | ConsentPrivate
  | ConsentTherapeutic
  | ConsentEntangled
  | ConsentWithdrawn
  | ConsentEmergency
  deriving (Generic, NFDataX, Eq, Show)

-- | Bridge mode
data BridgeMode
  = ModeMirror      -- Identical experience
  | ModeComplement  -- Dual perspectives
  | ModeAsymmetric  -- Leader/follower
  | ModeBroadcast   -- One-to-many
  deriving (Generic, NFDataX, Eq, Show)

-- | Bridge FSM state
data BridgeState
  = BridgeIdle
  | BridgeEntangled
  | BridgeSuspended
  | BridgeTerminated
  deriving (Generic, NFDataX, Eq, Show)

-- | Emergence type
data EmergenceType
  = EmergenceShared        -- MIRROR mode
  | EmergenceDualReflect   -- COMPLEMENT mode
  | EmergenceRelayAccess   -- ASYMMETRIC mode
  | EmergenceMulticastSync -- BROADCAST mode
  deriving (Generic, NFDataX, Eq, Show)

-- | User snapshot for entanglement
data UserSnapshot = UserSnapshot
  { userId      :: Unsigned 8    -- User ID
  , uCoherence  :: Unsigned 8    -- Coherence 0-255
  , uHarmonicL  :: Unsigned 4    -- l harmonic index
  , uHarmonicM  :: Unsigned 4    -- m harmonic index
  , uConsent    :: ConsentState  -- Current consent
  , uPhiPhase   :: Unsigned 8    -- phi^n phase (0-255)
  , uVeto       :: Bool          -- Override veto
  } deriving (Generic, NFDataX, Show)

-- | Bridge descriptor
data BridgeDescriptor = BridgeDescriptor
  { bridgeId       :: Unsigned 12
  , userA          :: Unsigned 8
  , userB          :: Unsigned 8
  , bMode          :: BridgeMode
  , bState         :: BridgeState
  , cohDelta       :: Unsigned 8
  , phaseDelta     :: Unsigned 8
  , harmCompat     :: Bool
  , entScore       :: Unsigned 8
  , bStable        :: Bool
  } deriving (Generic, NFDataX, Show)

-- | Bridge input
data BridgeInput = BridgeInput
  { biUserA      :: UserSnapshot
  , biUserB      :: UserSnapshot
  , biMode       :: BridgeMode
  , biCommand    :: BridgeCommand
  } deriving (Generic, NFDataX)

-- | Bridge commands
data BridgeCommand
  = CmdNone
  | CmdCreate
  | CmdSuspend
  | CmdTerminate
  | CmdEmerge
  deriving (Generic, NFDataX, Eq, Show)

-- | Bridge output
data BridgeOutput = BridgeOutput
  { boState       :: BridgeState
  , boEntScore    :: Unsigned 8
  , boStable      :: Bool
  , boCanEmerge   :: Bool
  , boEmergType   :: EmergenceType
  , boFluxBoost   :: Unsigned 8
  , boSafetyOk    :: Bool
  , boReason      :: Unsigned 4  -- Encoded reason
  } deriving (Generic, NFDataX, Show)

-- ============================================================================
-- Harmonic Compatibility
-- ============================================================================

-- | Check harmonic compatibility (l-match AND m-diff <= 1)
checkHarmonicCompat :: UserSnapshot -> UserSnapshot -> Bool
checkHarmonicCompat a b =
  let
    lMatch = uHarmonicL a == uHarmonicL b
    mDiff  = if uHarmonicM a >= uHarmonicM b
             then uHarmonicM a - uHarmonicM b
             else uHarmonicM b - uHarmonicM a
    mNear  = mDiff <= 1
  in lMatch && mNear

-- ============================================================================
-- Entanglement Score
-- ============================================================================

-- | Calculate entanglement score
-- Formula: max(0, 1.0 - (coherence_delta + phase_delta / 2))
-- Scaled to 0-255
calculateEntanglementScore :: UserSnapshot -> UserSnapshot -> Unsigned 8
calculateEntanglementScore a b =
  let
    cohA = resize (uCoherence a) :: Unsigned 16
    cohB = resize (uCoherence b) :: Unsigned 16
    cohDelta = if cohA >= cohB then cohA - cohB else cohB - cohA

    phaseA = resize (uPhiPhase a) :: Unsigned 16
    phaseB = resize (uPhiPhase b) :: Unsigned 16
    phaseDelta = if phaseA >= phaseB then phaseA - phaseB else phaseB - phaseA

    -- raw = 255 - (cohDelta + phaseDelta / 2)
    halfPhase = phaseDelta `shiftR` 1
    penalty = cohDelta + halfPhase
    score = if penalty >= 255 then 0 else 255 - penalty
  in truncateB score

-- | Calculate coherence delta
calcCohDelta :: UserSnapshot -> UserSnapshot -> Unsigned 8
calcCohDelta a b =
  if uCoherence a >= uCoherence b
  then uCoherence a - uCoherence b
  else uCoherence b - uCoherence a

-- | Calculate phase delta
calcPhaseDelta :: UserSnapshot -> UserSnapshot -> Unsigned 8
calcPhaseDelta a b =
  if uPhiPhase a >= uPhiPhase b
  then uPhiPhase a - uPhiPhase b
  else uPhiPhase b - uPhiPhase a

-- ============================================================================
-- Consent Validation
-- ============================================================================

-- | Check if consent allows entanglement
consentAllowsEntangle :: ConsentState -> Bool
consentAllowsEntangle ConsentEntangled = True
consentAllowsEntangle _                = False

-- | Check if consent allows shadow sharing
consentAllowsShadow :: ConsentState -> Bool
consentAllowsShadow ConsentTherapeutic = True
consentAllowsShadow ConsentEntangled   = True
consentAllowsShadow _                  = False

-- ============================================================================
-- Safety Enforcement
-- ============================================================================

-- | Check bridge safety
-- Must validate within 250ms
checkBridgeSafety :: UserSnapshot -> UserSnapshot -> BridgeState -> Bool
checkBridgeSafety a b state = case state of
  BridgeEntangled ->
    let
      consentOk = consentAllowsEntangle (uConsent a) && consentAllowsEntangle (uConsent b)
      vetoOk    = not (uVeto a) && not (uVeto b)
    in consentOk && vetoOk
  _ -> False

-- | Check if entanglement is possible
entanglementPossible :: UserSnapshot -> UserSnapshot -> (Bool, Unsigned 4)
entanglementPossible a b
  -- Consent check
  | not (consentAllowsEntangle (uConsent a)) = (False, 1)  -- User A consent
  | not (consentAllowsEntangle (uConsent b)) = (False, 2)  -- User B consent
  -- Veto check
  | uVeto a = (False, 3)  -- User A veto
  | uVeto b = (False, 4)  -- User B veto
  -- Phase alignment
  | calcPhaseDelta a b > phiThreshold = (False, 5)  -- Phase mismatch
  -- Coherence tolerance
  | calcCohDelta a b > coherenceTolerance = (False, 6)  -- Coherence mismatch
  -- Harmonic compatibility
  | not (checkHarmonicCompat a b) = (False, 7)  -- Harmonic incompatible
  -- Entanglement score
  | calculateEntanglementScore a b < entanglementScoreMin = (False, 8)  -- Score too low
  | otherwise = (True, 0)  -- OK

-- ============================================================================
-- Emergence Logic
-- ============================================================================

-- | Get emergence type from bridge mode
emergenceFromMode :: BridgeMode -> EmergenceType
emergenceFromMode ModeMirror     = EmergenceShared
emergenceFromMode ModeComplement = EmergenceDualReflect
emergenceFromMode ModeAsymmetric = EmergenceRelayAccess
emergenceFromMode ModeBroadcast  = EmergenceMulticastSync

-- | Calculate flux boost
-- flux = (coh_a + coh_b) / 2 * score / 255
calcFluxBoost :: UserSnapshot -> UserSnapshot -> Unsigned 8 -> Unsigned 8
calcFluxBoost a b score =
  let
    avgCoh = (resize (uCoherence a) + resize (uCoherence b)) `shiftR` 1 :: Unsigned 16
    flux = (avgCoh * resize score) `shiftR` 8
  in truncateB flux

-- | Check if shadow sharing is allowed
canShareShadow :: UserSnapshot -> UserSnapshot -> Bool
canShareShadow a b =
  let
    consentOk = consentAllowsShadow (uConsent a) && consentAllowsShadow (uConsent b)
    avgCoh = ((resize (uCoherence a) :: Unsigned 16) + resize (uCoherence b)) `shiftR` 1
  in consentOk && avgCoh >= resize shadowCoherenceMin

-- ============================================================================
-- Bridge FSM
-- ============================================================================

-- | Bridge FSM state
data BridgeFSMState = BridgeFSMState
  { fsmBridgeState :: BridgeState
  , fsmEntScore    :: Unsigned 8
  , fsmStable      :: Bool
  , fsmCohDelta    :: Unsigned 8
  , fsmPhaseDelta  :: Unsigned 8
  , fsmHarmCompat  :: Bool
  , fsmMode        :: BridgeMode
  , fsmCycles      :: Unsigned 24  -- Safety cycle counter
  } deriving (Generic, NFDataX)

-- | Initial FSM state
initBridgeFSM :: BridgeFSMState
initBridgeFSM = BridgeFSMState
  { fsmBridgeState = BridgeIdle
  , fsmEntScore    = 0
  , fsmStable      = False
  , fsmCohDelta    = 0
  , fsmPhaseDelta  = 0
  , fsmHarmCompat  = False
  , fsmMode        = ModeMirror
  , fsmCycles      = 0
  }

-- | Bridge step logic
bridgeStep :: BridgeFSMState -> BridgeInput -> (BridgeFSMState, BridgeOutput)
bridgeStep st@BridgeFSMState{..} BridgeInput{..} =
  let
    -- Calculate metrics
    entScore   = calculateEntanglementScore biUserA biUserB
    cohDelta   = calcCohDelta biUserA biUserB
    phaseDelta = calcPhaseDelta biUserA biUserB
    harmCompat = checkHarmonicCompat biUserA biUserB
    (possible, reason) = entanglementPossible biUserA biUserB
    safetyOk   = checkBridgeSafety biUserA biUserB fsmBridgeState
    stable     = entScore >= entanglementScoreMin

    -- State transitions
    (newState, newScore, newStable, newMode) = case (fsmBridgeState, biCommand) of
      -- From IDLE
      (BridgeIdle, CmdCreate) ->
        if possible
        then (BridgeEntangled, entScore, stable, biMode)
        else (BridgeIdle, 0, False, fsmMode)

      -- From ENTANGLED
      (BridgeEntangled, CmdSuspend) ->
        (BridgeSuspended, fsmEntScore, False, fsmMode)
      (BridgeEntangled, CmdTerminate) ->
        (BridgeTerminated, 0, False, fsmMode)
      (BridgeEntangled, _) ->
        -- Re-evaluate safety continuously
        if safetyOk
        then (BridgeEntangled, entScore, stable, fsmMode)
        else (BridgeSuspended, fsmEntScore, False, fsmMode)

      -- From SUSPENDED
      (BridgeSuspended, CmdCreate) ->
        if possible
        then (BridgeEntangled, entScore, stable, fsmMode)
        else (BridgeSuspended, fsmEntScore, False, fsmMode)
      (BridgeSuspended, CmdTerminate) ->
        (BridgeTerminated, 0, False, fsmMode)

      -- Default: stay in current state
      _ -> (fsmBridgeState, fsmEntScore, fsmStable, fsmMode)

    -- Emergence
    canEmerge = fsmBridgeState == BridgeEntangled && fsmStable && biCommand == CmdEmerge
    emergType = emergenceFromMode fsmMode
    fluxBoost = if canEmerge then calcFluxBoost biUserA biUserB fsmEntScore else 0

    -- Update state
    newSt = st
      { fsmBridgeState = newState
      , fsmEntScore    = newScore
      , fsmStable      = newStable
      , fsmCohDelta    = cohDelta
      , fsmPhaseDelta  = phaseDelta
      , fsmHarmCompat  = harmCompat
      , fsmMode        = newMode
      , fsmCycles      = if fsmCycles >= safetyLatencyCycles then 0 else fsmCycles + 1
      }

    -- Output
    output = BridgeOutput
      { boState     = newState
      , boEntScore  = newScore
      , boStable    = newStable
      , boCanEmerge = canEmerge
      , boEmergType = emergType
      , boFluxBoost = fluxBoost
      , boSafetyOk  = safetyOk || fsmBridgeState == BridgeIdle
      , boReason    = reason
      }
  in (newSt, output)

-- ============================================================================
-- Top-Level Entity
-- ============================================================================

-- | Main bridge sync FSM
bridgeSyncFSM :: HiddenClockResetEnable dom
              => Signal dom BridgeInput
              -> Signal dom BridgeOutput
bridgeSyncFSM = mealy bridgeStep initBridgeFSM

-- | Top entity with port annotations
{-# ANN bridgeSyncTop
  (Synthesize
    { t_name   = "bridge_sync_top"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "userA_id"
                 , PortName "userA_coherence"
                 , PortName "userA_harmonic_l"
                 , PortName "userA_harmonic_m"
                 , PortName "userA_consent"
                 , PortName "userA_phi_phase"
                 , PortName "userA_veto"
                 , PortName "userB_id"
                 , PortName "userB_coherence"
                 , PortName "userB_harmonic_l"
                 , PortName "userB_harmonic_m"
                 , PortName "userB_consent"
                 , PortName "userB_phi_phase"
                 , PortName "userB_veto"
                 , PortName "bridge_mode"
                 , PortName "command"
                 ]
    , t_output = PortProduct "output"
                 [ PortName "state"
                 , PortName "ent_score"
                 , PortName "stable"
                 , PortName "can_emerge"
                 , PortName "emerg_type"
                 , PortName "flux_boost"
                 , PortName "safety_ok"
                 , PortName "reason"
                 ]
    }) #-}
bridgeSyncTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 8)  -- userA_id
  -> Signal System (Unsigned 8)  -- userA_coherence
  -> Signal System (Unsigned 4)  -- userA_harmonic_l
  -> Signal System (Unsigned 4)  -- userA_harmonic_m
  -> Signal System (Unsigned 3)  -- userA_consent
  -> Signal System (Unsigned 8)  -- userA_phi_phase
  -> Signal System Bool          -- userA_veto
  -> Signal System (Unsigned 8)  -- userB_id
  -> Signal System (Unsigned 8)  -- userB_coherence
  -> Signal System (Unsigned 4)  -- userB_harmonic_l
  -> Signal System (Unsigned 4)  -- userB_harmonic_m
  -> Signal System (Unsigned 3)  -- userB_consent
  -> Signal System (Unsigned 8)  -- userB_phi_phase
  -> Signal System Bool          -- userB_veto
  -> Signal System (Unsigned 2)  -- bridge_mode
  -> Signal System (Unsigned 3)  -- command
  -> Signal System ( Unsigned 2  -- state
                   , Unsigned 8  -- ent_score
                   , Bool        -- stable
                   , Bool        -- can_emerge
                   , Unsigned 2  -- emerg_type
                   , Unsigned 8  -- flux_boost
                   , Bool        -- safety_ok
                   , Unsigned 4  -- reason
                   )
bridgeSyncTop clk rst en
              aId aCoh aL aM aCons aPhase aVeto
              bId bCoh bL bM bCons bPhase bVeto
              mode cmd =
  withClockResetEnable clk rst en $
    let
      -- Decode consent
      decConsent c = case c of
        0 -> ConsentNone
        1 -> ConsentPrivate
        2 -> ConsentTherapeutic
        3 -> ConsentEntangled
        4 -> ConsentWithdrawn
        _ -> ConsentEmergency

      -- Decode mode
      decMode m = case m of
        0 -> ModeMirror
        1 -> ModeComplement
        2 -> ModeAsymmetric
        _ -> ModeBroadcast

      -- Decode command
      decCommand c = case c of
        0 -> CmdNone
        1 -> CmdCreate
        2 -> CmdSuspend
        3 -> CmdTerminate
        _ -> CmdEmerge

      -- Build user A
      userA = UserSnapshot
        <$> aId <*> aCoh <*> aL <*> aM
        <*> fmap decConsent aCons
        <*> aPhase <*> aVeto

      -- Build user B
      userB = UserSnapshot
        <$> bId <*> bCoh <*> bL <*> bM
        <*> fmap decConsent bCons
        <*> bPhase <*> bVeto

      -- Build input
      input = BridgeInput
        <$> userA
        <*> userB
        <*> fmap decMode mode
        <*> fmap decCommand cmd

      -- Run FSM
      output = bridgeSyncFSM input

      -- Extract output
      extractOut BridgeOutput{..} =
        ( encState boState
        , boEntScore
        , boStable
        , boCanEmerge
        , encEmergence boEmergType
        , boFluxBoost
        , boSafetyOk
        , boReason
        )
    in fmap extractOut output

-- | Encode bridge state
encState :: BridgeState -> Unsigned 2
encState BridgeIdle       = 0
encState BridgeEntangled  = 1
encState BridgeSuspended  = 2
encState BridgeTerminated = 3

-- | Encode emergence type
encEmergence :: EmergenceType -> Unsigned 2
encEmergence EmergenceShared        = 0
encEmergence EmergenceDualReflect   = 1
encEmergence EmergenceRelayAccess   = 2
encEmergence EmergenceMulticastSync = 3

-- ============================================================================
-- Group Bridge Support (RaCohereNet)
-- ============================================================================

-- | Group coherence calculator (average of N users, max 8)
calcGroupCoherence :: Vec 8 (Unsigned 8) -> Unsigned 4 -> Unsigned 8
calcGroupCoherence cohVec count =
  let
    validCount = if count == 0 then 1 else count
    sum16 = fold (+) (map (resize :: Unsigned 8 -> Unsigned 16) cohVec)
    avg = sum16 `div` resize validCount
  in truncateB avg

-- | Check group phi stability (max deviation <= 0.04 * 255 = 10)
checkGroupPhiStability :: Vec 8 (Unsigned 8) -> Unsigned 4 -> Bool
checkGroupPhiStability phases count =
  let
    maxPhase = fold max phases
    minPhase = fold min phases
    deviation = maxPhase - minPhase
  in count >= 2 && deviation <= 10

-- | Group bridge output
data GroupOutput = GroupOutput
  { goGroupCoh   :: Unsigned 8
  , goPhiStable  :: Bool
  , goCanBroadcast :: Bool
  } deriving (Generic, NFDataX, Show)

-- | Calculate group metrics
calcGroupMetrics :: Vec 8 (Unsigned 8) -> Vec 8 (Unsigned 8) -> Unsigned 4 -> GroupOutput
calcGroupMetrics cohVec phaseVec count =
  let
    groupCoh  = calcGroupCoherence cohVec count
    phiStable = checkGroupPhiStability phaseVec count
    canBcast  = groupCoh >= groupCoherenceMin && phiStable
  in GroupOutput groupCoh phiStable canBcast

-- ============================================================================
-- Testbench
-- ============================================================================

-- | Test users
testUserA :: UserSnapshot
testUserA = UserSnapshot
  { userId     = 1
  , uCoherence = 217  -- 0.85
  , uHarmonicL = 3
  , uHarmonicM = 5
  , uConsent   = ConsentEntangled
  , uPhiPhase  = 158  -- ~0.618
  , uVeto      = False
  }

testUserB :: UserSnapshot
testUserB = UserSnapshot
  { userId     = 2
  , uCoherence = 209  -- 0.82
  , uHarmonicL = 3
  , uHarmonicM = 6
  , uConsent   = ConsentEntangled
  , uPhiPhase  = 159
  , uVeto      = False
  }

-- | Test input vectors
testInputs :: Vec 5 BridgeInput
testInputs =
     BridgeInput testUserA testUserB ModeMirror CmdCreate
  :> BridgeInput testUserA testUserB ModeMirror CmdNone
  :> BridgeInput testUserA testUserB ModeMirror CmdEmerge
  :> BridgeInput testUserA testUserB ModeMirror CmdSuspend
  :> BridgeInput testUserA testUserB ModeMirror CmdTerminate
  :> Nil

-- | Testbench entity
testBench :: Signal System BridgeOutput
testBench = withClockResetEnable systemClockGen systemResetGen enableGen $
  bridgeSyncFSM (fromList (toList testInputs))
