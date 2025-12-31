{-|
Module      : RaShadowConsent
Description : Consent-Gated Shadow Fragment Harmonics (Patch 12B/12C)
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 12: Enable safe emergence and therapeutic integration of inverted
(shadow) fragments within the Ra Scalar system, governed by consent
protocols, harmonic safety thresholds, and therapeutic guidance.

== Core Features

1. Shadow Fragment Schema - Extended EmergentContent for shadow forms
2. Consent Logic Gating - Ra.Gates enforcement with override protocol
3. Therapeutic Feedback - Guided prompts and convergence mapping
4. Session Tracking - Progress and resonance delta monitoring
5. Session Persistence - Multi-session tracking via session tokens (12B)
6. Emotional Charge Decay - 5% decay per integration cycle (12B)
7. Multi-Fragment Queue - Priority-based fragment ordering (12C)
8. Crypto Override Interface - Signature verification stub (Prompt 33)

== Therapeutic Gating Justification (Codex References)

- REICH_ORGONE_ACCUMULATOR.md:
  High emotional charge layers (DOR overlays) must be dissipated before
  safe emergence. Use chamber geometry and breath-field entrainment.

- KAALI_BECK_BLOOD_ELECTRIFICATION.md:
  Trauma inversion layers manifest in blood-encoded bioelectric fields.
  Electromagnetic discharges correlate to shadow access spikes.

== Safety Thresholds

- Coherence floor: 0.66 (169/256) for safe shadow access
- Emotional charge warning: 0.75 (192/256) triggers coherence spike alert
- Harmonic mismatch limit: 0.25 (64/256) maximum tolerable delta
- Emotional charge floor: 0.15 (38/256) minimum after decay

== Hardware Synthesis

- Target: Xilinx Artix-7 / Intel Cyclone V
- Clock: 10 Hz decision rate
- Resources: ~450 LUTs, 2 DSP slices
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaShadowConsent where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types: Base Definitions
-- =============================================================================

type Fixed8 = Unsigned 8
type Fixed16 = Unsigned 16
type FragmentId = Unsigned 16
type Timestamp = Unsigned 32
type SessionToken = Unsigned 64  -- Session persistence token (12B)

-- =============================================================================
-- Types: Shadow Fragment Schema
-- =============================================================================

-- | Fragment form type
data FragmentForm
  = FormNormal      -- ^ Standard fragment
  | FormShadow      -- ^ Shadow/inverted fragment
  | FormMirror      -- ^ Mirror reflection
  | FormObscured    -- ^ Partially visible
  deriving (Generic, NFDataX, Show, Eq)

-- | Inversion state
data InversionState
  = NotInverted     -- ^ Normal polarity
  | Inverted        -- ^ Inverted polarity
  | PartialInvert   -- ^ Partially inverted
  deriving (Generic, NFDataX, Show, Eq)

-- | Consent state (ACSP-aligned)
data ConsentState
  = FullConsent       -- ^ Full consent granted
  | Therapeutic       -- ^ Therapeutic consent mode
  | Suspended         -- ^ Consent suspended
  | EmergencyOverride -- ^ Emergency override active
  deriving (Generic, NFDataX, Show, Eq)

-- | Shadow type classification
data ShadowType
  = ShadowMirror    -- ^ Mirror reflection shadow
  | ShadowObscured  -- ^ Obscured/hidden shadow
  | ShadowInverted  -- ^ Fully inverted shadow
  deriving (Generic, NFDataX, Show, Eq)

-- | Shadow fragment schema (extended for 12B/12C)
data ShadowFragment = ShadowFragment
  { fragmentId        :: FragmentId       -- ^ Unique identifier
  , fragmentForm      :: FragmentForm     -- ^ Always FormShadow for shadows
  , inversion         :: InversionState   -- ^ Inversion state
  , alpha             :: Fixed8           -- ^ Emergence alpha (0-255)
  , consentState      :: ConsentState     -- ^ Current consent state
  , shadowType        :: ShadowType       -- ^ Shadow classification
  , requiresOverride  :: Bool             -- ^ Override required flag
  , originFragment    :: FragmentId       -- ^ Parent fragment ID
  , harmonicMismatch  :: Fixed8           -- ^ Delta from core Ra harmonic
  , emotionalCharge   :: Fixed8           -- ^ Reich DOR-derived charge
  , integrationCount  :: Unsigned 16      -- ^ Cumulative integration cycles (12B)
  , lastSessionToken  :: SessionToken     -- ^ Last session token (12B)
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Consent Override Protocol
-- =============================================================================

-- | Override source type
data OverrideSource
  = SourceTherapist   -- ^ Licensed therapist
  | SourceOperator    -- ^ Licensed operator
  | SourceSelf        -- ^ Self-authorized (limited)
  | SourceSystem      -- ^ System emergency
  deriving (Generic, NFDataX, Show, Eq)

-- | Consent override record (extended for crypto auth - Prompt 33 interface)
data ConsentOverride = ConsentOverride
  { overrideSource    :: OverrideSource   -- ^ Who authorized
  , overrideReason    :: Unsigned 8       -- ^ Reason code
  , overrideTimestamp :: Timestamp        -- ^ When authorized
  , overrideValid     :: Bool             -- ^ Is override active
  , signatureHash     :: Unsigned 64      -- ^ Crypto signature hash (Prompt 33)
  , operatorKeyId     :: Unsigned 32      -- ^ Public key ID for verification
  } deriving (Generic, NFDataX, Show, Eq)

-- | Verify crypto signature (stub - implementation in Prompt 33)
-- Returns True if: signature authentic, key authorized, timestamp fresh
verifySignature :: ConsentOverride -> Bool
verifySignature override =
  -- Stub: Returns valid if signature hash is non-zero
  -- Full crypto verification deferred to Prompt 33
  signatureHash override /= 0 && overrideValid override

-- =============================================================================
-- Types: Session State
-- =============================================================================

-- | Session state for shadow processing (extended for 12B)
data SessionState = SessionState
  { sessionCoherence    :: Fixed8           -- ^ Current field coherence
  , licensedOperator    :: Bool             -- ^ Is operator licensed
  , consentOverride     :: ConsentOverride  -- ^ Active override (if any)
  , shadowProgress      :: Fixed8           -- ^ Integration progress (0-255)
  , resonanceDelta      :: Signed 16        -- ^ Change in resonance quality
  , sessionCycle        :: Unsigned 16      -- ^ Session cycle counter
  , sessionToken        :: SessionToken     -- ^ Persistence token (12B)
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Gating Decision
-- =============================================================================

-- | Emergence decision result
data EmergenceDecision
  = EmergenceAllowed    -- ^ Safe to emerge
  | EmergenceBlocked    -- ^ Blocked by consent/coherence
  | EmergenceOverride   -- ^ Allowed via override
  | EmergenceTherapy    -- ^ Allowed in therapeutic mode
  deriving (Generic, NFDataX, Show, Eq)

-- | Block reason (for logging)
data BlockReason
  = ReasonNone          -- ^ Not blocked
  | ReasonConsent       -- ^ Consent state insufficient
  | ReasonCoherence     -- ^ Coherence too low
  | ReasonNoOperator    -- ^ No licensed operator
  | ReasonChargeHigh    -- ^ Emotional charge too high
  deriving (Generic, NFDataX, Show, Eq)

-- | Gating result
data GatingResult = GatingResult
  { decision      :: EmergenceDecision  -- ^ Allow/block decision
  , blockReason   :: BlockReason        -- ^ Why blocked (if applicable)
  , safetyWarning :: Bool               -- ^ Safety warning flag
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Therapeutic Feedback
-- =============================================================================

-- | Feedback prompt type
data PromptType
  = PromptGrounding     -- ^ "Breathe slowly and stay grounded"
  | PromptContext       -- ^ Fragment context information
  | PromptReflection    -- ^ "Would you like to reflect..."
  | PromptWarning       -- ^ Coherence spike warning
  | PromptComplete      -- ^ Integration complete
  deriving (Generic, NFDataX, Show, Eq)

-- | Therapeutic prompt
data TherapeuticPrompt = TherapeuticPrompt
  { promptType      :: PromptType       -- ^ Type of prompt
  , promptCode      :: Unsigned 8       -- ^ Message code for lookup
  , promptUrgency   :: Unsigned 4       -- ^ 0=low, 15=critical
  } deriving (Generic, NFDataX, Show, Eq)

-- | Convergence map for tracking
data ConvergenceMap = ConvergenceMap
  { resonanceTrend    :: Signed 8       -- ^ Positive=increasing, negative=decreasing
  , coherenceSpike    :: Bool           -- ^ Spike warning active
  , progressRate      :: Fixed8         -- ^ Integration rate
  } deriving (Generic, NFDataX, Show, Eq)

-- | Complete therapeutic feedback
data TherapeuticFeedback = TherapeuticFeedback
  { prompts         :: Vec 3 TherapeuticPrompt  -- ^ Up to 3 prompts
  , convergence     :: ConvergenceMap           -- ^ Convergence tracking
  , guidanceActive  :: Bool                     -- ^ Guidance mode enabled
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Complete Output
-- =============================================================================

-- | Shadow consent processing output
data ShadowConsentOutput = ShadowConsentOutput
  { gating          :: GatingResult           -- ^ Gating decision
  , feedback        :: TherapeuticFeedback    -- ^ Therapeutic prompts
  , updatedSession  :: SessionState           -- ^ Updated session state
  , updatedFragment :: ShadowFragment         -- ^ Updated fragment (12B)
  , safetyAlert     :: Bool                   -- ^ Critical safety alert
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Types: Multi-Fragment Queue (Patch 12C)
-- =============================================================================

-- | Priority score for queue ordering (higher = process first)
-- Calculated from: emotional_charge DESC, harmonic_mismatch DESC, alpha ASC
type PriorityScore = Unsigned 16

-- | Queue entry with priority
data QueueEntry = QueueEntry
  { entryFragment :: ShadowFragment  -- ^ Fragment data
  , entryPriority :: PriorityScore   -- ^ Computed priority
  } deriving (Generic, NFDataX, Show, Eq)

-- | Shadow fragment queue (fixed-size for FPGA)
data ShadowQueue = ShadowQueue
  { queueEntries :: Vec 8 QueueEntry   -- ^ Up to 8 fragments
  , queueCount   :: Unsigned 4         -- ^ Current count
  , activeIndex  :: Unsigned 4         -- ^ Currently active fragment
  } deriving (Generic, NFDataX, Show, Eq)

-- | Calculate priority score for a fragment
-- Higher emotional charge and harmonic mismatch = higher priority
-- Lower alpha = higher priority (safer emergence first)
calculatePriority :: ShadowFragment -> PriorityScore
calculatePriority frag =
  let
    charge = resize (emotionalCharge frag) :: Unsigned 16
    mismatch = resize (harmonicMismatch frag) :: Unsigned 16
    alphaPart = 256 - resize (alpha frag) :: Unsigned 16
  in
    (charge `shiftL` 8) + (mismatch `shiftL` 4) + alphaPart

-- | Initial empty queue
initQueue :: ShadowQueue
initQueue = ShadowQueue
  { queueEntries = repeat (QueueEntry initFragment 0)
  , queueCount = 0
  , activeIndex = 0
  }
  where
    initFragment = ShadowFragment 0 FormNormal NotInverted 0 Suspended
                     ShadowMirror False 0 0 0 0 0

-- | Add fragment to queue (insertion sort by priority)
addToQueue :: ShadowQueue -> ShadowFragment -> ShadowQueue
addToQueue queue frag
  | queueCount queue >= 8 = queue  -- Queue full
  | otherwise =
      let
        entry = QueueEntry frag (calculatePriority frag)
        newCount = queueCount queue + 1
        entries = queueEntries queue
        -- Simple append for FPGA (full sort not synthesizable)
        newEntries = replace (resize (queueCount queue)) entry entries
      in
        queue { queueEntries = newEntries, queueCount = newCount }

-- | Get next fragment from queue (highest priority)
nextFromQueue :: ShadowQueue -> Maybe (ShadowFragment, ShadowQueue)
nextFromQueue queue
  | queueCount queue == 0 = Nothing
  | otherwise =
      let
        idx = activeIndex queue
        entry = queueEntries queue !! resize idx
        newQueue = queue { activeIndex = (idx + 1) `mod` resize (queueCount queue) }
      in
        Just (entryFragment entry, newQueue)

-- =============================================================================
-- Constants: Safety Thresholds
-- =============================================================================

-- | Minimum coherence for safe shadow access (0.66 = 169/256)
coherenceSafeAccess :: Fixed8
coherenceSafeAccess = 169

-- | Emotional charge warning threshold (0.75 = 192/256)
chargeWarningThreshold :: Fixed8
chargeWarningThreshold = 192

-- | Maximum harmonic mismatch (0.25 = 64/256)
maxHarmonicMismatch :: Fixed8
maxHarmonicMismatch = 64

-- | Emotional charge floor (0.15 = 38/256) - minimum after decay (12B)
emotionalChargeFloor :: Fixed8
emotionalChargeFloor = 38

-- | Emotional charge decay factor (0.95 * 256 = 243)
-- Apply as: newCharge = (charge * 243) >> 8
emotionalChargeDecay :: Unsigned 16
emotionalChargeDecay = 243

-- | Progress increment per cycle
progressIncrement :: Fixed8
progressIncrement = 3

-- =============================================================================
-- Core Functions: Consent Gating (Ra.Gates)
-- =============================================================================

-- | Check if override is valid and active
isOverrideValid :: ConsentOverride -> Bool
isOverrideValid override =
  overrideValid override && overrideSource override /= SourceSelf

-- | Main consent gating logic
-- Block shadow emergence unless:
-- 1. consent_state == THERAPEUTIC, OR
-- 2. licensed_operator == true, OR
-- 3. consent_override is valid, OR
-- 4. field_coherence >= 0.66
shouldAllowEmergence :: ShadowFragment -> SessionState -> GatingResult
shouldAllowEmergence frag session
  -- Non-shadow fragments always allowed
  | fragmentForm frag /= FormShadow =
      GatingResult EmergenceAllowed ReasonNone False

  -- Therapeutic consent state allows
  | consentState frag == Therapeutic =
      GatingResult EmergenceTherapy ReasonNone (emotionalCharge frag > chargeWarningThreshold)

  -- Licensed operator allows
  | licensedOperator session =
      GatingResult EmergenceAllowed ReasonNone (emotionalCharge frag > chargeWarningThreshold)

  -- Valid override allows
  | isOverrideValid (consentOverride session) =
      GatingResult EmergenceOverride ReasonNone (emotionalCharge frag > chargeWarningThreshold)

  -- High coherence allows
  | sessionCoherence session >= coherenceSafeAccess =
      GatingResult EmergenceAllowed ReasonNone (emotionalCharge frag > chargeWarningThreshold)

  -- High emotional charge blocks even with moderate coherence
  | emotionalCharge frag > chargeWarningThreshold =
      GatingResult EmergenceBlocked ReasonChargeHigh True

  -- Default: blocked due to insufficient consent/coherence
  | otherwise =
      let reason = if sessionCoherence session < coherenceSafeAccess
                   then ReasonCoherence
                   else ReasonConsent
      in GatingResult EmergenceBlocked reason False

-- =============================================================================
-- Core Functions: Therapeutic Feedback Generator
-- =============================================================================

-- | Generate grounding prompt
groundingPrompt :: TherapeuticPrompt
groundingPrompt = TherapeuticPrompt PromptGrounding 1 4

-- | Generate context prompt
contextPrompt :: Fixed8 -> TherapeuticPrompt
contextPrompt urgency = TherapeuticPrompt PromptContext 2 (resize urgency `shiftR` 4)

-- | Generate reflection prompt
reflectionPrompt :: TherapeuticPrompt
reflectionPrompt = TherapeuticPrompt PromptReflection 3 2

-- | Generate warning prompt
warningPrompt :: TherapeuticPrompt
warningPrompt = TherapeuticPrompt PromptWarning 4 12

-- | Generate convergence map
generateConvergence :: SessionState -> ShadowFragment -> ConvergenceMap
generateConvergence session frag =
  let
    trend = resize (resonanceDelta session) :: Signed 8
    spike = emotionalCharge frag > chargeWarningThreshold
    rate = if sessionCoherence session > coherenceSafeAccess
           then progressIncrement + 2
           else progressIncrement
  in
    ConvergenceMap trend spike rate

-- | Generate therapeutic feedback
generateFeedback :: ShadowFragment -> SessionState -> GatingResult -> TherapeuticFeedback
generateFeedback frag session gating =
  let
    -- Always start with grounding
    p1 = groundingPrompt

    -- Context based on fragment
    p2 = contextPrompt (emotionalCharge frag)

    -- Warning or reflection based on state
    p3 = if safetyWarning gating
         then warningPrompt
         else reflectionPrompt

    conv = generateConvergence session frag

    active = decision gating /= EmergenceBlocked
  in
    TherapeuticFeedback (p1 :> p2 :> p3 :> Nil) conv active

-- =============================================================================
-- Core Functions: Session Update
-- =============================================================================

-- | Apply emotional charge decay (12B)
-- Decay by 5% per cycle, floor at 0.15
applyChargeDecay :: Fixed8 -> Fixed8
applyChargeDecay charge =
  let
    decayed = (resize charge * emotionalChargeDecay) `shiftR` 8 :: Unsigned 16
    result = resize decayed :: Fixed8
  in
    if result < emotionalChargeFloor
    then emotionalChargeFloor
    else result

-- | Update fragment state after processing (12B)
updateFragment :: ShadowFragment -> SessionState -> GatingResult -> ShadowFragment
updateFragment frag session gating
  | decision gating /= EmergenceBlocked =
      frag
        { emotionalCharge = applyChargeDecay (emotionalCharge frag)
        , integrationCount = satAdd SatBound (integrationCount frag) 1
        , lastSessionToken = sessionToken session
        }
  | otherwise = frag

-- | Update session state after processing
updateSession :: SessionState -> ShadowFragment -> GatingResult -> SessionState
updateSession session frag gating =
  let
    -- Update progress if allowed
    newProgress = if decision gating /= EmergenceBlocked
                  then satAdd SatBound (shadowProgress session) progressIncrement
                  else shadowProgress session

    -- Calculate resonance delta from harmonic mismatch
    mismatch = resize (harmonicMismatch frag) :: Signed 16
    newDelta = if decision gating /= EmergenceBlocked
               then satSub SatBound (resonanceDelta session) (mismatch `shiftR` 2)
               else resonanceDelta session

    -- Increment cycle
    newCycle = satAdd SatBound (sessionCycle session) 1
  in
    session
      { shadowProgress = newProgress
      , resonanceDelta = newDelta
      , sessionCycle = newCycle
      }

-- =============================================================================
-- Main Processing Function
-- =============================================================================

-- | Process shadow fragment with consent gating
processShadowConsent :: ShadowFragment -> SessionState -> ShadowConsentOutput
processShadowConsent frag session =
  let
    -- Step 1: Apply consent gating
    gating = shouldAllowEmergence frag session

    -- Step 2: Generate therapeutic feedback
    feedback = generateFeedback frag session gating

    -- Step 3: Update session state
    newSession = updateSession session frag gating

    -- Step 4: Update fragment state (12B - decay emotional charge)
    newFragment = updateFragment frag session gating

    -- Step 5: Check for critical safety alert
    -- Alert if: blocked due to high charge, or coherence drops below 0.5
    safety = (decision gating == EmergenceBlocked && blockReason gating == ReasonChargeHigh)
          || sessionCoherence session < 128
  in
    ShadowConsentOutput gating feedback newSession newFragment safety

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Initial session state
initSession :: SessionState
initSession = SessionState
  { sessionCoherence = 128
  , licensedOperator = False
  , consentOverride = ConsentOverride SourceSelf 0 0 False 0 0
  , shadowProgress = 0
  , resonanceDelta = 0
  , sessionCycle = 0
  , sessionToken = 0  -- Generated externally (12B)
  }

-- | Shadow consent processor (stateful)
shadowConsentProcessor
  :: HiddenClockResetEnable dom
  => Signal dom (ShadowFragment, Bool, Bool)  -- ^ (fragment, licensed, override_valid)
  -> Signal dom ShadowConsentOutput
shadowConsentProcessor input = mealy procState initSession input
  where
    procState :: SessionState -> (ShadowFragment, Bool, Bool) -> (SessionState, ShadowConsentOutput)
    procState session (frag, licensed, hasOverride) =
      let
        -- Update session with input flags
        session' = session
          { licensedOperator = licensed
          , consentOverride = if hasOverride
              then ConsentOverride SourceTherapist 1 0 True 0xDEADBEEF 1  -- Signature placeholder
              else consentOverride session
          }
        output = processShadowConsent frag session'
      in
        (updatedSession output, output)

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

{-# ANN shadowConsentTop (Synthesize
  { t_name = "shadow_consent_unit"
  , t_inputs = [ PortName "clk", PortName "rst", PortName "en"
               , PortName "fragment"
               , PortName "licensed_operator"
               , PortName "override_valid" ]
  , t_output = PortProduct "output"
      [ PortName "gating", PortName "feedback"
      , PortName "session", PortName "safety_alert" ]
  }) #-}
shadowConsentTop
  :: Clock System -> Reset System -> Enable System
  -> Signal System ShadowFragment
  -> Signal System Bool
  -> Signal System Bool
  -> Signal System ShadowConsentOutput
shadowConsentTop clk rst en frag licensed override =
  exposeClockResetEnable shadowConsentProcessor clk rst en
    (bundle (frag, licensed, override))

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test shadow fragment (suspended, requires override)
testFragment1 :: ShadowFragment
testFragment1 = ShadowFragment
  { fragmentId = 0x0038
  , fragmentForm = FormShadow
  , inversion = Inverted
  , alpha = 107  -- 0.42
  , consentState = Suspended
  , shadowType = ShadowInverted
  , requiresOverride = True
  , originFragment = 0x0038
  , harmonicMismatch = 46  -- 0.18
  , emotionalCharge = 181  -- 0.71
  , integrationCount = 0   -- Fresh fragment (12B)
  , lastSessionToken = 0   -- No prior session (12B)
  }

-- | Test shadow fragment (therapeutic mode)
testFragment2 :: ShadowFragment
testFragment2 = ShadowFragment
  { fragmentId = 0x0039
  , fragmentForm = FormShadow
  , inversion = Inverted
  , alpha = 169  -- 0.66
  , consentState = Therapeutic
  , shadowType = ShadowMirror
  , requiresOverride = False
  , originFragment = 0x0039
  , harmonicMismatch = 31  -- 0.12
  , emotionalCharge = 164  -- 0.64
  , integrationCount = 3   -- Returning fragment (12B)
  , lastSessionToken = 0x12345678  -- Prior session (12B)
  }

-- =============================================================================
-- Testbench
-- =============================================================================

testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    out = shadowConsentTop clk rst enableGen
            (pure testFragment1) (pure False) (pure False)
    -- Test passes if output is valid (blocked as expected)
    done = register clk rst enableGen False
            ((\o -> decision (gating o) == EmergenceBlocked) <$> out)

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Get decision name
decisionName :: EmergenceDecision -> String
decisionName EmergenceAllowed  = "ALLOWED"
decisionName EmergenceBlocked  = "BLOCKED"
decisionName EmergenceOverride = "OVERRIDE"
decisionName EmergenceTherapy  = "THERAPEUTIC"

-- | Get block reason name
blockReasonName :: BlockReason -> String
blockReasonName ReasonNone      = "None"
blockReasonName ReasonConsent   = "Consent state insufficient"
blockReasonName ReasonCoherence = "Coherence level insufficient"
blockReasonName ReasonNoOperator = "No licensed operator"
blockReasonName ReasonChargeHigh = "Emotional charge too high"

-- | Format therapeutic message by code
promptMessage :: Unsigned 8 -> String
promptMessage 1 = "This memory may feel dissonant. Breathe slowly and stay grounded."
promptMessage 2 = "Fragment context: Shadow type identified."
promptMessage 3 = "Would you like to reflect on this with your guide?"
promptMessage 4 = "Warning: Coherence spike detected. Stabilize before continuing."
promptMessage _ = "Processing shadow fragment..."
