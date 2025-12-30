{-|
Module      : Ra.Shadow
Description : Consent-gated shadow harmonics for therapeutic emergence
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Safe integration of suppressed or inverted content (trauma fragments)
via consent-aware field thresholds and resonance matching.

== Shadow Fragment Theory

From Reich's orgone research and Keely's sympathetic physics:

* Emotional charge accumulates in layered somatic structures
* Suppressed content exists in "inverted" scalar fields
* Safe integration requires gradual resonance matching
* Emergency overrides protect against overwhelming discharge

== Kaali/Beck Bioelectrical Gating

Trauma patterns manifest as bioelectrical signatures:

* Blood electrification affects emotional accessibility
* Microcurrent patterns correlate with suppression depth
* Gradual field harmonization enables safe surfacing

== Safety Protocol

Shadow fragments ONLY emerge when:

1. Explicit THERAPEUTIC consent state
2. Licensed operator flag (session is guided)
3. Field coherence > COHERENCE_SAFE_ACCESS (0.66)
4. No emergency override conditions present
-}
module Ra.Shadow
  ( -- * Shadow Fragment Types
    ShadowType(..)
  , ShadowFragment(..)
  , mkShadowFragment
  , shadowToNormal

    -- * Consent Extension
  , TherapeuticConsent(..)
  , ConsentOverride(..)
  , isTherapeuticSession
  , verifyConsentForShadow

    -- * Shadow Detection
  , detectShadowEmergence
  , shadowIndicators
  , emotionalCharge
  , harmonicMismatch

    -- * Safety Gating
  , SafetyGate(..)
  , GateResult(..)
  , evaluateSafetyGate
  , coherenceSafeAccess
  , emergencyOverrideCheck

    -- * Therapeutic Feedback
  , TherapeuticFeedback(..)
  , generateFeedback
  , integrationProgress
  , resonanceMatchDelta

    -- * Shadow Integration
  , IntegrationState(..)
  , ShadowSession(..)
  , initShadowSession
  , updateShadowSession
  , recordShadowInteraction

    -- * Constants
  , coherenceSafeAccessThreshold
  , shadowEmergenceFloor
  , maxChargeRate
  ) where

import Data.Time (UTCTime, getCurrentTime)

import Ra.Constants.Extended
  ( coherenceFloorPOR, phi )

-- =============================================================================
-- Shadow Fragment Types
-- =============================================================================

-- | Type of shadow fragment
data ShadowType
  = Mirror     -- ^ Reflected/opposite of normal fragment
  | Obscured   -- ^ Partially visible, details hidden
  | Inverted   -- ^ Fully inverted field polarity
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Shadow fragment with metadata
data ShadowFragment = ShadowFragment
  { sfFragmentId     :: !String        -- ^ Fragment identifier
  , sfShadowType     :: !ShadowType    -- ^ Type of shadow
  , sfAlpha          :: !Double        -- ^ Emergence intensity [0,1]
  , sfConsentState   :: !ConsentState  -- ^ Current consent
  , sfOriginFragment :: !(Maybe String) -- ^ Normal version this was inverted from
  , sfHarmonicDelta  :: !Double        -- ^ Mismatch from core field
  , sfEmotionalCharge :: !Double       -- ^ Inferred charge level [0,1]
  , sfRequiresOverride :: !Bool        -- ^ Needs explicit override
  } deriving (Eq, Show)

-- | Consent state for shadow work
data ConsentState
  = FullConsent
  | DiminishedConsent
  | SuspendedConsent
  | InTherapy  -- ^ Special state for shadow work
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create shadow fragment from parameters
mkShadowFragment :: String -> ShadowType -> Double -> Maybe String -> ShadowFragment
mkShadowFragment fid stype charge origin = ShadowFragment
  { sfFragmentId = fid ++ "-Shadow"
  , sfShadowType = stype
  , sfAlpha = 0.0  -- Starts suppressed
  , sfConsentState = SuspendedConsent
  , sfOriginFragment = origin
  , sfHarmonicDelta = charge * 0.5  -- Charge affects mismatch
  , sfEmotionalCharge = clamp01 charge
  , sfRequiresOverride = charge > 0.7  -- High charge needs override
  }

-- | Convert shadow to normal fragment (after integration)
shadowToNormal :: ShadowFragment -> Maybe String
shadowToNormal sf =
  if sfAlpha sf > 0.9 && sfEmotionalCharge sf < 0.2
  then sfOriginFragment sf
  else Nothing

-- =============================================================================
-- Consent Extension
-- =============================================================================

-- | Therapeutic consent with operator verification
data TherapeuticConsent = TherapeuticConsent
  { tcIsActive        :: !Bool    -- ^ Therapeutic mode active
  , tcLicensedOperator :: !Bool   -- ^ Session guided by licensed operator
  , tcCoherence       :: !Double  -- ^ Current field coherence
  , tcTimestamp       :: !UTCTime -- ^ When consent was established
  } deriving (Eq, Show)

-- | Override for shadow access
data ConsentOverride = ConsentOverride
  { coSource    :: !String   -- ^ Who approved (e.g., "avatar_therapist")
  , coReason    :: !String   -- ^ Why override was granted
  , coTimestamp :: !UTCTime  -- ^ When approved
  , coExpiry    :: !(Maybe UTCTime) -- ^ When override expires
  } deriving (Eq, Show)

-- | Check if session is in therapeutic mode
isTherapeuticSession :: TherapeuticConsent -> Bool
isTherapeuticSession tc =
  tcIsActive tc && tcLicensedOperator tc && tcCoherence tc >= coherenceSafeAccessThreshold

-- | Verify consent allows shadow access
verifyConsentForShadow :: TherapeuticConsent -> Maybe ConsentOverride -> ShadowFragment -> Bool
verifyConsentForShadow tc mOverride sf =
  let baseConsent = isTherapeuticSession tc

      -- High-charge fragments need override
      needsOverride = sfRequiresOverride sf
      hasOverride = case mOverride of
        Nothing -> False
        Just _ -> True

      -- Check coherence is sufficient
      coherenceOK = tcCoherence tc >= coherenceSafeAccessThreshold
  in baseConsent && coherenceOK && (not needsOverride || hasOverride)

-- =============================================================================
-- Shadow Detection
-- =============================================================================

-- | Detect if a fragment would emerge as shadow
detectShadowEmergence :: Double -> Double -> Double -> Bool
detectShadowEmergence coherence harmonicMatch emotionalTension =
  let -- Low coherence with high tension suggests shadow
      shadowIndicator = emotionalTension - coherence
      -- Harmonic mismatch also indicates shadow
      mismatchIndicator = 1.0 - harmonicMatch
  in shadowIndicator > 0.3 || mismatchIndicator > 0.5

-- | Get shadow emergence indicators
shadowIndicators :: ShadowFragment -> (Double, Double, Double)
shadowIndicators sf =
  ( sfAlpha sf           -- Emergence intensity
  , sfHarmonicDelta sf   -- Field mismatch
  , sfEmotionalCharge sf -- Charge level
  )

-- | Get emotional charge level
emotionalCharge :: ShadowFragment -> Double
emotionalCharge = sfEmotionalCharge

-- | Get harmonic mismatch from core field
harmonicMismatch :: ShadowFragment -> Double
harmonicMismatch = sfHarmonicDelta

-- =============================================================================
-- Safety Gating
-- =============================================================================

-- | Safety gate configuration
data SafetyGate = SafetyGate
  { sgCoherenceThreshold :: !Double  -- ^ Minimum coherence
  , sgMaxChargeExposure  :: !Double  -- ^ Maximum charge to surface
  , sgRequiresOperator   :: !Bool    -- ^ Licensed operator required
  , sgEmergencyCheck     :: !Bool    -- ^ Enable emergency override detection
  } deriving (Eq, Show)

-- | Result of safety gate evaluation
data GateResult
  = GateOpen             -- ^ Safe to proceed
  | GatePartial !Double  -- ^ Partial access with intensity limit
  | GateClosed !String   -- ^ Blocked with reason
  | GateEmergency        -- ^ Emergency override triggered
  deriving (Eq, Show)

-- | Evaluate safety gate for shadow fragment
evaluateSafetyGate :: SafetyGate -> TherapeuticConsent -> ShadowFragment -> GateResult
evaluateSafetyGate gate tc sf
  -- Emergency check first
  | sgEmergencyCheck gate && sfEmotionalCharge sf > 0.95 =
      GateEmergency

  -- Coherence check
  | tcCoherence tc < sgCoherenceThreshold gate =
      GateClosed $ "Coherence below threshold: " ++
                   show (tcCoherence tc) ++ " < " ++
                   show (sgCoherenceThreshold gate)

  -- Operator check
  | sgRequiresOperator gate && not (tcLicensedOperator tc) =
      GateClosed "Licensed operator required for shadow work"

  -- Charge exposure check
  | sfEmotionalCharge sf > sgMaxChargeExposure gate =
      GatePartial (sgMaxChargeExposure gate / sfEmotionalCharge sf)

  -- All checks passed
  | otherwise = GateOpen

-- | Safe access coherence threshold
coherenceSafeAccess :: Double
coherenceSafeAccess = coherenceSafeAccessThreshold

-- | Check for emergency override conditions
emergencyOverrideCheck :: ShadowFragment -> Double -> Bool
emergencyOverrideCheck sf currentHRV =
  let -- HRV collapse
      hrvCollapse = currentHRV < 0.2
      -- Charge spike
      chargeSpike = sfEmotionalCharge sf > 0.9
  in hrvCollapse || chargeSpike

-- =============================================================================
-- Therapeutic Feedback
-- =============================================================================

-- | Feedback for therapeutic shadow work
data TherapeuticFeedback = TherapeuticFeedback
  { tfInstruction      :: !String   -- ^ Guidance text
  , tfCoherenceGuide   :: !String   -- ^ Breathing/focus instruction
  , tfIntegrationPrompt :: !(Maybe String) -- ^ Optional reflection prompt
  , tfSafetyNote       :: !(Maybe String)  -- ^ Safety warning if needed
  } deriving (Eq, Show)

-- | Generate therapeutic feedback
generateFeedback :: ShadowFragment -> Double -> TherapeuticFeedback
generateFeedback sf coherence =
  let charge = sfEmotionalCharge sf
      stype = sfShadowType sf

      -- Instruction based on shadow type
      instruction = case stype of
        Mirror -> "This memory may feel dissonant. Breathe slowly..."
        Obscured -> "Something is surfacing. Allow it to clarify..."
        Inverted -> "You may notice resistance. Stay present..."

      -- Coherence guidance
      guide
        | coherence < 0.5 = "Focus on your breath. Inhale for 4, hold for 4, exhale for 4."
        | coherence < 0.7 = "Good. Maintain this steady rhythm."
        | otherwise = "Your field is stable. You may explore deeper."

      -- Integration prompt for lower charge
      integrationPrompt
        | charge < 0.4 = Just "Would you like to reflect on this with your guide?"
        | otherwise = Nothing

      -- Safety note for high charge
      safetyNote
        | charge > 0.8 = Just "If overwhelmed, say 'pause' to suspend emergence."
        | otherwise = Nothing
  in TherapeuticFeedback
      { tfInstruction = instruction
      , tfCoherenceGuide = guide
      , tfIntegrationPrompt = integrationPrompt
      , tfSafetyNote = safetyNote
      }

-- | Calculate integration progress
integrationProgress :: ShadowFragment -> Double
integrationProgress sf =
  let alpha = sfAlpha sf
      chargeReduction = 1.0 - sfEmotionalCharge sf
  in (alpha + chargeReduction) / 2

-- | Calculate resonance match improvement
resonanceMatchDelta :: Double -> Double -> Double
resonanceMatchDelta prev curr = curr - prev

-- =============================================================================
-- Shadow Integration Session
-- =============================================================================

-- | Integration state tracking
data IntegrationState = IntegrationState
  { isProgress       :: !Double  -- ^ Integration percentage [0,1]
  , isResonanceMatch :: !Double  -- ^ Current resonance match [0,1]
  , isChargeRemaining :: !Double -- ^ Remaining emotional charge
  , isSessionCount   :: !Int     -- ^ Number of sessions
  } deriving (Eq, Show)

-- | Shadow work session
data ShadowSession = ShadowSession
  { ssFragment     :: !ShadowFragment
  , ssConsent      :: !TherapeuticConsent
  , ssOverride     :: !(Maybe ConsentOverride)
  , ssIntegration  :: !IntegrationState
  , ssInteractions :: ![(UTCTime, String)]  -- ^ Logged interactions
  , ssFeedback     :: !TherapeuticFeedback
  } deriving (Eq, Show)

-- | Initialize shadow session
initShadowSession :: ShadowFragment -> TherapeuticConsent -> Maybe ConsentOverride -> ShadowSession
initShadowSession sf tc mOverride =
  let initialState = IntegrationState
        { isProgress = 0.0
        , isResonanceMatch = 0.3  -- Start with some match
        , isChargeRemaining = sfEmotionalCharge sf
        , isSessionCount = 1
        }
      feedback = generateFeedback sf (tcCoherence tc)
  in ShadowSession
      { ssFragment = sf
      , ssConsent = tc
      , ssOverride = mOverride
      , ssIntegration = initialState
      , ssInteractions = []
      , ssFeedback = feedback
      }

-- | Update shadow session with new coherence reading
updateShadowSession :: Double -> Double -> ShadowSession -> ShadowSession
updateShadowSession newCoherence dt session =
  let sf = ssFragment session
      integration = ssIntegration session

      -- Update progress based on coherence
      progressDelta = if newCoherence > coherenceSafeAccessThreshold
                      then 0.01 * dt  -- Progress when coherent
                      else 0.0
      newProgress = min 1.0 (isProgress integration + progressDelta)

      -- Charge reduces with progress
      chargeReduction = progressDelta * phi  -- Golden reduction
      newCharge = max 0.0 (isChargeRemaining integration - chargeReduction)

      -- Resonance improves with progress
      newResonance = min 1.0 (isResonanceMatch integration + progressDelta * 0.5)

      -- Update fragment alpha
      newAlpha = min 1.0 (sfAlpha sf + progressDelta)
      newFragment = sf { sfAlpha = newAlpha, sfEmotionalCharge = newCharge }

      -- Update integration state
      newIntegration = integration
        { isProgress = newProgress
        , isResonanceMatch = newResonance
        , isChargeRemaining = newCharge
        }

      -- Update consent coherence
      newConsent = (ssConsent session) { tcCoherence = newCoherence }

      -- Generate new feedback
      newFeedback = generateFeedback newFragment newCoherence
  in session
      { ssFragment = newFragment
      , ssConsent = newConsent
      , ssIntegration = newIntegration
      , ssFeedback = newFeedback
      }

-- | Record interaction for session log
recordShadowInteraction :: String -> ShadowSession -> IO ShadowSession
recordShadowInteraction description session = do
  now <- getCurrentTime
  let newInteractions = (now, description) : ssInteractions session
  return $ session { ssInteractions = newInteractions }

-- =============================================================================
-- Constants
-- =============================================================================

-- | Minimum coherence for safe shadow access
coherenceSafeAccessThreshold :: Double
coherenceSafeAccessThreshold = 0.66

-- | Minimum emergence level for shadow to surface
shadowEmergenceFloor :: Double
shadowEmergenceFloor = coherenceFloorPOR  -- 0.618

-- | Maximum charge discharge rate per second
maxChargeRate :: Double
maxChargeRate = 0.05  -- 5% per second max

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
