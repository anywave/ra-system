{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

-- | Prompt 16: Quantum Radionics Overlay for Scalar Pathway Tuning
-- FPGA module for real-time overlay modulation, rate encoding,
-- and intention-guided scalar emergence.
--
-- Codex References:
-- - Ra.Gates: Consent gating
-- - Ra.Emergence: Fragment emergence modulation
-- - Prompt 12: ShadowConsent integration
-- - Prompt 15: ConsentState integration
--
-- Features:
-- - Hybrid rate encoding (l, m, shell + decimal skew)
-- - Radius-scaled mode modifiers
-- - Overlay stacking with weighted modulation
-- - Consent gate integration

module RaOverlay where

import Clash.Prelude
import GHC.Generics (Generic)

-- ============================================================================
-- Constants (Codex-aligned)
-- ============================================================================

-- | Golden ratio (Fixed 4.12 representation)
phiConst :: SFixed 4 12
phiConst = 1.618

-- | Minimum coherence for partial overlay (0.4 * 4096)
coherenceMinPartial :: Unsigned 12
coherenceMinPartial = 1638  -- 0.4

-- | Maximum phi_boost after stacking (0.5 * 4096)
phiBoostMax :: Unsigned 12
phiBoostMax = 2048  -- 0.5

-- | Fixed epsilon for division safety
epsilon :: SFixed 4 12
epsilon = 0.000244140625  -- 2^-12

-- ============================================================================
-- Types
-- ============================================================================

-- | Overlay mode
data OverlayMode
  = ModeConstructive  -- phi potential boost
  | ModeDiffusive     -- fragment soft dispersion
  | ModeDirective     -- targeted emergence vector bias
  deriving (Generic, NFDataX, Eq, Show)

-- | Consent state (unified P12 + P15)
data ConsentState
  = ConsentNone
  | ConsentPrivate
  | ConsentTherapeutic
  | ConsentEntangled
  | ConsentWithdrawn
  | ConsentEmergency
  deriving (Generic, NFDataX, Eq, Show)

-- | Harmonic indices from rate encoding
data HarmonicIndices = HarmonicIndices
  { hiL     :: Unsigned 4     -- Primary harmonic (0-9)
  , hiM     :: Unsigned 4     -- Secondary harmonic (0-9)
  , hiShell :: Unsigned 4     -- Shell zone (0-9)
  , hiSkew  :: SFixed 4 12    -- Decimal skew
  } deriving (Generic, NFDataX, Show)

-- | Ra coordinate in normalized space
data RaCoord = RaCoord
  { rcX :: SFixed 4 12        -- X position (0.0-1.0)
  , rcY :: SFixed 4 12        -- Y position (0.0-1.0)
  } deriving (Generic, NFDataX, Show)

-- | Overlay descriptor
data Overlay = Overlay
  { ovRateInt   :: Unsigned 16    -- Integer part of rate
  , ovRateFrac  :: Unsigned 8     -- Fractional part * 100
  , ovRadius    :: SFixed 4 12    -- Radius in normalized space
  , ovMode      :: OverlayMode    -- Modulation mode
  , ovCenterX   :: SFixed 4 12    -- Center X
  , ovCenterY   :: SFixed 4 12    -- Center Y
  , ovCouplingL :: Unsigned 4     -- Harmonic coupling L
  , ovCouplingM :: Unsigned 4     -- Harmonic coupling M
  , ovActive    :: Bool           -- Is overlay active
  } deriving (Generic, NFDataX, Show)

-- | User state for consent gating
data UserState = UserState
  { usCoherence :: Unsigned 12    -- Coherence (0-4095 maps to 0.0-1.0)
  , usConsent   :: ConsentState   -- Current consent state
  } deriving (Generic, NFDataX, Show)

-- | Field modifiers (output)
data Modifiers = Modifiers
  { modPhiBoost   :: Unsigned 12  -- phi boost (0-4095)
  , modFlux       :: Unsigned 12  -- flux stabilization
  , modAnkhShift  :: Signed 2     -- ankh shift (-1, 0, +1)
  , modModulated  :: Bool         -- was modulation applied
  , modVetoed     :: Bool         -- was overlay vetoed by consent
  , modPartial    :: Bool         -- was partial application
  } deriving (Generic, NFDataX, Show)

-- | Overlay input bundle
data OverlayInput = OverlayInput
  { oiOverlay  :: Overlay
  , oiFragment :: RaCoord
  , oiUser     :: UserState
  } deriving (Generic, NFDataX)

-- ============================================================================
-- Rate Encoding
-- ============================================================================

-- | Encode rate into harmonic indices
-- Rate format: XXXY.ZZ where X=l, Y=m, Z=shell, .ZZ=skew
encodeRate :: Unsigned 16 -> Unsigned 8 -> HarmonicIndices
encodeRate rateInt rateFrac =
  let
    -- Extract digits from integer part
    l = resize ((rateInt `div` 100) `mod` 10) :: Unsigned 4
    m = resize ((rateInt `div` 10) `mod` 10) :: Unsigned 4
    shell = resize (rateInt `mod` 10) :: Unsigned 4
    -- Convert fractional to skew
    skew = (fromIntegral rateFrac :: SFixed 4 12) * 0.001
  in HarmonicIndices l m shell skew

-- ============================================================================
-- Distance Calculation
-- ============================================================================

-- | Calculate squared Euclidean distance
distanceSquared :: RaCoord -> RaCoord -> SFixed 4 12
distanceSquared (RaCoord x1 y1) (RaCoord x2 y2) =
  let dx = x1 - x2
      dy = y1 - y2
  in dx * dx + dy * dy

-- | Check if fragment is within overlay radius
withinRadius :: RaCoord -> Overlay -> Bool
withinRadius frag Overlay{..} =
  let center = RaCoord ovCenterX ovCenterY
      distSq = distanceSquared frag center
      radSq = ovRadius * ovRadius
  in distSq <= radSq

-- ============================================================================
-- Consent Gate
-- ============================================================================

-- | Check if consent allows overlay application
consentAllows :: ConsentState -> Bool
consentAllows ConsentTherapeutic = True
consentAllows ConsentEntangled   = True
consentAllows _                  = False

-- | Check consent gate
checkConsentGate :: UserState -> (Bool, Bool)  -- (allowed, partial)
checkConsentGate UserState{..} =
  let
    consentOk = consentAllows usConsent
    coherenceOk = usCoherence >= coherenceMinPartial
    partial = usCoherence < 2458  -- 0.6 * 4096
  in (consentOk && coherenceOk, partial && consentOk && coherenceOk)

-- ============================================================================
-- Mode Modifiers
-- ============================================================================

-- | Get base modifiers for each mode
-- Returns (phi_boost_base, flux_base, ankh)
getBaseModifiers :: OverlayMode -> (Unsigned 12, Unsigned 12, Signed 2)
getBaseModifiers ModeConstructive = (614, 205, 1)   -- (0.15, 0.05) * 4096
getBaseModifiers ModeDiffusive    = (205, 410, -1)  -- (0.05, 0.10) * 4096
getBaseModifiers ModeDirective    = (410, 82, 1)    -- (0.10, 0.02) * 4096

-- | Scale modifiers by radius and rate
scaleModifiers :: Unsigned 12 -> Unsigned 12 -> SFixed 4 12 -> HarmonicIndices
               -> (Unsigned 12, Unsigned 12)
scaleModifiers basePhi baseFlux radius HarmonicIndices{..} =
  let
    -- Scale factor: 1.0 - (radius * 0.5)
    -- At radius=1.0: scale=0.5, at radius=0.0: scale=1.0
    scale = 1.0 - (radius * 0.5)
    rateFactor = 1.0 + hiSkew

    -- Apply scaling
    scaledPhi = truncateB $ (fromIntegral basePhi :: SFixed 8 12) * scale * rateFactor
    scaledFlux = truncateB $ (fromIntegral baseFlux :: SFixed 8 12) * scale * rateFactor
  in (scaledPhi, scaledFlux)

-- ============================================================================
-- Overlay Application
-- ============================================================================

-- | Apply single overlay to fragment
applyOverlay :: OverlayInput -> Modifiers
applyOverlay OverlayInput{..} =
  let
    -- Check intersection
    inRadius = withinRadius oiFragment oiOverlay

    -- Check consent
    (allowed, partial) = checkConsentGate oiUser

    -- Get modifiers
    (basePhi, baseFlux, ankh) = getBaseModifiers (ovMode oiOverlay)
    indices = encodeRate (ovRateInt oiOverlay) (ovRateFrac oiOverlay)
    (scaledPhi, scaledFlux) = scaleModifiers basePhi baseFlux (ovRadius oiOverlay) indices

    -- Apply partial reduction if needed
    finalPhi = if partial then scaledPhi `shiftR` 1 else scaledPhi
    finalFlux = if partial then scaledFlux `shiftR` 1 else scaledFlux

    -- Determine output
    shouldApply = ovActive oiOverlay && inRadius && allowed
  in
    if shouldApply
    then Modifiers
      { modPhiBoost  = min finalPhi phiBoostMax
      , modFlux      = finalFlux
      , modAnkhShift = ankh
      , modModulated = True
      , modVetoed    = False
      , modPartial   = partial
      }
    else Modifiers
      { modPhiBoost  = 0
      , modFlux      = 0
      , modAnkhShift = 0
      , modModulated = False
      , modVetoed    = not allowed && inRadius && ovActive oiOverlay
      , modPartial   = False
      }

-- ============================================================================
-- Multi-Overlay Stacking (4 concurrent)
-- ============================================================================

-- | Stack up to 4 overlays with weighted modulation
stackOverlays :: Vec 4 Overlay -> RaCoord -> UserState -> Modifiers
stackOverlays overlays frag user =
  let
    -- Apply each overlay
    inputs = map (\ov -> OverlayInput ov frag user) overlays
    results = map applyOverlay inputs

    -- Sum phi_boost and flux
    totalPhi = fold (+) $ map modPhiBoost results
    totalFlux = fold (+) $ map modFlux results

    -- Vote on ankh shift
    ankhVotes = map modAnkhShift results
    posVotes = length $ filter (> 0) $ toList ankhVotes
    negVotes = length $ filter (< 0) $ toList ankhVotes
    finalAnkh = if posVotes > negVotes then 1
                else if negVotes > posVotes then -1
                else 0

    -- Any modulated?
    anyMod = or $ map modModulated results
  in Modifiers
    { modPhiBoost  = min totalPhi phiBoostMax
    , modFlux      = totalFlux
    , modAnkhShift = finalAnkh
    , modModulated = anyMod
    , modVetoed    = False
    , modPartial   = False
    }

-- ============================================================================
-- Top-Level Entities
-- ============================================================================

-- | Single overlay FSM
overlayFSM :: HiddenClockResetEnable dom
           => Signal dom OverlayInput
           -> Signal dom Modifiers
overlayFSM = fmap applyOverlay

-- | Top entity for single overlay
{-# ANN singleOverlayTop
  (Synthesize
    { t_name   = "single_overlay_top"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "rate_int"
                 , PortName "rate_frac"
                 , PortName "radius"
                 , PortName "mode"
                 , PortName "center_x"
                 , PortName "center_y"
                 , PortName "coupling_l"
                 , PortName "coupling_m"
                 , PortName "active"
                 , PortName "frag_x"
                 , PortName "frag_y"
                 , PortName "user_coherence"
                 , PortName "user_consent"
                 ]
    , t_output = PortProduct "output"
                 [ PortName "phi_boost"
                 , PortName "flux"
                 , PortName "ankh_shift"
                 , PortName "modulated"
                 , PortName "vetoed"
                 , PortName "partial"
                 ]
    }) #-}
singleOverlayTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 16)   -- rate_int
  -> Signal System (Unsigned 8)    -- rate_frac
  -> Signal System (SFixed 4 12)   -- radius
  -> Signal System (Unsigned 2)    -- mode
  -> Signal System (SFixed 4 12)   -- center_x
  -> Signal System (SFixed 4 12)   -- center_y
  -> Signal System (Unsigned 4)    -- coupling_l
  -> Signal System (Unsigned 4)    -- coupling_m
  -> Signal System Bool            -- active
  -> Signal System (SFixed 4 12)   -- frag_x
  -> Signal System (SFixed 4 12)   -- frag_y
  -> Signal System (Unsigned 12)   -- user_coherence
  -> Signal System (Unsigned 3)    -- user_consent
  -> Signal System ( Unsigned 12   -- phi_boost
                   , Unsigned 12   -- flux
                   , Signed 2      -- ankh_shift
                   , Bool          -- modulated
                   , Bool          -- vetoed
                   , Bool          -- partial
                   )
singleOverlayTop clk rst en
                 rateInt rateFrac radius mode
                 centerX centerY coupL coupM active
                 fragX fragY
                 userCoh userCons =
  withClockResetEnable clk rst en $
    let
      -- Decode mode
      decMode m = case m of
        0 -> ModeConstructive
        1 -> ModeDiffusive
        _ -> ModeDirective

      -- Decode consent
      decConsent c = case c of
        0 -> ConsentNone
        1 -> ConsentPrivate
        2 -> ConsentTherapeutic
        3 -> ConsentEntangled
        4 -> ConsentWithdrawn
        _ -> ConsentEmergency

      -- Build overlay
      overlay = Overlay
        <$> rateInt <*> rateFrac <*> radius
        <*> fmap decMode mode
        <*> centerX <*> centerY
        <*> coupL <*> coupM <*> active

      -- Build fragment coord
      frag = RaCoord <$> fragX <*> fragY

      -- Build user state
      user = UserState <$> userCoh <*> fmap decConsent userCons

      -- Build input
      input = OverlayInput <$> overlay <*> frag <*> user

      -- Apply overlay
      output = overlayFSM input

      -- Extract output
      extractOut Modifiers{..} =
        (modPhiBoost, modFlux, modAnkhShift, modModulated, modVetoed, modPartial)
    in fmap extractOut output

-- ============================================================================
-- Visualization Color Encoder
-- ============================================================================

-- | Encode mode to RGB color
modeToColor :: OverlayMode -> (Unsigned 8, Unsigned 8, Unsigned 8)
modeToColor ModeConstructive = (255, 215, 0)    -- Gold
modeToColor ModeDiffusive    = (0, 255, 255)    -- Cyan
modeToColor ModeDirective    = (139, 0, 255)    -- Violet

-- ============================================================================
-- Testbench
-- ============================================================================

-- | Test overlay
testOverlay :: Overlay
testOverlay = Overlay
  { ovRateInt   = 568
  , ovRateFrac  = 12
  , ovRadius    = 0.15
  , ovMode      = ModeConstructive
  , ovCenterX   = 0.5
  , ovCenterY   = 0.5
  , ovCouplingL = 5
  , ovCouplingM = 6
  , ovActive    = True
  }

-- | Test fragment (inside radius)
testFragInside :: RaCoord
testFragInside = RaCoord 0.52 0.48

-- | Test fragment (outside radius)
testFragOutside :: RaCoord
testFragOutside = RaCoord 0.8 0.8

-- | Test user (allowed)
testUserOk :: UserState
testUserOk = UserState 3481 ConsentTherapeutic  -- 0.85 * 4096

-- | Test inputs
testInputs :: Vec 4 OverlayInput
testInputs =
     OverlayInput testOverlay testFragInside testUserOk
  :> OverlayInput testOverlay testFragOutside testUserOk
  :> OverlayInput testOverlay testFragInside (UserState 820 ConsentNone)  -- denied
  :> OverlayInput testOverlay testFragInside (UserState 2048 ConsentTherapeutic)  -- partial
  :> Nil

-- | Testbench entity
testBench :: Signal System Modifiers
testBench = withClockResetEnable systemClockGen systemResetGen enableGen $
  overlayFSM (fromList (toList testInputs))
