{-|
Module      : RaVisualizerShellHUD
Description : Scalar Field HUD Overlay Interface
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 63: Modular scalar field HUD overlay system for live avatar
resonance feedback with shell-text glyph output and streamable
JSON HUD packets.

Supports multiple view modes with static layer priorities.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaVisualizerShellHUD where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Default FPS
defaultFPS :: Unsigned 8
defaultFPS = 60

-- | HUD view modes
data HUDMode
  = Diagnostic
  | EmergenceTracking
  | ChamberTuning
  | PointerGuidance
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | HUD layer types
data HUDLayerType
  = FieldSliceLayer
  | AuraBandLayer
  | CoherenceGaugeLayer
  | FragmentGlowLayer
  | HarmonicCompassLayer
  | AnkhBalanceLayer
  | DomainPulseLayer
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Shell glyph symbol index
type GlyphSymbol = Unsigned 4

-- | Shell glyph palette (index to symbol mapping)
-- 0=◯ 1=◐ 2=● 3=△ 4=▽ 5=◆ 6=◎ 7=↻
-- 8=✸ 9=░ 10=▓ 11=◇ 12=⬡ 13=⬢ 14=⊕ 15=⊗

-- | Scalar slice
data ScalarSlice = ScalarSlice
  { ssTheta :: Unsigned 16   -- 0-32767 = 0-π
  , ssPhi   :: Unsigned 16   -- 0-65535 = 0-2π
  , ssValue :: Unsigned 8    -- 0-255 amplitude
  } deriving (Generic, NFDataX, Eq, Show)

-- | Aura pattern
data AuraPattern = AuraPattern
  { apRings      :: Vec 6 (Unsigned 8)  -- Ring intensities
  , apHue        :: Unsigned 16          -- 0-65535 = 0-360°
  , apSaturation :: Unsigned 8           -- 0-255
  } deriving (Generic, NFDataX, Eq, Show)

-- | Coherence envelope
data CoherenceEnvelope = CoherenceEnvelope
  { ceCurrent   :: Unsigned 8   -- Current coherence
  , ceMinBand   :: Unsigned 8   -- Threshold
  , ceMaxBand   :: Unsigned 8   -- Ceiling
  , ceStability :: Unsigned 8   -- Stability
  } deriving (Generic, NFDataX, Eq, Show)

-- | Glow anchor
data GlowAnchor = GlowAnchor
  { gaFragmentId :: Unsigned 32
  , gaIntensity  :: Unsigned 8
  , gaPosX       :: Signed 16
  , gaPosY       :: Signed 16
  , gaPosZ       :: Signed 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Domain pulse frame
data DomainPulseFrame = DomainPulseFrame
  { dpPulseRate :: Unsigned 16   -- Hz * 256
  , dpAmplitude :: Unsigned 8    -- 0-255
  , dpPhase     :: Unsigned 16   -- 0-65535 = 0-2π
  } deriving (Generic, NFDataX, Eq, Show)

-- | HUD layer
data HUDLayer = HUDLayer
  { hlType     :: HUDLayerType
  , hlPriority :: Unsigned 4
  , hlVisible  :: Bool
  -- Layer-specific data encoded as fixed fields
  , hlValue1   :: Unsigned 16
  , hlValue2   :: Unsigned 16
  , hlValue3   :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Avatar field frame
data AvatarFieldFrame = AvatarFieldFrame
  { affCoherence     :: Unsigned 8
  , affFlux          :: Signed 8
  , affTorsionBias   :: Signed 8
  , affAnkhDelta     :: Signed 8
  , affHarmonicDepth :: Unsigned 4
  , affPhasePhiN     :: Unsigned 8    -- 0-255 normalized
  , affCompassTheta  :: Unsigned 16
  , affCompassPhi    :: Unsigned 16
  , affAura          :: AuraPattern
  } deriving (Generic, NFDataX, Eq, Show)

-- | Shell glyph line
data GlyphLine = GlyphLine
  { glSymbols :: Vec 40 GlyphSymbol
  , glLength  :: Unsigned 6
  } deriving (Generic, NFDataX, Eq, Show)

-- | Shell glyph overlay
data ShellGlyphOverlay = ShellGlyphOverlay
  { sgoLines     :: Vec 10 GlyphLine
  , sgoNumLines  :: Unsigned 4
  , sgoWidth     :: Unsigned 6
  , sgoHeight    :: Unsigned 4
  , sgoAnsiEnabled :: Bool
  } deriving (Generic, NFDataX, Eq, Show)

-- | HUD packet
data HUDPacket = HUDPacket
  { hpTimestamp    :: Unsigned 32
  , hpSessionId    :: Unsigned 32
  , hpMode         :: HUDMode
  , hpOverlay      :: ShellGlyphOverlay
  -- Scalar metrics
  , hpCoherence    :: Unsigned 8
  , hpFlux         :: Signed 8
  , hpAnkhDelta    :: Signed 8
  , hpTorsionBias  :: Signed 8
  , hpHarmonicDepth :: Unsigned 4
  , hpPhasePhiN    :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | HUD config
data HUDConfig = HUDConfig
  { hcMode                :: HUDMode
  , hcCoherenceThreshold  :: Unsigned 8
  , hcCompassSensitivity  :: Unsigned 8
  , hcFragmentGlowIntensity :: Unsigned 8
  , hcFPS                 :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Get layer priority for Diagnostic mode
diagnosticPriority :: Vec 4 HUDLayerType
diagnosticPriority = CoherenceGaugeLayer :> FieldSliceLayer :> DomainPulseLayer :> AuraBandLayer :> Nil

-- | Get layer priority for EmergenceTracking mode
emergencePriority :: Vec 4 HUDLayerType
emergencePriority = FragmentGlowLayer :> AuraBandLayer :> AnkhBalanceLayer :> CoherenceGaugeLayer :> Nil

-- | Get layer priority for ChamberTuning mode
chamberPriority :: Vec 4 HUDLayerType
chamberPriority = FieldSliceLayer :> HarmonicCompassLayer :> DomainPulseLayer :> AnkhBalanceLayer :> Nil

-- | Get layer priority for PointerGuidance mode
pointerPriority :: Vec 4 HUDLayerType
pointerPriority = HarmonicCompassLayer :> FragmentGlowLayer :> AuraBandLayer :> CoherenceGaugeLayer :> Nil

-- | Get layer priority for mode
getLayerPriority :: HUDMode -> Vec 4 HUDLayerType
getLayerPriority mode = case mode of
  Diagnostic       -> diagnosticPriority
  EmergenceTracking -> emergencePriority
  ChamberTuning    -> chamberPriority
  PointerGuidance  -> pointerPriority

-- | Default HUD config
defaultHUDConfig :: HUDMode -> HUDConfig
defaultHUDConfig mode = HUDConfig
  mode
  77     -- 0.3 * 255 coherence threshold
  26     -- 0.1 * 255 compass sensitivity
  255    -- Full fragment glow
  defaultFPS

-- | Map coherence to glyph symbol
coherenceToGlyph :: Unsigned 8 -> GlyphSymbol
coherenceToGlyph coh
  | coh < 51  = 0   -- ◯ very low
  | coh < 102 = 1   -- ◐ low
  | coh < 153 = 2   -- ● mid
  | coh < 204 = 6   -- ◎ high
  | otherwise = 8   -- ✸ very high

-- | Create empty glyph line
emptyGlyphLine :: GlyphLine
emptyGlyphLine = GlyphLine (repeat 0) 0

-- | Create layer from frame data
createLayer :: HUDLayerType -> AvatarFieldFrame -> Unsigned 4 -> HUDLayer
createLayer layerType frame priority = case layerType of
  CoherenceGaugeLayer ->
    HUDLayer layerType priority True
      (resize (affCoherence frame)) 0 (affCoherence frame)

  FieldSliceLayer ->
    HUDLayer layerType priority True
      (affCompassTheta frame) (affCompassPhi frame) (affCoherence frame)

  AuraBandLayer ->
    let aura = affAura frame
        avgRing = foldl (+) 0 (apRings aura) `div` 6
    in HUDLayer layerType priority True
         (apHue aura) (resize (apSaturation aura)) (resize avgRing)

  FragmentGlowLayer ->
    HUDLayer layerType priority True
      0 0 (affCoherence frame)

  HarmonicCompassLayer ->
    HUDLayer layerType priority True
      (affCompassTheta frame) (affCompassPhi frame) (resize (affHarmonicDepth frame))

  AnkhBalanceLayer ->
    HUDLayer layerType priority True
      (resize (pack (affAnkhDelta frame))) 0 (resize (pack (affAnkhDelta frame)))

  DomainPulseLayer ->
    HUDLayer layerType priority True
      (resize (affPhasePhiN frame)) (resize (pack (affFlux frame))) (affPhasePhiN frame)

-- | Render HUD layers
renderHUD :: HUDMode -> AvatarFieldFrame -> Vec 4 HUDLayer
renderHUD mode frame =
  let priorities = getLayerPriority mode
      makeLayers = \i t -> createLayer t frame i
  in imap makeLayers priorities

-- | Render layer to glyph line
renderLayerToGlyph :: HUDLayer -> GlyphLine
renderLayerToGlyph layer = case hlType layer of
  CoherenceGaugeLayer ->
    let coh = hlValue3 layer
        filled = resize coh `shiftR` 3 :: Unsigned 6  -- 0-31 chars
        symbols = map (\i -> if i < filled then 10 else 9) $(listToVecTH [0..39 :: Unsigned 6])
    in GlyphLine (map resize symbols) 40

  FragmentGlowLayer ->
    let glyph = coherenceToGlyph (hlValue3 layer)
        symbols = map (\i -> if i < 8 then glyph else 0) $(listToVecTH [0..39 :: Unsigned 6])
    in GlyphLine (map resize symbols) 8

  HarmonicCompassLayer ->
    let theta = hlValue1 layer `shiftR` 12 :: Unsigned 4
        phi = hlValue2 layer `shiftR` 12 :: Unsigned 4
        symbols = map (\_ -> 7) $(listToVecTH [0..39 :: Unsigned 6])  -- ↻
    in GlyphLine (map resize symbols) 10

  AnkhBalanceLayer ->
    let delta = unpack (resize (hlValue1 layer) :: Unsigned 8) :: Signed 8
        pos = resize ((resize delta + 128) `shiftR` 4) :: Unsigned 4
        symbols = map (\i -> if resize i == pos then 5 else 11) $(listToVecTH [0..39 :: Unsigned 6])
    in GlyphLine (map resize symbols) 12

  AuraBandLayer ->
    let intensity = hlValue3 layer
        glyph = coherenceToGlyph intensity
        symbols = map (\i -> if i < 6 then glyph else 0) $(listToVecTH [0..39 :: Unsigned 6])
    in GlyphLine (map resize symbols) 6

  FieldSliceLayer ->
    let coh = hlValue3 layer
        glyph = coherenceToGlyph coh
        symbols = map (\_ -> glyph) $(listToVecTH [0..39 :: Unsigned 6])
    in GlyphLine (map resize symbols) 20

  DomainPulseLayer ->
    let phase = hlValue3 layer
        glyph = if phase > 128 then 8 else 6
        symbols = map (\_ -> glyph) $(listToVecTH [0..39 :: Unsigned 6])
    in GlyphLine (map resize symbols) 15

-- | Render to shell glyphs
renderToShellGlyphs :: Vec 4 HUDLayer -> ShellGlyphOverlay
renderToShellGlyphs layers =
  let lines4 = map renderLayerToGlyph layers
      -- Pad to 10 lines
      emptyLines = repeat emptyGlyphLine :: Vec 6 GlyphLine
      allLines = lines4 ++ emptyLines
  in ShellGlyphOverlay allLines 4 40 10 True

-- | Compute ring expansion speed
computeRingExpansionSpeed :: Unsigned 8 -> Unsigned 16 -> Unsigned 16
computeRingExpansionSpeed phasePhiN baseSpeed =
  let -- Normalized phase (phi^n mod 1) approximation
      normalizedPhase = resize phasePhiN :: Unsigned 16
  in (baseSpeed * normalizedPhase) `shiftR` 8

-- | Create HUD packet
createHUDPacket
  :: Unsigned 32    -- Timestamp
  -> Unsigned 32    -- Session ID
  -> HUDMode
  -> AvatarFieldFrame
  -> HUDPacket
createHUDPacket ts sessionId mode frame =
  let layers = renderHUD mode frame
      overlay = renderToShellGlyphs layers
  in HUDPacket
       ts
       sessionId
       mode
       overlay
       (affCoherence frame)
       (affFlux frame)
       (affAnkhDelta frame)
       (affTorsionBias frame)
       (affHarmonicDepth frame)
       (affPhasePhiN frame)

-- | HUD state
data HUDState = HUDState
  { hsConfig    :: HUDConfig
  , hsTimestamp :: Unsigned 32
  } deriving (Generic, NFDataX)

-- | Initial HUD state
initialHUDState :: HUDState
initialHUDState = HUDState (defaultHUDConfig Diagnostic) 0

-- | HUD input
data HUDInput = HUDInput
  { hiSessionId :: Unsigned 32
  , hiMode      :: HUDMode
  , hiFrame     :: AvatarFieldFrame
  } deriving (Generic, NFDataX)

-- | HUD pipeline
hudPipeline
  :: HiddenClockResetEnable dom
  => Signal dom HUDInput
  -> Signal dom HUDPacket
hudPipeline input = mealy hudMealy initialHUDState input
  where
    hudMealy state inp =
      let ts = hsTimestamp state + 1
          packet = createHUDPacket ts (hiSessionId inp) (hiMode inp) (hiFrame inp)
          newState = state { hsTimestamp = ts }
      in (newState, packet)

-- | Layer rendering pipeline
layerPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (HUDMode, AvatarFieldFrame)
  -> Signal dom (Vec 4 HUDLayer)
layerPipeline input = uncurry renderHUD <$> input
