{-|
Module      : Ra.Visualizer.ShellHUD
Description : Scalar field HUD overlay interface
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Visualizes AvatarFieldFrame data in layered, animated 2D/3D UI overlays for
scalar navigation, emergence feedback, and coherence state awareness.

== HUD Theory

The ShellHUD projects live, navigable scalar data based on avatar resonance:

* Field cross-section visualizations (θ-φ slices, radial shells)
* Aura band readouts
* Coherence gauges
* Fragment glow anchors
* Harmonic flux vector compass

Users (or their AI proxies) can see, feel, and interact with their scalar
resonance state in real time.
-}
module Ra.Visualizer.ShellHUD
  ( -- * Core Types
    HUDMode(..)
  , HUDLayer(..)
  , HUDPacket(..)
  , HUDConfig(..)
  , ShellGlyphOverlay(..)

    -- * HUD Rendering
  , renderHUD
  , renderToShellGlyphs
  , renderLayer

    -- * Layer Types
  , CoherenceEnvelope(..)
  , DomainPulseFrame(..)

    -- * Configuration
  , defaultHUDConfig
  , diagnosticConfig
  , tuningConfig
  , emergenceConfig

    -- * Packet Management
  , mkHUDPacket
  , updatePacket
  , packetMetrics

    -- * Glyph Symbols
  , glyphSymbols
  , symbolForCoherence
  , symbolForAura
  , symbolForFragment

    -- * Layer Visibility
  , LayerVisibility(..)
  , visibleLayers
  , filterLayers
  ) where

import Data.Time (UTCTime, getCurrentTime)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Text (Text)
import qualified Data.Text as T

-- =============================================================================
-- Core Types
-- =============================================================================

-- | HUD rendering mode
data HUDMode
  = Diagnostic         -- ^ Shows coherence, aura bands, raw flux
  | ChamberTuning      -- ^ Focuses on harmonics and phase windows
  | EmergenceTracking  -- ^ Emphasizes fragment anchors and glow states
  | PointerGuidance    -- ^ Highlights vector appendage targetings
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | HUD layer types
data HUDLayer
  = FieldSliceLayer ![ScalarSlice]
  | AuraBandLayer !AuraPattern
  | CoherenceGaugeLayer !CoherenceEnvelope
  | FragmentGlowLayer ![GlowAnchor]
  | HarmonicCompassLayer !(Double, Double)  -- ^ (theta, phi)
  | AnkhBalanceOverlay !Double              -- ^ Δ(ankh) score
  | DomainPulseMap !DomainPulseFrame
  deriving (Eq, Show)

-- | Coherence envelope for gauge display
data CoherenceEnvelope = CoherenceEnvelope
  { ceValue     :: !Double        -- ^ Current coherence [0, 1]
  , ceMin       :: !Double        -- ^ Recent minimum
  , ceMax       :: !Double        -- ^ Recent maximum
  , ceAverage   :: !Double        -- ^ Running average
  , ceTrend     :: !CoherenceTrend
  } deriving (Eq, Show)

-- | Coherence trend indicator
data CoherenceTrend
  = TrendRising
  | TrendFalling
  | TrendStable
  | TrendFluctuating
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Domain pulse frame for flow visualization
data DomainPulseFrame = DomainPulseFrame
  { dpfSourceDomain :: !Text
  , dpfTargetDomain :: !Text
  , dpfFlowRate     :: !Double
  , dpfPhase        :: !Double
  } deriving (Eq, Show)

-- | Scalar slice (simplified import)
data ScalarSlice = ScalarSlice
  { ssData   :: ![[Double]]
  , ssCenter :: !(Double, Double, Double)
  , ssRadius :: !Double
  } deriving (Eq, Show)

-- | Aura pattern (simplified import)
data AuraPattern = AuraPattern
  { apBands     :: ![AuraBand]
  , apIntensity :: !Double
  } deriving (Eq, Show)

-- | Aura band
data AuraBand = AuraBand
  { abRadius    :: !Double
  , abIntensity :: !Double
  , abHue       :: !Double
  } deriving (Eq, Show)

-- | Glow anchor (simplified import)
data GlowAnchor = GlowAnchor
  { gaPosition   :: !(Double, Double, Double)
  , gaIntensity  :: !Double
  , gaFragmentId :: !(Maybe String)
  } deriving (Eq, Show)

-- | Session identifier
type SessionID = Text

-- | HUD packet for external streaming
data HUDPacket = HUDPacket
  { hpTimestamp    :: !UTCTime
  , hpSessionId    :: !SessionID
  , hpGlyphLayers  :: !ShellGlyphOverlay
  , hpHudMode      :: !HUDMode
  , hpScalarMetrics :: !(Map Text Double)
  } deriving (Eq, Show)

-- | Shell glyph overlay for text-based rendering
data ShellGlyphOverlay = ShellGlyphOverlay
  { sgoRows    :: ![[GlyphCell]]
  , sgoWidth   :: !Int
  , sgoHeight  :: !Int
  , sgoLegend  :: ![Text]
  } deriving (Eq, Show)

-- | Single glyph cell
data GlyphCell = GlyphCell
  { gcSymbol :: !Text
  , gcColor  :: !GlyphColor
  } deriving (Eq, Show)

-- | Glyph colors
data GlyphColor
  = ColorDefault
  | ColorRed
  | ColorGreen
  | ColorBlue
  | ColorYellow
  | ColorCyan
  | ColorMagenta
  | ColorWhite
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- HUD Configuration
-- =============================================================================

-- | HUD configuration
data HUDConfig = HUDConfig
  { hcMode           :: !HUDMode
  , hcLayerWeights   :: !(Map HUDLayerType Double)
  , hcRefreshRate    :: !Double              -- ^ Updates per second
  , hcGridSize       :: !(Int, Int)          -- ^ (width, height)
  , hcShowLegend     :: !Bool
  , hcAnimatePulse   :: !Bool
  } deriving (Eq, Show)

-- | Layer type for configuration
data HUDLayerType
  = LayerFieldSlice
  | LayerAuraBand
  | LayerCoherenceGauge
  | LayerFragmentGlow
  | LayerHarmonicCompass
  | LayerAnkhBalance
  | LayerDomainPulse
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Default HUD configuration
defaultHUDConfig :: HUDConfig
defaultHUDConfig = HUDConfig
  { hcMode = Diagnostic
  , hcLayerWeights = Map.fromList
      [ (LayerFieldSlice, 1.0)
      , (LayerAuraBand, 0.8)
      , (LayerCoherenceGauge, 1.0)
      , (LayerFragmentGlow, 0.9)
      , (LayerHarmonicCompass, 0.7)
      , (LayerAnkhBalance, 0.6)
      , (LayerDomainPulse, 0.5)
      ]
  , hcRefreshRate = 30.0
  , hcGridSize = (80, 24)
  , hcShowLegend = True
  , hcAnimatePulse = True
  }

-- | Diagnostic mode configuration
diagnosticConfig :: HUDConfig
diagnosticConfig = defaultHUDConfig
  { hcMode = Diagnostic
  , hcLayerWeights = Map.fromList
      [ (LayerCoherenceGauge, 1.0)
      , (LayerAuraBand, 1.0)
      , (LayerFieldSlice, 0.8)
      , (LayerFragmentGlow, 0.5)
      , (LayerHarmonicCompass, 0.5)
      , (LayerAnkhBalance, 0.8)
      , (LayerDomainPulse, 0.3)
      ]
  }

-- | Chamber tuning configuration
tuningConfig :: HUDConfig
tuningConfig = defaultHUDConfig
  { hcMode = ChamberTuning
  , hcLayerWeights = Map.fromList
      [ (LayerFieldSlice, 1.0)
      , (LayerHarmonicCompass, 1.0)
      , (LayerAuraBand, 0.7)
      , (LayerCoherenceGauge, 0.6)
      , (LayerFragmentGlow, 0.4)
      , (LayerAnkhBalance, 0.5)
      , (LayerDomainPulse, 0.8)
      ]
  }

-- | Emergence tracking configuration
emergenceConfig :: HUDConfig
emergenceConfig = defaultHUDConfig
  { hcMode = EmergenceTracking
  , hcLayerWeights = Map.fromList
      [ (LayerFragmentGlow, 1.0)
      , (LayerCoherenceGauge, 0.9)
      , (LayerAuraBand, 0.8)
      , (LayerAnkhBalance, 0.7)
      , (LayerFieldSlice, 0.5)
      , (LayerHarmonicCompass, 0.4)
      , (LayerDomainPulse, 0.6)
      ]
  }

-- =============================================================================
-- HUD Rendering
-- =============================================================================

-- | Render HUD from avatar field frame
renderHUD :: HUDMode -> AvatarFieldData -> [HUDLayer]
renderHUD mode fieldData =
  let coherenceLayer = CoherenceGaugeLayer (buildCoherenceEnvelope fieldData)
      auraLayer = AuraBandLayer (afdAura fieldData)
      sliceLayer = FieldSliceLayer (afdSlices fieldData)
      glowLayer = FragmentGlowLayer (afdGlows fieldData)
      compassLayer = HarmonicCompassLayer (afdMotion fieldData)
      ankhLayer = AnkhBalanceOverlay (afdAnkhDelta fieldData)
  in case mode of
    Diagnostic -> [coherenceLayer, auraLayer, sliceLayer, ankhLayer]
    ChamberTuning -> [sliceLayer, compassLayer, auraLayer]
    EmergenceTracking -> [glowLayer, coherenceLayer, auraLayer, ankhLayer]
    PointerGuidance -> [compassLayer, glowLayer, coherenceLayer]

-- | Avatar field data (simplified input structure)
data AvatarFieldData = AvatarFieldData
  { afdCoherence :: !Double
  , afdSlices    :: ![ScalarSlice]
  , afdAura      :: !AuraPattern
  , afdGlows     :: ![GlowAnchor]
  , afdMotion    :: !(Double, Double)
  , afdAnkhDelta :: !Double
  } deriving (Eq, Show)

-- | Build coherence envelope from field data
buildCoherenceEnvelope :: AvatarFieldData -> CoherenceEnvelope
buildCoherenceEnvelope fieldData = CoherenceEnvelope
  { ceValue = afdCoherence fieldData
  , ceMin = afdCoherence fieldData * 0.9
  , ceMax = min 1.0 (afdCoherence fieldData * 1.1)
  , ceAverage = afdCoherence fieldData
  , ceTrend = TrendStable
  }

-- | Render layers to shell glyph overlay
renderToShellGlyphs :: [HUDLayer] -> ShellGlyphOverlay
renderToShellGlyphs layers =
  let width = 80
      height = 24
      emptyGrid = replicate height (replicate width emptyCell)
      renderedGrid = foldr (overlayLayer width height) emptyGrid layers
      legend = buildLegend layers
  in ShellGlyphOverlay
    { sgoRows = renderedGrid
    , sgoWidth = width
    , sgoHeight = height
    , sgoLegend = legend
    }

-- | Empty glyph cell
emptyCell :: GlyphCell
emptyCell = GlyphCell " " ColorDefault

-- | Overlay a layer onto the grid
overlayLayer :: Int -> Int -> HUDLayer -> [[GlyphCell]] -> [[GlyphCell]]
overlayLayer width height layer grid = case layer of
  CoherenceGaugeLayer env ->
    overlayCoherenceGauge env width height grid

  AuraBandLayer aura ->
    overlayAuraBands aura width height grid

  FragmentGlowLayer glows ->
    overlayFragmentGlows glows width height grid

  HarmonicCompassLayer (theta, phi') ->
    overlayCompass theta phi' width height grid

  AnkhBalanceOverlay delta ->
    overlayAnkhBalance delta width height grid

  FieldSliceLayer _ -> grid  -- Complex rendering, simplified here
  DomainPulseMap _ -> grid   -- Complex rendering, simplified here

-- | Overlay coherence gauge
overlayCoherenceGauge :: CoherenceEnvelope -> Int -> Int -> [[GlyphCell]] -> [[GlyphCell]]
overlayCoherenceGauge env _width _height grid =
  let row = 0
      col = 0
      symbol = symbolForCoherence (ceValue env)
      color = colorForCoherence (ceValue env)
      label = "COH:" ++ show (round (ceValue env * 100) :: Int) ++ "%"
      cells = map (\c -> GlyphCell (T.singleton c) color) label ++ [GlyphCell symbol color]
  in updateRow row col cells grid

-- | Overlay aura bands
overlayAuraBands :: AuraPattern -> Int -> Int -> [[GlyphCell]] -> [[GlyphCell]]
overlayAuraBands aura _width _height grid =
  let row = 1
      col = 0
      symbols = map symbolForAura (apBands aura)
      label = "AURA:"
      cells = map (\c -> GlyphCell (T.singleton c) ColorCyan) label ++
              map (\s -> GlyphCell s ColorMagenta) symbols
  in updateRow row col cells grid

-- | Overlay fragment glows
overlayFragmentGlows :: [GlowAnchor] -> Int -> Int -> [[GlyphCell]] -> [[GlyphCell]]
overlayFragmentGlows glows width height grid =
  let glowCells = [(posToGrid ga width height, symbolForFragment (gaIntensity ga))
                  | ga <- glows]
  in foldr (\((r, c), sym) g -> setCell r c (GlyphCell sym ColorYellow) g) grid glowCells

-- | Overlay compass
overlayCompass :: Double -> Double -> Int -> Int -> [[GlyphCell]] -> [[GlyphCell]]
overlayCompass theta _phi _width _height grid =
  let row = 2
      col = 0
      arrow = compassArrow theta
      label = "DIR:"
      cells = map (\c -> GlyphCell (T.singleton c) ColorGreen) label ++
              [GlyphCell arrow ColorGreen]
  in updateRow row col cells grid

-- | Overlay ankh balance
overlayAnkhBalance :: Double -> Int -> Int -> [[GlyphCell]] -> [[GlyphCell]]
overlayAnkhBalance delta _width _height grid =
  let row = 3
      col = 0
      symbol = if abs delta < 0.1 then "◎" else if delta > 0 then "△" else "▽"
      color = if abs delta < 0.1 then ColorGreen else ColorRed
      label = "ANKH:" ++ show (round (delta * 100) :: Int)
      cells = map (\c -> GlyphCell (T.singleton c) color) label ++
              [GlyphCell symbol color]
  in updateRow row col cells grid

-- | Convert position to grid coordinates
posToGrid :: GlowAnchor -> Int -> Int -> (Int, Int)
posToGrid ga width height =
  let (x, y, _) = gaPosition ga
      row = height `div` 2 + round (y * fromIntegral height / 4)
      col = width `div` 2 + round (x * fromIntegral width / 4)
  in (clamp 0 (height - 1) row, clamp 0 (width - 1) col)

-- | Clamp value to range
clamp :: Int -> Int -> Int -> Int
clamp lo hi x = max lo (min hi x)

-- | Update row with cells
updateRow :: Int -> Int -> [GlyphCell] -> [[GlyphCell]] -> [[GlyphCell]]
updateRow row col cells grid
  | row < 0 || row >= length grid = grid
  | otherwise =
      case splitAt row grid of
        (before, targetRow:after) ->
          let updatedRow = updateCells col cells targetRow
          in before ++ [updatedRow] ++ after
        _ -> grid  -- Should not happen due to guard

-- | Update cells in a row
updateCells :: Int -> [GlyphCell] -> [GlyphCell] -> [GlyphCell]
updateCells col cells rowData =
  let (before, rest) = splitAt col rowData
      newCells = take (length cells) (cells ++ repeat emptyCell)
      after = drop (length cells) rest
  in before ++ newCells ++ after

-- | Set single cell
setCell :: Int -> Int -> GlyphCell -> [[GlyphCell]] -> [[GlyphCell]]
setCell row col cell grid
  | row < 0 || row >= length grid = grid
  | col < 0 || col >= length (grid !! row) = grid
  | otherwise =
      case splitAt row grid of
        (before, targetRow:after) ->
          case splitAt col targetRow of
            (colBefore, _:colAfter) ->
              let updatedRow = colBefore ++ [cell] ++ colAfter
              in before ++ [updatedRow] ++ after
            _ -> grid  -- Should not happen due to guard
        _ -> grid  -- Should not happen due to guard

-- | Render single layer
renderLayer :: HUDLayer -> Text
renderLayer layer = case layer of
  FieldSliceLayer slices -> T.pack $ "Field slices: " ++ show (length slices)
  AuraBandLayer aura -> T.pack $ "Aura intensity: " ++ show (apIntensity aura)
  CoherenceGaugeLayer env -> T.pack $ "Coherence: " ++ show (ceValue env)
  FragmentGlowLayer glows -> T.pack $ "Glows: " ++ show (length glows)
  HarmonicCompassLayer (t, p) -> T.pack $ "Compass: θ=" ++ show t ++ " φ=" ++ show p
  AnkhBalanceOverlay d -> T.pack $ "Ankh Δ: " ++ show d
  DomainPulseMap dpf -> T.pack $ "Domain pulse: " ++ T.unpack (dpfSourceDomain dpf)

-- | Build legend from layers
buildLegend :: [HUDLayer] -> [Text]
buildLegend layers = map layerLegend layers

-- | Layer legend entry
layerLegend :: HUDLayer -> Text
layerLegend layer = case layer of
  FieldSliceLayer _ -> "● Field Slice"
  AuraBandLayer _ -> "◐ Aura Band"
  CoherenceGaugeLayer _ -> "○ Coherence"
  FragmentGlowLayer _ -> "✸ Fragments"
  HarmonicCompassLayer _ -> "↻ Compass"
  AnkhBalanceOverlay _ -> "◎ Ankh Δ"
  DomainPulseMap _ -> "░ Domain"

-- =============================================================================
-- Glyph Symbols
-- =============================================================================

-- | Standard glyph symbols
glyphSymbols :: Map Text Text
glyphSymbols = Map.fromList
  [ ("coherence_peak", "●")
  , ("balanced", "◎")
  , ("half_aligned", "◐")
  , ("phase_loop", "↻")
  , ("high_ankh", "✸")
  , ("low_charge", "░")
  , ("high_charge", "▓")
  , ("fragment", "◆")
  , ("emergence", "○")
  ]

-- | Symbol for coherence level
symbolForCoherence :: Double -> Text
symbolForCoherence coh
  | coh >= 0.9 = "●"
  | coh >= 0.7 = "◐"
  | coh >= 0.5 = "○"
  | coh >= 0.3 = "◌"
  | otherwise = "·"

-- | Symbol for aura band
symbolForAura :: AuraBand -> Text
symbolForAura band
  | abIntensity band >= 0.8 = "▓"
  | abIntensity band >= 0.5 = "▒"
  | otherwise = "░"

-- | Symbol for fragment glow
symbolForFragment :: Double -> Text
symbolForFragment intensity
  | intensity >= 0.8 = "✸"
  | intensity >= 0.5 = "◆"
  | intensity >= 0.3 = "◇"
  | otherwise = "·"

-- | Color for coherence level
colorForCoherence :: Double -> GlyphColor
colorForCoherence coh
  | coh >= 0.8 = ColorGreen
  | coh >= 0.5 = ColorYellow
  | otherwise = ColorRed

-- | Compass arrow based on angle
compassArrow :: Double -> Text
compassArrow theta
  | theta < 0.39 = "→"
  | theta < 1.18 = "↗"
  | theta < 1.96 = "↑"
  | theta < 2.75 = "↖"
  | theta < 3.53 = "←"
  | theta < 4.32 = "↙"
  | theta < 5.11 = "↓"
  | theta < 5.89 = "↘"
  | otherwise = "→"

-- =============================================================================
-- Packet Management
-- =============================================================================

-- | Create HUD packet
mkHUDPacket :: SessionID -> HUDMode -> [HUDLayer] -> IO HUDPacket
mkHUDPacket sessionId mode layers = do
  now <- getCurrentTime
  let glyphs = renderToShellGlyphs layers
      metrics = computeMetrics layers
  return HUDPacket
    { hpTimestamp = now
    , hpSessionId = sessionId
    , hpGlyphLayers = glyphs
    , hpHudMode = mode
    , hpScalarMetrics = metrics
    }

-- | Update existing packet
updatePacket :: HUDPacket -> [HUDLayer] -> IO HUDPacket
updatePacket packet layers = do
  now <- getCurrentTime
  let glyphs = renderToShellGlyphs layers
      metrics = computeMetrics layers
  return packet
    { hpTimestamp = now
    , hpGlyphLayers = glyphs
    , hpScalarMetrics = metrics
    }

-- | Get packet metrics
packetMetrics :: HUDPacket -> Map Text Double
packetMetrics = hpScalarMetrics

-- | Compute metrics from layers
computeMetrics :: [HUDLayer] -> Map Text Double
computeMetrics layers = Map.fromList $ concatMap layerMetrics layers

-- | Extract metrics from layer
layerMetrics :: HUDLayer -> [(Text, Double)]
layerMetrics layer = case layer of
  CoherenceGaugeLayer env -> [("coherence", ceValue env)]
  AuraBandLayer aura -> [("aura_intensity", apIntensity aura)]
  FragmentGlowLayer glows -> [("fragment_count", fromIntegral (length glows))]
  AnkhBalanceOverlay delta -> [("ankh_delta", delta)]
  HarmonicCompassLayer (t, p) -> [("compass_theta", t), ("compass_phi", p)]
  _ -> []

-- =============================================================================
-- Layer Visibility
-- =============================================================================

-- | Layer visibility control
data LayerVisibility
  = Visible
  | Hidden
  | Faded
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Get visible layers based on mode
visibleLayers :: HUDMode -> [HUDLayerType]
visibleLayers mode = case mode of
  Diagnostic -> [LayerCoherenceGauge, LayerAuraBand, LayerFieldSlice, LayerAnkhBalance]
  ChamberTuning -> [LayerFieldSlice, LayerHarmonicCompass, LayerAuraBand, LayerDomainPulse]
  EmergenceTracking -> [LayerFragmentGlow, LayerCoherenceGauge, LayerAuraBand, LayerAnkhBalance]
  PointerGuidance -> [LayerHarmonicCompass, LayerFragmentGlow, LayerCoherenceGauge]

-- | Filter layers by visibility
filterLayers :: HUDMode -> [HUDLayer] -> [HUDLayer]
filterLayers mode layers =
  let visible = visibleLayers mode
      layerType layer = case layer of
        FieldSliceLayer _ -> LayerFieldSlice
        AuraBandLayer _ -> LayerAuraBand
        CoherenceGaugeLayer _ -> LayerCoherenceGauge
        FragmentGlowLayer _ -> LayerFragmentGlow
        HarmonicCompassLayer _ -> LayerHarmonicCompass
        AnkhBalanceOverlay _ -> LayerAnkhBalance
        DomainPulseMap _ -> LayerDomainPulse
  in filter (\l -> layerType l `elem` visible) layers
