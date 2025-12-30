{-|
Module      : Ra.Visualizer.Glyphs
Description : Harmonic signature renderer
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Converts chamber resonance states, avatar harmonic signatures, and fragment
emergence flows into visual glyphs suitable for screen rendering, symbolic
encoding, and AR overlays.

== Glyph Theory

Glyphs are visual representations of harmonic resonance, combining:

* Harmonic field structure (H_{l,m}, φ^n)
* Torsion phase and coherence gates
* Avatar signature overlays
* Fragment anchoring glyphs (emergence coordinates)

These glyphs serve as visual keys for:

* AVACHATTER interface
* Live biometric dashboards
* Chamber calibration visual feedback
* Field navigator tools
-}
module Ra.Visualizer.Glyphs
  ( -- * Core Types
    HarmonicGlyph(..)
  , GlyphID
  , GlyphImage(..)
  , GlowLevel(..)
  , TorsionPhase(..)

    -- * Glyph Generation
  , renderHarmonicGlyph
  , generateGlyphSVG
  , fragmentAnchorGlyph

    -- * Glyph Collections
  , GlyphSet(..)
  , buildGlyphSet
  , lookupGlyph
  , mergeGlyphSets

    -- * SVG Primitives
  , SVGElement(..)
  , svgCircle
  , svgSpiral
  , svgRing
  , svgPulse
  , svgGlow

    -- * SVG Rendering
  , renderSVGElement

    -- * Color Mapping
  , GlyphColor(..)
  , coherenceToColor
  , torsionToColor
  , harmonicToHue
  , colorToHex

    -- * Animation
  , GlyphAnimation(..)
  , animateGlyph
  , pulseAnimation
  , spiralAnimation

    -- * Text Rendering
  , glyphToText
  , glyphToUnicode
  , textGlyphOverlay
  ) where

import Data.ByteString (ByteString)
import qualified Data.ByteString.Char8 as BS
import Data.Time (UTCTime, getCurrentTime)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.List (intercalate)

import Ra.Omega (OmegaFormat(..))
-- Ra.Constants.Extended not needed for this module

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Glyph identifier
type GlyphID = String

-- | Fragment identifier
type FragmentID = String

-- | Torsion phase state
data TorsionPhase
  = TorsionNormal     -- ^ Standard field rotation
  | TorsionInverted   -- ^ Reversed field rotation
  | TorsionNull       -- ^ No rotation (neutral)
  | TorsionOscillating -- ^ Alternating rotation
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Glow intensity level
data GlowLevel
  = GlowNone
  | GlowDim
  | GlowMedium
  | GlowBright
  | GlowIntense
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Harmonic glyph representation
data HarmonicGlyph = HarmonicGlyph
  { glyphId        :: !GlyphID
  , phiPhase       :: !Int              -- ^ n from φ^n depth
  , harmonicRoot   :: !OmegaFormat
  , harmonicL      :: !Int              -- ^ Spherical harmonic degree
  , harmonicM      :: !Int              -- ^ Spherical harmonic order
  , torsionState   :: !TorsionPhase
  , coherenceGlow  :: !(Maybe GlowLevel)
  , associatedWith :: !(Maybe FragmentID)
  } deriving (Eq, Show)

-- | Rendered glyph image
data GlyphImage = GlyphImage
  { svgData    :: !ByteString
  , legend     :: ![String]
  , timestamp  :: !UTCTime
  , glyphMeta  :: !(Map String String)
  } deriving (Eq, Show)

-- =============================================================================
-- Scalar Layer Type (simplified for this module)
-- =============================================================================

-- | Scalar layer (simplified representation)
data ScalarLayer = ScalarLayer
  { slHarmonicL :: !Int
  , slHarmonicM :: !Int
  , slDepth     :: !Int
  , slCoherence :: !Double
  , slTorsion   :: !TorsionPhase
  } deriving (Eq, Show)

-- | Fragment metadata (simplified)
data FragmentMetadata = FragmentMetadata
  { fmFragmentId :: !FragmentID
  , fmHarmonicL  :: !Int
  , fmHarmonicM  :: !Int
  , fmCoherence  :: !Double
  , fmPhiDepth   :: !Int
  } deriving (Eq, Show)

-- =============================================================================
-- Glyph Generation
-- =============================================================================

-- | Render harmonic glyph from scalar layer
renderHarmonicGlyph :: ScalarLayer -> HarmonicGlyph
renderHarmonicGlyph layer = HarmonicGlyph
  { glyphId = "glyph-" ++ show (slHarmonicL layer) ++ "-" ++ show (slHarmonicM layer)
  , phiPhase = slDepth layer
  , harmonicRoot = depthToFormat (slDepth layer)
  , harmonicL = slHarmonicL layer
  , harmonicM = slHarmonicM layer
  , torsionState = slTorsion layer
  , coherenceGlow = Just (coherenceToGlow (slCoherence layer))
  , associatedWith = Nothing
  }

-- | Generate SVG glyph image
generateGlyphSVG :: HarmonicGlyph -> IO GlyphImage
generateGlyphSVG glyph = do
  now <- getCurrentTime
  let svg = buildSVG glyph
      legendItems = buildLegend glyph
  return GlyphImage
    { svgData = BS.pack svg
    , legend = legendItems
    , timestamp = now
    , glyphMeta = Map.fromList
        [ ("l", show (harmonicL glyph))
        , ("m", show (harmonicM glyph))
        , ("phi_n", show (phiPhase glyph))
        ]
    }

-- | Build SVG string for glyph
buildSVG :: HarmonicGlyph -> String
buildSVG glyph =
  let width = 200 :: Int
      height = 200 :: Int
      cx = 100 :: Int
      cy = 100 :: Int
      l = harmonicL glyph
      m = harmonicM glyph
      n = phiPhase glyph
      glowColor = glowToColor (coherenceGlow glyph)
      torsColor = torsionToColorStr (torsionState glyph)

      -- Generate spiral arms based on m
      spirals = generateSpirals m cx cy 80

      -- Generate rings based on l
      rings = generateRings l cx cy

      -- Generate pulse animation based on φ^n
      pulse = generatePulseStyle n

  in unlines
    [ "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    , "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" ++ show width ++ "\" height=\"" ++ show height ++ "\">"
    , "  <defs>"
    , "    <radialGradient id=\"glow\">"
    , "      <stop offset=\"0%\" stop-color=\"" ++ glowColor ++ "\" stop-opacity=\"1\"/>"
    , "      <stop offset=\"100%\" stop-color=\"" ++ glowColor ++ "\" stop-opacity=\"0\"/>"
    , "    </radialGradient>"
    , pulse
    , "  </defs>"
    , "  <circle cx=\"" ++ show cx ++ "\" cy=\"" ++ show cy ++ "\" r=\"90\" fill=\"url(#glow)\" opacity=\"0.3\"/>"
    , rings
    , spirals
    , "  <circle cx=\"" ++ show cx ++ "\" cy=\"" ++ show cy ++ "\" r=\"5\" fill=\"" ++ torsColor ++ "\"/>"
    , "</svg>"
    ]

-- | Generate spiral arms for SVG
generateSpirals :: Int -> Int -> Int -> Int -> String
generateSpirals m cx cy radius =
  let numArms = abs m + 1
      angleStep = 2 * pi / fromIntegral numArms
      arms = [generateSpiralArm i angleStep cx cy radius | i <- [0..numArms-1]]
  in unlines arms

-- | Generate single spiral arm
generateSpiralArm :: Int -> Double -> Int -> Int -> Int -> String
generateSpiralArm i angleStep cx cy radius =
  let startAngle = fromIntegral i * angleStep
      points = [spiralPoint startAngle t cx cy radius | t <- [0.0, 0.1 .. 1.0]]
      pathData = "M " ++ intercalate " L " [show x ++ "," ++ show y | (x, y) <- points]
  in "  <path d=\"" ++ pathData ++ "\" fill=\"none\" stroke=\"#4488ff\" stroke-width=\"2\" opacity=\"0.7\"/>"

-- | Compute spiral point
spiralPoint :: Double -> Double -> Int -> Int -> Int -> (Int, Int)
spiralPoint startAngle t cx cy radius =
  let angle = startAngle + t * 2 * pi
      r = fromIntegral radius * t
      x = cx + round (r * cos angle)
      y = cy + round (r * sin angle)
  in (x, y)

-- | Generate concentric rings for l
generateRings :: Int -> Int -> Int -> String
generateRings l cx cy =
  let numRings = l + 1
      radii = [20 + i * 15 | i <- [0..numRings-1]]
      rings = [ringElement r cx cy i | (r, i) <- zip radii [0..]]
  in unlines rings

-- | Generate ring SVG element
ringElement :: Int -> Int -> Int -> Int -> String
ringElement radius cx cy idx =
  let opacity = 0.3 + fromIntegral idx * 0.1 :: Double
      color = ringColor idx
  in "  <circle cx=\"" ++ show cx ++ "\" cy=\"" ++ show cy ++
     "\" r=\"" ++ show radius ++ "\" fill=\"none\" stroke=\"" ++ color ++
     "\" stroke-width=\"1\" opacity=\"" ++ show opacity ++ "\"/>"

-- | Ring color based on index
ringColor :: Int -> String
ringColor i = case i `mod` 7 of
  0 -> "#ff4444"
  1 -> "#ff8844"
  2 -> "#ffff44"
  3 -> "#44ff44"
  4 -> "#4444ff"
  5 -> "#8844ff"
  _ -> "#ff44ff"

-- | Generate pulse animation style
generatePulseStyle :: Int -> String
generatePulseStyle n =
  let duration = max 0.5 (2.0 / (1.0 + fromIntegral n * 0.1)) :: Double
  in unlines
    [ "    <style>"
    , "      @keyframes pulse {"
    , "        0% { transform: scale(1); opacity: 1; }"
    , "        50% { transform: scale(1.1); opacity: 0.7; }"
    , "        100% { transform: scale(1); opacity: 1; }"
    , "      }"
    , "      .pulsing { animation: pulse " ++ show duration ++ "s infinite; }"
    , "    </style>"
    ]

-- | Build legend for glyph
buildLegend :: HarmonicGlyph -> [String]
buildLegend glyph =
  [ "Harmonic: H_{" ++ show (harmonicL glyph) ++ "," ++ show (harmonicM glyph) ++ "}"
  , "Depth: φ^" ++ show (phiPhase glyph)
  , "Format: " ++ show (harmonicRoot glyph)
  , "Torsion: " ++ show (torsionState glyph)
  , "Glow: " ++ show (coherenceGlow glyph)
  ]

-- | Fragment anchor glyph
fragmentAnchorGlyph :: FragmentMetadata -> IO GlyphImage
fragmentAnchorGlyph meta = do
  let glyph = HarmonicGlyph
        { glyphId = "anchor-" ++ fmFragmentId meta
        , phiPhase = fmPhiDepth meta
        , harmonicRoot = depthToFormat (fmPhiDepth meta)
        , harmonicL = fmHarmonicL meta
        , harmonicM = fmHarmonicM meta
        , torsionState = TorsionNormal
        , coherenceGlow = Just (coherenceToGlow (fmCoherence meta))
        , associatedWith = Just (fmFragmentId meta)
        }
  generateGlyphSVG glyph

-- =============================================================================
-- Glyph Collections
-- =============================================================================

-- | Collection of glyphs
data GlyphSet = GlyphSet
  { gsGlyphs   :: !(Map GlyphID HarmonicGlyph)
  , gsOrdering :: ![GlyphID]
  } deriving (Eq, Show)

-- | Build glyph set from layers
buildGlyphSet :: [ScalarLayer] -> GlyphSet
buildGlyphSet layers =
  let glyphs = map renderHarmonicGlyph layers
      glyphMap = Map.fromList [(glyphId g, g) | g <- glyphs]
      ordering = map glyphId glyphs
  in GlyphSet glyphMap ordering

-- | Lookup glyph by ID
lookupGlyph :: GlyphID -> GlyphSet -> Maybe HarmonicGlyph
lookupGlyph gid gs = Map.lookup gid (gsGlyphs gs)

-- | Merge two glyph sets
mergeGlyphSets :: GlyphSet -> GlyphSet -> GlyphSet
mergeGlyphSets gs1 gs2 = GlyphSet
  { gsGlyphs = Map.union (gsGlyphs gs1) (gsGlyphs gs2)
  , gsOrdering = gsOrdering gs1 ++ filter (`notElem` gsOrdering gs1) (gsOrdering gs2)
  }

-- =============================================================================
-- SVG Primitives
-- =============================================================================

-- | SVG element types
data SVGElement
  = SVGCircle !Int !Int !Int !String    -- ^ cx, cy, r, color
  | SVGSpiral !Int !Int !Int !Int !String -- ^ cx, cy, radius, arms, color
  | SVGRing !Int !Int !Int !Int !String -- ^ cx, cy, r, width, color
  | SVGPulse !Int !Int !Int !Double !String -- ^ cx, cy, r, duration, color
  | SVGGlow !Int !Int !Int !String      -- ^ cx, cy, r, color
  deriving (Eq, Show)

-- | Create circle element
svgCircle :: Int -> Int -> Int -> String -> SVGElement
svgCircle = SVGCircle

-- | Create spiral element
svgSpiral :: Int -> Int -> Int -> Int -> String -> SVGElement
svgSpiral = SVGSpiral

-- | Create ring element
svgRing :: Int -> Int -> Int -> Int -> String -> SVGElement
svgRing = SVGRing

-- | Create pulse element
svgPulse :: Int -> Int -> Int -> Double -> String -> SVGElement
svgPulse = SVGPulse

-- | Create glow element
svgGlow :: Int -> Int -> Int -> String -> SVGElement
svgGlow = SVGGlow

-- | Render SVG element to string
renderSVGElement :: SVGElement -> String
renderSVGElement el = case el of
  SVGCircle cx cy r color ->
    "<circle cx=\"" ++ show cx ++ "\" cy=\"" ++ show cy ++
    "\" r=\"" ++ show r ++ "\" fill=\"" ++ color ++ "\"/>"

  SVGSpiral cx cy radius arms _color ->
    generateSpirals arms cx cy radius

  SVGRing cx cy r w color ->
    "<circle cx=\"" ++ show cx ++ "\" cy=\"" ++ show cy ++
    "\" r=\"" ++ show r ++ "\" fill=\"none\" stroke=\"" ++ color ++
    "\" stroke-width=\"" ++ show w ++ "\"/>"

  SVGPulse cx cy r dur color ->
    "<circle cx=\"" ++ show cx ++ "\" cy=\"" ++ show cy ++
    "\" r=\"" ++ show r ++ "\" fill=\"" ++ color ++ "\">" ++
    "<animate attributeName=\"r\" values=\"" ++ show r ++ ";" ++ show (r + 5) ++
    ";" ++ show r ++ "\" dur=\"" ++ show dur ++ "s\" repeatCount=\"indefinite\"/></circle>"

  SVGGlow cx cy r color ->
    "<circle cx=\"" ++ show cx ++ "\" cy=\"" ++ show cy ++
    "\" r=\"" ++ show r ++ "\" fill=\"url(#glow-" ++ color ++ ")\" opacity=\"0.5\"/>"

-- =============================================================================
-- Color Mapping
-- =============================================================================

-- | Glyph colors
data GlyphColor
  = ColorRed
  | ColorOrange
  | ColorYellow
  | ColorGreen
  | ColorCyan
  | ColorBlue
  | ColorViolet
  | ColorWhite
  | ColorGray
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Map coherence to color
coherenceToColor :: Double -> GlyphColor
coherenceToColor coh
  | coh >= 0.9 = ColorWhite
  | coh >= 0.8 = ColorViolet
  | coh >= 0.7 = ColorBlue
  | coh >= 0.6 = ColorCyan
  | coh >= 0.5 = ColorGreen
  | coh >= 0.4 = ColorYellow
  | coh >= 0.3 = ColorOrange
  | coh >= 0.2 = ColorRed
  | otherwise = ColorGray

-- | Map torsion to color
torsionToColor :: TorsionPhase -> GlyphColor
torsionToColor TorsionNormal = ColorBlue
torsionToColor TorsionInverted = ColorRed
torsionToColor TorsionNull = ColorGray
torsionToColor TorsionOscillating = ColorViolet

-- | Map harmonic degree to hue angle
harmonicToHue :: Int -> Double
harmonicToHue l = fromIntegral (l `mod` 12) * 30.0

-- | Color to hex string
colorToHex :: GlyphColor -> String
colorToHex c = case c of
  ColorRed -> "#ff4444"
  ColorOrange -> "#ff8844"
  ColorYellow -> "#ffff44"
  ColorGreen -> "#44ff44"
  ColorCyan -> "#44ffff"
  ColorBlue -> "#4444ff"
  ColorViolet -> "#ff44ff"
  ColorWhite -> "#ffffff"
  ColorGray -> "#888888"

-- =============================================================================
-- Animation
-- =============================================================================

-- | Glyph animation types
data GlyphAnimation
  = AnimPulse !Double           -- ^ Pulse with duration
  | AnimSpiral !Double !Int     -- ^ Spiral with speed and direction
  | AnimGlow !Double !Double    -- ^ Glow fade (min, max opacity)
  | AnimRotate !Double          -- ^ Rotation with speed
  | AnimNone
  deriving (Eq, Show)

-- | Apply animation to glyph
animateGlyph :: GlyphAnimation -> HarmonicGlyph -> HarmonicGlyph
animateGlyph _ glyph = glyph  -- Animation affects rendering, not glyph data

-- | Create pulse animation
pulseAnimation :: Double -> GlyphAnimation
pulseAnimation = AnimPulse

-- | Create spiral animation
spiralAnimation :: Double -> Int -> GlyphAnimation
spiralAnimation = AnimSpiral

-- =============================================================================
-- Text Rendering
-- =============================================================================

-- | Convert glyph to text representation
glyphToText :: HarmonicGlyph -> String
glyphToText glyph = unlines
  [ "┌─────────────────────────┐"
  , "│ " ++ padRight 23 ("H_{" ++ show (harmonicL glyph) ++ "," ++ show (harmonicM glyph) ++ "}") ++ " │"
  , "│ " ++ padRight 23 ("φ^" ++ show (phiPhase glyph) ++ " depth") ++ " │"
  , "│ " ++ padRight 23 (show (torsionState glyph)) ++ " │"
  , "│ " ++ padRight 23 (glowIndicator (coherenceGlow glyph)) ++ " │"
  , "└─────────────────────────┘"
  ]

-- | Pad string to right
padRight :: Int -> String -> String
padRight n s = take n (s ++ repeat ' ')

-- | Glow indicator string
glowIndicator :: Maybe GlowLevel -> String
glowIndicator Nothing = "○○○○○"
glowIndicator (Just g) = case g of
  GlowNone -> "○○○○○"
  GlowDim -> "●○○○○"
  GlowMedium -> "●●○○○"
  GlowBright -> "●●●○○"
  GlowIntense -> "●●●●●"

-- | Convert glyph to unicode representation
glyphToUnicode :: HarmonicGlyph -> String
glyphToUnicode glyph =
  let l = harmonicL glyph
      m = harmonicM glyph
      n = phiPhase glyph
      tors = torsionSymbol (torsionState glyph)
      glow = glowSymbol (coherenceGlow glyph)
  in tors ++ harmonicSymbol l m ++ superscriptPhi n ++ glow

-- | Torsion symbol
torsionSymbol :: TorsionPhase -> String
torsionSymbol TorsionNormal = "◇"
torsionSymbol TorsionInverted = "◆"
torsionSymbol TorsionNull = "○"
torsionSymbol TorsionOscillating = "◎"

-- | Harmonic symbol
harmonicSymbol :: Int -> Int -> String
harmonicSymbol 0 _ = "●"
harmonicSymbol 1 m
  | m == 0 = "◐"
  | m > 0 = "◑"
  | otherwise = "◒"
harmonicSymbol 2 m
  | m == 0 = "◓"
  | m > 0 = "▣"
  | otherwise = "▤"
harmonicSymbol _ _ = "✦"

-- | Superscript phi notation
superscriptPhi :: Int -> String
superscriptPhi n = "φ" ++ superscript n

-- | Convert int to superscript
superscript :: Int -> String
superscript n
  | n == 0 = "⁰"
  | n == 1 = "¹"
  | n == 2 = "²"
  | n == 3 = "³"
  | n == 4 = "⁴"
  | n == 5 = "⁵"
  | n == 6 = "⁶"
  | n == 7 = "⁷"
  | n == 8 = "⁸"
  | n == 9 = "⁹"
  | otherwise = "^" ++ show n

-- | Glow symbol
glowSymbol :: Maybe GlowLevel -> String
glowSymbol Nothing = ""
glowSymbol (Just GlowNone) = ""
glowSymbol (Just GlowDim) = "·"
glowSymbol (Just GlowMedium) = "∘"
glowSymbol (Just GlowBright) = "○"
glowSymbol (Just GlowIntense) = "◉"

-- | Create text glyph overlay
textGlyphOverlay :: [HarmonicGlyph] -> String
textGlyphOverlay glyphs =
  let rows = map glyphToUnicode glyphs
  in unlines rows

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Map depth to omega format
depthToFormat :: Int -> OmegaFormat
depthToFormat n = case n `mod` 5 of
  0 -> Red
  1 -> OmegaMajor
  2 -> Green
  3 -> OmegaMinor
  _ -> Blue

-- | Map coherence to glow level
coherenceToGlow :: Double -> GlowLevel
coherenceToGlow coh
  | coh >= 0.9 = GlowIntense
  | coh >= 0.7 = GlowBright
  | coh >= 0.5 = GlowMedium
  | coh >= 0.3 = GlowDim
  | otherwise = GlowNone

-- | Glow level to color string
glowToColor :: Maybe GlowLevel -> String
glowToColor Nothing = "#888888"
glowToColor (Just g) = case g of
  GlowNone -> "#444444"
  GlowDim -> "#666688"
  GlowMedium -> "#8888aa"
  GlowBright -> "#aaaaff"
  GlowIntense -> "#ffffff"

-- | Torsion to color string
torsionToColorStr :: TorsionPhase -> String
torsionToColorStr TorsionNormal = "#4488ff"
torsionToColorStr TorsionInverted = "#ff4488"
torsionToColorStr TorsionNull = "#888888"
torsionToColorStr TorsionOscillating = "#ff88ff"
