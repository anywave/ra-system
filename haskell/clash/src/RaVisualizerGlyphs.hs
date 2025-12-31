{-|
Module      : RaVisualizerGlyphs
Description : Harmonic Signature Renderer
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 61: Render symbolic glyphs representing scalar field harmonics,
avatar resonance signatures, and fragment emergence anchors.

Uses torsion-based color mapping and coherence glow blending.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaVisualizerGlyphs where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Base depth (0.618 * 1024)
baseDepth :: Unsigned 16
baseDepth = 633

-- | Torsion state
data TorsionPhase
  = TorsionNormal
  | TorsionInverted
  | TorsionNull
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | RGB color (8-bit per channel)
data RGBColor = RGBColor
  { rgbR :: Unsigned 8
  , rgbG :: Unsigned 8
  , rgbB :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | OmegaFormat (l, m spherical harmonics)
data OmegaFormat = OmegaFormat
  { ofL :: Unsigned 3   -- 0-7
  , ofM :: Signed 4     -- -l to +l
  } deriving (Generic, NFDataX, Eq, Show)

-- | Glow level (0-255 intensity)
newtype GlowLevel = GlowLevel { glowIntensity :: Unsigned 8 }
  deriving (Generic, NFDataX, Eq, Show)

-- | Spiral direction
data SpiralDirection
  = SpiralNone
  | SpiralCW
  | SpiralCCW
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Harmonic glyph metadata
data HarmonicGlyph = HarmonicGlyph
  { hgGlyphId      :: Unsigned 32      -- Glyph ID
  , hgPhiPhase     :: Unsigned 4       -- phi^n phase level
  , hgHarmonicRoot :: OmegaFormat      -- spherical harmonic key
  , hgTorsionState :: TorsionPhase
  , hgCoherenceGlow :: Unsigned 8      -- 0-255 glow intensity
  , hgHasFragment  :: Bool             -- Has associated fragment
  , hgFragmentId   :: Unsigned 32      -- Fragment ID (if any)
  } deriving (Generic, NFDataX, Eq, Show)

-- | Spiral arm configuration
data SpiralConfig = SpiralConfig
  { scNumArms   :: Unsigned 4
  , scDirection :: SpiralDirection
  } deriving (Generic, NFDataX, Eq, Show)

-- | SVG element type
data SVGElementType
  = SVGCircle
  | SVGPath
  | SVGGroup
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | SVG primitive element
data SVGElement = SVGElement
  { seType     :: SVGElementType
  , seCenterX  :: Signed 16
  , seCenterY  :: Signed 16
  , seRadius   :: Unsigned 16
  , seColorR   :: Unsigned 8
  , seColorG   :: Unsigned 8
  , seColorB   :: Unsigned 8
  , seArmIndex :: Unsigned 4
  } deriving (Generic, NFDataX, Eq, Show)

-- | Glyph image (simplified for FPGA)
data GlyphImage = GlyphImage
  { giElements :: Vec 16 SVGElement
  , giNumElements :: Unsigned 5
  , giPhiPhase :: Unsigned 4
  , giLValue   :: Unsigned 3
  , giMValue   :: Signed 4
  , giTorsion  :: TorsionPhase
  , giGlowLevel :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Scalar layer input
data ScalarLayer = ScalarLayer
  { slDepth      :: Unsigned 16
  , slAmplitude  :: Unsigned 8
  , slPhase      :: Unsigned 16
  , slOmega      :: OmegaFormat
  , slLayerIndex :: Unsigned 4
  , slTorsion    :: TorsionPhase
  , slCoherence  :: Unsigned 8
  , slFragmentId :: Unsigned 32
  , slHasFragment :: Bool
  } deriving (Generic, NFDataX, Eq, Show)

-- | Fragment metadata
data FragmentMetadata = FragmentMetadata
  { fmFragmentId :: Unsigned 32
  , fmEmergenceX :: Signed 16
  , fmEmergenceY :: Signed 16
  , fmEmergenceZ :: Signed 16
  , fmCoherence  :: Unsigned 8
  , fmHarmonicRoot :: OmegaFormat
  } deriving (Generic, NFDataX, Eq, Show)

-- | Get base color for torsion state
getTorsionBaseColor :: TorsionPhase -> RGBColor
getTorsionBaseColor torsion = case torsion of
  TorsionNormal   -> RGBColor 0 0 255      -- Blue
  TorsionInverted -> RGBColor 255 0 0      -- Red
  TorsionNull     -> RGBColor 128 128 128  -- Grey

-- | Blend color with white based on glow intensity
blendWithWhite :: RGBColor -> Unsigned 8 -> RGBColor
blendWithWhite base intensity =
  let factor = resize intensity :: Unsigned 16
      invFactor = 255 - factor
      blendChan orig = resize ((resize invFactor * resize orig + factor * 255) `shiftR` 8 :: Unsigned 16)
  in RGBColor
       (blendChan (rgbR base))
       (blendChan (rgbG base))
       (blendChan (rgbB base))

-- | Compute spiral arm configuration from m value
computeSpiralConfig :: Signed 4 -> SpiralConfig
computeSpiralConfig m
  | m == 0    = SpiralConfig 0 SpiralNone
  | m > 0     = SpiralConfig (resize (pack m) :: Unsigned 4) SpiralCW
  | otherwise = SpiralConfig (resize (pack (negate m)) :: Unsigned 4) SpiralCCW

-- | Compute phi^n phase level from depth
computePhiPhase :: Unsigned 16 -> Unsigned 4
computePhiPhase depth =
  let -- Approximate log_phi calculation
      ratio = (depth * 1024) `div` baseDepth
  in if ratio < 1024 then 0
     else if ratio < 1657 then 1
     else if ratio < 2681 then 2
     else if ratio < 4338 then 3
     else if ratio < 7019 then 4
     else if ratio < 11357 then 5
     else if ratio < 18376 then 6
     else 7

-- | Generate glyph ID from phi phase and omega
generateGlyphId :: Unsigned 4 -> OmegaFormat -> Unsigned 32
generateGlyphId phase omega =
  let phiPart = resize phase `shiftL` 24 :: Unsigned 32
      lPart = resize (ofL omega) `shiftL` 16 :: Unsigned 32
      mPart = resize (pack (ofM omega)) `shiftL` 8 :: Unsigned 32
  in phiPart .|. lPart .|. mPart

-- | Render harmonic glyph from scalar layer
renderHarmonicGlyph :: ScalarLayer -> HarmonicGlyph
renderHarmonicGlyph layer =
  let phiPhase = computePhiPhase (slDepth layer)
      glyphId = if slHasFragment layer
                then slFragmentId layer
                else generateGlyphId phiPhase (slOmega layer)
  in HarmonicGlyph
       glyphId
       phiPhase
       (slOmega layer)
       (slTorsion layer)
       (slCoherence layer)
       (slHasFragment layer)
       (slFragmentId layer)

-- | Create empty SVG element
emptySVGElement :: SVGElement
emptySVGElement = SVGElement SVGCircle 0 0 0 0 0 0 0

-- | Create circle SVG element
createCircleElement :: Signed 16 -> Signed 16 -> Unsigned 16 -> RGBColor -> SVGElement
createCircleElement cx cy r color =
  SVGElement SVGCircle cx cy r (rgbR color) (rgbG color) (rgbB color) 0

-- | Create path SVG element for spiral arm
createArmElement :: Signed 16 -> Signed 16 -> Unsigned 16 -> Unsigned 4 -> SpiralDirection -> RGBColor -> SVGElement
createArmElement cx cy scale armIdx dir color =
  SVGElement SVGPath cx cy scale (rgbR color) (rgbG color) (rgbB color) armIdx

-- | Generate glyph SVG elements
generateGlyphSVG :: HarmonicGlyph -> Unsigned 16 -> GlyphImage
generateGlyphSVG glyph size =
  let cx = resize (size `shiftR` 1) :: Signed 16
      cy = cx

      -- Get base color from torsion
      baseColor = getTorsionBaseColor (hgTorsionState glyph)

      -- Apply coherence glow
      finalColor = blendWithWhite baseColor (hgCoherenceGlow glyph)

      -- Generate rings for l
      l = ofL (hgHarmonicRoot glyph)

      makeRing :: Unsigned 4 -> SVGElement
      makeRing i =
        let ringRadius = resize ((resize (i + 1) * resize size * 4 `div` 10) `div` (resize l + 1) :: Unsigned 16)
        in createCircleElement cx cy ringRadius finalColor

      rings = map makeRing $(listToVecTH [0..7 :: Unsigned 4])

      -- Generate spiral arms for m
      spiralConfig = computeSpiralConfig (ofM (hgHarmonicRoot glyph))
      numArms = scNumArms spiralConfig

      makeArm :: Unsigned 4 -> SVGElement
      makeArm i =
        if i < numArms
        then createArmElement cx cy (resize size `shiftR` 1) i (scDirection spiralConfig) finalColor
        else emptySVGElement

      arms = map makeArm $(listToVecTH [0..7 :: Unsigned 4])

      -- Combine elements (8 rings + 8 arm slots = 16)
      elements = rings ++ arms

      numElements = resize l + 1 + resize numArms

  in GlyphImage
       elements
       numElements
       (hgPhiPhase glyph)
       (ofL (hgHarmonicRoot glyph))
       (ofM (hgHarmonicRoot glyph))
       (hgTorsionState glyph)
       (hgCoherenceGlow glyph)

-- | Generate fragment anchor glyph
fragmentAnchorGlyph :: FragmentMetadata -> GlyphImage
fragmentAnchorGlyph meta =
  let glyph = HarmonicGlyph
        (fmFragmentId meta)
        0  -- Base phase for anchors
        (fmHarmonicRoot meta)
        TorsionNormal
        (fmCoherence meta)
        True
        (fmFragmentId meta)
  in generateGlyphSVG glyph 100

-- | Glyph rendering pipeline
glyphRenderPipeline
  :: HiddenClockResetEnable dom
  => Signal dom ScalarLayer
  -> Signal dom GlyphImage
glyphRenderPipeline input =
  let glyph = renderHarmonicGlyph <$> input
  in generateGlyphSVG <$> glyph <*> pure 100

-- | Fragment anchor pipeline
fragmentAnchorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom FragmentMetadata
  -> Signal dom GlyphImage
fragmentAnchorPipeline input = fragmentAnchorGlyph <$> input
