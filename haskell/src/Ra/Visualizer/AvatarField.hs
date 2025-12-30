{-|
Module      : Ra.Visualizer.AvatarField
Description : Live avatar harmonic field visualizer
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Synthesizes and renders the live scalar field around an avatar based on
biometric input, coherence state, chamber tuning, and emergence resonance.

== Avatar Field Theory

The avatar field is a dynamic, spatial scalar field projection combining:

* Biometric resonance data
* Avatar signature (base harmonic seed)
* Chamber configuration (field shaping)
* Fragment interactions and emergence activity

The output can be rendered in 3D (AR/VR), 2D (HUD), or encoded glyphically.
-}
module Ra.Visualizer.AvatarField
  ( -- * Core Types
    AvatarFieldFrame(..)
  , AvatarID
  , ScalarSlice(..)
  , AuraPattern(..)
  , GlowAnchor(..)

    -- * Field Computation
  , computeAvatarField
  , updateAvatarField
  , blendFields

    -- * Spatial Slices
  , fieldSlicesFromScalar
  , horizontalSlice
  , verticalSlice
  , radialShell

    -- * Aura Computation
  , deriveAuraPattern
  , auraIntensity
  , auraBands

    -- * Emergence Mapping
  , mapEmergencePoints
  , glowFromEmergence
  , emergenceIntensity

    -- * Avatar Signatures
  , AvatarSignature(..)
  , mkAvatarSignature
  , signatureHarmonic

    -- * Chamber Integration
  , ChamberState(..)
  , applyToField

    -- * Coherence Packets
  , CoherencePacket(..)
  , packetToField

    -- * Emergent Content
  , EmergentContent(..)
  , contentToGlow
  ) where

import Data.Time (UTCTime, getCurrentTime)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

import Ra.Omega (OmegaFormat(..))
import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Avatar identifier
type AvatarID = String

-- | Theta angle (0 to 2π)
type Theta = Double

-- | Phi angle (0 to π)
type Phi = Double

-- | Avatar field frame (single snapshot)
data AvatarFieldFrame = AvatarFieldFrame
  { avatarId       :: !AvatarID
  , timestamp      :: !UTCTime
  , fieldSlices    :: ![ScalarSlice]
  , coherenceAura  :: !(Maybe AuraPattern)
  , emergenceGlow  :: ![GlowAnchor]
  , motionVector   :: !(Maybe (Theta, Phi))
  , frameCoherence :: !Double
  , frameMetadata  :: !(Map String String)
  } deriving (Eq, Show)

-- | Scalar slice (2D cross-section of field)
data ScalarSlice = ScalarSlice
  { sliceType    :: !SliceType
  , sliceData    :: ![[Double]]      -- ^ 2D grid of values
  , sliceCenter  :: !(Double, Double, Double)
  , sliceNormal  :: !(Double, Double, Double)
  , sliceRadius  :: !Double
  } deriving (Eq, Show)

-- | Type of slice
data SliceType
  = HorizontalSlice  -- ^ XY plane at fixed Z
  | VerticalSlice    -- ^ XZ or YZ plane
  | RadialShell      -- ^ Spherical shell at fixed radius
  | ArbitrarySlice   -- ^ Custom orientation
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Aura pattern around avatar
data AuraPattern = AuraPattern
  { apBands     :: ![AuraBand]
  , apIntensity :: !Double
  , apPulseRate :: !Double
  , apColor     :: !AuraColor
  } deriving (Eq, Show)

-- | Single aura band
data AuraBand = AuraBand
  { abRadius    :: !Double
  , abThickness :: !Double
  , abIntensity :: !Double
  , abHue       :: !Double       -- ^ 0-360 degrees
  } deriving (Eq, Show)

-- | Aura color (dominant)
data AuraColor
  = AuraRed
  | AuraOrange
  | AuraYellow
  | AuraGreen
  | AuraCyan
  | AuraBlue
  | AuraViolet
  | AuraWhite
  | AuraGold
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Glow anchor point (emergence location)
data GlowAnchor = GlowAnchor
  { gaPosition   :: !(Double, Double, Double)
  , gaIntensity  :: !Double
  , gaPulse      :: !Double       -- ^ Pulse phase [0, 1]
  , gaFragmentId :: !(Maybe String)
  } deriving (Eq, Show)

-- =============================================================================
-- Avatar Signature
-- =============================================================================

-- | Avatar's harmonic signature (base resonance)
data AvatarSignature = AvatarSignature
  { asAvatarId   :: !AvatarID
  , asHarmonicL  :: !Int           -- ^ Base spherical harmonic l
  , asHarmonicM  :: !Int           -- ^ Base spherical harmonic m
  , asBaseFreq   :: !Double        -- ^ Base frequency (Hz)
  , asPhiDepth   :: !Int           -- ^ φ^n depth index
  , asFormat     :: !OmegaFormat
  } deriving (Eq, Show)

-- | Create avatar signature
mkAvatarSignature :: AvatarID -> Int -> Int -> Double -> AvatarSignature
mkAvatarSignature aid l m freq = AvatarSignature
  { asAvatarId = aid
  , asHarmonicL = l
  , asHarmonicM = m
  , asBaseFreq = freq
  , asPhiDepth = 0
  , asFormat = Green
  }

-- | Get signature harmonic as tuple
signatureHarmonic :: AvatarSignature -> (Int, Int)
signatureHarmonic sig = (asHarmonicL sig, asHarmonicM sig)

-- =============================================================================
-- Chamber State
-- =============================================================================

-- | Current chamber state
data ChamberState = ChamberState
  { csCoherence   :: !Double
  , csTorsion     :: !TorsionState
  , csHarmonics   :: ![(Int, Int)]   -- ^ Active harmonics (l, m)
  , csRadius      :: !Double
  , csDepth       :: !Int
  } deriving (Eq, Show)

-- | Torsion state
data TorsionState
  = TorsionPositive
  | TorsionNegative
  | TorsionNeutral
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Apply chamber state to field
applyToField :: ChamberState -> AvatarFieldFrame -> AvatarFieldFrame
applyToField cs frame =
  let scaleFactor = csCoherence cs
      newSlices = map (scaleSlice scaleFactor) (fieldSlices frame)
      newAura = fmap (scaleAura scaleFactor) (coherenceAura frame)
  in frame
    { fieldSlices = newSlices
    , coherenceAura = newAura
    , frameCoherence = frameCoherence frame * scaleFactor
    }

-- | Scale slice values
scaleSlice :: Double -> ScalarSlice -> ScalarSlice
scaleSlice factor slice =
  slice { sliceData = map (map (* factor)) (sliceData slice) }

-- | Scale aura intensity
scaleAura :: Double -> AuraPattern -> AuraPattern
scaleAura factor aura =
  aura { apIntensity = apIntensity aura * factor }

-- =============================================================================
-- Coherence Packet
-- =============================================================================

-- | Coherence packet from biometric input
data CoherencePacket = CoherencePacket
  { cpCoherence   :: !Double
  , cpHRVPhase    :: !Double
  , cpBreathPhase :: !Double
  , cpTimestamp   :: !UTCTime
  } deriving (Eq, Show)

-- | Convert packet to field contribution
packetToField :: CoherencePacket -> AvatarSignature -> IO AvatarFieldFrame
packetToField packet sig = do
  now <- getCurrentTime
  let baseSlices = generateBaseSlices sig
      aura = deriveAuraPattern packet
  return AvatarFieldFrame
    { avatarId = asAvatarId sig
    , timestamp = now
    , fieldSlices = baseSlices
    , coherenceAura = Just aura
    , emergenceGlow = []
    , motionVector = Nothing
    , frameCoherence = cpCoherence packet
    , frameMetadata = Map.empty
    }

-- =============================================================================
-- Emergent Content
-- =============================================================================

-- | Emergent content for glow mapping
data EmergentContent = EmergentContent
  { ecContentId :: !String
  , ecPosition  :: !(Double, Double, Double)
  , ecAlpha     :: !Double           -- ^ Emergence alpha [0, 1]
  , ecHarmonic  :: !(Int, Int)
  } deriving (Eq, Show)

-- | Convert content to glow anchor
contentToGlow :: EmergentContent -> GlowAnchor
contentToGlow ec = GlowAnchor
  { gaPosition = ecPosition ec
  , gaIntensity = ecAlpha ec
  , gaPulse = 0.0
  , gaFragmentId = Just (ecContentId ec)
  }

-- =============================================================================
-- Field Computation
-- =============================================================================

-- | Compute complete avatar field
computeAvatarField :: AvatarSignature
                   -> ChamberState
                   -> CoherencePacket
                   -> [EmergentContent]
                   -> IO AvatarFieldFrame
computeAvatarField sig chamber packet contents = do
  now <- getCurrentTime
  let -- Generate field slices based on signature and chamber
      slices = generateFieldSlices sig chamber

      -- Derive aura from coherence
      aura = deriveAuraPattern packet

      -- Map emergence points
      glows = mapEmergencePoints contents

      -- Compute motion vector from harmonics
      motion = computeMotionVector sig chamber

  return AvatarFieldFrame
    { avatarId = asAvatarId sig
    , timestamp = now
    , fieldSlices = slices
    , coherenceAura = Just aura
    , emergenceGlow = glows
    , motionVector = motion
    , frameCoherence = cpCoherence packet * csCoherence chamber
    , frameMetadata = Map.fromList
        [ ("l", show (asHarmonicL sig))
        , ("m", show (asHarmonicM sig))
        , ("depth", show (asPhiDepth sig))
        ]
    }

-- | Update existing field with new data
updateAvatarField :: AvatarFieldFrame
                  -> CoherencePacket
                  -> [EmergentContent]
                  -> IO AvatarFieldFrame
updateAvatarField frame packet contents = do
  now <- getCurrentTime
  let newAura = deriveAuraPattern packet
      newGlows = mapEmergencePoints contents
      blendedGlows = blendGlows (emergenceGlow frame) newGlows
  return frame
    { timestamp = now
    , coherenceAura = Just newAura
    , emergenceGlow = blendedGlows
    , frameCoherence = cpCoherence packet
    }

-- | Blend two avatar fields
blendFields :: Double -> AvatarFieldFrame -> AvatarFieldFrame -> AvatarFieldFrame
blendFields factor f1 f2 =
  let blendedSlices = zipWith (blendSlices factor) (fieldSlices f1) (fieldSlices f2)
      blendedAura = case (coherenceAura f1, coherenceAura f2) of
        (Just a1, Just a2) -> Just (blendAuras factor a1 a2)
        (Just a, Nothing) -> Just a
        (Nothing, Just a) -> Just a
        _ -> Nothing
  in f1
    { fieldSlices = blendedSlices
    , coherenceAura = blendedAura
    , frameCoherence = lerp factor (frameCoherence f1) (frameCoherence f2)
    }

-- =============================================================================
-- Spatial Slices
-- =============================================================================

-- | Generate slices from scalar field data
fieldSlicesFromScalar :: [[Double]] -> [ScalarSlice]
fieldSlicesFromScalar gridData =
  [ horizontalSlice gridData 0.0
  , verticalSlice gridData 0.0
  ]

-- | Create horizontal slice at height
horizontalSlice :: [[Double]] -> Double -> ScalarSlice
horizontalSlice gridData z = ScalarSlice
  { sliceType = HorizontalSlice
  , sliceData = gridData
  , sliceCenter = (0, 0, z)
  , sliceNormal = (0, 0, 1)
  , sliceRadius = fromIntegral (length gridData) / 2
  }

-- | Create vertical slice at angle
verticalSlice :: [[Double]] -> Double -> ScalarSlice
verticalSlice gridData angle = ScalarSlice
  { sliceType = VerticalSlice
  , sliceData = gridData
  , sliceCenter = (0, 0, 0)
  , sliceNormal = (cos angle, sin angle, 0)
  , sliceRadius = fromIntegral (length gridData) / 2
  }

-- | Create radial shell at radius
radialShell :: Int -> Double -> ScalarSlice
radialShell resolution radius =
  let thetaSteps = resolution
      phiSteps = resolution `div` 2
      grid = [[shellValue theta phi' radius
              | phi' <- [0, pi / fromIntegral phiSteps .. pi]]
             | theta <- [0, 2 * pi / fromIntegral thetaSteps .. 2 * pi]]
  in ScalarSlice
    { sliceType = RadialShell
    , sliceData = grid
    , sliceCenter = (0, 0, 0)
    , sliceNormal = (0, 0, 0)  -- Not applicable for shell
    , sliceRadius = radius
    }

-- | Compute shell value at angles
shellValue :: Double -> Double -> Double -> Double
shellValue _theta phi' radius =
  -- Simplified spherical harmonic-like value
  let y00 = 1.0 / sqrt (4 * pi)
      y10 = sqrt (3 / (4 * pi)) * cos phi'
  in (y00 + 0.3 * y10) * exp (-radius / 10)

-- =============================================================================
-- Aura Computation
-- =============================================================================

-- | Derive aura pattern from coherence packet
deriveAuraPattern :: CoherencePacket -> AuraPattern
deriveAuraPattern packet =
  let coh = cpCoherence packet
      bands = generateAuraBands coh
      color = coherenceToAuraColor coh
      pulseRate = 1.0 / (0.5 + coh)  -- Higher coherence = slower pulse
  in AuraPattern
    { apBands = bands
    , apIntensity = coh
    , apPulseRate = pulseRate
    , apColor = color
    }

-- | Compute aura intensity
auraIntensity :: AuraPattern -> Double
auraIntensity = apIntensity

-- | Get aura bands
auraBands :: AuraPattern -> [AuraBand]
auraBands = apBands

-- | Generate aura bands based on coherence
generateAuraBands :: Double -> [AuraBand]
generateAuraBands coh =
  let numBands = max 1 (round (coh * 5) :: Int)
      bandRadius i = 1.0 + fromIntegral i * phi
      bandThickness = 0.2 * coh
      bandIntensity i = coh * (1.0 - fromIntegral i * 0.15)
      bandHue i = fromIntegral (i * 60) `mod'` 360
  in [AuraBand (bandRadius i) bandThickness (bandIntensity i) (bandHue i)
     | i <- [0..numBands-1]]
  where
    mod' :: Double -> Double -> Double
    mod' a b = a - b * fromIntegral (floor (a / b) :: Int)

-- | Map coherence to aura color
coherenceToAuraColor :: Double -> AuraColor
coherenceToAuraColor coh
  | coh >= 0.95 = AuraGold
  | coh >= 0.9 = AuraWhite
  | coh >= 0.8 = AuraViolet
  | coh >= 0.7 = AuraBlue
  | coh >= 0.6 = AuraCyan
  | coh >= 0.5 = AuraGreen
  | coh >= 0.4 = AuraYellow
  | coh >= 0.3 = AuraOrange
  | otherwise = AuraRed

-- =============================================================================
-- Emergence Mapping
-- =============================================================================

-- | Map emergence points to glow anchors
mapEmergencePoints :: [EmergentContent] -> [GlowAnchor]
mapEmergencePoints = map contentToGlow

-- | Compute glow from emergence alpha
glowFromEmergence :: Double -> Double
glowFromEmergence alpha = alpha ** phiInverse  -- Non-linear glow curve

-- | Compute emergence intensity from multiple sources
emergenceIntensity :: [EmergentContent] -> Double
emergenceIntensity [] = 0.0
emergenceIntensity contents =
  let alphas = map ecAlpha contents
  in sum alphas / fromIntegral (length alphas)

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Generate base slices from signature
generateBaseSlices :: AvatarSignature -> [ScalarSlice]
generateBaseSlices sig =
  let resolution = 20
      baseGrid = [[harmonicValue (asHarmonicL sig) (asHarmonicM sig) i j
                  | j <- [0..resolution-1]]
                 | i <- [0..resolution-1]]
  in [horizontalSlice baseGrid 0.0, verticalSlice baseGrid 0.0]

-- | Generate field slices from signature and chamber
generateFieldSlices :: AvatarSignature -> ChamberState -> [ScalarSlice]
generateFieldSlices _sig chamber =
  let resolution = 20
      scale = csCoherence chamber
      harmonics = csHarmonics chamber
      grid = [[sum [harmonicValue l m i j * scale | (l, m) <- harmonics]
              | j <- [0..resolution-1]]
             | i <- [0..resolution-1]]
      shell = radialShell resolution (csRadius chamber)
  in [horizontalSlice grid 0.0, verticalSlice grid (pi/4), shell]

-- | Simplified harmonic value at grid position
harmonicValue :: Int -> Int -> Int -> Int -> Double
harmonicValue l m i j =
  let theta = fromIntegral i * 2 * pi / 20
      phi' = fromIntegral j * pi / 20
      -- Very simplified spherical harmonic approximation
      y_lm = cos (fromIntegral l * phi') * cos (fromIntegral m * theta)
  in y_lm * 0.5 + 0.5

-- | Compute motion vector from harmonics
computeMotionVector :: AvatarSignature -> ChamberState -> Maybe (Theta, Phi)
computeMotionVector sig chamber =
  let l = asHarmonicL sig
      m = asHarmonicM sig
      coh = csCoherence chamber
  in if coh > 0.3
     then Just (fromIntegral m * pi / 6, fromIntegral l * pi / 12)
     else Nothing

-- | Blend two slices
blendSlices :: Double -> ScalarSlice -> ScalarSlice -> ScalarSlice
blendSlices factor s1 s2 =
  let blendedData = zipWith (zipWith (lerp factor)) (sliceData s1) (sliceData s2)
  in s1 { sliceData = blendedData }

-- | Blend two auras
blendAuras :: Double -> AuraPattern -> AuraPattern -> AuraPattern
blendAuras factor a1 a2 = a1
  { apIntensity = lerp factor (apIntensity a1) (apIntensity a2)
  , apPulseRate = lerp factor (apPulseRate a1) (apPulseRate a2)
  }

-- | Blend glow anchors
blendGlows :: [GlowAnchor] -> [GlowAnchor] -> [GlowAnchor]
blendGlows old new =
  -- Keep new glows, fade old ones
  let fadedOld = [g { gaIntensity = gaIntensity g * 0.9 }
                 | g <- old, gaIntensity g > 0.1]
  in new ++ fadedOld

-- | Linear interpolation
lerp :: Double -> Double -> Double -> Double
lerp t a b = a + t * (b - a)
