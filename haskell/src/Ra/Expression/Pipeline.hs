{-|
Module      : Ra.Expression.Pipeline
Description : Biometric-to-avatar scalar expression pipeline
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Real-time pipeline that maps live biometric input to visible avatar field
expressions using scalar torsion physics, morphogenetic coherence, and
radiant frequency modulation. Powers expressive biofield halos and auras.

== Expression Theory

=== Torsion Shell Model

* Avatar fields expressed as nested torsion shells
* Each shell oscillates at a BioHarmonic frequency
* Clockwise spin = coherent, counter = disrupted

=== Resonance Zones

* Crown: respiratory coherence -> bloom expansion
* Chest: HRV -> torsion turbulence
* Solar plexus: breath-hold -> spin halt
* Root: grounding coherence -> stability
-}
module Ra.Expression.Pipeline
  ( -- * Core Types
    ExpressionPipeline(..)
  , AvatarFieldModulation(..)
  , TorsionShell(..)
  , ResonanceZone(..)

    -- * Pipeline Operations
  , createPipeline
  , expressAvatarField
  , updatePipeline

    -- * Biometric Input
  , BiometricSnapshot(..)
  , processSnapshot
  , snapshotDeltas

    -- * Field Expression
  , BioHarmonic(..)
  , expressField
  , shellOscillation
  , spinPolarity

    -- * Zone Operations
  , ZoneState(..)
  , zoneFromBiometric
  , zoneReaction
  , zoneBlend

    -- * Visual Output
  , VisualParams(..)
  , toVisualParams
  , hueFromHarmonic
  , bloomRadius

    -- * Fallback Rendering
  , StaticFieldState(..)
  , renderStatic
  , lastKnownState
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete expression pipeline
data ExpressionPipeline = ExpressionPipeline
  { epShells       :: ![TorsionShell]      -- ^ Nested torsion shells
  , epZones        :: ![ResonanceZone]     -- ^ Body resonance zones
  , epLastSnapshot :: !(Maybe BiometricSnapshot)  -- ^ Last processed input
  , epLatency      :: !Double              -- ^ Current latency (ms)
  , epActive       :: !Bool                -- ^ Pipeline active
  } deriving (Eq, Show)

-- | Avatar field modulation output
data AvatarFieldModulation = AvatarFieldModulation
  { afmHueShift    :: !Double              -- ^ Color hue shift [0, 360]
  , afmTorsionDensity :: !Double           -- ^ Spiral density [0, 1]
  , afmOscillationTempo :: !Double         -- ^ Oscillation rate (Hz)
  , afmBloomExpansion :: !Double           -- ^ Radiant bloom [0, 2]
  , afmSpinDirection :: !SpinDirection     -- ^ Current spin
  , afmIntensity   :: !Double              -- ^ Overall intensity [0, 1]
  } deriving (Eq, Show)

-- | Spin direction
data SpinDirection
  = SpinClockwise      -- ^ Coherent spin
  | SpinCounter        -- ^ Disrupted spin
  | SpinNeutral        -- ^ No spin
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Single torsion shell
data TorsionShell = TorsionShell
  { tsRadius       :: !Double              -- ^ Shell radius (normalized)
  , tsHarmonic     :: !BioHarmonic         -- ^ Operating harmonic
  , tsSpin         :: !SpinDirection       -- ^ Current spin
  , tsPhase        :: !Double              -- ^ Phase [0, 2pi]
  , tsIntensity    :: !Double              -- ^ Shell intensity [0, 1]
  , tsColor        :: !(Double, Double, Double)  -- ^ RGB color
  } deriving (Eq, Show)

-- | Bio-harmonic frequency bands
data BioHarmonic
  = HarmonicAlpha      -- ^ Alpha band (8-12 Hz)
  | HarmonicTheta      -- ^ Theta band (4-8 Hz)
  | HarmonicPhiCrown   -- ^ Phi-tuned crown
  | HarmonicRootShift  -- ^ Root grounding
  | HarmonicAnkhFold   -- ^ Ankh coherence fold
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Body resonance zone
data ResonanceZone = ResonanceZone
  { rzName         :: !String              -- ^ Zone name
  , rzPosition     :: !ZonePosition        -- ^ Body position
  , rzState        :: !ZoneState           -- ^ Current state
  , rzSensitivity  :: !Double              -- ^ Response sensitivity
  , rzColor        :: !(Double, Double, Double)  -- ^ Zone color
  } deriving (Eq, Show)

-- | Zone position on body
data ZonePosition
  = ZoneCrown          -- ^ Crown/head
  | ZoneThirdEye       -- ^ Third eye
  | ZoneThroat         -- ^ Throat
  | ZoneHeart          -- ^ Heart/chest
  | ZoneSolarPlexus    -- ^ Solar plexus
  | ZoneSacral         -- ^ Sacral
  | ZoneRoot           -- ^ Root/base
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Zone state
data ZoneState = ZoneState
  { zsActivity     :: !Double              -- ^ Activity level [0, 1]
  , zsTurbulence   :: !Double              -- ^ Turbulence [0, 1]
  , zsBloom        :: !Double              -- ^ Bloom expansion [0, 2]
  , zsSpinHalt     :: !Bool                -- ^ Spin halted
  } deriving (Eq, Show)

-- =============================================================================
-- Pipeline Operations
-- =============================================================================

-- | Create default expression pipeline
createPipeline :: ExpressionPipeline
createPipeline = ExpressionPipeline
  { epShells = defaultShells
  , epZones = defaultZones
  , epLastSnapshot = Nothing
  , epLatency = 0
  , epActive = True
  }

-- | Main expression function
expressAvatarField :: BiometricSnapshot -> AvatarFieldModulation
expressAvatarField snapshot =
  let -- Calculate torsion from scalar tension
      torsionDensity = bsTension snapshot * phi

      -- Oscillation from respiration
      oscTempo = bsRespiration snapshot / 6  -- Normalize to ~2-4 Hz

      -- Bloom from coherence
      bloom = bsCoherence snapshot * 2

      -- Spin from coherence threshold
      spin = if bsCoherence snapshot > phiInverse
             then SpinClockwise
             else if bsCoherence snapshot < phiInverse * phiInverse
                  then SpinCounter
                  else SpinNeutral

      -- Hue from dominant harmonic
      hue = hueFromHarmonic (dominantHarmonic snapshot)

      -- Overall intensity
      intensity = (bsCoherence snapshot + bsHRV snapshot / 100) / 2

  in AvatarFieldModulation
    { afmHueShift = hue
    , afmTorsionDensity = clamp01 torsionDensity
    , afmOscillationTempo = oscTempo
    , afmBloomExpansion = clamp02 bloom
    , afmSpinDirection = spin
    , afmIntensity = clamp01 intensity
    }

-- | Update pipeline with new snapshot
updatePipeline :: ExpressionPipeline -> BiometricSnapshot -> ExpressionPipeline
updatePipeline pipeline snapshot =
  let -- Update shells based on snapshot
      newShells = map (updateShell snapshot) (epShells pipeline)

      -- Update zones
      newZones = map (updateZone snapshot) (epZones pipeline)

  in pipeline
    { epShells = newShells
    , epZones = newZones
    , epLastSnapshot = Just snapshot
    }

-- =============================================================================
-- Biometric Input
-- =============================================================================

-- | Biometric snapshot for expression
data BiometricSnapshot = BiometricSnapshot
  { bsCoherence    :: !Double              -- ^ Overall coherence [0, 1]
  , bsHRV          :: !Double              -- ^ Heart rate variability (ms)
  , bsRespiration  :: !Double              -- ^ Breaths per minute
  , bsRespirationPhase :: !Double          -- ^ Breath phase [0, 2pi]
  , bsTension      :: !Double              -- ^ Scalar tension [0, 1]
  , bsBreathHold   :: !Bool                -- ^ Breath held
  , bsTimestamp    :: !Int                 -- ^ Capture time
  } deriving (Eq, Show)

-- | Process snapshot through pipeline
processSnapshot :: ExpressionPipeline -> BiometricSnapshot -> (ExpressionPipeline, AvatarFieldModulation)
processSnapshot pipeline snapshot =
  let modulation = expressAvatarField snapshot
      newPipeline = updatePipeline pipeline snapshot
  in (newPipeline, modulation)

-- | Calculate biometric deltas from history
snapshotDeltas :: BiometricSnapshot -> BiometricSnapshot -> BiometricDeltas
snapshotDeltas prev curr = BiometricDeltas
  { bdCoherence = bsCoherence curr - bsCoherence prev
  , bdHRV = bsHRV curr - bsHRV prev
  , bdRespiration = bsRespiration curr - bsRespiration prev
  , bdTension = bsTension curr - bsTension prev
  }

-- | Biometric change deltas
data BiometricDeltas = BiometricDeltas
  { bdCoherence    :: !Double
  , bdHRV          :: !Double
  , bdRespiration  :: !Double
  , bdTension      :: !Double
  } deriving (Eq, Show)

-- =============================================================================
-- Field Expression
-- =============================================================================

-- | Express field based on harmonic
expressField :: BioHarmonic -> BiometricSnapshot -> TorsionShell
expressField harmonic snapshot =
  let radius = harmonicRadius harmonic
      spin = if bsCoherence snapshot > phiInverse then SpinClockwise else SpinCounter
      phase = bsRespirationPhase snapshot
      intensity = bsCoherence snapshot * harmonicWeight harmonic
      color = harmonicColor harmonic
  in TorsionShell radius harmonic spin phase intensity color

-- | Calculate shell oscillation parameters
shellOscillation :: TorsionShell -> Double -> (Double, Double)
shellOscillation shell time =
  let freq = harmonicFrequency (tsHarmonic shell)
      phase = tsPhase shell + time * freq * 2 * pi
      amplitude = tsIntensity shell * (1 + sin phase * 0.3)
  in (freq, amplitude)

-- | Determine spin polarity from coherence
spinPolarity :: Double -> SpinDirection
spinPolarity coherence
  | coherence > phi * 0.5 = SpinClockwise
  | coherence < phiInverse * 0.5 = SpinCounter
  | otherwise = SpinNeutral

-- =============================================================================
-- Zone Operations
-- =============================================================================

-- | Convert biometric to zone state
zoneFromBiometric :: ZonePosition -> BiometricSnapshot -> ZoneState
zoneFromBiometric pos snapshot = case pos of
  ZoneCrown -> ZoneState
    { zsActivity = bsCoherence snapshot
    , zsTurbulence = 0
    , zsBloom = bsCoherence snapshot * 1.5
    , zsSpinHalt = False
    }
  ZoneHeart -> ZoneState
    { zsActivity = bsHRV snapshot / 100
    , zsTurbulence = if bsHRV snapshot < 30 then 0.8 else 0.2
    , zsBloom = bsHRV snapshot / 100
    , zsSpinHalt = False
    }
  ZoneSolarPlexus -> ZoneState
    { zsActivity = if bsBreathHold snapshot then 0.1 else 0.8
    , zsTurbulence = bsTension snapshot
    , zsBloom = 0.5
    , zsSpinHalt = bsBreathHold snapshot
    }
  ZoneRoot -> ZoneState
    { zsActivity = 1 - bsTension snapshot
    , zsTurbulence = bsTension snapshot * 0.5
    , zsBloom = (1 - bsTension snapshot) * 0.8
    , zsSpinHalt = False
    }
  _ -> ZoneState 0.5 0.2 0.5 False

-- | Calculate zone reaction to delta
zoneReaction :: ResonanceZone -> BiometricDeltas -> ZoneState
zoneReaction zone deltas =
  let currentState = rzState zone
      sensitivity = rzSensitivity zone

      -- React based on zone type
      newActivity = clamp01 (zsActivity currentState + bdCoherence deltas * sensitivity)
      newTurbulence = case rzPosition zone of
        ZoneHeart -> clamp01 (zsTurbulence currentState - bdHRV deltas * 0.01)
        _ -> clamp01 (zsTurbulence currentState + bdTension deltas * sensitivity)
      newBloom = clamp02 (zsBloom currentState + bdCoherence deltas * sensitivity)

  in currentState
    { zsActivity = newActivity
    , zsTurbulence = newTurbulence
    , zsBloom = newBloom
    }

-- | Blend multiple zone states
zoneBlend :: [ZoneState] -> ZoneState
zoneBlend [] = ZoneState 0 0 0 False
zoneBlend states =
  let n = fromIntegral (length states)
      avgActivity = sum (map zsActivity states) / n
      avgTurbulence = sum (map zsTurbulence states) / n
      avgBloom = sum (map zsBloom states) / n
      anyHalt = any zsSpinHalt states
  in ZoneState avgActivity avgTurbulence avgBloom anyHalt

-- =============================================================================
-- Visual Output
-- =============================================================================

-- | Visual rendering parameters
data VisualParams = VisualParams
  { vpHue          :: !Double              -- ^ Hue [0, 360]
  , vpSaturation   :: !Double              -- ^ Saturation [0, 1]
  , vpBrightness   :: !Double              -- ^ Brightness [0, 1]
  , vpSpiralCount  :: !Int                 -- ^ Number of spirals
  , vpSpiralSpeed  :: !Double              -- ^ Rotation speed
  , vpBloomRadius  :: !Double              -- ^ Bloom radius
  , vpAlpha        :: !Double              -- ^ Transparency [0, 1]
  } deriving (Eq, Show)

-- | Convert modulation to visual parameters
toVisualParams :: AvatarFieldModulation -> VisualParams
toVisualParams modulation = VisualParams
  { vpHue = afmHueShift modulation
  , vpSaturation = afmIntensity modulation
  , vpBrightness = 0.5 + afmIntensity modulation * 0.5
  , vpSpiralCount = round (afmTorsionDensity modulation * 8)
  , vpSpiralSpeed = afmOscillationTempo modulation * spinMultiplier (afmSpinDirection modulation)
  , vpBloomRadius = afmBloomExpansion modulation
  , vpAlpha = afmIntensity modulation
  }

-- | Get hue from harmonic band
hueFromHarmonic :: BioHarmonic -> Double
hueFromHarmonic HarmonicAlpha = 240      -- Blue
hueFromHarmonic HarmonicTheta = 280      -- Purple
hueFromHarmonic HarmonicPhiCrown = 60    -- Yellow/gold
hueFromHarmonic HarmonicRootShift = 0    -- Red
hueFromHarmonic HarmonicAnkhFold = 120   -- Green

-- | Calculate bloom radius
bloomRadius :: AvatarFieldModulation -> Double
bloomRadius modulation =
  afmBloomExpansion modulation * (1 + afmIntensity modulation * 0.5)

-- =============================================================================
-- Fallback Rendering
-- =============================================================================

-- | Static field state for fallback rendering
data StaticFieldState = StaticFieldState
  { sfsModulation  :: !AvatarFieldModulation
  , sfsTimestamp   :: !Int
  , sfsValid       :: !Bool
  } deriving (Eq, Show)

-- | Render static field from last known state
renderStatic :: ExpressionPipeline -> Maybe StaticFieldState
renderStatic pipeline = case epLastSnapshot pipeline of
  Nothing -> Nothing
  Just snapshot ->
    let modulation = expressAvatarField snapshot
    in Just StaticFieldState
      { sfsModulation = modulation
      , sfsTimestamp = bsTimestamp snapshot
      , sfsValid = True
      }

-- | Get last known state
lastKnownState :: ExpressionPipeline -> Maybe AvatarFieldModulation
lastKnownState pipeline =
  expressAvatarField <$> epLastSnapshot pipeline

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Default torsion shells
defaultShells :: [TorsionShell]
defaultShells =
  [ TorsionShell 0.3 HarmonicAlpha SpinClockwise 0 0.8 (0.2, 0.4, 0.9)
  , TorsionShell 0.5 HarmonicTheta SpinClockwise (pi/4) 0.6 (0.5, 0.2, 0.8)
  , TorsionShell 0.7 HarmonicPhiCrown SpinClockwise (pi/2) 0.9 (0.9, 0.8, 0.2)
  , TorsionShell 0.9 HarmonicRootShift SpinNeutral 0 0.5 (0.8, 0.2, 0.2)
  ]

-- | Default resonance zones
defaultZones :: [ResonanceZone]
defaultZones =
  [ ResonanceZone "Crown" ZoneCrown (ZoneState 0.5 0 0.5 False) 1.2 (0.9, 0.9, 1.0)
  , ResonanceZone "Heart" ZoneHeart (ZoneState 0.5 0.2 0.5 False) 1.5 (0.2, 0.9, 0.4)
  , ResonanceZone "Solar" ZoneSolarPlexus (ZoneState 0.5 0.3 0.5 False) 1.0 (0.9, 0.9, 0.2)
  , ResonanceZone "Root" ZoneRoot (ZoneState 0.5 0.1 0.4 False) 0.8 (0.9, 0.2, 0.2)
  ]

-- | Get dominant harmonic from snapshot
dominantHarmonic :: BiometricSnapshot -> BioHarmonic
dominantHarmonic snapshot
  | bsCoherence snapshot > phi * 0.5 = HarmonicPhiCrown
  | bsHRV snapshot > 60 = HarmonicAlpha
  | bsRespiration snapshot < 8 = HarmonicTheta
  | bsTension snapshot > 0.7 = HarmonicRootShift
  | otherwise = HarmonicAnkhFold

-- | Get radius for harmonic
harmonicRadius :: BioHarmonic -> Double
harmonicRadius HarmonicAlpha = 0.4
harmonicRadius HarmonicTheta = 0.5
harmonicRadius HarmonicPhiCrown = 0.8
harmonicRadius HarmonicRootShift = 0.3
harmonicRadius HarmonicAnkhFold = 0.6

-- | Get weight for harmonic
harmonicWeight :: BioHarmonic -> Double
harmonicWeight HarmonicPhiCrown = phi
harmonicWeight HarmonicAnkhFold = phi * phiInverse
harmonicWeight _ = 1.0

-- | Get frequency for harmonic
harmonicFrequency :: BioHarmonic -> Double
harmonicFrequency HarmonicAlpha = 10
harmonicFrequency HarmonicTheta = 6
harmonicFrequency HarmonicPhiCrown = 7.83 * phi
harmonicFrequency HarmonicRootShift = 4
harmonicFrequency HarmonicAnkhFold = 7.83

-- | Get color for harmonic
harmonicColor :: BioHarmonic -> (Double, Double, Double)
harmonicColor HarmonicAlpha = (0.2, 0.4, 0.9)
harmonicColor HarmonicTheta = (0.5, 0.2, 0.8)
harmonicColor HarmonicPhiCrown = (0.9, 0.8, 0.2)
harmonicColor HarmonicRootShift = (0.8, 0.2, 0.2)
harmonicColor HarmonicAnkhFold = (0.2, 0.8, 0.4)

-- | Update shell from snapshot
updateShell :: BiometricSnapshot -> TorsionShell -> TorsionShell
updateShell snapshot shell =
  let newSpin = spinPolarity (bsCoherence snapshot)
      newPhase = tsPhase shell + bsRespirationPhase snapshot * 0.1
      newIntensity = clamp01 (tsIntensity shell * 0.9 + bsCoherence snapshot * 0.1)
  in shell { tsSpin = newSpin, tsPhase = newPhase, tsIntensity = newIntensity }

-- | Update zone from snapshot
updateZone :: BiometricSnapshot -> ResonanceZone -> ResonanceZone
updateZone snapshot zone =
  let newState = zoneFromBiometric (rzPosition zone) snapshot
  in zone { rzState = newState }

-- | Spin direction multiplier
spinMultiplier :: SpinDirection -> Double
spinMultiplier SpinClockwise = 1
spinMultiplier SpinCounter = (-1)
spinMultiplier SpinNeutral = 0

-- | Clamp to [0, 1]
clamp01 :: Double -> Double
clamp01 = max 0 . min 1

-- | Clamp to [0, 2]
clamp02 :: Double -> Double
clamp02 = max 0 . min 2
