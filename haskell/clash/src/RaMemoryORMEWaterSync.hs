{-|
Module      : RaMemoryORMEWaterSync
Description : ORME-Water Consciousness Loopback
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 72: Models quantum coherence coupling between structured water
and user memory/emotional imprinting. Simulates loopback dynamics between
memory states, emotional signatures, and avatar-water phase resonance.

Supports emotion intensity threshold (>0.75), phase transitions based on α,
and proximity-based or intent-driven entanglement.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaMemoryORMEWaterSync where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Thresholds (scaled to 255)
emotionIntensityThreshold :: Unsigned 8
emotionIntensityThreshold = 191   -- 0.75 * 255

sustainedDurationTicks :: Unsigned 4
sustainedDurationTicks = 8

alphaEntanglementMin :: Unsigned 8
alphaEntanglementMin = 153        -- 0.6 * 255

alphaLoopbackMin :: Unsigned 8
alphaLoopbackMin = 217            -- 0.85 * 255

-- | ORME phase states
data ORMEPhase
  = PhaseDormant
  | PhaseEntangled
  | PhaseLoopback
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Emotion valence tags
data EmotionValence
  = ValenceNeutral
  | ValenceJoy
  | ValenceAwe
  | ValenceGrief
  | ValenceLove
  | ValenceFear
  | ValencePeace
  | ValenceAnger
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Entanglement mode
data EntanglementMode
  = ModeProximity
  | ModeIntent
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Emotion waveform
data EmotionWaveform = EmotionWaveform
  { ewValence      :: EmotionValence
  , ewIntensity    :: Unsigned 8   -- 0-255
  , ewFrequency    :: Unsigned 16  -- Hz * 256
  , ewDuration     :: Unsigned 8   -- Ticks
  } deriving (Generic, NFDataX, Eq, Show)

-- | Fragment ID for memory linking
data FragmentID = FragmentID
  { fidNamespace :: Unsigned 8
  , fidIndex     :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Biometric state
data BioState = BioState
  { bsHRV        :: Unsigned 8
  , bsCoherence  :: Unsigned 8
  , bsFocusLevel :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | 3D coordinate (simplified)
data RaCoordinate = RaCoordinate
  { rcX :: Signed 16
  , rcY :: Signed 16
  , rcZ :: Signed 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Water coherence field
data WaterCoherenceField = WaterCoherenceField
  { wcfPhase           :: ORMEPhase
  , wcfHasImprint      :: Bool
  , wcfImprintValence  :: EmotionValence
  , wcfImprintIntensity :: Unsigned 8
  , wcfHasMemoryLink   :: Bool
  , wcfMemoryLink      :: FragmentID
  , wcfLocalAlpha      :: Unsigned 8
  , wcfLatticeCoherence :: Unsigned 8  -- Molecular lattice order
  } deriving (Generic, NFDataX, Eq, Show)

-- | Check emotion imprint trigger conditions
checkImportTrigger :: EmotionWaveform -> Bool
checkImportTrigger ew =
  ewIntensity ew >= emotionIntensityThreshold &&
  ewDuration ew >= resize sustainedDurationTicks

-- | Check if valence supports strong imprinting
isImprintValence :: EmotionValence -> Bool
isImprintValence v = case v of
  ValenceJoy   -> True
  ValenceAwe   -> True
  ValenceGrief -> True
  ValenceLove  -> True
  ValencePeace -> True
  _            -> False

-- | Compute ORME phase from coherence
computeORMEPhase :: BioState -> Unsigned 8 -> ORMEPhase -> ORMEPhase
computeORMEPhase bio fieldAlpha currentPhase =
  let -- Combined coherence (bio + field) / 2
      combined = (resize (bsCoherence bio) + resize fieldAlpha) `shiftR` 1 :: Unsigned 16
      combinedU8 = resize combined :: Unsigned 8
  in if combinedU8 >= alphaLoopbackMin then
       PhaseLoopback
     else if combinedU8 >= alphaEntanglementMin then
       -- Hysteresis: stay in loopback if already there
       if currentPhase == PhaseLoopback then PhaseLoopback
       else PhaseEntangled
     else
       -- Low coherence returns to dormant (with hysteresis)
       if currentPhase == PhaseEntangled then PhaseEntangled
       else PhaseDormant

-- | Compute lattice coherence (order of molecular structure)
computeLatticeCoherence :: Unsigned 8 -> Unsigned 8
computeLatticeCoherence coherence =
  -- Higher coherence = more ordered lattice (closer to 109.5° tetrahedral)
  -- Return as order metric (255 = perfect order)
  coherence

-- | Check proximity entanglement conditions
checkProximityEntanglement
  :: Unsigned 16  -- Distance
  -> Unsigned 8   -- Alpha 1
  -> Unsigned 8   -- Alpha 2
  -> ORMEPhase    -- Phase 1
  -> ORMEPhase    -- Phase 2
  -> Bool
checkProximityEntanglement dist alpha1 alpha2 phase1 phase2 =
  let radiusThreshold = 256 :: Unsigned 16  -- Scaled distance
      alphaDiff = if alpha1 > alpha2
                  then alpha1 - alpha2
                  else alpha2 - alpha1
      alphaAligned = alphaDiff < 51  -- Within 0.2 of each other
      bothActive = phase1 /= PhaseDormant && phase2 /= PhaseDormant
  in dist <= radiusThreshold && alphaAligned && bothActive

-- | Check intent entanglement conditions
checkIntentEntanglement :: BioState -> Bool
checkIntentEntanglement bio =
  bsFocusLevel bio >= 179 &&  -- 0.7 * 255
  bsHRV bio >= 153            -- 0.6 * 255

-- | Generate water coherence field
generateWaterLoopback
  :: BioState
  -> EmotionWaveform
  -> Unsigned 8        -- Field alpha
  -> ORMEPhase         -- Current phase
  -> WaterCoherenceField
generateWaterLoopback bio emotion fieldAlpha currentPhase =
  let -- Compute new phase
      newPhase = computeORMEPhase bio fieldAlpha currentPhase

      -- Check for imprint
      canImprint = checkImportTrigger emotion &&
                   isImprintValence (ewValence emotion) &&
                   newPhase /= PhaseDormant

      -- Compute lattice coherence
      combined = (bsCoherence bio + fieldAlpha) `shiftR` 1
      lattice = computeLatticeCoherence combined

  in WaterCoherenceField
       newPhase
       canImprint
       (if canImprint then ewValence emotion else ValenceNeutral)
       (if canImprint then ewIntensity emotion else 0)
       False                    -- Memory link set separately
       (FragmentID 0 0)
       fieldAlpha
       lattice

-- | Compute loopback strength
computeLoopbackStrength :: WaterCoherenceField -> Unsigned 8
computeLoopbackStrength wcf
  | wcfPhase wcf /= PhaseLoopback = 0
  | otherwise =
      let baseStrength = wcfLocalAlpha wcf
          -- Imprint amplifies (1 + intensity * 0.5)
          imprintFactor = if wcfHasImprint wcf
                          then 128 + (wcfImprintIntensity wcf `shiftR` 1)
                          else 128
          -- Memory link amplifies by 1.2
          linkFactor = if wcfHasMemoryLink wcf then 154 else 128
          -- Combine (scaled by 128)
          strength = (resize baseStrength * resize imprintFactor * resize linkFactor) `shiftR` 14 :: Unsigned 16
      in min 255 (resize strength)

-- | Water sync pipeline state
data WaterSyncState = WaterSyncState
  { wssCurrentPhase :: ORMEPhase
  , wssHasImprint   :: Bool
  } deriving (Generic, NFDataX)

-- | Initial state
initialWaterSyncState :: WaterSyncState
initialWaterSyncState = WaterSyncState PhaseDormant False

-- | Water sync input
data WaterSyncInput = WaterSyncInput
  { wsiBioState   :: BioState
  , wsiEmotion    :: EmotionWaveform
  , wsiFieldAlpha :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Water sync pipeline
waterSyncPipeline
  :: HiddenClockResetEnable dom
  => Signal dom WaterSyncInput
  -> Signal dom WaterCoherenceField
waterSyncPipeline input = mealy waterMealy initialWaterSyncState input
  where
    waterMealy state inp =
      let field = generateWaterLoopback
            (wsiBioState inp)
            (wsiEmotion inp)
            (wsiFieldAlpha inp)
            (wssCurrentPhase state)

          newState = WaterSyncState
            (wcfPhase field)
            (wcfHasImprint field)

      in (newState, field)

-- | Phase transition pipeline
phaseTransitionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BioState, Unsigned 8, ORMEPhase)
  -> Signal dom ORMEPhase
phaseTransitionPipeline = fmap (\(bio, alpha, phase) -> computeORMEPhase bio alpha phase)

-- | Loopback strength pipeline
loopbackStrengthPipeline
  :: HiddenClockResetEnable dom
  => Signal dom WaterCoherenceField
  -> Signal dom Unsigned 8
loopbackStrengthPipeline = fmap computeLoopbackStrength
