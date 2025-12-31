{-|
Module      : RaChamberTeslaVortex
Description : Scalar-Torsion Reactor Model
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 55: Scalar-torsion reactor model based on Tesla turbine vortex
dynamics, Keely harmonic overlays, and torsion field modulation.
Simulates layered scalar shells with configurable spin bias, torsion
envelope, and Keely-mode sympathetic resonance.

Based on φ-nested layer scaling.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaChamberTeslaVortex where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Base frequency (432 Hz scaled to fixed point)
baseFrequency :: Unsigned 16
baseFrequency = 432

-- | Coherence threshold for harmonic lock (0.72 * 255)
harmonicLockThreshold :: Unsigned 8
harmonicLockThreshold = 184

-- | Keely mode type (3, 6, or 9)
data KeelyMode
  = KeelyNone
  | Keely3     -- Thirds (1/3 ratio)
  | Keely6     -- Sixths (1/6 ratio)
  | Keely9     -- Ninths (1/9 ratio)
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Vortex profile
data VortexProfile = VortexProfile
  { vpBaseSpin     :: Unsigned 8    -- Base angular velocity (0-255)
  , vpTorsionBias  :: Signed 8      -- Field chirality (-128 to 127)
  , vpLayerCount   :: Unsigned 4    -- Number of layers (1-15)
  , vpKeelyMode    :: KeelyMode     -- Sympathetic resonance mode
  } deriving (Generic, NFDataX)

-- | Ra coordinate (spherical position)
data RaCoordinate = RaCoordinate
  { rcTheta :: Unsigned 9         -- 0-511 maps to 0-π
  , rcPhi   :: Unsigned 9         -- 0-511 maps to 0-2π
  , rcH     :: Unsigned 8         -- Height/shell (0-255)
  } deriving (Generic, NFDataX)

-- | Scalar field component
data ScalarFieldComponent = ScalarFieldComponent
  { sfcAmplitude     :: Unsigned 8   -- Field strength (0-255)
  , sfcPhase         :: Unsigned 16  -- Phase angle (0-65535 = 0-2π)
  , sfcFrequency     :: Unsigned 16  -- Oscillation frequency
  , sfcSpinDirection :: Signed 2     -- +1 = CW, -1 = CCW
  , sfcLayerIndex    :: Unsigned 4   -- Which shell layer
  } deriving (Generic, NFDataX)

-- | Emergence condition
data EmergenceCondition = EmergenceCondition
  { ecMinCoherence    :: Unsigned 8  -- Minimum coherence (0-255)
  , ecHarmonicLock    :: Bool        -- Whether harmonic lock achieved
  , ecTorsionAligned  :: Bool        -- Whether torsion field-aligned
  , ecResonanceDepth  :: Unsigned 4  -- Number of resonant layers
  } deriving (Generic, NFDataX)

-- | Vortex state
data VortexState = VortexState
  { vsProfile       :: VortexProfile
  , vsCurrentPhase  :: Unsigned 16  -- Current rotation phase
  , vsCoherence     :: Unsigned 8   -- Current coherence (0-255)
  , vsActiveLayers  :: Unsigned 4   -- Number of active layers
  } deriving (Generic, NFDataX)

-- | Sine lookup table (quarter wave, 64 entries, scaled to 127)
sineLUT :: Vec 64 (Signed 8)
sineLUT = $(listToVecTH
  [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
   48, 51, 54, 57, 59, 62, 65, 67, 70, 73, 75, 78, 80, 82, 85, 87,
   89, 91, 93, 95, 97, 99, 101, 102, 104, 105, 107, 108, 109, 110, 112, 113,
   114, 115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 120, 121, 121, 121, 121])

-- | Cosine lookup table (quarter wave, 64 entries, scaled to 127)
cosineLUT :: Vec 64 (Signed 8)
cosineLUT = $(listToVecTH
  [127, 126, 126, 126, 125, 125, 124, 123, 122, 121, 120, 119, 118, 116, 115, 113,
   112, 110, 108, 107, 105, 103, 101, 99, 97, 94, 92, 90, 87, 85, 82, 80,
   77, 75, 72, 69, 67, 64, 61, 58, 55, 52, 49, 46, 43, 40, 37, 34,
   31, 28, 25, 22, 19, 16, 12, 9, 6, 3, 0, -3, -6, -9, -12, -15])

-- | Get sine value from angle (0-65535 = 0-2π)
getSine :: Unsigned 16 -> Signed 8
getSine angle =
  let idx = resize (angle `shiftR` 10) :: Unsigned 6
  in sineLUT !! idx

-- | Get cosine value from angle
getCosine :: Unsigned 16 -> Signed 8
getCosine angle =
  let idx = resize (angle `shiftR` 10) :: Unsigned 6
  in cosineLUT !! idx

-- | Get Keely ratio numerator (for division)
-- Returns (numerator, denominator_shift) where ratio = numerator >> denominator_shift
getKeelyRatio :: KeelyMode -> (Unsigned 8, Unsigned 2)
getKeelyRatio KeelyNone = (255, 0)  -- 1.0
getKeelyRatio Keely3    = (85, 0)   -- ~1/3 * 255
getKeelyRatio Keely6    = (43, 0)   -- ~1/6 * 255
getKeelyRatio Keely9    = (28, 0)   -- ~1/9 * 255

-- | Compute layer frequency with φ scaling and Keely modulation
computeLayerFrequency
  :: Unsigned 16     -- Base frequency
  -> Unsigned 4      -- Layer index
  -> KeelyMode       -- Keely mode
  -> Unsigned 16     -- Result frequency
computeLayerFrequency baseFreq layerIdx keely =
  let -- φ-scale: multiply by 1657/1024 per layer
      phiFactor = iterate (* phi16) 1024 !! resize layerIdx
      phiScaled = (resize baseFreq * phiFactor) `shiftR` 10 :: Unsigned 32

      -- Apply Keely ratio
      (keelyNum, _) = getKeelyRatio keely
      result = (phiScaled * resize keelyNum) `shiftR` 8

  in resize $ min 65535 result

-- | Compute layer amplitude
-- Inner layers higher amplitude, modulated by torsion
computeLayerAmplitude
  :: Unsigned 4      -- Layer index
  -> Unsigned 4      -- Total layers
  -> Signed 8        -- Torsion bias
  -> Unsigned 8      -- Result amplitude
computeLayerAmplitude layerIdx layerCount torsionBias =
  let -- Base amplitude decreases with layer
      baseAmp = 255 - resize (layerIdx * 16)  -- Simple linear decay

      -- Torsion modulation based on parity
      parity = if layerIdx .&. 1 == 0 then 1 else -1 :: Signed 8
      torsionMod = 128 + resize ((torsionBias * parity) `shiftR` 3) :: Unsigned 16

      -- Apply modulation
      result = (resize baseAmp * torsionMod) `shiftR` 7

  in resize $ min 255 result

-- | Compute layer phase at given time tick
computeLayerPhase
  :: Unsigned 4      -- Layer index
  -> Unsigned 8      -- Base spin
  -> Signed 8        -- Torsion bias
  -> Unsigned 16     -- Time tick
  -> Unsigned 16     -- Result phase
computeLayerPhase layerIdx baseSpin torsionBias timeTick =
  let -- Layer spin rate scales with √φ ≈ 1.272
      layerFactor = 1024 + resize layerIdx * 100 :: Unsigned 16
      layerSpin = (resize baseSpin * layerFactor) `shiftR` 10 :: Unsigned 16

      -- Direction based on torsion and layer parity
      direction = if torsionBias >= 0 then 1 else -1 :: Signed 16
      parityFlip = if layerIdx .&. 1 == 1 then -1 else 1 :: Signed 16
      signedPhase = resize layerSpin * resize timeTick * direction * parityFlip

  in resize signedPhase

-- | Generate scalar field component for a layer
generateLayerComponent
  :: VortexProfile
  -> Unsigned 4      -- Layer index
  -> Unsigned 16     -- Time tick
  -> ScalarFieldComponent
generateLayerComponent profile layerIdx timeTick =
  let freq = computeLayerFrequency baseFrequency layerIdx (vpKeelyMode profile)
      amp = computeLayerAmplitude layerIdx (vpLayerCount profile) (vpTorsionBias profile)
      phase = computeLayerPhase layerIdx (vpBaseSpin profile) (vpTorsionBias profile) timeTick

      -- Spin direction
      spin = if vpTorsionBias profile >= 0
             then if layerIdx .&. 1 == 0 then 1 else -1
             else if layerIdx .&. 1 == 0 then -1 else 1

  in ScalarFieldComponent amp phase freq spin layerIdx

-- | Compute coherence at coordinate from field components
-- Uses phase alignment weighted by distance
computeCoherence
  :: Vec 8 ScalarFieldComponent  -- Up to 8 components
  -> Unsigned 4                   -- Active component count
  -> Unsigned 8                   -- Coordinate height (h)
  -> Unsigned 8                   -- Result coherence
computeCoherence components activeCount coordH =
  let -- Accumulate weighted phase vectors
      accumPhase :: (Signed 16, Signed 16, Unsigned 16)
                 -> ScalarFieldComponent
                 -> (Signed 16, Signed 16, Unsigned 16)
      accumPhase (sumX, sumY, totalW) comp =
        if sfcLayerIndex comp < activeCount
        then
          let layerH = resize (sfcLayerIndex comp) * 32 :: Unsigned 8
              dist = if coordH > layerH
                     then coordH - layerH
                     else layerH - coordH
              weight = resize (sfcAmplitude comp) `shiftR` (resize dist `shiftR` 3) :: Unsigned 16

              cosPhase = getCosine (sfcPhase comp)
              sinPhase = getSine (sfcPhase comp)

              newSumX = sumX + resize (resize weight * resize cosPhase `shiftR` 7 :: Signed 32)
              newSumY = sumY + resize (resize weight * resize sinPhase `shiftR` 7 :: Signed 32)
              newTotalW = totalW + weight
          in (newSumX, newSumY, newTotalW)
        else (sumX, sumY, totalW)

      (phaseX, phaseY, totalWeight) = foldl accumPhase (0, 0, 0) components

      -- Coherence = magnitude of average phase vector
      avgX = if totalWeight > 0
             then resize (phaseX * 256 `div` resize totalWeight) :: Signed 16
             else 0
      avgY = if totalWeight > 0
             then resize (phaseY * 256 `div` resize totalWeight) :: Signed 16
             else 0

      -- Approximate magnitude (|x| + |y|) * 0.7
      absX = if avgX < 0 then -avgX else avgX
      absY = if avgY < 0 then -avgY else avgY
      magnitude = (resize absX + resize absY) * 180 `shiftR` 8 :: Unsigned 16

  in resize $ min 255 magnitude

-- | Bind vortex chamber to coordinate
bindVortexChamber
  :: VortexProfile
  -> RaCoordinate
  -> Vec 8 ScalarFieldComponent
  -> Unsigned 4                   -- Active count
  -> EmergenceCondition
bindVortexChamber profile coord components activeCount =
  let coherence = computeCoherence components activeCount (rcH coord)

      -- Harmonic lock at coherence > 0.72
      harmonicLock = coherence > harmonicLockThreshold

      -- Torsion alignment based on coordinate phi
      torsionPhase = resize (rcPhi coord) :: Unsigned 8
      targetPhase = 128 + resize (vpTorsionBias profile) :: Unsigned 8
      torsionDiff = if torsionPhase > targetPhase
                    then torsionPhase - targetPhase
                    else targetPhase - torsionPhase
      torsionAligned = torsionDiff < 51  -- ~0.2 * 255

      -- Count resonant layers (amplitude > 127)
      countResonant acc comp =
        if sfcAmplitude comp > 127 && sfcLayerIndex comp < activeCount
        then acc + 1
        else acc
      resonanceDepth = foldl countResonant 0 components

  in EmergenceCondition coherence harmonicLock torsionAligned resonanceDepth

-- | Compute total field energy
computeFieldEnergy
  :: Vec 8 ScalarFieldComponent
  -> Unsigned 4
  -> Unsigned 16
computeFieldEnergy components activeCount =
  let accEnergy acc comp =
        if sfcLayerIndex comp < activeCount
        then
          let ampSq = resize (sfcAmplitude comp) * resize (sfcAmplitude comp) :: Unsigned 32
              freqScale = resize (sfcFrequency comp) `shiftR` 4 :: Unsigned 32
              energy = (ampSq * freqScale) `shiftR` 12
          in acc + resize energy
        else acc
  in foldl accEnergy 0 components

-- | Compute torsion moment
computeTorsionMoment
  :: Vec 8 ScalarFieldComponent
  -> Unsigned 4
  -> Signed 16
computeTorsionMoment components activeCount =
  let accMoment acc comp =
        if sfcLayerIndex comp < activeCount
        then
          let layerWeight = 256 `div` (resize (sfcLayerIndex comp) + 1) :: Unsigned 16
              spinVal = resize (sfcSpinDirection comp) :: Signed 16
              contribution = resize (sfcAmplitude comp) * spinVal * resize layerWeight `shiftR` 8
          in acc + contribution
        else acc
  in foldl accMoment 0 components

-- | Vortex simulation step
vortexStep
  :: VortexState
  -> Unsigned 16     -- Time tick
  -> RaCoordinate    -- Sample coordinate
  -> (VortexState, EmergenceCondition)
vortexStep state timeTick coord =
  let profile = vsProfile state

      -- Generate components for each layer
      genComp i = generateLayerComponent profile (resize i) timeTick
      components = map genComp (iterateI (+1) (0 :: Unsigned 4))

      -- Compute coherence
      newCoherence = computeCoherence components (vpLayerCount profile) (rcH coord)

      -- Bind chamber
      emergence = bindVortexChamber profile coord components (vpLayerCount profile)

      -- Update state
      newState = state
        { vsCurrentPhase = vsCurrentPhase state + resize (vpBaseSpin profile)
        , vsCoherence = newCoherence
        , vsActiveLayers = ecResonanceDepth emergence
        }

  in (newState, emergence)

-- | Vortex chamber pipeline
vortexPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (VortexProfile, Unsigned 16, RaCoordinate)
  -> Signal dom EmergenceCondition
vortexPipeline input = mealy vortexMealy initialState input
  where
    initialState = VortexState
      (VortexProfile 128 0 4 KeelyNone)
      0 128 4

    vortexMealy state (profile, tick, coord) =
      let (newState, emergence) = vortexStep (state { vsProfile = profile }) tick coord
      in (newState, emergence)

-- | Tesla phase map generation
teslaPhaseMap
  :: VortexProfile
  -> Vec 8 (Unsigned 16, Unsigned 8)  -- (phase_delta, amplitude_mod)
teslaPhaseMap profile =
  let genEntry i =
        let layerIdx = resize i :: Unsigned 4
            -- Phase delta based on layer position
            baseDelta = (65536 * resize layerIdx) `div` resize (vpLayerCount profile)
            torsionOffset = resize (vpTorsionBias profile) * 512 `shiftR` 7 :: Signed 32
            phaseDelta = resize (resize baseDelta + torsionOffset) :: Unsigned 16

            -- Amplitude modulation from Keely
            (keelyNum, _) = getKeelyRatio (vpKeelyMode profile)
            keelyAngle = resize layerIdx * resize keelyNum `shiftR` 2 :: Unsigned 16
            ampMod = 128 + resize (getCosine keelyAngle) :: Unsigned 8

        in (phaseDelta, ampMod)
  in map genEntry (iterateI (+1) (0 :: Unsigned 4))
