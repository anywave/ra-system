{-|
Module      : RaChamberSynthesis
Description : Multi-Layer Scalar Chamber Generator
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 60: Multi-layer scalar chamber generator for fragment emergence,
avatar stability, and scalar-based experience rendering.

Uses φ^n depth sequencing with OmegaFormat harmonic mapping.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaChamberSynthesis where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Base depth (0.618 * 1024)
baseDepth :: Unsigned 16
baseDepth = 633

-- | Max chamber layers
maxChamberLayers :: Unsigned 4
maxChamberLayers = 8

-- | Coherence thresholds
coherenceEmergence :: Unsigned 8
coherenceEmergence = 230  -- 0.9 * 255

coherenceResonant :: Unsigned 8
coherenceResonant = 184   -- 0.72 * 255

coherenceActive :: Unsigned 8
coherenceActive = 128     -- 0.5 * 255

coherenceAwakening :: Unsigned 8
coherenceAwakening = 77   -- 0.3 * 255

-- | Torsion state
data TorsionState
  = TorsionNormal
  | TorsionInverted
  | TorsionNull
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Coherence band
data CoherenceBand
  = BandDormant
  | BandAwakening
  | BandActive
  | BandResonant
  | BandEmergence
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | OmegaFormat (l, m spherical harmonics)
data OmegaFormat = OmegaFormat
  { ofL :: Unsigned 3   -- 0-7
  , ofM :: Signed 4     -- -l to +l
  } deriving (Generic, NFDataX, Eq)

-- | Scalar layer
data ScalarLayer = ScalarLayer
  { slDepth      :: Unsigned 16   -- Depth (fixed point)
  , slAmplitude  :: Unsigned 8    -- 0-255 amplitude
  , slPhase      :: Unsigned 16   -- 0-65535 = 0-2π
  , slOmega      :: OmegaFormat
  , slLayerIndex :: Unsigned 4
  } deriving (Generic, NFDataX)

-- | Torsion signature
data TorsionSignature = TorsionSignature
  { tsState        :: TorsionState
  , tsBias         :: Signed 8       -- -128 to 127
  , tsRotationRate :: Unsigned 8     -- Rotation rate
  } deriving (Generic, NFDataX)

-- | Biometric field input
data BiometricField = BiometricField
  { bfCoherence      :: Unsigned 8   -- 0-255
  , bfHRV            :: Unsigned 8   -- HRV (scaled)
  , bfBreathRate     :: Unsigned 8   -- Breaths per minute
  , bfPhaseAlignment :: Unsigned 8   -- 0-255
  } deriving (Generic, NFDataX)

-- | Fragment requirement
data FragmentRequirement = FragmentRequirement
  { frFragId        :: Unsigned 8
  , frMinCoherence  :: Unsigned 8
  , frPreferredL    :: Unsigned 3    -- Preferred harmonic l
  , frTorsionNormal :: Bool          -- Allow normal torsion
  , frTorsionInvert :: Bool          -- Allow inverted torsion
  , frTorsionNull   :: Bool          -- Allow null torsion
  , frIsValid       :: Bool          -- Entry is valid
  } deriving (Generic, NFDataX)

-- | Scalar chamber
data ScalarChamber = ScalarChamber
  { scChamberId     :: Unsigned 32
  , scDepthProfile  :: Vec 8 ScalarLayer
  , scCoherenceZone :: CoherenceBand
  , scResonancePeak :: Unsigned 8
  , scTorsionTune   :: TorsionSignature
  , scHarmonicRoots :: Vec 5 OmegaFormat
  , scSafeFor       :: Vec 4 Unsigned 8   -- FragmentIDs that fit
  , scSafeCount     :: Unsigned 3
  } deriving (Generic, NFDataX)

-- | Synthesis result
data SynthesisResult = SynthesisResult
  { srChamber      :: ScalarChamber
  , srSuccess      :: Bool
  , srUnplaceable  :: Vec 4 Unsigned 8   -- Unplaceable fragment IDs
  , srUnplaceCount :: Unsigned 3
  } deriving (Generic, NFDataX)

-- | Visualization element type
data VisElementType
  = VisPulseRing
  | VisNodalGeometry
  | VisHarmonicLoop
  | VisTorsionSwirl
  | VisSymmetryRing
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Visualization element
data VisElement = VisElement
  { veType       :: VisElementType
  , vePositionX  :: Signed 16
  , vePositionY  :: Signed 16
  , vePositionZ  :: Signed 16
  , veIntensity  :: Unsigned 8
  , vePhase      :: Unsigned 16
  , veLayerIndex :: Unsigned 4
  } deriving (Generic, NFDataX)

-- | Compute φ^n depth for a layer
computePhiDepth :: Unsigned 4 -> Unsigned 16
computePhiDepth layerIdx =
  let -- Iterative φ multiplication
      phiPowers = iterate (\d -> (d * phi16) `shiftR` 10) baseDepth
  in phiPowers !! layerIdx

-- | Classify coherence into band
classifyCoherenceBand :: Unsigned 8 -> CoherenceBand
classifyCoherenceBand coh
  | coh >= coherenceEmergence = BandEmergence
  | coh >= coherenceResonant  = BandResonant
  | coh >= coherenceActive    = BandActive
  | coh >= coherenceAwakening = BandAwakening
  | otherwise                 = BandDormant

-- | Generate default harmonic roots
generateHarmonicRoots :: Vec 5 OmegaFormat
generateHarmonicRoots = $(listToVecTH
  [ OmegaFormat 0 0
  , OmegaFormat 1 0
  , OmegaFormat 2 0
  , OmegaFormat 3 0
  , OmegaFormat 4 0
  ])

-- | Compute layer amplitude
computeLayerAmplitude
  :: Unsigned 4    -- Layer index
  -> Unsigned 8    -- Coherence
  -> Signed 8      -- Torsion bias
  -> Unsigned 8
computeLayerAmplitude layerIdx coherence torsionBias =
  let -- Base amplitude (inner = higher)
      baseAmp = 255 - resize layerIdx * 16 :: Unsigned 16

      -- Coherence factor (0.5 + 0.5 * coh)
      cohFactor = 128 + (resize coherence `shiftR` 1) :: Unsigned 16

      -- Torsion modulation based on parity
      parity = if layerIdx .&. 1 == 0 then 1 else -1 :: Signed 16
      torsionMod = 256 + resize (resize torsionBias * parity `shiftR` 2) :: Unsigned 16

      -- Combined
      combined = (baseAmp * cohFactor * torsionMod) `shiftR` 16

  in resize $ min 255 combined

-- | Compute layer phase
computeLayerPhase :: Unsigned 4 -> Unsigned 8 -> Unsigned 16
computeLayerPhase layerIdx bioPhase =
  let -- Phase offset based on layer (φ * 0.5 per layer)
      phaseOffset = resize layerIdx * 829 :: Unsigned 16  -- ~π/4 per layer
      basePhase = resize bioPhase `shiftL` 8 :: Unsigned 16
  in basePhase + phaseOffset

-- | Select omega for layer
selectOmegaForLayer :: Unsigned 4 -> OmegaFormat
selectOmegaForLayer layerIdx =
  let l = resize (layerIdx .&. 7) :: Unsigned 3
  in OmegaFormat l 0

-- | Create torsion signature from biometric field
createTorsionSignature :: BiometricField -> TorsionSignature
createTorsionSignature bio =
  let state = if bfCoherence bio < coherenceAwakening
              then TorsionNull
              else if bfPhaseAlignment bio < 77
                   then TorsionInverted
                   else TorsionNormal

      -- Bias from HRV (normalized around 128)
      bias = resize (bfHRV bio) - 128 :: Signed 8

      -- Rotation rate from breath
      rotRate = (resize (bfBreathRate bio) * 21) `shiftR` 4 :: Unsigned 8

  in TorsionSignature state bias rotRate

-- | Check if fragment is compatible with chamber
checkFragmentCompatible
  :: FragmentRequirement
  -> Unsigned 8        -- Chamber resonance peak
  -> TorsionState      -- Chamber torsion state
  -> Vec 5 OmegaFormat -- Chamber harmonic roots
  -> Bool
checkFragmentCompatible frag resonance torsion roots
  | not (frIsValid frag) = False
  | resonance < frMinCoherence frag = False
  | otherwise =
      let -- Check torsion
          torsionOk = case torsion of
            TorsionNormal   -> frTorsionNormal frag
            TorsionInverted -> frTorsionInvert frag
            TorsionNull     -> frTorsionNull frag

          -- Check shell match (any root with matching l)
          shellMatch = foldl
            (\acc omega -> acc || ofL omega == frPreferredL frag)
            False
            roots

      in torsionOk && shellMatch

-- | Synthesize scalar chamber
synthesizeChamber
  :: Unsigned 8              -- User ID
  -> BiometricField
  -> Vec 4 FragmentRequirement
  -> SynthesisResult
synthesizeChamber userId bio fragments =
  let -- Create torsion signature
      torsion = createTorsionSignature bio

      -- Generate harmonic roots
      harmonicRoots = generateHarmonicRoots

      -- Build layers
      buildLayer :: Unsigned 4 -> ScalarLayer
      buildLayer i = ScalarLayer
        (computePhiDepth i)
        (computeLayerAmplitude i (bfCoherence bio) (tsBias torsion))
        (computeLayerPhase i (bfPhaseAlignment bio))
        (selectOmegaForLayer i)
        i

      layers = map buildLayer $(listToVecTH [0..7 :: Unsigned 4])

      -- Compute resonance peak
      avgAmp = foldl (\s l -> s + resize (slAmplitude l)) (0 :: Unsigned 16) layers `div` 8
      resonancePeak = resize ((resize (bfCoherence bio) * avgAmp) `shiftR` 8 :: Unsigned 16)

      -- Classify coherence zone
      coherenceZone = classifyCoherenceBand resonancePeak

      -- Check fragment compatibility
      checkFrag :: FragmentRequirement -> Bool
      checkFrag f = checkFragmentCompatible f resonancePeak (tsState torsion) harmonicRoots

      -- Collect safe and unplaceable
      collectResults
        :: (Vec 4 (Unsigned 8), Unsigned 3, Vec 4 (Unsigned 8), Unsigned 3)
        -> Unsigned 2
        -> (Vec 4 (Unsigned 8), Unsigned 3, Vec 4 (Unsigned 8), Unsigned 3)
      collectResults (safe, safeC, unp, unpC) i =
        let frag = fragments !! i
        in if not (frIsValid frag)
           then (safe, safeC, unp, unpC)
           else if checkFrag frag
                then (replace (resize safeC) (frFragId frag) safe, safeC + 1, unp, unpC)
                else (safe, safeC, replace (resize unpC) (frFragId frag) unp, unpC + 1)

      (safeFor, safeCount, unplaceable, unplaceCount) =
        foldl collectResults
              (repeat 0, 0, repeat 0, 0)
              $(listToVecTH [0..3 :: Unsigned 2])

      -- Build chamber
      chamber = ScalarChamber
        (resize userId `shiftL` 24)  -- Simple chamber ID
        layers
        coherenceZone
        resonancePeak
        torsion
        harmonicRoots
        safeFor
        safeCount

      success = unplaceCount == 0

  in SynthesisResult chamber success unplaceable unplaceCount

-- | Export chamber visualization
exportChamberVisualization
  :: ScalarChamber
  -> Vec 16 VisElement
exportChamberVisualization chamber =
  let -- Create pulse ring for each layer
      createPulseRing :: Unsigned 4 -> VisElement
      createPulseRing i =
        let layer = scDepthProfile chamber !! i
        in VisElement
             VisPulseRing
             0 0 (resize $ slDepth layer)
             (slAmplitude layer)
             (slPhase layer)
             i

      pulseRings = map createPulseRing $(listToVecTH [0..7 :: Unsigned 4])

      -- Create nodal geometry (simplified)
      createNodalElement :: Unsigned 4 -> VisElement
      createNodalElement i =
        let layer = scDepthProfile chamber !! (i .&. 7)
            angle = resize i * 8192 :: Unsigned 16  -- Distribute around circle
            cosA = if angle < 16384 then 127 else -127  -- Simplified
            sinA = if angle >= 8192 && angle < 49152 then 127 else -127
            depth = slDepth layer
            x = (resize cosA * resize depth) `shiftR` 10 :: Signed 16
            y = (resize sinA * resize depth) `shiftR` 10 :: Signed 16
        in VisElement
             VisNodalGeometry
             x y (resize depth)
             (slAmplitude layer `shiftR` 1)
             (slPhase layer)
             (i .&. 7)

      nodalElements = map createNodalElement $(listToVecTH [8..15 :: Unsigned 4])

  in pulseRings ++ nodalElements

-- | Synthesis pipeline
synthesisPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, BiometricField, Vec 4 FragmentRequirement)
  -> Signal dom SynthesisResult
synthesisPipeline input =
  (\(uid, bio, frags) -> synthesizeChamber uid bio frags) <$> input

-- | Visualization pipeline
visualizationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom ScalarChamber
  -> Signal dom (Vec 16 VisElement)
visualizationPipeline input = exportChamberVisualization <$> input
