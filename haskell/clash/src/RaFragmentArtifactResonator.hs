{-|
Module      : RaFragmentArtifactResonator
Description : Physical Object Memory Fragment System
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 78: Allows physical objects to emit/contain memory fragments.
Fragments activate upon coherence match with scalar field.
Supports material signatures, decay, mutation, and grid/envelope tethering.

Static frequency tables for materials, decay with field cycles, leyline + contact tethering.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaFragmentArtifactResonator where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Thresholds
activationTolerance :: Unsigned 8
activationTolerance = 38       -- 0.15 * 255

decayRatePerCycle :: Unsigned 8
decayRatePerCycle = 13         -- 0.05 * 255

mutationThreshold :: Unsigned 8
mutationThreshold = 77         -- 0.3 * 255

-- | Material types
data MaterialType
  = MatQuartz
  | MatObsidian
  | MatGranite
  | MatCopper
  | MatGold
  | MatSilver
  | MatIron
  | MatLimestone
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Fragment origin types
data FragmentOrigin
  = OriginUser
  | OriginArtifact
  | OriginGridNode
  | OriginContact
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Tether types
data TetherType
  = TetherNone
  | TetherEarthNode
  | TetherContactEnvelope
  | TetherAvatarLineage
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Material → Base Frequency (Hz * 256, static lookup)
materialFrequency :: MaterialType -> Unsigned 16
materialFrequency mat = case mat of
  MatQuartz    -> 8388608    -- 32768 * 256
  MatObsidian  -> 2097152    -- 8192 * 256
  MatGranite   -> 1048576    -- 4096 * 256
  MatCopper    -> 4194304    -- 16384 * 256
  MatGold      -> 6291456    -- 24576 * 256
  MatSilver    -> 5242880    -- 20480 * 256
  MatIron      -> 3145728    -- 12288 * 256
  MatLimestone -> 524288     -- 2048 * 256

-- | Material → Alpha Affinity (scaled 0-255)
materialAlphaAffinity :: MaterialType -> Unsigned 8
materialAlphaAffinity mat = case mat of
  MatQuartz    -> 230   -- 0.9
  MatObsidian  -> 179   -- 0.7
  MatGranite   -> 128   -- 0.5
  MatCopper    -> 204   -- 0.8
  MatGold      -> 242   -- 0.95
  MatSilver    -> 217   -- 0.85
  MatIron      -> 153   -- 0.6
  MatLimestone -> 102   -- 0.4

-- | Material → Conductivity (scaled 0-255)
materialConductivity :: MaterialType -> Unsigned 8
materialConductivity mat = case mat of
  MatQuartz    -> 77    -- 0.3 (piezoelectric)
  MatObsidian  -> 26    -- 0.1
  MatGranite   -> 13    -- 0.05
  MatCopper    -> 217   -- 0.85
  MatGold      -> 242   -- 0.95
  MatSilver    -> 230   -- 0.9
  MatIron      -> 128   -- 0.5
  MatLimestone -> 5     -- 0.02

-- | Scalar signature
data ScalarSignature = ScalarSignature
  { ssBaseFrequency  :: Unsigned 16
  , ssAlphaAffinity  :: Unsigned 8
  , ssConductivity   :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute scalar signature from material
computeScalarSignature :: MaterialType -> ScalarSignature
computeScalarSignature mat = ScalarSignature
  (materialFrequency mat)
  (materialAlphaAffinity mat)
  (materialConductivity mat)

-- | Activation window
data ActivationWindow = ActivationWindow
  { awMinAlpha :: Unsigned 8
  , awMaxAlpha :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute activation window from signature
computeActivationWindow :: ScalarSignature -> ActivationWindow
computeActivationWindow sig =
  let center = ssAlphaAffinity sig
      -- Width based on conductivity (0.1 + cond * 0.2)
      widthBase = 26 :: Unsigned 16  -- 0.1 * 255
      widthCond = (resize (ssConductivity sig) * 51) `shiftR` 8 :: Unsigned 16  -- * 0.2
      halfWidth = (widthBase + widthCond) `shiftR` 1

      minA = if resize center < halfWidth then 0
             else resize center - resize halfWidth
      maxA = if resize center + halfWidth > 255 then 255
             else resize center + resize halfWidth
  in ActivationWindow (resize minA) (resize maxA)

-- | Scalar field
data ScalarField = ScalarField
  { sfAlpha     :: Unsigned 8
  , sfFrequency :: Unsigned 16
  , sfPhase     :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Fragment node
data FragmentNode = FragmentNode
  { fnOrigin      :: FragmentOrigin
  , fnCoherence   :: Unsigned 8     -- 0-255
  , fnPayloadHash :: Unsigned 16
  , fnDecayCycles :: Unsigned 8
  , fnMutated     :: Bool
  } deriving (Generic, NFDataX, Eq, Show)

-- | Artifact resonator
data ArtifactResonator = ArtifactResonator
  { arMaterialType    :: MaterialType
  , arSignature       :: ScalarSignature
  , arActivationWindow :: ActivationWindow
  , arCurrentlyActive :: Bool
  , arTetherType      :: TetherType
  , arFragmentCount   :: Unsigned 4
  } deriving (Generic, NFDataX, Eq, Show)

-- | Check coherence match
checkCoherenceMatch :: ScalarField -> ArtifactResonator -> Bool
checkCoherenceMatch field artifact =
  let minA = awMinAlpha (arActivationWindow artifact)
      maxA = awMaxAlpha (arActivationWindow artifact)
      alpha = sfAlpha field
  in alpha >= minA && alpha <= maxA

-- | Check frequency resonance
checkFrequencyResonance :: ScalarField -> ScalarSignature -> Bool
checkFrequencyResonance field sig =
  let baseFreq = ssBaseFrequency sig
      fieldFreq = sfFrequency field

      -- Check ratios: 1.0, φ, 0.5, 2.0
      -- Compute ratio * 1024
      ratio = if baseFreq == 0 then 0
              else (resize fieldFreq * 1024) `div` resize baseFreq :: Unsigned 32

      -- Check near 1024 (ratio 1.0)
      near1 = ratio > 870 && ratio < 1178   -- ±15%
      -- Check near 1657 (ratio φ)
      nearPhi = ratio > 1409 && ratio < 1905
      -- Check near 512 (ratio 0.5)
      nearHalf = ratio > 435 && ratio < 589
      -- Check near 2048 (ratio 2.0)
      near2 = ratio > 1741 && ratio < 2355

  in near1 || nearPhi || nearHalf || near2

-- | Apply field decay
applyFieldDecay :: FragmentNode -> Unsigned 8 -> FragmentNode
applyFieldDecay frag cycles =
  let newCycles = if fnDecayCycles frag > cycles
                  then fnDecayCycles frag - cycles
                  else 0

      -- Decay coherence by rate * cycles
      decayAmount = resize cycles * resize decayRatePerCycle :: Unsigned 16
      newCoh = if resize (fnCoherence frag) > decayAmount
               then fnCoherence frag - resize decayAmount
               else 0

  in FragmentNode
       (fnOrigin frag)
       newCoh
       (fnPayloadHash frag)
       newCycles
       (fnMutated frag)

-- | Check field alignment
checkFieldAlignment :: ScalarField -> ArtifactResonator -> Unsigned 8
checkFieldAlignment field artifact =
  let affinity = ssAlphaAffinity (arSignature artifact)
      fieldAlpha = sfAlpha field
      diff = if fieldAlpha > affinity
             then fieldAlpha - affinity
             else affinity - fieldAlpha
  in 255 - diff

-- | Apply misaligned reactivation (mutation)
applyMisalignedReactivation :: FragmentNode -> Unsigned 8 -> FragmentNode
applyMisalignedReactivation frag alignment =
  if alignment < mutationThreshold
  then FragmentNode
         (fnOrigin frag)
         ((fnCoherence frag * 204) `shiftR` 8)  -- * 0.8
         (fnPayloadHash frag `xor` 255)         -- Corrupt
         (fnDecayCycles frag)
         True                                   -- Mark mutated
  else frag

-- | Create artifact resonator
createArtifactResonator :: MaterialType -> TetherType -> ArtifactResonator
createArtifactResonator mat tether =
  let sig = computeScalarSignature mat
      window = computeActivationWindow sig
  in ArtifactResonator mat sig window False tether 0

-- | Artifact scanner input
data ScannerInput = ScannerInput
  { siField       :: ScalarField
  , siArtifact    :: ArtifactResonator
  , siFragment    :: FragmentNode
  , siDecayCycles :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Artifact scanner output
data ScannerOutput = ScannerOutput
  { soFragment   :: FragmentNode
  , soEmitted    :: Bool
  , soActive     :: Bool
  } deriving (Generic, NFDataX)

-- | Artifact scanner pipeline
artifactScannerPipeline
  :: HiddenClockResetEnable dom
  => Signal dom ScannerInput
  -> Signal dom ScannerOutput
artifactScannerPipeline = fmap scanArtifact
  where
    scanArtifact inp =
      let field = siField inp
          artifact = siArtifact inp
          frag = siFragment inp

          -- Check activation conditions
          cohMatch = checkCoherenceMatch field artifact
          freqRes = checkFrequencyResonance field (arSignature artifact)
          activated = cohMatch && freqRes

          -- Apply decay
          decayedFrag = applyFieldDecay frag (siDecayCycles inp)

          -- Check for emission
          canEmit = fnCoherence decayedFrag > 0 && fnDecayCycles decayedFrag > 0

          -- Check alignment for mutation
          alignment = checkFieldAlignment field artifact
          finalFrag = if activated
                      then applyMisalignedReactivation decayedFrag alignment
                      else decayedFrag

      in ScannerOutput finalFrag (activated && canEmit) activated

-- | Coherence match pipeline
coherenceMatchPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ScalarField, ArtifactResonator)
  -> Signal dom Bool
coherenceMatchPipeline = fmap (uncurry checkCoherenceMatch)

-- | Frequency resonance pipeline
frequencyResonancePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ScalarField, ScalarSignature)
  -> Signal dom Bool
frequencyResonancePipeline = fmap (uncurry checkFrequencyResonance)

-- | Material signature pipeline
materialSignaturePipeline
  :: HiddenClockResetEnable dom
  => Signal dom MaterialType
  -> Signal dom ScalarSignature
materialSignaturePipeline = fmap computeScalarSignature
