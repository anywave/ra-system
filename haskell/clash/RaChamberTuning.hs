{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

-- | Prompt 20: Ra.Chamber.Tuning - Biometric Lattice & Coherence Tensor
-- FPGA module for real-time chamber tuning based on multi-vector
-- biometric input with coherence tensor derivation.
--
-- Codex References:
-- - Ra.Emergence: Shell activation based on coherence
-- - Ra.Coherence: Tensor derivation from biometric vectors
-- - P16: Overlay integration
-- - P19: Domain safety
--
-- Features:
-- - 4-vector BiometricLattice input
-- - CoherenceTensor with tensor-sum torsion
-- - Coherence-scaled frequencies (432 + c*88)
-- - Shell overlap with priority logic
-- - Legacy BiometricState adapter

module RaChamberTuning where

import Clash.Prelude
import GHC.Generics (Generic)

-- ============================================================================
-- Constants (Codex-aligned)
-- ============================================================================

-- | Base frequency for coherence scaling (432 Hz)
baseFrequency :: SFixed 16 16
baseFrequency = 432.0

-- | Frequency scale per coherence unit (88 Hz)
frequencyScale :: SFixed 16 16
frequencyScale = 88.0

-- | Octave anchor frequencies (fallback)
octaveAnchors :: Vec 4 (SFixed 16 16)
octaveAnchors = 111.0 :> 222.0 :> 444.0 :> 888.0 :> Nil

-- | Torsion threshold (0.2)
torsionThreshold :: SFixed 4 12
torsionThreshold = 0.2

-- | Therapeutic mode threshold (1/pi ~ 0.3183)
therapeuticThreshold :: SFixed 4 12
therapeuticThreshold = 0.3183

-- | Coherence band overlap (0.15)
bandOverlap :: SFixed 4 12
bandOverlap = 0.15

-- ============================================================================
-- Types - Biometric Vectors
-- ============================================================================

-- | Cardiovascular biometric vector
data CardiovascularVector = CardiovascularVector
  { cvHeartRate      :: Unsigned 8      -- bpm (40-220)
  , cvHrvRatio       :: SFixed 4 12     -- LF/HF ratio
  , cvHeartCoherence :: Unsigned 12     -- 0-4095 (0.0-1.0)
  } deriving (Generic, NFDataX, Show)

-- | Neuroelectrical biometric vector
data NeuroVector = NeuroVector
  { nvEegAlpha       :: SFixed 4 12     -- Alpha power
  , nvEegTheta       :: SFixed 4 12     -- Theta power
  , nvEegDelta       :: SFixed 4 12     -- Delta power
  , nvNeuroCoherence :: Unsigned 12     -- 0-4095
  } deriving (Generic, NFDataX, Show)

-- | Respiratory biometric vector
data RespirationVector = RespirationVector
  { rvBreathRate     :: Unsigned 8      -- breaths per minute
  , rvInhaleExhale   :: SFixed 4 12     -- [-1 to +1] symmetry
  , rvBreathCoherence :: Unsigned 12    -- 0-4095
  } deriving (Generic, NFDataX, Show)

-- | Electrodermal activity vector
data ElectrodermalVector = ElectrodermalVector
  { evGsrLevel       :: SFixed 4 12     -- GSR level
  , evTensionIndex   :: Unsigned 12     -- 0-4095 (0-1)
  , evSkinCoherence  :: Unsigned 12     -- 0-4095
  } deriving (Generic, NFDataX, Show)

-- | Unified biometric lattice (4 vectors)
data BiometricLattice = BiometricLattice
  { blCardio  :: CardiovascularVector
  , blNeuro   :: NeuroVector
  , blBreath  :: RespirationVector
  , blEda     :: ElectrodermalVector
  , blSeed    :: Unsigned 32            -- Lattice seed hash
  } deriving (Generic, NFDataX, Show)

-- | Legacy flat biometric state (adapter target)
data BiometricState = BiometricState
  { bsCoherence   :: Unsigned 12        -- Overall coherence
  , bsHeartRate   :: Unsigned 8
  , bsBreathRate  :: Unsigned 8
  , bsGsrLevel    :: SFixed 4 12
  , bsSeed        :: Unsigned 32
  } deriving (Generic, NFDataX, Show)

-- ============================================================================
-- Types - Coherence Tensor
-- ============================================================================

-- | Coherence tensor derived from lattice
data CoherenceTensor = CoherenceTensor
  { ctHeartScore   :: Unsigned 12       -- Heart coherence
  , ctNeuroScore   :: Unsigned 12       -- Neuro coherence
  , ctBreathScore  :: Unsigned 12       -- Breath coherence
  , ctSkinScore    :: Unsigned 12       -- Skin coherence
  , ctOverallScore :: Unsigned 12       -- Average
  , ctTorsionBias  :: Signed 2          -- -1, 0, +1
  , ctHarmonicSkew :: SFixed 4 12       -- Phase angle shift
  } deriving (Generic, NFDataX, Show)

-- ============================================================================
-- Types - Chamber & Shell
-- ============================================================================

-- | Unified chamber flavor (P20 + P21)
data ChamberFlavor
  = FlavorHealing
  | FlavorRetrieval
  | FlavorNavigation
  | FlavorDream
  | FlavorArchive
  | FlavorTherapeutic
  | FlavorExploratory
  deriving (Generic, NFDataX, Eq, Show)

-- | Resonance shell (single layer)
data ResonanceShell = ResonanceShell
  { rsLayerId        :: Unsigned 4      -- 0-15
  , rsHarmonicCarrier :: SFixed 16 16   -- Hz
  , rsBandLow        :: Unsigned 12     -- Band low (0-4095)
  , rsBandHigh       :: Unsigned 12     -- Band high (0-4095)
  , rsTorsionFactor  :: Signed 2        -- -1, 0, +1
  , rsFluidMemory    :: Bool            -- Has memory seed
  } deriving (Generic, NFDataX, Show)

-- | Chamber tuning profile output
data ChamberTuningProfile = ChamberTuningProfile
  { tpActiveShellCount :: Unsigned 4    -- Number of active shells
  , tpShell0           :: ResonanceShell
  , tpShell1           :: ResonanceShell
  , tpShell2           :: ResonanceShell
  , tpShell3           :: ResonanceShell
  , tpAmbientMode      :: ChamberFlavor
  , tpFeedbackTone     :: SFixed 16 16  -- Hz
  , tpTensor           :: CoherenceTensor
  } deriving (Generic, NFDataX, Show)

-- ============================================================================
-- Core Functions
-- ============================================================================

-- | Convert 12-bit unsigned to fixed point (0.0-1.0)
toFixed :: Unsigned 12 -> SFixed 4 12
toFixed x = fromIntegral x / 4096.0

-- | Convert fixed point to 12-bit unsigned
fromFixed :: SFixed 4 12 -> Unsigned 12
fromFixed x = truncateB $ x * 4096.0

-- | Coherence-scaled frequency: 432.0 + (c * 88.0)
coherenceScaledFreq :: Unsigned 12 -> SFixed 16 16
coherenceScaledFreq coh =
  let c = fromIntegral coh / 4096.0 :: SFixed 16 16
  in baseFrequency + (c * frequencyScale)

-- | Get octave anchor for fallback
getOctaveAnchor :: Unsigned 4 -> SFixed 16 16
getOctaveAnchor idx = octaveAnchors !! (resize idx :: Index 4)

-- | Generate coherence tensor from lattice
generateCoherenceTensor :: BiometricLattice -> CoherenceTensor
generateCoherenceTensor BiometricLattice{..} =
  let
    hc = cvHeartCoherence blCardio
    nc = nvNeuroCoherence blNeuro
    bc = rvBreathCoherence blBreath
    sc = evSkinCoherence blEda

    -- Overall = average
    total = resize hc + resize nc + resize bc + resize sc :: Unsigned 16
    overall = resize (total `shiftR` 2) :: Unsigned 12

    -- Tensor sum for torsion (centered around 2048 per vector = 8192 total)
    tensorSum = (fromIntegral total :: Signed 17) - 8192
    -- Threshold is 0.2 * 4096 * 4 = 3276.8 ~ 3277
    torsion = if tensorSum < -3277 then -1
              else if tensorSum > 3277 then 1
              else 0

    -- Harmonic skew = (alpha - delta) + inhale_exhale_sym
    alphaMinusDelta = nvEegAlpha blNeuro - nvEegDelta blNeuro
    skew = alphaMinusDelta + rvInhaleExhale blBreath
  in CoherenceTensor
       { ctHeartScore = hc
       , ctNeuroScore = nc
       , ctBreathScore = bc
       , ctSkinScore = sc
       , ctOverallScore = overall
       , ctTorsionBias = torsion
       , ctHarmonicSkew = skew
       }

-- | Build resonance shell for a vector
buildShell :: Unsigned 4 -> Unsigned 12 -> Signed 2 -> Bool -> ResonanceShell
buildShell layerId coherence torsion useOctave =
  let
    -- Frequency
    freq = if useOctave
           then getOctaveAnchor layerId
           else coherenceScaledFreq coherence

    -- Band with overlap (coherence ± 614 which is ~0.15 * 4096)
    bandLow = if coherence > 614 then coherence - 614 else 0
    bandHigh = if coherence < 3481 then coherence + 614 else 4095

    -- Fluid memory for high coherence (> 0.85 = 3481)
    hasMemory = coherence > 3481
  in ResonanceShell
       { rsLayerId = layerId
       , rsHarmonicCarrier = freq
       , rsBandLow = bandLow
       , rsBandHigh = bandHigh
       , rsTorsionFactor = torsion
       , rsFluidMemory = hasMemory
       }

-- | Check if coherence activates a shell
coherenceInBand :: Unsigned 12 -> ResonanceShell -> Bool
coherenceInBand coh shell =
  coh >= rsBandLow shell && coh <= rsBandHigh shell

-- | Generate tuning profile from lattice
generateTuningProfile :: BiometricLattice -> ChamberTuningProfile
generateTuningProfile lattice =
  let
    tensor = generateCoherenceTensor lattice

    -- Build shells for each vector
    shell0 = buildShell 0 (ctHeartScore tensor) (ctTorsionBias tensor) False
    shell1 = buildShell 1 (ctNeuroScore tensor) (ctTorsionBias tensor) False
    shell2 = buildShell 2 (ctBreathScore tensor) (ctTorsionBias tensor) False
    shell3 = buildShell 3 (ctSkinScore tensor) (ctTorsionBias tensor) False

    -- Count active shells (those where overall coherence falls in band)
    overall = ctOverallScore tensor
    active0 = if coherenceInBand overall shell0 then 1 else 0
    active1 = if coherenceInBand overall shell1 then 1 else 0
    active2 = if coherenceInBand overall shell2 then 1 else 0
    active3 = if coherenceInBand overall shell3 then 1 else 0
    activeCount = active0 + active1 + active2 + active3

    -- Ambient mode based on overall coherence
    mode = if overall < 1304  -- 0.3183 * 4096
           then FlavorTherapeutic
           else FlavorExploratory

    -- Feedback tone from max coherence vector
    maxCoh = maximum (ctHeartScore tensor :> ctNeuroScore tensor :>
                      ctBreathScore tensor :> ctSkinScore tensor :> Nil)
    tone = coherenceScaledFreq maxCoh
  in ChamberTuningProfile
       { tpActiveShellCount = activeCount
       , tpShell0 = shell0
       , tpShell1 = shell1
       , tpShell2 = shell2
       , tpShell3 = shell3
       , tpAmbientMode = mode
       , tpFeedbackTone = tone
       , tpTensor = tensor
       }

-- ============================================================================
-- Legacy Adapters
-- ============================================================================

-- | Convert lattice to legacy state
latticeToState :: BiometricLattice -> BiometricState
latticeToState lattice =
  let tensor = generateCoherenceTensor lattice
  in BiometricState
       { bsCoherence = ctOverallScore tensor
       , bsHeartRate = cvHeartRate (blCardio lattice)
       , bsBreathRate = rvBreathRate (blBreath lattice)
       , bsGsrLevel = evGsrLevel (blEda lattice)
       , bsSeed = blSeed lattice
       }

-- | Convert legacy state to lattice (with defaults)
stateToLattice :: BiometricState -> BiometricLattice
stateToLattice BiometricState{..} =
  BiometricLattice
    { blCardio = CardiovascularVector
        { cvHeartRate = bsHeartRate
        , cvHrvRatio = 1.5
        , cvHeartCoherence = bsCoherence
        }
    , blNeuro = NeuroVector
        { nvEegAlpha = 0.4
        , nvEegTheta = 0.3
        , nvEegDelta = 0.2
        , nvNeuroCoherence = bsCoherence
        }
    , blBreath = RespirationVector
        { rvBreathRate = bsBreathRate
        , rvInhaleExhale = 0.0
        , rvBreathCoherence = bsCoherence
        }
    , blEda = ElectrodermalVector
        { evGsrLevel = bsGsrLevel
        , evTensionIndex = 2048
        , evSkinCoherence = bsCoherence
        }
    , blSeed = bsSeed
    }

-- ============================================================================
-- Top-Level Entity
-- ============================================================================

-- | Chamber tuning FSM
chamberTuningFSM :: HiddenClockResetEnable dom
                 => Signal dom BiometricLattice
                 -> Signal dom ChamberTuningProfile
chamberTuningFSM = fmap generateTuningProfile

-- | Top entity for chamber tuning
{-# ANN chamberTuningTop
  (Synthesize
    { t_name   = "chamber_tuning_top"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 -- Cardio vector
                 , PortName "heart_rate"
                 , PortName "hrv_ratio"
                 , PortName "heart_coh"
                 -- Neuro vector
                 , PortName "eeg_alpha"
                 , PortName "eeg_theta"
                 , PortName "eeg_delta"
                 , PortName "neuro_coh"
                 -- Breath vector
                 , PortName "breath_rate"
                 , PortName "inhale_exhale"
                 , PortName "breath_coh"
                 -- EDA vector
                 , PortName "gsr_level"
                 , PortName "tension_idx"
                 , PortName "skin_coh"
                 -- Seed
                 , PortName "seed"
                 ]
    , t_output = PortProduct "output"
                 [ PortName "active_shell_count"
                 , PortName "shell0_freq"
                 , PortName "shell1_freq"
                 , PortName "shell2_freq"
                 , PortName "shell3_freq"
                 , PortName "ambient_mode"
                 , PortName "feedback_tone"
                 , PortName "overall_coherence"
                 , PortName "torsion_bias"
                 ]
    }) #-}
chamberTuningTop
  :: Clock System
  -> Reset System
  -> Enable System
  -- Cardio
  -> Signal System (Unsigned 8)     -- heart_rate
  -> Signal System (SFixed 4 12)    -- hrv_ratio
  -> Signal System (Unsigned 12)    -- heart_coh
  -- Neuro
  -> Signal System (SFixed 4 12)    -- eeg_alpha
  -> Signal System (SFixed 4 12)    -- eeg_theta
  -> Signal System (SFixed 4 12)    -- eeg_delta
  -> Signal System (Unsigned 12)    -- neuro_coh
  -- Breath
  -> Signal System (Unsigned 8)     -- breath_rate
  -> Signal System (SFixed 4 12)    -- inhale_exhale
  -> Signal System (Unsigned 12)    -- breath_coh
  -- EDA
  -> Signal System (SFixed 4 12)    -- gsr_level
  -> Signal System (Unsigned 12)    -- tension_idx
  -> Signal System (Unsigned 12)    -- skin_coh
  -- Seed
  -> Signal System (Unsigned 32)    -- seed
  -> Signal System ( Unsigned 4     -- active_shell_count
                   , SFixed 16 16   -- shell0_freq
                   , SFixed 16 16   -- shell1_freq
                   , SFixed 16 16   -- shell2_freq
                   , SFixed 16 16   -- shell3_freq
                   , Unsigned 3     -- ambient_mode (encoded)
                   , SFixed 16 16   -- feedback_tone
                   , Unsigned 12    -- overall_coherence
                   , Signed 2       -- torsion_bias
                   )
chamberTuningTop clk rst en
                 heartRate hrvRatio heartCoh
                 eegAlpha eegTheta eegDelta neuroCoh
                 breathRate inhaleExhale breathCoh
                 gsrLevel tensionIdx skinCoh
                 seed =
  withClockResetEnable clk rst en $
    let
      -- Build cardio vector
      cardio = CardiovascularVector <$> heartRate <*> hrvRatio <*> heartCoh

      -- Build neuro vector
      neuro = NeuroVector <$> eegAlpha <*> eegTheta <*> eegDelta <*> neuroCoh

      -- Build breath vector
      breath = RespirationVector <$> breathRate <*> inhaleExhale <*> breathCoh

      -- Build EDA vector
      eda = ElectrodermalVector <$> gsrLevel <*> tensionIdx <*> skinCoh

      -- Build lattice
      lattice = BiometricLattice <$> cardio <*> neuro <*> breath <*> eda <*> seed

      -- Generate profile
      profile = chamberTuningFSM lattice

      -- Encode ambient mode
      encodeMode m = case m of
        FlavorHealing     -> 0
        FlavorRetrieval   -> 1
        FlavorNavigation  -> 2
        FlavorDream       -> 3
        FlavorArchive     -> 4
        FlavorTherapeutic -> 5
        FlavorExploratory -> 6

      -- Extract output
      extractOut ChamberTuningProfile{..} =
        ( tpActiveShellCount
        , rsHarmonicCarrier tpShell0
        , rsHarmonicCarrier tpShell1
        , rsHarmonicCarrier tpShell2
        , rsHarmonicCarrier tpShell3
        , encodeMode tpAmbientMode
        , tpFeedbackTone
        , ctOverallScore tpTensor
        , ctTorsionBias tpTensor
        )
    in fmap extractOut profile

-- ============================================================================
-- Testbench
-- ============================================================================

-- | Test lattice (mid coherence)
testLatticeMid :: BiometricLattice
testLatticeMid = BiometricLattice
  { blCardio = CardiovascularVector 72 1.8 2253  -- 0.55
  , blNeuro = NeuroVector 0.45 0.33 0.22 2458    -- 0.60
  , blBreath = RespirationVector 13 0.05 2376    -- 0.58
  , blEda = ElectrodermalVector 0.7 1229 2130    -- 0.52
  , blSeed = 12345
  }

-- | Test lattice (high coherence)
testLatticeHigh :: BiometricLattice
testLatticeHigh = BiometricLattice
  { blCardio = CardiovascularVector 65 1.5 3481  -- 0.85
  , blNeuro = NeuroVector 0.5 0.3 0.2 3604       -- 0.88
  , blBreath = RespirationVector 12 0.1 3358     -- 0.82
  , blEda = ElectrodermalVector 0.5 819 3277     -- 0.80
  , blSeed = 54321
  }

-- | Test lattice (low coherence)
testLatticeLow :: BiometricLattice
testLatticeLow = BiometricLattice
  { blCardio = CardiovascularVector 90 2.5 614   -- 0.15
  , blNeuro = NeuroVector 0.2 0.5 0.4 737        -- 0.18
  , blBreath = RespirationVector 18 (-0.3) 491   -- 0.12
  , blEda = ElectrodermalVector 1.2 3277 819     -- 0.20
  , blSeed = 99999
  }

-- | Testbench inputs
testInputs :: Vec 3 BiometricLattice
testInputs = testLatticeMid :> testLatticeHigh :> testLatticeLow :> Nil

-- | Testbench entity
testBench :: Signal System ChamberTuningProfile
testBench = withClockResetEnable systemClockGen systemResetGen enableGen $
  chamberTuningFSM (fromList (toList testInputs))
