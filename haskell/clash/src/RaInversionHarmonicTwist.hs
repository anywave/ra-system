{-|
Module      : RaInversionHarmonicTwist
Description : Torsional Field Forces During Scalar Inversions
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 49: Models torsional field forces during harmonic inversions,
computing twist vectors, risk indices, and φ^n decay durations.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaInversionHarmonicTwist where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Risk thresholds (8-bit scaled)
riskLowThreshold :: Unsigned 8
riskLowThreshold = 51       -- 0.2 * 255

riskModerateThreshold :: Unsigned 8
riskModerateThreshold = 128 -- 0.5 * 255

riskHighThreshold :: Unsigned 8
riskHighThreshold = 191     -- 0.75 * 255

-- | Base duration for φ^n ticks
baseDurationPhiN :: Unsigned 4
baseDurationPhiN = 8

-- | Coherence threshold for negligible twist
negligibleTwistCoherence :: Unsigned 8
negligibleTwistCoherence = 242  -- 0.95 * 255

-- | Inversion polarity
data InversionPolarity
  = PolNormal
  | PolInverted
  deriving (Generic, NFDataX, Eq, Show)

-- | Risk level classification
data RiskLevel
  = RiskLow
  | RiskModerate
  | RiskHigh
  | RiskCritical
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Omega format (harmonic signature)
data OmegaFormat = OmegaFormat
  { ofOmegaL     :: Unsigned 4    -- Spherical harmonic l (0-9)
  , ofOmegaM     :: Signed 8      -- Spherical harmonic m
  , ofPhaseAngle :: Unsigned 16   -- Phase angle (0-65535 = 0-2π)
  , ofAmplitude  :: Unsigned 8    -- Field intensity (0-255)
  } deriving (Generic, NFDataX)

-- | Flux coherence state
data FluxCoherence = FluxCoherence
  { fcCoherence :: Unsigned 8
  , fcFlux      :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Twist vector
data TwistVector = TwistVector
  { tvThetaForce  :: Signed 16    -- Pull along θ
  , tvPhiForce    :: Signed 16    -- Pull along φ
  , tvHarmonicL   :: Unsigned 4   -- Harmonic mode l
  , tvHarmonicM   :: Signed 8     -- Harmonic mode m
  , tvPolarity    :: InversionPolarity
  , tvCoherenceMod :: Unsigned 8  -- Damping (0-255)
  } deriving (Generic, NFDataX)

-- | Twist envelope
data TwistEnvelope = TwistEnvelope
  { teNetTwist     :: TwistVector
  , teDurationPhiN :: Unsigned 4   -- Time in φ^n ticks to decay
  , teRiskIndex    :: Unsigned 8   -- Risk rating (0-255)
  } deriving (Generic, NFDataX)

-- | Sine lookup table for phase (quarter wave, 64 entries)
sineLUT :: Vec 64 (Signed 8)
sineLUT = $(listToVecTH
  [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
   48, 51, 54, 57, 59, 62, 65, 67, 70, 73, 75, 78, 80, 82, 85, 87,
   89, 91, 93, 95, 97, 99, 101, 102, 104, 105, 107, 108, 109, 110, 112, 113,
   114, 115, 116, 116, 117, 118, 118, 119, 119, 120, 120, 120, 121, 121, 121, 121])

-- | Cosine lookup table for phase (quarter wave, 64 entries)
cosineLUT :: Vec 64 (Signed 8)
cosineLUT = $(listToVecTH
  [127, 126, 126, 126, 125, 125, 124, 123, 122, 121, 120, 119, 118, 116, 115, 113,
   112, 110, 108, 107, 105, 103, 101, 99, 97, 94, 92, 90, 87, 85, 82, 80,
   77, 75, 72, 69, 67, 64, 61, 58, 55, 52, 49, 46, 43, 40, 37, 34,
   31, 28, 25, 22, 19, 16, 12, 9, 6, 3, 0, -3, -6, -9, -12, -15])

-- | Exponential decay lookup (exp(-x) * 255 for x = 0..8)
expDecayLUT :: Vec 9 (Unsigned 8)
expDecayLUT = $(listToVecTH
  [255, 155, 94, 57, 35, 21, 13, 8, 5])

-- | Get sine value from phase (0-65535 = 0-2π)
getSine :: Unsigned 16 -> Signed 8
getSine phase =
  let idx = resize (phase `shiftR` 10) :: Unsigned 6  -- Map to 0-63
  in sineLUT !! idx

-- | Get cosine value from phase (0-65535 = 0-2π)
getCosine :: Unsigned 16 -> Signed 8
getCosine phase =
  let idx = resize (phase `shiftR` 10) :: Unsigned 6
  in cosineLUT !! idx

-- | Compute theta force
computeThetaForce :: OmegaFormat -> InversionPolarity -> Signed 16
computeThetaForce omega polarity =
  let -- Base force from phase offset
      sinVal = getSine (ofPhaseAngle omega)
      baseForce = resize sinVal * resize (ofAmplitude omega) `shiftR` 7

      -- Scale by harmonic order (1 + l * 0.1 ≈ 1 + l/10)
      harmonicScale = 10 + resize (ofOmegaL omega)
      scaledForce = (baseForce * resize harmonicScale) `div` 10

      -- Inversion flips sign
      finalForce = case polarity of
        PolNormal   -> scaledForce
        PolInverted -> -scaledForce

  in max (-127) (min 127 finalForce)

-- | Compute phi force
computePhiForce :: OmegaFormat -> InversionPolarity -> Signed 16
computePhiForce omega polarity =
  let -- Base force from phase offset
      cosVal = getCosine (ofPhaseAngle omega)
      baseForce = resize cosVal * resize (ofAmplitude omega) `shiftR` 7

      -- Scale by m index (1 + |m| * 0.15 ≈ 1 + |m|*3/20)
      mAbs = if ofOmegaM omega < 0 then -(ofOmegaM omega) else ofOmegaM omega
      angularScale = 20 + resize mAbs * 3
      scaledForce = (baseForce * resize angularScale) `div` 20

      -- Inversion flips sign
      finalForce = case polarity of
        PolNormal   -> scaledForce
        PolInverted -> -scaledForce

  in max (-127) (min 127 finalForce)

-- | Compute twist vector magnitude (simplified: |theta| + |phi|)
computeTwistMagnitude :: TwistVector -> Unsigned 8
computeTwistMagnitude tv =
  let thetaAbs = if tvThetaForce tv < 0 then resize (-tvThetaForce tv) else resize (tvThetaForce tv)
      phiAbs = if tvPhiForce tv < 0 then resize (-tvPhiForce tv) else resize (tvPhiForce tv)
      sum = thetaAbs + phiAbs :: Unsigned 16
      -- Normalize: max sum is ~254, scale to 0-255
  in resize $ min 255 sum

-- | Compute risk index: |twist| * (1 - coherence) / 255
computeRiskIndex :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
computeRiskIndex twistMag coherence =
  let invCoherence = 255 - coherence
      product = resize twistMag * resize invCoherence :: Unsigned 16
  in resize (product `shiftR` 8)

-- | Get risk level from index
getRiskLevel :: Unsigned 8 -> RiskLevel
getRiskLevel idx
  | idx >= riskHighThreshold     = RiskCritical
  | idx >= riskModerateThreshold = RiskHigh
  | idx >= riskLowThreshold      = RiskModerate
  | otherwise                    = RiskLow

-- | Compute duration in φ^n ticks using exponential scaling
-- durationPhiN = baseDuration * exp(-coherence/255)
computeDurationPhiN :: Unsigned 8 -> Unsigned 4
computeDurationPhiN coherence =
  let idx = coherence `shiftR` 5  -- Map 0-255 to 0-7
      expVal = expDecayLUT !! resize (min 8 idx)
      duration = (resize baseDurationPhiN * resize expVal) `shiftR` 8 :: Unsigned 8
  in max 1 (resize $ min 15 duration)

-- | Compute harmonic twist envelope
computeHarmonicTwist :: OmegaFormat -> InversionPolarity -> FluxCoherence -> TwistEnvelope
computeHarmonicTwist omega polarity fc =
  let coh = fcCoherence fc

      -- Skip for negligible twist (high coherence)
      isNegligible = coh >= negligibleTwistCoherence

      thetaF = if isNegligible then 0 else computeThetaForce omega polarity
      phiF = if isNegligible then 0 else computePhiForce omega polarity

      twist = TwistVector thetaF phiF (ofOmegaL omega) (ofOmegaM omega) polarity coh

      mag = computeTwistMagnitude twist
      risk = if isNegligible then 0 else computeRiskIndex mag coh
      dur = if isNegligible then 1 else computeDurationPhiN coh

  in TwistEnvelope twist dur risk

-- | Theta force pipeline
thetaForcePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (OmegaFormat, InversionPolarity)
  -> Signal dom (Signed 16)
thetaForcePipeline input = uncurry computeThetaForce <$> input

-- | Phi force pipeline
phiForcePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (OmegaFormat, InversionPolarity)
  -> Signal dom (Signed 16)
phiForcePipeline input = uncurry computePhiForce <$> input

-- | Risk index pipeline
riskIndexPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8)
  -> Signal dom (Unsigned 8)
riskIndexPipeline input = uncurry computeRiskIndex <$> input

-- | Full twist computation pipeline
twistPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (OmegaFormat, InversionPolarity, FluxCoherence)
  -> Signal dom TwistEnvelope
twistPipeline input = (\(o, p, f) -> computeHarmonicTwist o p f) <$> input
