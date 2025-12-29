{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DerivingStrategies #-}

{-|
Module      : Ra.Biometric
Description : Biometric to coherence pipeline
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Reference implementation for biometric → coherence pipeline.
Maps embodied signals to coherence scalar [0,1] for SCO gating.
-}
module Ra.Biometric
  ( -- * Data Types
    BioFrame(..)
  , CalibrationParams(..)
  , CoherenceScore(..)
  , SignalScore(..)
  , SmoothingAlpha(..)

    -- * Signal Weights
  , SignalWeights(..)
  , defaultWeights

    -- * Calibration
  , calibrateUser

    -- * Normalization (each → [0,1])
  , normHrv
  , normHr
  , normGsr
  , normResp
  , normMotion

    -- * Combination
  , combineSignals

    -- * Smoothing
  , smooth

    -- * Main Computation
  , computeCoherence

    -- * Invariants
  , coherenceInvariant
  ) where

import GHC.Generics (Generic)
import Data.List (sort)

-- | Single biometric measurement frame
data BioFrame = BioFrame
  { bfHrvMs    :: !Double           -- ^ Heart rate variability (milliseconds)
  , bfHrBpm    :: !Double           -- ^ Heart rate (beats per minute)
  , bfGsrUS    :: !Double           -- ^ Galvanic skin response (microsiemens)
  , bfRespBpm  :: !Double           -- ^ Respiration rate (breaths per minute)
  , bfAccelMag :: !(Maybe Double)   -- ^ Movement magnitude (optional)
  } deriving (Show, Eq, Generic)

-- | Per-user calibration parameters (anchor to individual physiology)
data CalibrationParams = CalibrationParams
  { cpHrvMean       :: !Double  -- ^ HRV mean
  , cpHrvStd        :: !Double  -- ^ HRV standard deviation
  , cpHrMean        :: !Double  -- ^ HR mean
  , cpHrStd         :: !Double  -- ^ HR standard deviation
  , cpGsrBase       :: !Double  -- ^ GSR 40th percentile
  , cpGsrPeak       :: !Double  -- ^ GSR 80th percentile
  , cpRespMean      :: !Double  -- ^ Respiration mean
  , cpAccelBaseline :: !Double  -- ^ Acceleration baseline
  } deriving (Show, Eq, Generic)

-- | Coherence score, bounded [0,1]
newtype CoherenceScore = CoherenceScore { unCoherence :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Individual signal score, bounded [0,1]
newtype SignalScore = SignalScore { unSignal :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Smoothing alpha parameter, bounded (0,1)
newtype SmoothingAlpha = SmoothingAlpha { unAlpha :: Double }
  deriving (Show, Eq, Ord)

-- | Signal weights for combination
data SignalWeights = SignalWeights
  { swHrv    :: !Double  -- ^ HRV weight (default 0.35)
  , swHr     :: !Double  -- ^ HR weight (default 0.20)
  , swGsr    :: !Double  -- ^ GSR weight (default 0.20)
  , swResp   :: !Double  -- ^ Respiration weight (default 0.15)
  , swMotion :: !Double  -- ^ Motion weight (default 0.10)
  } deriving (Show, Eq, Generic)

-- | Default weights per Architect spec
defaultWeights :: SignalWeights
defaultWeights = SignalWeights 0.35 0.20 0.20 0.15 0.10

-- | Sigmoid function
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

-- | Clamp value to [0,1]
clamp01 :: Double -> Double
clamp01 = max 0.0 . min 1.0

-- | Safe division (avoid div by zero)
safeDiv :: Double -> Double -> Double
safeDiv _ 0 = 0
safeDiv x y = x / y

-- | Calibrate user from sample frames
calibrateUser :: [BioFrame] -> CalibrationParams
calibrateUser samples = CalibrationParams
  { cpHrvMean       = mean hrvs
  , cpHrvStd        = stdev hrvs
  , cpHrMean        = mean hrs
  , cpHrStd         = stdev hrs
  , cpGsrBase       = percentile 40 gsrs
  , cpGsrPeak       = percentile 80 gsrs
  , cpRespMean      = mean resps
  , cpAccelBaseline = median accels
  }
  where
    hrvs   = map bfHrvMs samples
    hrs    = map bfHrBpm samples
    gsrs   = map bfGsrUS samples
    resps  = map bfRespBpm samples
    accels = [a | Just a <- map bfAccelMag samples]

    mean [] = 0
    mean xs = sum xs / fromIntegral (length xs)

    stdev [] = 1
    stdev xs =
      let m = mean xs
          n = fromIntegral (length xs)
      in sqrt (sum [(x - m)^(2::Int) | x <- xs] / n)

    percentile _ [] = 0
    percentile p xs =
      let sorted = sort xs
          idx = (p * length xs) `div` 100
      in sorted !! max 0 (min idx (length sorted - 1))

    median [] = 0
    median xs = percentile 50 xs

-- | Normalize HRV → calmness score. Higher HRV = more coherent.
normHrv :: Double -> CalibrationParams -> SignalScore
normHrv hrvMs params = SignalScore $ clamp01 $ sigmoid z
  where
    z = safeDiv (hrvMs - cpHrvMean params) (max 1e-6 $ cpHrvStd params)

-- | Normalize HR → inverse arousal. Higher HR = less coherent.
normHr :: Double -> CalibrationParams -> SignalScore
normHr hrBpm params = SignalScore $ 1.0 - clamp01 (sigmoid z)
  where
    z = safeDiv (hrBpm - cpHrMean params) (max 1e-6 $ cpHrStd params)

-- | Normalize GSR → inverse emotional load. Higher GSR = less coherent.
normGsr :: Double -> CalibrationParams -> SignalScore
normGsr gsrUS params = SignalScore $ clamp01 (1.0 - x)
  where
    lo = cpGsrBase params
    hi = cpGsrPeak params
    x  = if hi <= lo then 0.5 else safeDiv (gsrUS - lo) (hi - lo)

-- | Normalize respiration regularity. Deviation from baseline = less coherent.
normResp :: Double -> CalibrationParams -> SignalScore
normResp respBpm params = SignalScore $ clamp01 (1.0 - err)
  where
    m   = cpRespMean params
    err = if m <= 0 then 0.5 else abs (respBpm - m) / m

-- | Normalize motion. Movement deviation = less coherent. Missing = no penalty.
normMotion :: Maybe Double -> CalibrationParams -> SignalScore
normMotion Nothing _ = SignalScore 1.0
normMotion (Just accel) params = SignalScore $ clamp01 (1.0 - err)
  where
    baseline = cpAccelBaseline params
    err = if baseline <= 0 then 0 else abs (accel - baseline) / baseline

-- | Combine normalized signals with weights
combineSignals
  :: SignalScore  -- ^ HRV
  -> SignalScore  -- ^ HR
  -> SignalScore  -- ^ GSR
  -> SignalScore  -- ^ Resp
  -> SignalScore  -- ^ Motion
  -> SignalWeights
  -> CoherenceScore
combineSignals hrv hr gsr resp motion weights =
  CoherenceScore $ safeDiv num den
  where
    num = unSignal hrv * swHrv weights
        + unSignal hr  * swHr weights
        + unSignal gsr * swGsr weights
        + unSignal resp * swResp weights
        + unSignal motion * swMotion weights
    den = swHrv weights + swHr weights + swGsr weights
        + swResp weights + swMotion weights

-- | Temporal smoothing (low-pass filter)
-- alpha=0.3 → slow shift (stable)
-- alpha=0.7 → responsive (jittery)
smooth :: CoherenceScore -> CoherenceScore -> SmoothingAlpha -> CoherenceScore
smooth (CoherenceScore prev) (CoherenceScore curr) (SmoothingAlpha alpha) =
  CoherenceScore $ (1 - alpha) * prev + alpha * curr

-- | Compute coherence from biometric frame
computeCoherence
  :: BioFrame
  -> CalibrationParams
  -> CoherenceScore      -- ^ Previous coherence
  -> SmoothingAlpha
  -> SignalWeights
  -> CoherenceScore
computeCoherence bio params prev alpha weights =
  smooth prev raw alpha
  where
    hrvScore    = normHrv (bfHrvMs bio) params
    hrScore     = normHr (bfHrBpm bio) params
    gsrScore    = normGsr (bfGsrUS bio) params
    respScore   = normResp (bfRespBpm bio) params
    motionScore = normMotion (bfAccelMag bio) params
    raw = combineSignals hrvScore hrScore gsrScore respScore motionScore weights

-- | Invariant: coherence always in [0,1]
coherenceInvariant :: CoherenceScore -> Bool
coherenceInvariant (CoherenceScore c) = c >= 0 && c <= 1
