{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DerivingStrategies #-}

{-|
Module      : Ra.Temporal
Description : φ-phased temporal window scheduler
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Reference implementation for φ-phased temporal window scheduler.
Emergence only occurs during valid windows.

Duty cycle: configurable (default 0.5)
Time source: monotonic (not wall clock)
Base period: configurable (default 1.0s)
Max depth: configurable (default 10)
-}
module Ra.Temporal
  ( -- * Configuration
    SchedulerConfig(..)
  , defaultConfig

    -- * Types
  , WindowState(..)
  , WindowDepth(..)
  , WindowPhase(..)
  , WindowDuration(..)
  , AlignmentMultiplier(..)

    -- * Core Functions
  , windowDuration
  , currentWindow
  , inWindow
  , timeToNextWindow
  , nestedAlignment

    -- * Invariants
  , phiScalingInvariant
  , phaseMonotonicInvariant
  , alignmentBoundedInvariant
  ) where

import GHC.Generics (Generic)
import Ra.Constants (GreenPhi(..), greenPhi)

-- | Scheduler configuration
data SchedulerConfig = SchedulerConfig
  { scBasePeriod :: !Double  -- ^ Root window duration in seconds
  , scMaxDepth   :: !Int     -- ^ Number of φ^n levels
  , scDutyCycle  :: !Double  -- ^ Fraction of window considered "open" ∈ (0,1)
  , scPhi        :: !Double  -- ^ Golden ratio (default φ)
  } deriving (Show, Eq, Generic)

-- | Default configuration per Architect spec
defaultConfig :: SchedulerConfig
defaultConfig = SchedulerConfig
  { scBasePeriod = 1.0
  , scMaxDepth   = 10
  , scDutyCycle  = 0.5
  , scPhi        = unGreenPhi greenPhi  -- 1.62
  }

-- | Window depth index (0 = fastest, max_depth-1 = slowest)
newtype WindowDepth = WindowDepth { unDepth :: Int }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Phase within window ∈ [0, 1)
newtype WindowPhase = WindowPhase { unPhase :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Window duration in seconds
newtype WindowDuration = WindowDuration { unDuration :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Alignment multiplier for nested windows (≥ 1.0)
newtype AlignmentMultiplier = AlignmentMultiplier { unMultiplier :: Double }
  deriving stock (Show, Eq, Ord)
  deriving newtype (Num)

-- | Current window state
data WindowState = WindowState
  { wsWindowIndex    :: !Int             -- ^ Which window number
  , wsPhase          :: !WindowPhase     -- ^ Phase within window [0, 1)
  , wsWindowStart    :: !Double          -- ^ Timestamp of window start
  , wsWindowDuration :: !WindowDuration
  , wsIsOpen         :: !Bool            -- ^ Whether emergence permitted
  } deriving (Show, Eq, Generic)

-- | Compute window duration at given depth
-- duration(n) = base_period × φ^n
windowDuration :: SchedulerConfig -> WindowDepth -> WindowDuration
windowDuration cfg (WindowDepth n)
  | n < 0 = WindowDuration (scBasePeriod cfg)
  | n >= scMaxDepth cfg = WindowDuration (scBasePeriod cfg * scPhi cfg ^ (scMaxDepth cfg - 1))
  | otherwise = WindowDuration (scBasePeriod cfg * scPhi cfg ^ n)

-- | Get current window state at given depth
-- Uses elapsed time from start reference
currentWindow
  :: SchedulerConfig
  -> Double           -- ^ Start time (monotonic reference)
  -> Double           -- ^ Current time (monotonic)
  -> WindowDepth
  -> WindowState
currentWindow cfg startTime now depth =
  WindowState
    { wsWindowIndex = windowIndex
    , wsPhase = WindowPhase phase
    , wsWindowStart = startTime + fromIntegral windowIndex * dur
    , wsWindowDuration = WindowDuration dur
    , wsIsOpen = phase < scDutyCycle cfg
    }
  where
    WindowDuration dur = windowDuration cfg depth
    elapsed = now - startTime
    windowIndex = floor (elapsed / dur)
    phase = (elapsed - fromIntegral windowIndex * dur) / dur

-- | Check if currently in an open emergence window
inWindow :: SchedulerConfig -> Double -> Double -> WindowDepth -> Bool
inWindow cfg start now depth = wsIsOpen (currentWindow cfg start now depth)

-- | Seconds until next window opens
timeToNextWindow :: SchedulerConfig -> Double -> Double -> WindowDepth -> Double
timeToNextWindow cfg start now depth
  | wsIsOpen state = 0.0
  | otherwise = (1.0 - unPhase (wsPhase state)) * unDuration (wsWindowDuration state)
  where
    state = currentWindow cfg start now depth

-- | Compute alignment multiplier for multiple open depths
-- More depths aligned = stronger emergence
-- Returns 1.0 if no alignment, >1.0 if nested alignment
nestedAlignment
  :: SchedulerConfig
  -> Double           -- ^ Start time
  -> Double           -- ^ Current time
  -> [WindowDepth]    -- ^ Depths to check
  -> AlignmentMultiplier
nestedAlignment cfg start now depths =
  AlignmentMultiplier $ 1.0 + (fromIntegral openCount - 1) * 0.2
  where
    openCount = length [d | d <- depths, inWindow cfg start now d]

-- | Invariant: φ scaling between adjacent depths
phiScalingInvariant :: SchedulerConfig -> WindowDepth -> Bool
phiScalingInvariant cfg (WindowDepth n)
  | n < 0 || n >= scMaxDepth cfg - 1 = True
  | otherwise =
      let d1 = windowDuration cfg (WindowDepth n)
          d2 = windowDuration cfg (WindowDepth (n + 1))
          ratio = unDuration d2 / unDuration d1
      in abs (ratio - scPhi cfg) < 0.001

-- | Invariant: phase values are in valid range
phaseMonotonicInvariant :: WindowPhase -> Bool
phaseMonotonicInvariant (WindowPhase p) = p >= 0 && p < 1

-- | Invariant: alignment multiplier ≥ 1.0
alignmentBoundedInvariant :: AlignmentMultiplier -> Bool
alignmentBoundedInvariant (AlignmentMultiplier m) = m >= 1.0
