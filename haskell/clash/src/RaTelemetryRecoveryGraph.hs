{-|
Module      : RaTelemetryRecoveryGraph
Description : Field Recovery Tracking with Emergence Trail
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 48: Temporal topological map of field coherence disruption and
healing, with emergence transitions, inversion resolutions, and
harmonic axis realignment tracking.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaTelemetryRecoveryGraph where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Recovery thresholds (8-bit scaled)
coherenceDeltaThreshold :: Unsigned 8
coherenceDeltaThreshold = 13   -- 0.05 * 255

loopClosureTolerance :: Unsigned 8
loopClosureTolerance = 13      -- 0.05 * 255

-- | Phase cycle (2π scaled to 16-bit)
phaseCycle2Pi :: Unsigned 16
phaseCycle2Pi = 65535          -- Full 16-bit range represents 2π

-- | Emergence result states
data EmergenceResult
  = Incoherent
  | Reintegrating
  | HarmonicallyStable
  | Collapsed
  | PhaseLocked
  | GhostEmergent
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Harmonic axis options
data HarmonicAxis
  = AxisTheta
  | AxisPhi
  | AxisThetaPhi
  | AxisH
  | AxisLM
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Inversion polarity
data InversionPolarity
  = InvNormal
  | InvInverted
  deriving (Generic, NFDataX, Eq, Show)

-- | Ra coordinate
data RaCoordinate = RaCoordinate
  { rcTheta :: Unsigned 16
  , rcPhi   :: Unsigned 16
  , rcH     :: Unsigned 8
  } deriving (Generic, NFDataX, Eq)

-- | Inversion shift
data InversionShift = InversionShift
  { isAxisFlipped     :: Bool
  , isTorsionIntensity :: Unsigned 8
  , isPolarity        :: InversionPolarity
  } deriving (Generic, NFDataX)

-- | Scalar field snapshot
data ScalarFieldSnapshot = ScalarFieldSnapshot
  { sfsTimestamp  :: Unsigned 32
  , sfsCoherence  :: Unsigned 8
  , sfsFlux       :: Unsigned 8
  , sfsPhaseAngle :: Unsigned 16
  , sfsDominantL  :: Unsigned 4
  , sfsDominantM  :: Signed 8
  } deriving (Generic, NFDataX)

-- | Recovery event
data RecoveryEvent = RecoveryEvent
  { reTimestamp      :: Unsigned 32
  , reLocation       :: RaCoordinate
  , reCoherenceValue :: Unsigned 8
  , reFluxValue      :: Unsigned 8
  , reStateBefore    :: EmergenceResult
  , reStateAfter     :: EmergenceResult
  , reHasInversion   :: Bool
  , reInversionShift :: InversionShift
  } deriving (Generic, NFDataX)

-- | Recovery graph summary
data RecoveryGraphSummary = RecoveryGraphSummary
  { rgsEventCount       :: Unsigned 8
  , rgsLoopClosed       :: Bool
  , rgsDominantAxis     :: HarmonicAxis
  , rgsCoherenceDelta   :: Signed 16
  , rgsBaselineCoherence :: Unsigned 8
  , rgsFinalPhase       :: Unsigned 16
  } deriving (Generic, NFDataX)

-- | Determine emergence state from coherence
determineEmergenceState :: Unsigned 8 -> EmergenceResult
determineEmergenceState coh
  | coh < 51   = Collapsed          -- < 0.20
  | coh < 102  = Incoherent         -- < 0.40
  | coh < 184  = Reintegrating      -- < 0.72
  | coh < 230  = PhaseLocked        -- < 0.90
  | otherwise  = HarmonicallyStable

-- | Detect inversion shift between snapshots
detectInversionShift :: ScalarFieldSnapshot -> ScalarFieldSnapshot -> Maybe InversionShift
detectInversionShift prev curr =
  let phaseDiff = if sfsPhaseAngle curr > sfsPhaseAngle prev
                  then sfsPhaseAngle curr - sfsPhaseAngle prev
                  else sfsPhaseAngle prev - sfsPhaseAngle curr
      phaseFlip = phaseDiff > 52428  -- > 0.8 * 65535

      cohDrop = if sfsCoherence prev > sfsCoherence curr
                then sfsCoherence prev - sfsCoherence curr
                else 0
      significantDrop = cohDrop > 38  -- > 0.15 * 255

      torsion = resize (phaseDiff `shiftR` 8) * (255 - sfsCoherence curr) `shiftR` 8

  in if phaseFlip || (significantDrop && phaseDiff > 8192)
     then Just $ InversionShift phaseFlip (resize $ min 255 torsion) InvInverted
     else Nothing

-- | Check loop closure
checkLoopClosure :: Unsigned 8 -> Unsigned 8 -> Unsigned 16 -> Unsigned 16 -> Bool
checkLoopClosure baseline final initPhase finalPhase =
  let coherenceRestored = if final > baseline
                          then final - baseline <= loopClosureTolerance
                          else baseline - final <= loopClosureTolerance

      phaseDelta = if finalPhase > initPhase
                   then finalPhase - initPhase
                   else initPhase - finalPhase

      -- Check nearly full cycle or returned to start
      phaseCycleComplete = phaseDelta >= 62258 || phaseDelta <= 3277

  in coherenceRestored && phaseCycleComplete

-- | Compute dominant axis from movement
computeDominantAxis :: Unsigned 16 -> Unsigned 16 -> Unsigned 8 -> HarmonicAxis
computeDominantAxis thetaMove phiMove hMove =
  let maxMove = max (max thetaMove phiMove) (resize hMove)
  in if maxMove == 0 then AxisTheta
     else if thetaMove == maxMove && phiMove > maxMove `shiftR` 1
          then AxisThetaPhi
          else if thetaMove >= phiMove && thetaMove >= resize hMove
               then AxisTheta
               else if phiMove >= resize hMove
                    then AxisPhi
                    else AxisH

-- | Should log recovery event
shouldLogEvent :: Unsigned 8 -> Unsigned 8 -> EmergenceResult -> EmergenceResult -> Bool
shouldLogEvent prevCoh currCoh prevState currState =
  let delta = if currCoh > prevCoh
              then currCoh - prevCoh
              else prevCoh - currCoh
  in delta > coherenceDeltaThreshold || prevState /= currState

-- | Create recovery event
createRecoveryEvent
  :: Unsigned 32             -- Timestamp
  -> ScalarFieldSnapshot     -- Current snapshot
  -> EmergenceResult         -- State before
  -> EmergenceResult         -- State after
  -> Maybe InversionShift    -- Inversion
  -> RecoveryEvent
createRecoveryEvent ts snapshot stateBefore stateAfter maybeInv =
  let coord = RaCoordinate
        (sfsPhaseAngle snapshot)
        (sfsPhaseAngle snapshot `shiftR` 1)
        (resize $ sfsDominantL snapshot)
      (hasInv, inv) = case maybeInv of
        Just i  -> (True, i)
        Nothing -> (False, InversionShift False 0 InvNormal)
  in RecoveryEvent ts coord (sfsCoherence snapshot) (sfsFlux snapshot)
                   stateBefore stateAfter hasInv inv

-- | Emergence state pipeline
emergenceStatePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom EmergenceResult
emergenceStatePipeline = fmap determineEmergenceState

-- | Loop closure check pipeline
loopClosurePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8, Unsigned 16, Unsigned 16)
  -> Signal dom Bool
loopClosurePipeline input = (\(b, f, i, fp) -> checkLoopClosure b f i fp) <$> input

-- | Event logging decision pipeline
eventLoggingPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8, EmergenceResult, EmergenceResult)
  -> Signal dom Bool
eventLoggingPipeline input = (\(pc, cc, ps, cs) -> shouldLogEvent pc cc ps cs) <$> input
