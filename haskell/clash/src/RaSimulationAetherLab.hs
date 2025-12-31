{-|
Module      : RaSimulationAetherLab
Description : Full Ra-Field Simulation in Morphogenic Chambers
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 53: Full Ra-field simulations in synthetic morphogenic chambers.
Blends biometric data, scalar topologies, inversion nodes, and avatar
traversal. Enables non-realtime emergence testing, debug coherence breaks,
and prototype fragment alignments.

Based on φ-nested triple pyramid chamber topology.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaSimulationAetherLab where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Coherence emergence threshold (0.72 * 255)
coherenceEmergenceThreshold :: Unsigned 8
coherenceEmergenceThreshold = 184

-- | Fragment proximity threshold (squared, for efficiency)
fragmentProximityThresholdSq :: Unsigned 16
fragmentProximityThresholdSq = 9  -- 3^2

-- | Default field size
defaultFieldSize :: Unsigned 8
defaultFieldSize = 27

-- | Emergence event types
data EmergenceType
  = EtCoherenceSpike
  | EtFragmentAlignment
  | EtHarmonicLock
  | EtInversionResponse
  | EtAvatarResonance
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Avatar resonance states
data AvatarState
  = AsStable
  | AsSeeking
  | AsResonant
  | AsDisrupted
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | 3D position in chamber space (grid indices)
data Position = Position
  { posTheta :: Unsigned 8
  , posPhi   :: Unsigned 8
  , posH     :: Unsigned 8
  } deriving (Generic, NFDataX, Eq)

-- | Biometric state
data BioState = BioState
  { bsHRV        :: Unsigned 16   -- Heart rate variability (ms)
  , bsCoherence  :: Unsigned 8    -- Biometric coherence (0-255)
  , bsRhythmPhase :: Unsigned 16  -- Phase in rhythm cycle (0-65535 = 0-2π)
  } deriving (Generic, NFDataX)

-- | Fragment
data Fragment = Fragment
  { frFragmentId :: Unsigned 8
  , frPosition   :: Position
  , frHarmonicL  :: Unsigned 4
  , frHarmonicM  :: Signed 8
  , frIntensity  :: Unsigned 8    -- 0-255
  , frIsManifest :: Bool
  } deriving (Generic, NFDataX)

-- | Avatar
data Avatar = Avatar
  { avAvatarId       :: Unsigned 8
  , avPosition       :: Position
  , avResonanceState :: AvatarState
  , avCoherenceLevel :: Unsigned 8  -- 0-255
  , avVelocityX      :: Signed 8
  , avVelocityY      :: Signed 8
  , avVelocityZ      :: Signed 8
  } deriving (Generic, NFDataX)

-- | Inversion region
data InversionRegion = InversionRegion
  { irCenter    :: Position
  , irRadius    :: Unsigned 4
  , irIntensity :: Unsigned 8    -- 0-255
  , irIsActive  :: Bool
  } deriving (Generic, NFDataX)

-- | Phi clock (timing)
data PhiClock = PhiClock
  { pcTick     :: Unsigned 32
  , pcPhiPhase :: Unsigned 16    -- 0-65535 = 0-2π
  } deriving (Generic, NFDataX)

-- | Emergence event
data EmergenceEvent = EmergenceEvent
  { eeType          :: EmergenceType
  , eePosition      :: Position
  , eeTimestamp     :: Unsigned 32
  , eeCoherenceLevel :: Unsigned 8
  , eeHasEvent      :: Bool
  } deriving (Generic, NFDataX)

-- | Chamber state (simplified for FPGA)
data ChamberState = ChamberState
  { csAvatarState    :: Avatar
  , csPhiClock       :: PhiClock
  , csAvgCoherence   :: Unsigned 8    -- Average field coherence
  , csManifestCount  :: Unsigned 8    -- Number of manifest fragments
  , csEmergenceCount :: Unsigned 8    -- Emergence events this step
  } deriving (Generic, NFDataX)

-- | Simulation result
data SimulationResult = SimulationResult
  { srEmergenceEvent :: EmergenceEvent
  , srNewCoherence   :: Unsigned 8
  , srNewAvatar      :: Avatar
  , srNewClock       :: PhiClock
  } deriving (Generic, NFDataX)

-- | Distance squared between positions (for proximity check)
distanceSquared :: Position -> Position -> Unsigned 16
distanceSquared p1 p2 =
  let dt = if posTheta p1 > posTheta p2
           then resize (posTheta p1) - resize (posTheta p2)
           else resize (posTheta p2) - resize (posTheta p1) :: Unsigned 16
      dp = if posPhi p1 > posPhi p2
           then resize (posPhi p1) - resize (posPhi p2)
           else resize (posPhi p2) - resize (posPhi p1) :: Unsigned 16
      dh = if posH p1 > posH p2
           then resize (posH p1) - resize (posH p2)
           else resize (posH p2) - resize (posH p1) :: Unsigned 16
  in dt * dt + dp * dp + dh * dh

-- | Advance phi clock
advancePhiClock :: PhiClock -> PhiClock
advancePhiClock clock =
  let newTick = pcTick clock + 1
      -- phi_phase = (tick * phi) mod 2π
      -- Approximation: phi * 1024 = 1657, so phase advances by 1657 * 64 = 106048 per tick
      -- Scaled to fit in 16 bits: advance by ~1600 per tick
      phaseInc = 1600 :: Unsigned 16
      newPhase = pcPhiPhase clock + phaseInc
  in PhiClock newTick newPhase

-- | Compute local coherence from field value and neighbors
-- Simplified: coherence = value * (0.5 + 0.5 * stability)
computeLocalCoherence :: Unsigned 8 -> Unsigned 8 -> Unsigned 8 -> Unsigned 8
computeLocalCoherence centerVal neighborAvg stability =
  let -- Stability factor (0-255 where 255 = perfectly stable)
      -- stability = 255 - variance (passed in)
      stabilityFactor = 128 + (resize stability `shiftR` 1) :: Unsigned 16
      -- coherence = centerVal * stabilityFactor / 256
      coherence = (resize centerVal * stabilityFactor) `shiftR` 8
  in resize $ min 255 coherence

-- | Compute coherence gradient direction
-- Returns direction to move toward higher coherence (-1, 0, or 1 per axis)
computeCoherenceGradient
  :: Unsigned 8  -- Coherence at current position
  -> Unsigned 8  -- Coherence at theta+1
  -> Unsigned 8  -- Coherence at phi+1
  -> Unsigned 8  -- Coherence at h+1
  -> (Signed 8, Signed 8, Signed 8)
computeCoherenceGradient current cohT cohP cohH =
  let dt = if cohT > current then 1 else if cohT < current then -1 else 0
      dp = if cohP > current then 1 else if cohP < current then -1 else 0
      dh = if cohH > current then 1 else if cohH < current then -1 else 0
  in (dt, dp, dh)

-- | Move avatar following coherence gradient
moveAvatar :: Avatar -> (Signed 8, Signed 8, Signed 8) -> Unsigned 8 -> Unsigned 8 -> Avatar
moveAvatar avatar (gradT, gradP, gradH) newCoherence fieldSize =
  let -- Move position by gradient
      newTheta = if gradT > 0
                 then (posTheta (avPosition avatar) + 1) `mod` fieldSize
                 else if gradT < 0
                      then (posTheta (avPosition avatar) - 1) `mod` fieldSize
                      else posTheta (avPosition avatar)
      newPhi = if gradP > 0
               then (posPhi (avPosition avatar) + 1) `mod` fieldSize
               else if gradP < 0
                    then (posPhi (avPosition avatar) - 1) `mod` fieldSize
                    else posPhi (avPosition avatar)
      newH = if gradH > 0
             then (posH (avPosition avatar) + 1) `mod` fieldSize
             else if gradH < 0
                  then (posH (avPosition avatar) - 1) `mod` fieldSize
                  else posH (avPosition avatar)

      newPos = Position newTheta newPhi newH

      -- Determine resonance state based on coherence
      newState = if newCoherence >= 217     -- > 0.85 * 255
                 then AsResonant
                 else if newCoherence >= 128 -- > 0.50 * 255
                      then AsStable
                      else if newCoherence >= 77   -- > 0.30 * 255
                           then AsSeeking
                           else AsDisrupted

  in Avatar
       (avAvatarId avatar)
       newPos
       newState
       newCoherence
       gradT gradP gradH

-- | Check if fragment should manifest
shouldManifest :: Unsigned 8 -> Bool
shouldManifest localCoherence = localCoherence >= coherenceEmergenceThreshold

-- | Update fragment manifestation
updateFragment :: Fragment -> Unsigned 8 -> Fragment
updateFragment frag localCoherence =
  let manifest = shouldManifest localCoherence
      newIntensity = if manifest
                     then frIntensity frag
                     else frIntensity frag `shiftR` 1  -- Halve intensity if not manifest
  in Fragment
       (frFragmentId frag)
       (frPosition frag)
       (frHarmonicL frag)
       (frHarmonicM frag)
       newIntensity
       manifest

-- | Check for fragment proximity (within threshold)
fragmentsInProximity :: Position -> Position -> Bool
fragmentsInProximity p1 p2 =
  distanceSquared p1 p2 <= fragmentProximityThresholdSq

-- | Detect emergence event
detectEmergence
  :: Unsigned 8      -- Avatar coherence
  -> AvatarState     -- Avatar state
  -> Position        -- Avatar position
  -> Bool            -- Has nearby manifest fragment
  -> Unsigned 32     -- Timestamp
  -> EmergenceEvent
detectEmergence avatarCoh avatarState avatarPos hasNearbyFragment timestamp =
  let -- Coherence spike with fragment
      isCohSpike = avatarCoh >= coherenceEmergenceThreshold && hasNearbyFragment

      -- Avatar resonance
      isResonance = avatarState == AsResonant

      (eventType, hasEvent) =
        if isCohSpike then (EtCoherenceSpike, True)
        else if isResonance then (EtAvatarResonance, True)
        else (EtCoherenceSpike, False)

  in EmergenceEvent eventType avatarPos timestamp avatarCoh hasEvent

-- | Single simulation step
simulateStep
  :: ChamberState
  -> BioState
  -> Unsigned 8      -- Local coherence at avatar
  -> Unsigned 8      -- Coherence at theta+1
  -> Unsigned 8      -- Coherence at phi+1
  -> Unsigned 8      -- Coherence at h+1
  -> Bool            -- Has nearby manifest fragment
  -> SimulationResult
simulateStep chamber bio localCoh cohT cohP cohH hasNearbyFrag =
  let -- Bio-modulated coherence
      bioFactor = 230 + (resize (bsCoherence bio) `shiftR` 3) :: Unsigned 16
      modCoh = (resize localCoh * bioFactor) `shiftR` 8 :: Unsigned 16
      newCoherence = resize $ min 255 modCoh

      -- Compute gradient
      gradient = computeCoherenceGradient newCoherence cohT cohP cohH

      -- Move avatar
      newAvatar = moveAvatar (csAvatarState chamber) gradient newCoherence defaultFieldSize

      -- Advance clock
      newClock = advancePhiClock (csPhiClock chamber)

      -- Detect emergence
      emergence = detectEmergence
        newCoherence
        (avResonanceState newAvatar)
        (avPosition newAvatar)
        hasNearbyFrag
        (pcTick newClock)

  in SimulationResult emergence newCoherence newAvatar newClock

-- | Initial chamber state
initialChamberState :: ChamberState
initialChamberState = ChamberState
  { csAvatarState = Avatar 1 (Position 13 3 2) AsStable 128 0 0 0
  , csPhiClock = PhiClock 0 0
  , csAvgCoherence = 128
  , csManifestCount = 0
  , csEmergenceCount = 0
  }

-- | Simulation step pipeline
simulationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (BioState, Unsigned 8, Unsigned 8, Unsigned 8, Unsigned 8, Bool)
  -> Signal dom SimulationResult
simulationPipeline input = mealy simMealy initialChamberState input
  where
    simMealy state (bio, localCoh, cohT, cohP, cohH, hasNearby) =
      let result = simulateStep state bio localCoh cohT cohP cohH hasNearby
          newState = ChamberState
            { csAvatarState = srNewAvatar result
            , csPhiClock = srNewClock result
            , csAvgCoherence = srNewCoherence result
            , csManifestCount = csManifestCount state  -- Would need fragment tracking
            , csEmergenceCount = if eeHasEvent (srEmergenceEvent result)
                                 then csEmergenceCount state + 1
                                 else csEmergenceCount state
            }
      in (newState, result)

-- | Avatar state pipeline (just avatar updates)
avatarStatePipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Avatar, (Signed 8, Signed 8, Signed 8), Unsigned 8)
  -> Signal dom Avatar
avatarStatePipeline input = (\(a, g, c) -> moveAvatar a g c defaultFieldSize) <$> input

-- | Emergence detection pipeline
emergenceDetectionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, AvatarState, Position, Bool, Unsigned 32)
  -> Signal dom EmergenceEvent
emergenceDetectionPipeline input =
  (\(coh, state, pos, frag, ts) -> detectEmergence coh state pos frag ts) <$> input

-- | Phi clock pipeline
phiClockPipeline
  :: HiddenClockResetEnable dom
  => Signal dom Bool     -- Tick enable
  -> Signal dom PhiClock
phiClockPipeline enable = mealy clockMealy (PhiClock 0 0) enable
  where
    clockMealy clock tick = if tick
                            then (advancePhiClock clock, advancePhiClock clock)
                            else (clock, clock)
