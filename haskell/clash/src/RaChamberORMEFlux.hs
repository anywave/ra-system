{-|
Module      : RaChamberORMEFlux
Description : ORME Flux Chamber Tuning
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 65: Maps scalar coherence zones to ORME (Orbitally Rearranged
Monoatomic Elements) stabilization states including inertial shielding,
mass-phase inversion, and superfluid field behavior.

Uses dual-condition superfluid triggering and geometry multipliers.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaChamberORMEFlux where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Alpha window thresholds (scaled to 255)
alphaGroundMax :: Unsigned 8
alphaGroundMax = 128   -- 0.50 * 255

alphaInvertedMax :: Unsigned 8
alphaInvertedMax = 184  -- 0.72 * 255

alphaSuperfluidMin :: Unsigned 8
alphaSuperfluidMin = 184  -- 0.72 * 255

-- | Shield level thresholds (scaled to 255)
shieldNoneMax :: Unsigned 8
shieldNoneMax = 26     -- 0.10 * 255

shieldLowMax :: Unsigned 8
shieldLowMax = 77      -- 0.30 * 255

shieldMediumMax :: Unsigned 8
shieldMediumMax = 153  -- 0.60 * 255

shieldHighMax :: Unsigned 8
shieldHighMax = 217    -- 0.85 * 255

-- | Mass phase states
data MassPhaseState
  = PhaseGround
  | PhaseInverted
  | PhaseSuperfluid
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Shield levels
data ShieldLevel
  = ShieldNone
  | ShieldLow
  | ShieldMedium
  | ShieldHigh
  | ShieldCritical
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Torsion phase
data TorsionPhase
  = TorsionNormal
  | TorsionInverted
  | TorsionNull
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Geometry types
data GeometryType
  = GeoDodecahedral
  | GeoSpherical
  | GeoToroidal
  | GeoCustom
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Scalar field point
data ScalarFieldPoint = ScalarFieldPoint
  { sfpX       :: Signed 16
  , sfpY       :: Signed 16
  , sfpZ       :: Signed 16
  , sfpAlpha   :: Unsigned 8    -- Coherence (0-255)
  , sfpTorsion :: TorsionPhase
  } deriving (Generic, NFDataX, Eq, Show)

-- | ORME flux zone
data ORMEFluxZone = ORMEFluxZone
  { ofzZoneId          :: Unsigned 8
  , ofzResonanceBandLo :: Unsigned 8
  , ofzResonanceBandHi :: Unsigned 8
  , ofzMassPhase       :: MassPhaseState
  , ofzShieldLevel     :: ShieldLevel
  , ofzShieldContinuous :: Unsigned 8  -- 0-255
  } deriving (Generic, NFDataX, Eq, Show)

-- | ORME coherence map
data ORMECoherenceMap = ORMECoherenceMap
  { ocmZones          :: Vec 8 ORMEFluxZone
  , ocmNumZones       :: Unsigned 4
  , ocmTotalShielding :: Unsigned 8
  , ocmDominantPhase  :: MassPhaseState
  } deriving (Generic, NFDataX, Eq, Show)

-- | Scalar field
data ScalarField = ScalarField
  { sfPoints       :: Vec 8 ScalarFieldPoint
  , sfNumPoints    :: Unsigned 4
  , sfGeometry     :: GeometryType
  , sfAverageAlpha :: Unsigned 8
  } deriving (Generic, NFDataX, Eq, Show)

-- | Get geometry multiplier (scaled to 256 = 1.0)
getGeometryMultiplier :: GeometryType -> Unsigned 16
getGeometryMultiplier geo = case geo of
  GeoDodecahedral -> 294  -- 1.15 * 256
  GeoSpherical    -> 276  -- 1.08 * 256
  GeoToroidal     -> 256  -- 1.00 * 256
  GeoCustom       -> 256  -- 1.00 * 256

-- | Classify mass phase from alpha and torsion
classifyMassPhase :: Unsigned 8 -> TorsionPhase -> MassPhaseState
classifyMassPhase alpha torsion
  | alpha < alphaGroundMax = PhaseGround
  | alpha < alphaSuperfluidMin = PhaseInverted
  | torsion == TorsionInverted = PhaseSuperfluid
  | otherwise = PhaseInverted

-- | Map continuous value to shield level
continuousToShieldLevel :: Unsigned 8 -> ShieldLevel
continuousToShieldLevel continuous
  | continuous < shieldNoneMax   = ShieldNone
  | continuous < shieldLowMax    = ShieldLow
  | continuous < shieldMediumMax = ShieldMedium
  | continuous < shieldHighMax   = ShieldHigh
  | otherwise                    = ShieldCritical

-- | Get alpha band for mass phase
getAlphaBandForPhase :: MassPhaseState -> (Unsigned 8, Unsigned 8)
getAlphaBandForPhase phase = case phase of
  PhaseGround    -> (0, alphaGroundMax)
  PhaseInverted  -> (alphaGroundMax, alphaInvertedMax)
  PhaseSuperfluid -> (alphaSuperfluidMin, 255)

-- | Compute zone shielding
computeZoneShielding
  :: Unsigned 8      -- Alpha
  -> MassPhaseState
  -> GeometryType
  -> Unsigned 8
computeZoneShielding alpha phase geometry =
  let -- Base shielding from alpha (0.8 * alpha)
      baseShielding = (resize alpha * 204) `shiftR` 8 :: Unsigned 16

      -- Phase bonus
      phaseBonus = case phase of
        PhaseGround    -> 0
        PhaseInverted  -> 26   -- 0.1 * 255
        PhaseSuperfluid -> 64  -- 0.25 * 255

      -- Geometry multiplier
      geoMult = getGeometryMultiplier geometry

      -- Combined shielding
      combined = ((baseShielding + phaseBonus) * geoMult) `shiftR` 8

  in resize $ min 255 combined

-- | Create empty ORME zone
emptyORMEZone :: ORMEFluxZone
emptyORMEZone = ORMEFluxZone 0 0 0 PhaseGround ShieldNone 0

-- | Create ORME zone from field point
createORMEZone :: Unsigned 8 -> ScalarFieldPoint -> GeometryType -> ORMEFluxZone
createORMEZone zoneId point geometry =
  let massPhase = classifyMassPhase (sfpAlpha point) (sfpTorsion point)
      (bandLo, bandHi) = getAlphaBandForPhase massPhase
      shieldContinuous = computeZoneShielding (sfpAlpha point) massPhase geometry
      shieldLevel = continuousToShieldLevel shieldContinuous
  in ORMEFluxZone
       zoneId
       bandLo
       bandHi
       massPhase
       shieldLevel
       shieldContinuous

-- | Count phase occurrences
countPhase :: Vec 8 ORMEFluxZone -> Unsigned 4 -> MassPhaseState -> Unsigned 4
countPhase zones numZones targetPhase =
  foldl (\acc i -> if i < numZones && ofzMassPhase (zones !! i) == targetPhase
                   then acc + 1 else acc)
        0
        $(listToVecTH [0..7 :: Unsigned 4])

-- | Find dominant phase
findDominantPhase :: Vec 8 ORMEFluxZone -> Unsigned 4 -> MassPhaseState
findDominantPhase zones numZones =
  let groundCount = countPhase zones numZones PhaseGround
      invertedCount = countPhase zones numZones PhaseInverted
      superfluidCount = countPhase zones numZones PhaseSuperfluid
  in if superfluidCount >= invertedCount && superfluidCount >= groundCount
     then PhaseSuperfluid
     else if invertedCount >= groundCount
          then PhaseInverted
          else PhaseGround

-- | Generate ORME coherence map
generateORMEMap :: ScalarField -> ORMECoherenceMap
generateORMEMap field =
  let numPoints = sfNumPoints field
      geometry = sfGeometry field

      -- Create zones
      createZone :: Unsigned 4 -> ORMEFluxZone
      createZone i =
        if i < numPoints
        then createORMEZone (resize i) (sfPoints field !! i) geometry
        else emptyORMEZone

      zones = map createZone $(listToVecTH [0..7 :: Unsigned 4])

      -- Compute total shielding (average)
      sumShielding = foldl
        (\s i -> if i < numPoints
                 then s + resize (ofzShieldContinuous (zones !! i))
                 else s)
        (0 :: Unsigned 16)
        $(listToVecTH [0..7 :: Unsigned 4])

      avgShielding = if numPoints > 0
                     then resize (sumShielding `div` resize numPoints)
                     else 0

      -- Find dominant phase
      dominant = if numPoints > 0
                 then findDominantPhase zones numPoints
                 else PhaseGround

  in ORMECoherenceMap zones numPoints avgShielding dominant

-- | Check if superfluid is active in zone
isSuperfluidActive :: ORMEFluxZone -> Bool
isSuperfluidActive zone = ofzMassPhase zone == PhaseSuperfluid

-- | Compute inertial shielding from ORME map
computeInertialShielding :: ORMECoherenceMap -> Unsigned 8
computeInertialShielding ormeMap
  | ocmNumZones ormeMap == 0 = 0
  | otherwise =
      let numZones = ocmNumZones ormeMap

          -- Count superfluid zones for bonus
          superfluidCount = foldl
            (\c i -> if i < numZones && isSuperfluidActive (ocmZones ormeMap !! i)
                     then c + 1 else c)
            (0 :: Unsigned 4)
            $(listToVecTH [0..7 :: Unsigned 4])

          -- Superfluid bonus (0.2 per zone normalized)
          superfluidBonus = (resize superfluidCount * 51) `div` resize numZones :: Unsigned 16

          -- Combined shielding
          combined = resize (ocmTotalShielding ormeMap) + superfluidBonus

      in resize $ min 255 combined

-- | ORME tuning state
data ORMETuningState = ORMETuningState
  { otsCurrentMap :: ORMECoherenceMap
  } deriving (Generic, NFDataX)

-- | Initial ORME state
initialORMEState :: ORMETuningState
initialORMEState = ORMETuningState
  (ORMECoherenceMap (repeat emptyORMEZone) 0 0 PhaseGround)

-- | ORME tuning pipeline
ormeTuningPipeline
  :: HiddenClockResetEnable dom
  => Signal dom ScalarField
  -> Signal dom ORMECoherenceMap
ormeTuningPipeline input = mealy tuningMealy initialORMEState input
  where
    tuningMealy state field =
      let newMap = generateORMEMap field
          newState = state { otsCurrentMap = newMap }
      in (newState, newMap)

-- | Inertial shielding pipeline
inertialShieldingPipeline
  :: HiddenClockResetEnable dom
  => Signal dom ORMECoherenceMap
  -> Signal dom Unsigned 8
inertialShieldingPipeline input = computeInertialShielding <$> input

-- | Zone classification pipeline
zoneClassificationPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (ScalarFieldPoint, GeometryType)
  -> Signal dom ORMEFluxZone
zoneClassificationPipeline input =
  (\(point, geo) -> createORMEZone 0 point geo) <$> input
