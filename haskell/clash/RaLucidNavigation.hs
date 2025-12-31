{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}

-- | Prompt 14: Lucid Scalar Navigation via Harmonic Field Wayfinding
-- FPGA module for real-time coordinate translation, coherence gating,
-- and resonance scoring in lucid scalar traversal.
--
-- Codex References:
-- - KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md: Harmonic intention guidance
-- - RADIONICS_RATES_DOWSING.md: Scalar targeting
-- - GOLOD_RUSSIAN_PYRAMIDS.md: Field stabilization
--
-- Integration:
-- - Prompt 12: Shadow consent gating
-- - Prompt 13A: Gamma spike amplification

module RaLucidNavigation where

import Clash.Prelude
import GHC.Generics (Generic)

-- ============================================================================
-- Constants (Codex-aligned)
-- ============================================================================

-- | Golden ratio constants (Fixed8 = 0-255 scale)
phiInverse :: Unsigned 8
phiInverse = 158  -- 0.618 * 255

goldenTolerance :: Unsigned 8
goldenTolerance = 13  -- 0.05 * 255

-- | Coherence thresholds (0-255 scale)
coherenceFull :: Unsigned 8
coherenceFull = 204     -- 0.80 * 255

coherencePartial :: Unsigned 8
coherencePartial = 128  -- 0.50 * 255

coherenceDistorted :: Unsigned 8
coherenceDistorted = 77 -- 0.30 * 255

coherenceBlocked :: Unsigned 8
coherenceBlocked = 71   -- 0.28 * 255

-- | Resonance thresholds
resonanceFull :: Unsigned 8
resonanceFull = 224     -- 0.88 * 255

resonancePartial :: Unsigned 8
resonancePartial = 166  -- 0.65 * 255

-- | Return vector thresholds
driftThreshold :: Unsigned 8
driftThreshold = 107    -- 0.42 * 255

-- ============================================================================
-- Types
-- ============================================================================

-- | Access tier for coherence gating
data AccessTier = TierFull | TierPartial | TierDistorted | TierBlocked
  deriving (Generic, NFDataX, Eq, Show)

-- | Emergence form for fragments
data EmergenceForm = FormFull | FormSummary | FormSymbolic | FormDreamglyph | FormEcho
  deriving (Generic, NFDataX, Eq, Show)

-- | Navigation direction
data NavDirection = DirAscend | DirDescend | DirSpiral | DirAttune | DirExit | DirNone
  deriving (Generic, NFDataX, Eq, Show)

-- | RaCoordinate - Spherical lattice position
data RaCoordinate = RaCoordinate
  { coordTheta :: Unsigned 4   -- Azimuth sector 1-13
  , coordPhi   :: Unsigned 4   -- Polar stratum 1-12
  , coordH     :: Unsigned 3   -- Harmonic shell 0-7
  , coordR     :: Unsigned 8   -- Scalar depth 0-255 (maps to 0.0-1.0)
  } deriving (Generic, NFDataX, Show)

-- | User biometric state
data UserState = UserState
  { userHrv        :: Unsigned 8  -- HRV resonance 0-255
  , userBreath     :: Unsigned 8  -- Breath rate (normalized)
  , userCoherence  :: Unsigned 8  -- Coherence score 0-255
  , userGamma      :: Unsigned 8  -- Gamma power (lucid marker)
  } deriving (Generic, NFDataX, Show)

-- | Fragment anchor at coordinate
data FragmentAnchor = FragmentAnchor
  { fragId        :: Unsigned 12  -- Fragment ID
  , fragResonance :: Unsigned 8   -- Resonance score
  , fragAccess    :: AccessTier   -- Access level
  , fragForm      :: EmergenceForm
  , fragIsShadow  :: Bool
  } deriving (Generic, NFDataX, Show)

-- | Navigation input
data NavInput = NavInput
  { navDirection :: NavDirection
  , navUser      :: UserState
  , navCurrent   :: RaCoordinate
  } deriving (Generic, NFDataX)

-- | Navigation output
data NavOutput = NavOutput
  { outCoord      :: RaCoordinate
  , outTier       :: AccessTier
  , outCanMove    :: Bool
  , outInGolden   :: Bool
  , outReturnTrig :: Bool
  , outMetaphor   :: Unsigned 4  -- Metaphor index
  } deriving (Generic, NFDataX, Show)

-- | Return beacon data
data ReturnBeacon = ReturnBeacon
  { beaconOrigin  :: RaCoordinate
  , beaconTarget  :: RaCoordinate
  , beaconPhi     :: Unsigned 8   -- phi^n harmonic (3 or 5)
  , beaconActive  :: Bool
  } deriving (Generic, NFDataX, Show)

-- ============================================================================
-- Coherence Gate
-- ============================================================================

-- | Calculate weighted coherence vector
-- Formula: 0.5 * coherence + 0.3 * hrv + 0.2 * breath
coherenceVector :: UserState -> Unsigned 8
coherenceVector UserState{..} =
  let
    -- Weighted components (using shift for multiplication)
    cohPart = resize userCoherence `shiftR` 1  -- * 0.5
    hrvPart = (resize userHrv * 77) `shiftR` 8  -- * 0.3 (77/256)
    brPart  = (resize userBreath * 51) `shiftR` 8  -- * 0.2 (51/256)
  in truncateB (cohPart + hrvPart + brPart)

-- | Determine access tier from coherence vector
accessTier :: Unsigned 8 -> AccessTier
accessTier cv
  | cv >= coherenceFull     = TierFull
  | cv >= coherencePartial  = TierPartial
  | cv >= coherenceDistorted = TierDistorted
  | otherwise               = TierBlocked

-- | Apply gamma amplification (Prompt 13A integration)
gammaAmplify :: UserState -> Unsigned 8 -> Unsigned 8
gammaAmplify UserState{..} cv
  | userGamma >= 64 = satAdd SatBound cv 26  -- +0.1 boost if gamma >= 0.25
  | otherwise       = cv

-- | Full coherence gate evaluation
evaluateCoherence :: UserState -> (AccessTier, Unsigned 8, Bool)
evaluateCoherence user =
  let
    rawCv = coherenceVector user
    ampCv = gammaAmplify user rawCv
    tier  = accessTier ampCv
    canMove = tier /= TierBlocked
  in (tier, ampCv, canMove)

-- ============================================================================
-- Coordinate Operations
-- ============================================================================

-- | Check if coordinate is in golden corridor (r near 0.618)
isGoldenCorridor :: RaCoordinate -> Bool
isGoldenCorridor RaCoordinate{..} =
  let diff = if coordR >= phiInverse
             then coordR - phiInverse
             else phiInverse - coordR
  in diff <= goldenTolerance

-- | Clamp coordinate values to valid ranges
clampCoord :: RaCoordinate -> RaCoordinate
clampCoord RaCoordinate{..} = RaCoordinate
  { coordTheta = max 1 (min 13 coordTheta)
  , coordPhi   = max 1 (min 12 coordPhi)
  , coordH     = min 7 coordH
  , coordR     = coordR  -- Already 0-255
  }

-- | Apply navigation direction to coordinate
applyDirection :: NavDirection -> RaCoordinate -> RaCoordinate
applyDirection dir coord@RaCoordinate{..} = clampCoord $ case dir of
  DirAscend  -> coord { coordPhi = satSub SatBound coordPhi 1
                      , coordH   = satSub SatBound coordH 1
                      , coordR   = satSub SatBound coordR 26 }  -- -0.1
  DirDescend -> coord { coordPhi = satAdd SatBound coordPhi 1
                      , coordH   = satAdd SatBound coordH 1
                      , coordR   = satAdd SatBound coordR 26 }  -- +0.1
  DirSpiral  -> coord { coordTheta = if coordTheta >= 13 then 1 else coordTheta + 1 }
  DirAttune  -> coord { coordR = phiInverse }  -- Align to golden corridor
  DirExit    -> RaCoordinate 1 1 0 0  -- Return to origin
  DirNone    -> coord

-- | Check if user can access target depth
canAccessDepth :: AccessTier -> Unsigned 8 -> Bool
canAccessDepth tier targetR = case tier of
  TierFull     -> True
  TierPartial  -> targetR <= 179  -- 0.7 * 255
  TierDistorted -> targetR <= 102 -- 0.4 * 255
  TierBlocked  -> False

-- ============================================================================
-- Resonance Scorer
-- ============================================================================

-- | Simple dot product for 4-element vectors (reduced for FPGA)
dotProduct4 :: Vec 4 (Unsigned 8) -> Vec 4 (Unsigned 8) -> Unsigned 16
dotProduct4 a b = fold (+) $ zipWith (\x y -> resize x * resize y) a b

-- | Magnitude squared of 4-element vector
magnitude4Sq :: Vec 4 (Unsigned 8) -> Unsigned 16
magnitude4Sq v = dotProduct4 v v

-- | Calculate resonance score (simplified cosine similarity)
-- Returns 0-255 scale
resonanceScore :: Vec 4 (Unsigned 8) -> Vec 4 (Unsigned 8) -> Unsigned 8
resonanceScore userVec fragVec =
  let
    dot = dotProduct4 userVec fragVec
    magU = magnitude4Sq userVec
    magF = magnitude4Sq fragVec
    -- Simplified: score = dot / sqrt(magU * magF)
    -- Approximate with: score = dot * 255 / max(magU, magF)
    maxMag = max magU magF
    scaledScore = if maxMag == 0 then 0 else (dot * 255) `div` maxMag
  in truncateB scaledScore

-- | Determine fragment access from resonance score
fragmentAccess :: Unsigned 8 -> AccessTier
fragmentAccess score
  | score >= resonanceFull   = TierFull
  | score >= resonancePartial = TierPartial
  | otherwise                = TierBlocked

-- ============================================================================
-- Return Vector
-- ============================================================================

-- | Check if return should be triggered
shouldTriggerReturn :: Unsigned 8 -> Unsigned 8 -> Bool
shouldTriggerReturn currentCv baselineCv =
  let drift = if currentCv >= baselineCv
              then currentCv - baselineCv
              else baselineCv - currentCv
  in drift >= driftThreshold || currentCv < coherenceBlocked

-- | Create return beacon
createBeacon :: RaCoordinate -> RaCoordinate -> ReturnBeacon
createBeacon origin target = ReturnBeacon
  { beaconOrigin = origin
  , beaconTarget = target
  , beaconPhi    = if coordR origin > 179 || coordH origin > 5
                   then 5  -- phi^5 for deep returns
                   else 3  -- phi^3 for shallow
  , beaconActive = True
  }

-- ============================================================================
-- Navigation FSM
-- ============================================================================

-- | Navigation state
data NavState = NavState
  { stCurrent   :: RaCoordinate
  , stEntry     :: RaCoordinate
  , stBaseline  :: Unsigned 8
  , stActive    :: Bool
  , stSteps     :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Initial navigation state
initNavState :: NavState
initNavState = NavState
  { stCurrent  = RaCoordinate 1 1 0 0
  , stEntry    = RaCoordinate 1 1 0 0
  , stBaseline = 128
  , stActive   = False
  , stSteps    = 0
  }

-- | Navigation step logic
navStep :: NavState -> NavInput -> (NavState, NavOutput)
navStep st@NavState{..} NavInput{..} =
  let
    -- Evaluate coherence
    (tier, cv, canMove) = evaluateCoherence navUser

    -- Apply direction if can move
    newCoord = if canMove && stActive
               then applyDirection navDirection stCurrent
               else stCurrent

    -- Check depth access
    depthOk = canAccessDepth tier (coordR newCoord)

    -- Final coordinate
    finalCoord = if depthOk then newCoord else stCurrent

    -- Check return trigger
    returnTrig = shouldTriggerReturn cv stBaseline

    -- Update state
    newState = if stActive
               then st { stCurrent = finalCoord
                       , stSteps   = stSteps + 1 }
               else st

    -- Output
    output = NavOutput
      { outCoord      = finalCoord
      , outTier       = tier
      , outCanMove    = canMove && depthOk
      , outInGolden   = isGoldenCorridor finalCoord
      , outReturnTrig = returnTrig
      , outMetaphor   = encodeMetaphor navDirection
      }
  in (newState, output)

-- | Encode direction to metaphor index
encodeMetaphor :: NavDirection -> Unsigned 4
encodeMetaphor DirAscend  = 0  -- luminous_path
encodeMetaphor DirDescend = 1  -- spiral_staircase
encodeMetaphor DirSpiral  = 2  -- angular_bridge
encodeMetaphor DirAttune  = 3  -- golden_resonance
encodeMetaphor DirExit    = 4  -- emergence_portal
encodeMetaphor DirNone    = 5  -- stillness

-- ============================================================================
-- Top-Level Entity
-- ============================================================================

-- | Main navigation FSM
lucidNavFSM :: HiddenClockResetEnable dom
            => Signal dom NavInput
            -> Signal dom NavOutput
lucidNavFSM = mealy navStep initNavState

-- | Top entity with port annotations
{-# ANN lucidNavTop
  (Synthesize
    { t_name   = "lucid_nav_top"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "direction"
                 , PortName "user_hrv"
                 , PortName "user_breath"
                 , PortName "user_coherence"
                 , PortName "user_gamma"
                 , PortName "current_theta"
                 , PortName "current_phi"
                 , PortName "current_h"
                 , PortName "current_r"
                 ]
    , t_output = PortProduct "output"
                 [ PortName "out_theta"
                 , PortName "out_phi"
                 , PortName "out_h"
                 , PortName "out_r"
                 , PortName "tier"
                 , PortName "can_move"
                 , PortName "in_golden"
                 , PortName "return_trig"
                 , PortName "metaphor"
                 ]
    }) #-}
lucidNavTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 3)   -- direction (0-5)
  -> Signal System (Unsigned 8)   -- user_hrv
  -> Signal System (Unsigned 8)   -- user_breath
  -> Signal System (Unsigned 8)   -- user_coherence
  -> Signal System (Unsigned 8)   -- user_gamma
  -> Signal System (Unsigned 4)   -- current_theta
  -> Signal System (Unsigned 4)   -- current_phi
  -> Signal System (Unsigned 3)   -- current_h
  -> Signal System (Unsigned 8)   -- current_r
  -> Signal System ( Unsigned 4   -- out_theta
                   , Unsigned 4   -- out_phi
                   , Unsigned 3   -- out_h
                   , Unsigned 8   -- out_r
                   , Unsigned 2   -- tier
                   , Bool         -- can_move
                   , Bool         -- in_golden
                   , Bool         -- return_trig
                   , Unsigned 4   -- metaphor
                   )
lucidNavTop clk rst en dir hrv breath coh gamma theta phi h r =
  withClockResetEnable clk rst en $
    let
      -- Decode direction
      decDir = fmap decodeDirection dir

      -- Build input signal
      input = NavInput <$> decDir
                       <*> (UserState <$> hrv <*> breath <*> coh <*> gamma)
                       <*> (RaCoordinate <$> theta <*> phi <*> h <*> r)

      -- Run FSM
      output = lucidNavFSM input

      -- Extract output fields
      extractOut NavOutput{..} =
        ( coordTheta outCoord
        , coordPhi outCoord
        , coordH outCoord
        , coordR outCoord
        , encodeTier outTier
        , outCanMove
        , outInGolden
        , outReturnTrig
        , outMetaphor
        )
    in fmap extractOut output

-- | Decode direction from unsigned
decodeDirection :: Unsigned 3 -> NavDirection
decodeDirection 0 = DirAscend
decodeDirection 1 = DirDescend
decodeDirection 2 = DirSpiral
decodeDirection 3 = DirAttune
decodeDirection 4 = DirExit
decodeDirection _ = DirNone

-- | Encode tier to unsigned
encodeTier :: AccessTier -> Unsigned 2
encodeTier TierFull      = 0
encodeTier TierPartial   = 1
encodeTier TierDistorted = 2
encodeTier TierBlocked   = 3

-- ============================================================================
-- Testbench
-- ============================================================================

-- | Test input vectors
testInputs :: Vec 5 NavInput
testInputs =
     NavInput DirDescend (UserState 200 50 220 0) (RaCoordinate 1 1 0 25)
  :> NavInput DirDescend (UserState 180 60 200 100) (RaCoordinate 1 2 1 51)
  :> NavInput DirAttune (UserState 190 55 210 0) (RaCoordinate 1 3 2 77)
  :> NavInput DirSpiral (UserState 100 80 120 0) (RaCoordinate 1 3 2 158)
  :> NavInput DirExit (UserState 220 40 240 50) (RaCoordinate 5 5 3 180)
  :> Nil

-- | Testbench entity
testBench :: Signal System NavOutput
testBench = withClockResetEnable systemClockGen systemResetGen enableGen $
  lucidNavFSM (fromList (toList testInputs))
