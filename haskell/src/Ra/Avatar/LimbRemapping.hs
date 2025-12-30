{-|
Module      : Ra.Avatar.LimbRemapping
Description : Non-physical limb representation and coherence mapping
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements non-physical limb remapping for avatars with non-human anatomical
features. Maintains coherence between physical biometric sources and virtual
appendage representations.

== Limb Remapping Theory

=== Non-Physical Appendages

* Phantom limb extensions beyond physical form
* Multi-appendage avatar configurations (wings, tails, etc.)
* Telepresence limb projection
* Gesture-field coherence mapping

=== Biometric-Avatar Coupling

1. Physical movement detection from biometric sources
2. Coherence-weighted mapping to virtual appendages
3. Feedback loop for proprioceptive calibration
4. Consent-gated appendage activation
-}
module Ra.Avatar.LimbRemapping
  ( -- * Core Types
    LimbMap(..)
  , VirtualLimb(..)
  , LimbType(..)
  , LimbState(..)

    -- * Map Creation
  , createLimbMap
  , defaultHumanoidMap
  , addVirtualLimb
  , removeLimb

    -- * Limb Configuration
  , configureLimb
  , setLimbCoherence
  , activateLimb
  , deactivateLimb

    -- * Biometric Mapping
  , BiometricSource(..)
  , mapBiometricToLimb
  , syncBiometricSources
  , calibrateLimbResponse

    -- * Gesture Translation
  , GestureField(..)
  , translateGesture
  , gestureToLimbMotion
  , fieldCoherence

    -- * Phantom Extensions
  , PhantomLimb(..)
  , createPhantomExtension
  , phantomFeedback
  , phantomCoherence

    -- * Multi-Appendage Systems
  , AppendageSystem(..)
  , createWingSystem
  , createTailSystem
  , createTentacleSystem
  , syncAppendages
  ) where

import Ra.Constants.Extended (phi, phiInverse)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Complete limb mapping configuration
data LimbMap = LimbMap
  { lmLimbs        :: ![VirtualLimb]       -- ^ All virtual limbs
  , lmPhantoms     :: ![PhantomLimb]       -- ^ Phantom extensions
  , lmBaseCoherence :: !Double             -- ^ Base coherence level
  , lmSourceCount  :: !Int                 -- ^ Number of biometric sources
  , lmSymmetry     :: !SymmetryMode        -- ^ Symmetry enforcement
  , lmActive       :: !Bool                -- ^ System active flag
  } deriving (Eq, Show)

-- | Single virtual limb definition
data VirtualLimb = VirtualLimb
  { vlId          :: !String               -- ^ Limb identifier
  , vlType        :: !LimbType             -- ^ Limb classification
  , vlState       :: !LimbState            -- ^ Current state
  , vlPosition    :: !(Double, Double, Double)  -- ^ Base position
  , vlOrientation :: !(Double, Double, Double)  -- ^ Orientation vector
  , vlLength      :: !Double               -- ^ Extended length
  , vlSegments    :: !Int                  -- ^ Number of segments
  , vlCoherence   :: !Double               -- ^ Limb-specific coherence
  , vlBiometricSource :: !(Maybe String)   -- ^ Linked biometric source
  } deriving (Eq, Show)

-- | Limb type classification
data LimbType
  = LimbArm           -- ^ Standard arm
  | LimbLeg           -- ^ Standard leg
  | LimbWing          -- ^ Wing appendage
  | LimbTail          -- ^ Tail appendage
  | LimbTentacle      -- ^ Tentacle/flexible limb
  | LimbAntenna       -- ^ Sensory antenna
  | LimbPseudopod     -- ^ Amorphous extension
  | LimbEthereal      -- ^ Non-physical energy projection
  | LimbCustom        -- ^ User-defined type
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Limb operational state
data LimbState
  = StateInactive     -- ^ Not responding
  | StateCalibrating  -- ^ Being calibrated
  | StateActive       -- ^ Fully operational
  | StateLocked       -- ^ Temporarily locked
  | StatePhantom      -- ^ Phantom mode (sensing only)
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Symmetry enforcement mode
data SymmetryMode
  = SymmetryNone        -- ^ No symmetry enforcement
  | SymmetryBilateral   -- ^ Left-right symmetry
  | SymmetryRadial      -- ^ Radial symmetry (n-fold)
  | SymmetryMirror      -- ^ Full mirror symmetry
  deriving (Eq, Ord, Show, Enum, Bounded)

-- =============================================================================
-- Map Creation
-- =============================================================================

-- | Create empty limb map
createLimbMap :: Double -> LimbMap
createLimbMap baseCoherence = LimbMap
  { lmLimbs = []
  , lmPhantoms = []
  , lmBaseCoherence = baseCoherence
  , lmSourceCount = 0
  , lmSymmetry = SymmetryNone
  , lmActive = False
  }

-- | Default humanoid limb configuration
defaultHumanoidMap :: LimbMap
defaultHumanoidMap = LimbMap
  { lmLimbs =
    [ VirtualLimb "left_arm" LimbArm StateInactive (-0.3, 0, 0) (0, -1, 0) 0.7 3 0.8 Nothing
    , VirtualLimb "right_arm" LimbArm StateInactive (0.3, 0, 0) (0, -1, 0) 0.7 3 0.8 Nothing
    , VirtualLimb "left_leg" LimbLeg StateInactive (-0.15, -0.5, 0) (0, -1, 0) 0.9 3 0.8 Nothing
    , VirtualLimb "right_leg" LimbLeg StateInactive (0.15, -0.5, 0) (0, -1, 0) 0.9 3 0.8 Nothing
    ]
  , lmPhantoms = []
  , lmBaseCoherence = phiInverse
  , lmSourceCount = 0
  , lmSymmetry = SymmetryBilateral
  , lmActive = False
  }

-- | Add virtual limb to map
addVirtualLimb :: LimbMap -> VirtualLimb -> LimbMap
addVirtualLimb lmap limb =
  lmap { lmLimbs = limb : lmLimbs lmap }

-- | Remove limb by ID
removeLimb :: LimbMap -> String -> LimbMap
removeLimb lmap limbId =
  lmap { lmLimbs = filter ((/= limbId) . vlId) (lmLimbs lmap) }

-- =============================================================================
-- Limb Configuration
-- =============================================================================

-- | Configure specific limb parameters
configureLimb :: LimbMap -> String -> (VirtualLimb -> VirtualLimb) -> LimbMap
configureLimb lmap limbId f =
  let updated = map (\l -> if vlId l == limbId then f l else l) (lmLimbs lmap)
  in lmap { lmLimbs = updated }

-- | Set limb coherence level
setLimbCoherence :: LimbMap -> String -> Double -> LimbMap
setLimbCoherence lmap limbId coherence =
  configureLimb lmap limbId (\l -> l { vlCoherence = clamp01 coherence })

-- | Activate limb
activateLimb :: LimbMap -> String -> LimbMap
activateLimb lmap limbId =
  configureLimb lmap limbId (\l -> l { vlState = StateActive })

-- | Deactivate limb
deactivateLimb :: LimbMap -> String -> LimbMap
deactivateLimb lmap limbId =
  configureLimb lmap limbId (\l -> l { vlState = StateInactive })

-- =============================================================================
-- Biometric Mapping
-- =============================================================================

-- | Biometric source definition
data BiometricSource = BiometricSource
  { bsId         :: !String          -- ^ Source identifier
  , bsType       :: !SourceType      -- ^ Type of biometric
  , bsPosition   :: !(Double, Double, Double)  -- ^ Physical position
  , bsSignal     :: !Double          -- ^ Current signal value
  , bsCalibrated :: !Bool            -- ^ Calibration status
  } deriving (Eq, Show)

-- | Biometric source type
data SourceType
  = SourceEMG       -- ^ Electromyography (muscle)
  | SourceEEG       -- ^ Electroencephalography (brain)
  | SourceIMU       -- ^ Inertial measurement unit
  | SourceOptical   -- ^ Optical tracking
  | SourcePressure  -- ^ Pressure sensors
  | SourceCustom    -- ^ Custom sensor
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Map biometric source to virtual limb
mapBiometricToLimb :: LimbMap -> BiometricSource -> String -> LimbMap
mapBiometricToLimb lmap source limbId =
  let updated = map (\l -> if vlId l == limbId
                           then l { vlBiometricSource = Just (bsId source) }
                           else l) (lmLimbs lmap)
  in lmap { lmLimbs = updated, lmSourceCount = lmSourceCount lmap + 1 }

-- | Synchronize all biometric sources
syncBiometricSources :: LimbMap -> [BiometricSource] -> LimbMap
syncBiometricSources lmap sources =
  let coherenceFactor = if null sources then 1.0
                        else sum [if bsCalibrated s then 1 else 0.5 | s <- sources] / fromIntegral (length sources)
      newCoherence = lmBaseCoherence lmap * coherenceFactor
  in lmap { lmBaseCoherence = newCoherence, lmSourceCount = length sources }

-- | Calibrate limb response to biometric input
calibrateLimbResponse :: LimbMap -> String -> [Double] -> LimbMap
calibrateLimbResponse lmap limbId samples =
  let avgSignal = if null samples then 0.5 else sum samples / fromIntegral (length samples)
      calibratedCoherence = phiInverse + (avgSignal - 0.5) * 0.2
  in setLimbCoherence lmap limbId calibratedCoherence

-- =============================================================================
-- Gesture Translation
-- =============================================================================

-- | Gesture field for motion capture
data GestureField = GestureField
  { gfVectors     :: ![(Double, Double, Double)]  -- ^ Motion vectors
  , gfIntensity   :: !Double                      -- ^ Overall intensity
  , gfCoherence   :: !Double                      -- ^ Field coherence
  , gfTimestamp   :: !Int                         -- ^ Capture timestamp
  } deriving (Eq, Show)

-- | Translate gesture to limb-specific motion
translateGesture :: LimbMap -> GestureField -> [(String, (Double, Double, Double))]
translateGesture lmap gfield =
  [ (vlId limb, scaledVector limb vec)
  | limb <- lmLimbs lmap
  , vlState limb == StateActive
  , vec <- take 1 (gfVectors gfield)  -- Primary vector per limb
  ]
  where
    scaledVector limb (vx, vy, vz) =
      let scale = vlCoherence limb * gfCoherence gfield * gfIntensity gfield
      in (vx * scale, vy * scale, vz * scale)

-- | Convert gesture to specific limb motion
gestureToLimbMotion :: VirtualLimb -> GestureField -> (Double, Double, Double)
gestureToLimbMotion limb gfield =
  let coherence = vlCoherence limb * gfCoherence gfield
      (ox, oy, oz) = vlOrientation limb
      intensity = gfIntensity gfield * coherence
  in (ox * intensity, oy * intensity, oz * intensity)

-- | Calculate gesture field coherence
fieldCoherence :: GestureField -> Double
fieldCoherence gfield =
  let vectors = gfVectors gfield
      n = length vectors
  in if n <= 1 then gfCoherence gfield
     else let magnitudes = [sqrt (x*x + y*y + z*z) | (x, y, z) <- vectors]
              avgMag = sum magnitudes / fromIntegral n
              variance = sum [(m - avgMag)^(2::Int) | m <- magnitudes] / fromIntegral n
          in max 0 (gfCoherence gfield * (1 - variance))

-- =============================================================================
-- Phantom Extensions
-- =============================================================================

-- | Phantom limb extension (beyond physical form)
data PhantomLimb = PhantomLimb
  { plId          :: !String               -- ^ Identifier
  , plSourceLimb  :: !String               -- ^ Source limb ID
  , plExtension   :: !Double               -- ^ Extension distance
  , plIntensity   :: !Double               -- ^ Sensory intensity
  , plFeedback    :: !Double               -- ^ Feedback strength
  , plCoherence   :: !Double               -- ^ Extension coherence
  } deriving (Eq, Show)

-- | Create phantom extension from existing limb
createPhantomExtension :: VirtualLimb -> Double -> PhantomLimb
createPhantomExtension limb extension = PhantomLimb
  { plId = vlId limb ++ "_phantom"
  , plSourceLimb = vlId limb
  , plExtension = extension
  , plIntensity = vlCoherence limb * phi
  , plFeedback = 0.5
  , plCoherence = vlCoherence limb * phiInverse
  }

-- | Calculate phantom feedback signal
phantomFeedback :: PhantomLimb -> Double -> Double
phantomFeedback phantom stimulus =
  let response = stimulus * plIntensity phantom * plFeedback phantom
      coherenceModifier = plCoherence phantom
  in response * coherenceModifier

-- | Get phantom limb coherence relative to source
phantomCoherence :: LimbMap -> PhantomLimb -> Double
phantomCoherence lmap phantom =
  case filter ((== plSourceLimb phantom) . vlId) (lmLimbs lmap) of
    (source:_) -> vlCoherence source * plCoherence phantom
    [] -> plCoherence phantom * 0.5

-- =============================================================================
-- Multi-Appendage Systems
-- =============================================================================

-- | Multi-appendage system configuration
data AppendageSystem = AppendageSystem
  { asType        :: !AppendageType        -- ^ System type
  , asLimbs       :: ![VirtualLimb]        -- ^ Component limbs
  , asSyncMode    :: !SyncMode             -- ^ Synchronization mode
  , asCoherence   :: !Double               -- ^ System coherence
  , asSpan        :: !Double               -- ^ Total span/reach
  } deriving (Eq, Show)

-- | Appendage system type
data AppendageType
  = AppendageWings      -- ^ Wing pair or set
  | AppendageTail       -- ^ Single or multiple tails
  | AppendageTentacles  -- ^ Tentacle array
  | AppendageSpines     -- ^ Spine/fin array
  | AppendageAura       -- ^ Energy field projections
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Synchronization mode
data SyncMode
  = SyncIndependent     -- ^ Each appendage independent
  | SyncMirrored        -- ^ Mirrored motion
  | SyncSequential      -- ^ Wave-like sequential
  | SyncUnified         -- ^ All move as one
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Create wing system
createWingSystem :: Double -> AppendageSystem
createWingSystem wingSpan = AppendageSystem
  { asType = AppendageWings
  , asLimbs =
    [ VirtualLimb "left_wing" LimbWing StateInactive (-0.2, 0.3, -0.1) (-1, 0, 0.3) (wingSpan/2) 5 phi Nothing
    , VirtualLimb "right_wing" LimbWing StateInactive (0.2, 0.3, -0.1) (1, 0, 0.3) (wingSpan/2) 5 phi Nothing
    ]
  , asSyncMode = SyncMirrored
  , asCoherence = phi
  , asSpan = wingSpan
  }

-- | Create tail system
createTailSystem :: Int -> Double -> AppendageSystem
createTailSystem segments len = AppendageSystem
  { asType = AppendageTail
  , asLimbs =
    [ VirtualLimb "tail" LimbTail StateInactive (0, -0.4, -0.2) (0, 0, -1) len segments phiInverse Nothing
    ]
  , asSyncMode = SyncUnified
  , asCoherence = phiInverse
  , asSpan = len
  }

-- | Create tentacle system
createTentacleSystem :: Int -> Double -> AppendageSystem
createTentacleSystem count len = AppendageSystem
  { asType = AppendageTentacles
  , asLimbs =
    [ VirtualLimb ("tentacle_" ++ show i) LimbTentacle StateInactive
        (cos angle * 0.3, sin angle * 0.3, -0.2)
        (cos angle, sin angle, -0.5)
        len 8 (phiInverse * 0.9) Nothing
    | i <- [1..count]
    , let angle = (fromIntegral i / fromIntegral count) * 2 * pi
    ]
  , asSyncMode = SyncSequential
  , asCoherence = phiInverse * 0.8
  , asSpan = len * 2
  }

-- | Synchronize all appendages in system
syncAppendages :: AppendageSystem -> Double -> AppendageSystem
syncAppendages sys coherenceInput =
  let newCoherence = (asCoherence sys + coherenceInput) / 2
      syncedLimbs = case asSyncMode sys of
        SyncIndependent -> asLimbs sys
        SyncMirrored -> mirrorSync (asLimbs sys) newCoherence
        SyncSequential -> sequentialSync (asLimbs sys) newCoherence
        SyncUnified -> unifiedSync (asLimbs sys) newCoherence
  in sys { asLimbs = syncedLimbs, asCoherence = newCoherence }

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 = max 0 . min 1

-- | Mirror synchronization
mirrorSync :: [VirtualLimb] -> Double -> [VirtualLimb]
mirrorSync limbs coherence =
  map (\l -> l { vlCoherence = coherence }) limbs

-- | Sequential wave synchronization
sequentialSync :: [VirtualLimb] -> Double -> [VirtualLimb]
sequentialSync limbs coherence =
  zipWith (\l i -> l { vlCoherence = coherence * (1 - fromIntegral i * 0.1) }) limbs ([0..] :: [Int])

-- | Unified synchronization
unifiedSync :: [VirtualLimb] -> Double -> [VirtualLimb]
unifiedSync limbs coherence =
  map (\l -> l { vlCoherence = coherence * phi }) limbs
