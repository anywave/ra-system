{-|
Module      : RaDreamSurface
Description : Dream-Memory Buffer System
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 75: Manages fragments originating during sleep or liminal states.
Supports temporal fragility, reverse-inversion logic, non-linear coherence,
and reentry detection (dream → waking pipeline).

72-hour reentry window, ∇α flip triggers inversion, wake/dream cycle decay.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaDreamSurface where

import Clash.Prelude

-- | Phi constant scaled (1.618 * 1024)
phi16 :: Unsigned 16
phi16 = 1657

-- | Reentry window in ticks (72 hours * assumed tick rate)
reentryWindowTicks :: Unsigned 32
reentryWindowTicks = 259200  -- 72 * 60 * 60 (seconds)

-- | Fragility decay per cycle (0.2 * 256)
fragilityDecayPerCycle :: Unsigned 8
fragilityDecayPerCycle = 51

-- | Alpha thresholds
alphaReentryThreshold :: Unsigned 8
alphaReentryThreshold = 179   -- 0.7 * 255

alphaInversionThreshold :: Unsigned 8
alphaInversionThreshold = 128 -- 0.5 * 255

-- | Dream state phases
data DreamState
  = StateLiminal
  | StateREM
  | StateDeep
  | StateWaking
  | StateLucid
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Inversion trigger types
data InversionTrigger
  = TriggerAlphaFlip
  | TriggerValenceShift
  | TriggerSymbolicFlip
  deriving (Generic, NFDataX, Eq, Show, Enum, Bounded)

-- | Fragment ID
data FragmentID = FragmentID
  { fidNamespace :: Unsigned 8
  , fidIndex     :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Dream timestamp
data DreamTime = DreamTime
  { dtRealTicks   :: Unsigned 32
  , dtDreamTicks  :: Unsigned 16
  , dtCycleIndex  :: Unsigned 8
  , dtState       :: DreamState
  } deriving (Generic, NFDataX, Eq, Show)

-- | Dream fragment
data DreamFragment = DreamFragment
  { dfFragmentID     :: FragmentID
  , dfTimestamp      :: DreamTime
  , dfInversionFlag  :: Bool
  , dfFragilityScore :: Unsigned 8   -- 0-255
  , dfReentryDetected :: Bool
  , dfValence        :: Signed 8     -- -128 to 127
  , dfSymbolicHash   :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Waking fragment (for reentry detection)
data WakingFragment = WakingFragment
  { wfFragmentID   :: FragmentID
  , wfTimestamp    :: Unsigned 32
  , wfAlpha        :: Unsigned 8
  , wfSymbolicHash :: Unsigned 16
  } deriving (Generic, NFDataX, Eq, Show)

-- | Compute base fragility from dream state
baseFragility :: DreamState -> Unsigned 8
baseFragility state = case state of
  StateLiminal -> 230   -- 0.9 * 255
  StateREM     -> 179   -- 0.7 * 255
  StateDeep    -> 128   -- 0.5 * 255
  StateWaking  -> 77    -- 0.3 * 255
  StateLucid   -> 102   -- 0.4 * 255

-- | Compute fragility score from state and alpha
computeFragility :: DreamState -> Unsigned 8 -> Unsigned 8
computeFragility state alpha =
  let base = baseFragility state
      -- Higher alpha = more stable = lower fragility
      -- Factor = 1.0 - (alpha * 0.5) in fixed point
      alphaFactor = 256 - (resize alpha `shiftR` 1) :: Unsigned 16
      result = (resize base * alphaFactor) `shiftR` 8 :: Unsigned 16
  in min 255 (resize result)

-- | Check alpha flip inversion trigger
checkAlphaFlip :: Unsigned 8 -> Unsigned 8 -> Bool
checkAlphaFlip prevAlpha currAlpha =
  prevAlpha > alphaInversionThreshold &&
  currAlpha < alphaInversionThreshold

-- | Check valence shift trigger
checkValenceShift :: Signed 8 -> Signed 8 -> Bool
checkValenceShift prevVal currVal =
  (prevVal > 0 && currVal < 0) || (prevVal < 0 && currVal > 0)

-- | Check symbolic flip trigger
checkSymbolicFlip :: Unsigned 16 -> Unsigned 16 -> Bool
checkSymbolicFlip prev curr = prev /= curr

-- | Check inversion condition
checkInversion
  :: InversionTrigger
  -> Unsigned 8      -- Previous alpha
  -> Unsigned 8      -- Current alpha
  -> Signed 8        -- Previous valence
  -> Signed 8        -- Current valence
  -> Unsigned 16     -- Previous symbolic hash
  -> Unsigned 16     -- Current symbolic hash
  -> Bool
checkInversion trigger prevA currA prevV currV prevS currS =
  case trigger of
    TriggerAlphaFlip    -> checkAlphaFlip prevA currA
    TriggerValenceShift -> checkValenceShift prevV currV
    TriggerSymbolicFlip -> checkSymbolicFlip prevS currS

-- | Check reentry conditions
checkReentry
  :: Unsigned 32     -- Current time
  -> Unsigned 32     -- Dream timestamp
  -> Unsigned 8      -- Waking alpha
  -> Unsigned 16     -- Dream symbolic hash
  -> Unsigned 16     -- Waking symbolic hash
  -> Bool            -- Already detected
  -> Bool
checkReentry currTime dreamTime wakingAlpha dreamHash wakingHash alreadyDetected =
  let timeDelta = currTime - dreamTime
      withinWindow = timeDelta <= reentryWindowTicks
      alphaOK = wakingAlpha >= alphaReentryThreshold
      hashMatch = dreamHash == wakingHash
  in not alreadyDetected && withinWindow && alphaOK && hashMatch

-- | Apply fragility decay
applyFragilityDecay :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
applyFragilityDecay fragility cycles =
  let decay = resize cycles * resize fragilityDecayPerCycle :: Unsigned 16
  in if decay >= resize fragility then 0
     else fragility - resize decay

-- | Dream buffer state
data DreamBufferState = DreamBufferState
  { dbsFragmentCount :: Unsigned 8
  , dbsReentryCount  :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Initial buffer state
initialDreamBufferState :: DreamBufferState
initialDreamBufferState = DreamBufferState 0 0

-- | Dream buffer input
data DreamBufferInput = DreamBufferInput
  { dbiFragment      :: DreamFragment
  , dbiWakingFrag    :: WakingFragment
  , dbiCurrentTime   :: Unsigned 32
  , dbiDecayCycles   :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Dream buffer output
data DreamBufferOutput = DreamBufferOutput
  { dboFragment        :: DreamFragment
  , dboReentryDetected :: Bool
  , dboIsActive        :: Bool
  } deriving (Generic, NFDataX)

-- | Dream buffer pipeline
dreamBufferPipeline
  :: HiddenClockResetEnable dom
  => Signal dom DreamBufferInput
  -> Signal dom DreamBufferOutput
dreamBufferPipeline input = mealy dreamMealy initialDreamBufferState input
  where
    dreamMealy state inp =
      let frag = dbiFragment inp
          waking = dbiWakingFrag inp

          -- Check reentry
          reentry = checkReentry
            (dbiCurrentTime inp)
            (dtRealTicks (dfTimestamp frag))
            (wfAlpha waking)
            (dfSymbolicHash frag)
            (wfSymbolicHash waking)
            (dfReentryDetected frag)

          -- Apply decay
          newFragility = applyFragilityDecay
            (dfFragilityScore frag)
            (dbiDecayCycles inp)

          -- Update fragment
          updatedFrag = DreamFragment
            (dfFragmentID frag)
            (dfTimestamp frag)
            (dfInversionFlag frag)
            newFragility
            (dfReentryDetected frag || reentry)
            (dfValence frag)
            (dfSymbolicHash frag)

          -- Check if still active
          isActive = newFragility > 13  -- > 0.05 * 255

          -- Update state
          newReentryCount = if reentry
                            then dbsReentryCount state + 1
                            else dbsReentryCount state

          newState = DreamBufferState
            (dbsFragmentCount state)
            newReentryCount

          output = DreamBufferOutput updatedFrag reentry isActive

      in (newState, output)

-- | Fragility computation pipeline
fragilityPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (DreamState, Unsigned 8)
  -> Signal dom Unsigned 8
fragilityPipeline = fmap (uncurry computeFragility)

-- | Inversion check pipeline
inversionPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8)
  -> Signal dom Bool
inversionPipeline = fmap (uncurry checkAlphaFlip)

-- | Decay pipeline
decayPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8)
  -> Signal dom Unsigned 8
decayPipeline = fmap (uncurry applyFragilityDecay)
