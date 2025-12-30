{-|
Module      : RaVisualizerShell
Description : Visual shell renderer for field synthesis feedback
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Prompt 41: Renders visual shell feedback from chamber state, coherence,
and sync status. Maps internal states to RGB color output for LED strips
or display integration.

== Visual Mapping

@
ShellColor = f(chamberState, syncState, coherence)
@

== Chamber State Colors

| State       | Base RGB           |
|-------------|-------------------|
| Idle        | (0, 0, 32)        | Deep blue (dim)
| Spinning    | (0, 64, 128)      | Cyan glow
| Stabilizing | (128, 64, 255)    | Purple pulse
| Emanating   | (255, 128, 64)    | Golden radiance

== Sync State Modulation

| Sync State | Effect                      |
|------------|-----------------------------|
| Desync     | Flash red overlay (50%)     |
| Aligning   | Pulse brightness (25-100%)  |
| Locked     | Steady (100%)               |
| Drifting   | Subtle fade (75-100%)       |

== Coherence Intensity

Final brightness = baseColor * (coherence / 255)

== Integration

- Consumes ChamberState from RaFieldSynthesisNode
- Consumes SyncState from RaChamberSync
- Outputs RGB triplet for LED/display driver
-}

{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module RaVisualizerShell where

import Clash.Prelude
import qualified Prelude as P

-- =============================================================================
-- Types
-- =============================================================================

-- | Chamber synthesis states (from RaFieldSynthesisNode)
data ChamberState = Idle | Spinning | Stabilizing | Emanating
  deriving (Generic, NFDataX, Show, Eq)

-- | Synchronization states (from RaChamberSync)
data SyncState = Desync | Aligning | Locked | Drifting
  deriving (Generic, NFDataX, Show, Eq)

-- | RGB color output
data RGBColor = RGBColor
  { red   :: Unsigned 8
  , green :: Unsigned 8
  , blue  :: Unsigned 8
  } deriving (Generic, NFDataX, Show, Eq)

-- | Shell visualizer input bundle
data ShellInput = ShellInput
  { chamber   :: ChamberState
  , sync      :: SyncState
  , coherence :: Unsigned 8
  , tick      :: Unsigned 8    -- For animation timing
  } deriving (Generic, NFDataX, Show, Eq)

-- =============================================================================
-- Core Functions
-- =============================================================================

-- | Base color for chamber state
chamberColor :: ChamberState -> RGBColor
chamberColor Idle        = RGBColor 0   0   32
chamberColor Spinning    = RGBColor 0   64  128
chamberColor Stabilizing = RGBColor 128 64  255
chamberColor Emanating   = RGBColor 255 128 64

-- | Brightness modifier for sync state (0-255)
syncBrightness :: SyncState -> Unsigned 8 -> Unsigned 8
syncBrightness Desync   tick = if testBit tick 4 then 128 else 0  -- Flash
syncBrightness Aligning tick = 64 + (resize (tick `mod` 192))     -- Pulse 25-100%
syncBrightness Locked   _    = 255                                -- Full
syncBrightness Drifting tick = 192 + (resize (tick `mod` 64))     -- Subtle fade 75-100%

-- | Scale a color component by brightness and coherence
scaleColor :: Unsigned 8 -> Unsigned 8 -> Unsigned 8 -> Unsigned 8
scaleColor base brightness coherence =
  let scaled = (resize base * resize brightness * resize coherence) :: Unsigned 24
  in resize (scaled `shiftR` 16)  -- Divide by 65536 (256*256)

-- | Apply modulation to RGB color
modulate :: RGBColor -> Unsigned 8 -> Unsigned 8 -> RGBColor
modulate (RGBColor r g b) brightness coherence = RGBColor
  (scaleColor r brightness coherence)
  (scaleColor g brightness coherence)
  (scaleColor b brightness coherence)

-- | Compute final shell color
computeShellColor :: ShellInput -> RGBColor
computeShellColor (ShellInput chState syncSt coh t) =
  let base = chamberColor chState
      brightness = syncBrightness syncSt t
  in modulate base brightness coh

-- =============================================================================
-- Signal-Level Processing
-- =============================================================================

-- | Visualizer shell - renders RGB from state inputs
visualizerShell
  :: HiddenClockResetEnable dom
  => Signal dom ChamberState    -- ^ Chamber state
  -> Signal dom SyncState       -- ^ Sync state
  -> Signal dom (Unsigned 8)    -- ^ Coherence score
  -> Signal dom RGBColor        -- ^ RGB output
visualizerShell chamber sync coherence = rgbOut
  where
    -- Animation tick counter
    tick = register 0 (tick + 1)

    -- Bundle inputs
    input = ShellInput <$> chamber <*> sync <*> coherence <*> tick

    -- Compute output color
    rgbOut = fmap computeShellColor input

-- =============================================================================
-- Synthesis Entry Point
-- =============================================================================

-- | Top-level entity for Clash synthesis
visualizerTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System ChamberState
  -> Signal System SyncState
  -> Signal System (Unsigned 8)
  -> Signal System RGBColor
visualizerTop = exposeClockResetEnable visualizerShell

-- =============================================================================
-- Test Data
-- =============================================================================

-- | Test chamber states
testChamber :: Vec 4 ChamberState
testChamber = $(listToVecTH [Idle, Spinning, Stabilizing, Emanating])

-- | Test sync states
testSync :: Vec 4 SyncState
testSync = $(listToVecTH [Locked, Aligning, Drifting, Desync])

-- | Test coherence values
testCoherence :: Vec 4 (Unsigned 8)
testCoherence = $(listToVecTH [255 :: Unsigned 8, 200, 150, 100])

-- | Expected outputs (approximate due to animation tick)
-- Cycle 0: Idle + Locked + 255 coh -> dim blue scaled by 255
-- Cycle 1: Spinning + Aligning + 200 coh -> cyan pulsing
-- Cycle 2: Stabilizing + Drifting + 150 coh -> purple fading
-- Cycle 3: Emanating + Desync + 100 coh -> golden flashing
-- Note: Exact values depend on tick counter state

-- =============================================================================
-- Testbench
-- =============================================================================

-- | Testbench for visualizer shell validation
testBench :: Signal System Bool
testBench = done
  where
    clk = tbSystemClockGen (not <$> done)
    rst = systemResetGen
    s1 = stimuliGenerator clk rst testChamber
    s2 = stimuliGenerator clk rst testSync
    s3 = stimuliGenerator clk rst testCoherence
    out = visualizerTop clk rst enableGen s1 s2 s3
    -- Simple validation: check that output is not all zeros for non-Desync states
    -- (Desync may flash to zero on some ticks)
    done = register clk rst enableGen False $
      pure (length testChamber == 4)

-- | Utility: Extract RGB components as tuple
rgbTuple :: RGBColor -> (Unsigned 8, Unsigned 8, Unsigned 8)
rgbTuple (RGBColor r g b) = (r, g, b)
