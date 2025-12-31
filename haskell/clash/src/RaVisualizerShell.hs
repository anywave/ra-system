{-|
Module      : RaVisualizerShell
Description : Terminal Visualization Data Layer
Copyright   : (c) 2025 Anywave Creations
License     : MIT

Prompt 41: Data structures for terminal visualization of resonance chambers
and sync graphs. Generates rendering data for UART/terminal output.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RaVisualizerShell where

import Clash.Prelude

-- | Default terminal width
defaultWidth :: Unsigned 8
defaultWidth = 120

-- | Color mode for rendering
data ColorMode = ANSI256 | ANSI16 | PlainText
  deriving (Generic, NFDataX, Eq, Show)

-- | Character set for box drawing
data CharSet = UnicodeBox | ASCIIBox
  deriving (Generic, NFDataX, Eq, Show)

-- | Coherence color level
data CoherenceColor = ColorCritical | ColorLow | ColorPartial | ColorSyncing | ColorHigh | ColorOptimal
  deriving (Generic, NFDataX, Eq, Show)

-- | Terminal configuration
data TerminalConfig = TerminalConfig
  { tcWidth     :: Unsigned 8
  , tcHeight    :: Unsigned 8
  , tcColorMode :: ColorMode
  , tcCharSet   :: CharSet
  , tcShowLegend :: Bool
  } deriving (Generic, NFDataX)

-- | Node render data
data NodeRenderData = NodeRenderData
  { nrNodeId    :: Unsigned 8
  , nrCoherence :: Unsigned 8
  , nrSyncState :: Unsigned 4   -- Encoded state
  , nrFrequency :: Unsigned 16
  , nrGridX     :: Unsigned 8
  , nrGridY     :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Link render data
data LinkRenderData = LinkRenderData
  { lrSourceId  :: Unsigned 8
  , lrTargetId  :: Unsigned 8
  , lrStrength  :: Unsigned 8
  , lrLinkType  :: Unsigned 4   -- Encoded type
  } deriving (Generic, NFDataX)

-- | Render output packet (for UART transmission)
data RenderPacket = RenderPacket
  { rpPacketType :: Unsigned 4   -- 0=header, 1=node, 2=link, 3=footer
  , rpPayload0   :: Unsigned 8
  , rpPayload1   :: Unsigned 8
  , rpPayload2   :: Unsigned 8
  , rpPayload3   :: Unsigned 8
  } deriving (Generic, NFDataX)

-- | Default terminal config
defaultConfig :: TerminalConfig
defaultConfig = TerminalConfig
  { tcWidth = 120
  , tcHeight = 24
  , tcColorMode = PlainText
  , tcCharSet = ASCIIBox
  , tcShowLegend = True
  }

-- | Get coherence color level
getCoherenceColor :: Unsigned 8 -> CoherenceColor
getCoherenceColor coherence
  | coherence < 77  = ColorCritical  -- < 0.30
  | coherence < 140 = ColorLow       -- < 0.55
  | coherence < 184 = ColorPartial   -- < 0.72
  | coherence < 217 = ColorSyncing   -- < 0.85
  | coherence < 242 = ColorHigh      -- < 0.95
  | otherwise       = ColorOptimal

-- | Encode color as ANSI code (0-7 for basic colors)
colorToANSI :: CoherenceColor -> Unsigned 4
colorToANSI color = case color of
  ColorCritical -> 1  -- Red
  ColorLow      -> 3  -- Yellow
  ColorPartial  -> 4  -- Blue
  ColorSyncing  -> 6  -- Cyan
  ColorHigh     -> 2  -- Green
  ColorOptimal  -> 5  -- Magenta

-- | Generate progress bar value (0-width)
progressBarFilled :: Unsigned 8 -> Unsigned 8 -> Unsigned 8
progressBarFilled coherence barWidth =
  resize $ (resize coherence * resize barWidth) `div` (255 :: Unsigned 16)

-- | Create node render data
createNodeRenderData :: Unsigned 8 -> Unsigned 8 -> Unsigned 4 -> Unsigned 16 -> Unsigned 8 -> Unsigned 8 -> NodeRenderData
createNodeRenderData nodeId coh state freq gx gy = NodeRenderData
  { nrNodeId = nodeId
  , nrCoherence = coh
  , nrSyncState = state
  , nrFrequency = freq
  , nrGridX = gx
  , nrGridY = gy
  }

-- | Create link render data
createLinkRenderData :: Unsigned 8 -> Unsigned 8 -> Unsigned 8 -> Unsigned 4 -> LinkRenderData
createLinkRenderData src tgt str lt = LinkRenderData
  { lrSourceId = src
  , lrTargetId = tgt
  , lrStrength = str
  , lrLinkType = lt
  }

-- | Encode node as render packet
nodeToPacket :: NodeRenderData -> RenderPacket
nodeToPacket node = RenderPacket
  { rpPacketType = 1  -- Node packet
  , rpPayload0 = nrNodeId node
  , rpPayload1 = nrCoherence node
  , rpPayload2 = nrGridX node
  , rpPayload3 = nrGridY node
  }

-- | Encode link as render packet
linkToPacket :: LinkRenderData -> RenderPacket
linkToPacket link = RenderPacket
  { rpPacketType = 2  -- Link packet
  , rpPayload0 = lrSourceId link
  , rpPayload1 = lrTargetId link
  , rpPayload2 = lrStrength link
  , rpPayload3 = resize (lrLinkType link)
  }

-- | Create header packet with global sync
headerPacket :: Unsigned 8 -> Unsigned 8 -> Unsigned 32 -> RenderPacket
headerPacket globalSync nodeCount timestamp = RenderPacket
  { rpPacketType = 0  -- Header packet
  , rpPayload0 = globalSync
  , rpPayload1 = nodeCount
  , rpPayload2 = resize (timestamp `shiftR` 8)
  , rpPayload3 = resize timestamp
  }

-- | Create footer packet
footerPacket :: RenderPacket
footerPacket = RenderPacket
  { rpPacketType = 3  -- Footer packet
  , rpPayload0 = 0xFF
  , rpPayload1 = 0xFF
  , rpPayload2 = 0xFF
  , rpPayload3 = 0xFF
  }

-- | Box drawing character codes (for UART output)
-- 0=horizontal, 1=vertical, 2=top-left, 3=top-right, 4=bottom-left, 5=bottom-right
boxChar :: CharSet -> Unsigned 4 -> Unsigned 8
boxChar UnicodeBox idx = case idx of
  0 -> 0xC4  -- ─ (horizontal)
  1 -> 0xB3  -- │ (vertical)
  2 -> 0xDA  -- ┌ (top-left)
  3 -> 0xBF  -- ┐ (top-right)
  4 -> 0xC0  -- └ (bottom-left)
  5 -> 0xD9  -- ┘ (bottom-right)
  _ -> 0x20  -- space
boxChar ASCIIBox idx = case idx of
  0 -> 0x2D  -- -
  1 -> 0x7C  -- |
  2 -> 0x2B  -- +
  3 -> 0x2B  -- +
  4 -> 0x2B  -- +
  5 -> 0x2B  -- +
  _ -> 0x20  -- space

-- | Calculate grid position for node in visualization
calculateGridPosition :: Unsigned 8 -> Unsigned 8 -> Unsigned 8 -> (Unsigned 8, Unsigned 8)
calculateGridPosition nodeIdx termWidth termHeight =
  let nodesPerRow = termWidth `div` 12  -- Assuming 12 chars per node
      row = nodeIdx `div` max 1 nodesPerRow
      col = nodeIdx `mod` max 1 nodesPerRow
      gridX = col * 12 + 2
      gridY = row * 4 + 2
  in (gridX, min gridY (termHeight - 4))

-- | Clamp width to valid range
clampWidth :: Unsigned 8 -> Unsigned 8
clampWidth w
  | w < 40  = 40
  | w > 200 = 200
  | otherwise = w

-- | Node render pipeline
nodeRenderPipeline
  :: HiddenClockResetEnable dom
  => Signal dom NodeRenderData
  -> Signal dom RenderPacket
nodeRenderPipeline = fmap nodeToPacket

-- | Link render pipeline
linkRenderPipeline
  :: HiddenClockResetEnable dom
  => Signal dom LinkRenderData
  -> Signal dom RenderPacket
linkRenderPipeline = fmap linkToPacket

-- | Color mapping pipeline
colorPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8)
  -> Signal dom CoherenceColor
colorPipeline = fmap getCoherenceColor

-- | Progress bar pipeline
progressBarPipeline
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 8, Unsigned 8)  -- (coherence, width)
  -> Signal dom (Unsigned 8)              -- filled count
progressBarPipeline = fmap (uncurry progressBarFilled)
