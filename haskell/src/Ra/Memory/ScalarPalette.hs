{-|
Module      : Ra.Memory.ScalarPalette
Description : Scalar field memory patterns and palette management
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements scalar field memory palettes for storing and recalling
coherence patterns, harmonic configurations, and emergence templates.
Palettes enable rapid field configuration and pattern replay.

== Memory Palette Theory

=== Pattern Storage

Scalar field patterns can be captured and stored:

* Coherence maps preserve field state
* Harmonic signatures encode oscillation patterns
* Emergence templates capture fragment configurations

=== Palette Architecture

1. Pattern Library: Named pattern storage
2. Quick Slots: Fast-access pattern slots
3. Blend Engine: Pattern interpolation
4. History Stack: Undo/redo capability
-}
module Ra.Memory.ScalarPalette
  ( -- * Core Types
    ScalarPalette(..)
  , PaletteEntry(..)
  , CoherencePattern(..)
  , PatternSlot(..)

    -- * Palette Management
  , createPalette
  , clearPalette
  , paletteSize

    -- * Pattern Storage
  , storePattern
  , recallPattern
  , deletePattern
  , listPatterns

    -- * Quick Slots
  , assignSlot
  , recallSlot
  , clearSlot
  , slotStatus

    -- * Pattern Blending
  , BlendMode(..)
  , blendPatterns
  , crossfade
  , layerPatterns

    -- * History
  , HistoryEntry(..)
  , pushHistory
  , undoPattern
  , redoPattern
  , historyDepth

    -- * Pattern Analysis
  , analyzePattern
  , patternSimilarity
  , findSimilar
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.List (sortBy, maximumBy)
import Data.Ord (comparing)
import Ra.Constants.Extended (phi)

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Scalar memory palette
data ScalarPalette = ScalarPalette
  { spLibrary    :: !(Map String PaletteEntry)  -- ^ Named patterns
  , spSlots      :: ![PatternSlot]               -- ^ Quick access slots
  , spHistory    :: ![HistoryEntry]              -- ^ History stack
  , spUndoStack  :: ![HistoryEntry]              -- ^ Redo stack
  , spMaxHistory :: !Int                         -- ^ Max history depth
  } deriving (Eq, Show)

-- | Single palette entry
data PaletteEntry = PaletteEntry
  { pePattern    :: !CoherencePattern    -- ^ The pattern data
  , peName       :: !String              -- ^ Pattern name
  , peCreated    :: !Int                 -- ^ Creation timestamp
  , peAccess     :: !Int                 -- ^ Last access time
  , peTags       :: ![String]            -- ^ Pattern tags
  } deriving (Eq, Show)

-- | Coherence pattern data
data CoherencePattern = CoherencePattern
  { cpValues     :: ![Double]                    -- ^ Coherence values
  , cpHarmonics  :: ![(Int, Int, Double)]        -- ^ (l, m, amplitude)
  , cpPhase      :: !Double                      -- ^ Phase offset
  , cpCenter     :: !(Double, Double, Double)    -- ^ Pattern center
  , cpRadius     :: !Double                      -- ^ Pattern radius
  } deriving (Eq, Show)

-- | Quick access slot
data PatternSlot = PatternSlot
  { psIndex      :: !Int                 -- ^ Slot index (0-9)
  , psPattern    :: !(Maybe String)      -- ^ Pattern name or Nothing
  , psLabel      :: !String              -- ^ Slot label
  } deriving (Eq, Show)

-- =============================================================================
-- Palette Management
-- =============================================================================

-- | Create new palette with default slots
createPalette :: Int -> ScalarPalette
createPalette historySize = ScalarPalette
  { spLibrary = Map.empty
  , spSlots = [PatternSlot i Nothing ("Slot " ++ show i) | i <- [0..9]]
  , spHistory = []
  , spUndoStack = []
  , spMaxHistory = historySize
  }

-- | Clear all patterns from palette
clearPalette :: ScalarPalette -> ScalarPalette
clearPalette palette = palette
  { spLibrary = Map.empty
  , spSlots = [slot { psPattern = Nothing } | slot <- spSlots palette]
  , spHistory = []
  , spUndoStack = []
  }

-- | Get number of stored patterns
paletteSize :: ScalarPalette -> Int
paletteSize = Map.size . spLibrary

-- =============================================================================
-- Pattern Storage
-- =============================================================================

-- | Store pattern in library
storePattern :: ScalarPalette -> String -> CoherencePattern -> [String] -> Int -> ScalarPalette
storePattern palette name pattern' tags timestamp =
  let entry = PaletteEntry
        { pePattern = pattern'
        , peName = name
        , peCreated = timestamp
        , peAccess = timestamp
        , peTags = tags
        }
      newLibrary = Map.insert name entry (spLibrary palette)
      histEntry = StoreAction name pattern'
  in pushHistory (palette { spLibrary = newLibrary }) histEntry timestamp

-- | Recall pattern from library
recallPattern :: ScalarPalette -> String -> Int -> Maybe (ScalarPalette, CoherencePattern)
recallPattern palette name timestamp =
  case Map.lookup name (spLibrary palette) of
    Nothing -> Nothing
    Just entry ->
      let updatedEntry = entry { peAccess = timestamp }
          newLibrary = Map.insert name updatedEntry (spLibrary palette)
          histEntry = RecallAction name
          newPalette = pushHistory (palette { spLibrary = newLibrary }) histEntry timestamp
      in Just (newPalette, pePattern entry)

-- | Delete pattern from library
deletePattern :: ScalarPalette -> String -> Int -> ScalarPalette
deletePattern palette name timestamp =
  case Map.lookup name (spLibrary palette) of
    Nothing -> palette
    Just entry ->
      let newLibrary = Map.delete name (spLibrary palette)
          histEntry = DeleteAction name (pePattern entry)
          -- Clear from any slots
          newSlots = [if psPattern slot == Just name
                      then slot { psPattern = Nothing }
                      else slot
                     | slot <- spSlots palette]
      in pushHistory (palette { spLibrary = newLibrary, spSlots = newSlots }) histEntry timestamp

-- | List all pattern names
listPatterns :: ScalarPalette -> [String]
listPatterns = Map.keys . spLibrary

-- =============================================================================
-- Quick Slots
-- =============================================================================

-- | Assign pattern to slot
assignSlot :: ScalarPalette -> Int -> String -> ScalarPalette
assignSlot palette slotIdx patternName
  | slotIdx < 0 || slotIdx > 9 = palette
  | not (Map.member patternName (spLibrary palette)) = palette
  | otherwise =
      let newSlots = [if psIndex slot == slotIdx
                      then slot { psPattern = Just patternName }
                      else slot
                     | slot <- spSlots palette]
      in palette { spSlots = newSlots }

-- | Recall pattern from slot
recallSlot :: ScalarPalette -> Int -> Int -> Maybe (ScalarPalette, CoherencePattern)
recallSlot palette slotIdx timestamp
  | slotIdx < 0 || slotIdx > 9 = Nothing
  | otherwise =
      case filter ((== slotIdx) . psIndex) (spSlots palette) of
        [] -> Nothing
        (slot:_) -> case psPattern slot of
          Nothing -> Nothing
          Just name -> recallPattern palette name timestamp

-- | Clear slot assignment
clearSlot :: ScalarPalette -> Int -> ScalarPalette
clearSlot palette slotIdx
  | slotIdx < 0 || slotIdx > 9 = palette
  | otherwise =
      let newSlots = [if psIndex slot == slotIdx
                      then slot { psPattern = Nothing }
                      else slot
                     | slot <- spSlots palette]
      in palette { spSlots = newSlots }

-- | Get slot status summary
slotStatus :: ScalarPalette -> [(Int, String, Bool)]
slotStatus palette =
  [(psIndex s, psLabel s, psPattern s /= Nothing) | s <- spSlots palette]

-- =============================================================================
-- Pattern Blending
-- =============================================================================

-- | Blend mode types
data BlendMode
  = BlendLinear       -- ^ Linear interpolation
  | BlendGolden       -- ^ Golden ratio weighted
  | BlendHarmonic     -- ^ Harmonic average
  | BlendMax          -- ^ Maximum values
  | BlendMin          -- ^ Minimum values
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Blend two patterns
blendPatterns :: BlendMode -> CoherencePattern -> CoherencePattern -> Double -> CoherencePattern
blendPatterns mode p1 p2 factor =
  let blendFunc = case mode of
        BlendLinear   -> linearBlend
        BlendGolden   -> goldenBlend
        BlendHarmonic -> harmonicBlend
        BlendMax      -> maxBlend
        BlendMin      -> minBlend
      values1 = cpValues p1
      values2 = cpValues p2
      -- Pad shorter list
      len = max (length values1) (length values2)
      padded1 = values1 ++ replicate (len - length values1) 0
      padded2 = values2 ++ replicate (len - length values2) 0
      newValues = zipWith (blendFunc factor) padded1 padded2
      -- Blend harmonics
      newHarmonics = blendHarmonics (cpHarmonics p1) (cpHarmonics p2) factor
      -- Blend positions
      (x1, y1, z1) = cpCenter p1
      (x2, y2, z2) = cpCenter p2
      newCenter = (x1 + (x2-x1)*factor, y1 + (y2-y1)*factor, z1 + (z2-z1)*factor)
      newRadius = cpRadius p1 + (cpRadius p2 - cpRadius p1) * factor
      newPhase = cpPhase p1 + (cpPhase p2 - cpPhase p1) * factor
  in CoherencePattern
    { cpValues = newValues
    , cpHarmonics = newHarmonics
    , cpPhase = newPhase
    , cpCenter = newCenter
    , cpRadius = newRadius
    }

-- | Crossfade between patterns
crossfade :: CoherencePattern -> CoherencePattern -> Double -> CoherencePattern
crossfade = blendPatterns BlendLinear

-- | Layer patterns (additive)
layerPatterns :: [CoherencePattern] -> CoherencePattern
layerPatterns [] = emptyPattern
layerPatterns [p] = p
layerPatterns (p:ps) =
  let combined = layerPatterns ps
      values = zipWith (+) (cpValues p) (padValues (cpValues combined) (length (cpValues p)))
      harmonics = cpHarmonics p ++ cpHarmonics combined
  in p { cpValues = map (min 1.0) values, cpHarmonics = harmonics }

-- =============================================================================
-- History
-- =============================================================================

-- | History entry types
data HistoryEntry
  = StoreAction !String !CoherencePattern   -- ^ Stored pattern
  | RecallAction !String                    -- ^ Recalled pattern
  | DeleteAction !String !CoherencePattern  -- ^ Deleted pattern
  | BlendAction !String !String !Double     -- ^ Blended patterns
  deriving (Eq, Show)

-- | Push to history stack
pushHistory :: ScalarPalette -> HistoryEntry -> Int -> ScalarPalette
pushHistory palette entry _timestamp =
  let newHistory = entry : take (spMaxHistory palette - 1) (spHistory palette)
  in palette { spHistory = newHistory, spUndoStack = [] }

-- | Undo last action
undoPattern :: ScalarPalette -> Int -> ScalarPalette
undoPattern palette timestamp =
  case spHistory palette of
    [] -> palette
    (entry:rest) ->
      let newPalette = applyUndo palette entry timestamp
          newUndoStack = entry : spUndoStack palette
      in newPalette { spHistory = rest, spUndoStack = newUndoStack }

-- | Redo last undone action
redoPattern :: ScalarPalette -> Int -> ScalarPalette
redoPattern palette timestamp =
  case spUndoStack palette of
    [] -> palette
    (entry:rest) ->
      let newPalette = applyRedo palette entry timestamp
          newHistory = entry : spHistory palette
      in newPalette { spHistory = newHistory, spUndoStack = rest }

-- | Get history depth
historyDepth :: ScalarPalette -> Int
historyDepth = length . spHistory

-- =============================================================================
-- Pattern Analysis
-- =============================================================================

-- | Pattern analysis result
data PatternAnalysis = PatternAnalysis
  { paCoherenceAvg   :: !Double    -- ^ Average coherence
  , paHarmonicCount  :: !Int       -- ^ Number of harmonics
  , paDominantL      :: !Int       -- ^ Dominant l value
  , paComplexity     :: !Double    -- ^ Pattern complexity
  } deriving (Eq, Show)

-- | Analyze pattern characteristics
analyzePattern :: CoherencePattern -> PatternAnalysis
analyzePattern pattern' =
  let values = cpValues pattern'
      avg = if null values then 0 else sum values / fromIntegral (length values)
      harmonics = cpHarmonics pattern'
      dominantL = if null harmonics
                  then 0
                  else let (l, _, _) = maximumBy (comparing (\(_,_,a) -> a)) harmonics
                       in l
      complexity = fromIntegral (length harmonics) * avg * phi
  in PatternAnalysis
    { paCoherenceAvg = avg
    , paHarmonicCount = length harmonics
    , paDominantL = dominantL
    , paComplexity = complexity
    }

-- | Compute similarity between patterns
patternSimilarity :: CoherencePattern -> CoherencePattern -> Double
patternSimilarity p1 p2 =
  let values1 = cpValues p1
      values2 = cpValues p2
      len = max (length values1) (length values2)
      padded1 = values1 ++ replicate (len - length values1) 0
      padded2 = values2 ++ replicate (len - length values2) 0
      diffs = zipWith (\a b -> (a - b) ^ (2 :: Int)) padded1 padded2
      mse = if null diffs then 0 else sum diffs / fromIntegral (length diffs)
      similarity = 1 / (1 + mse)
      -- Phase similarity
      phaseDiff = abs (cpPhase p1 - cpPhase p2)
      phaseSim = 1 - phaseDiff / (2 * pi)
  in similarity * 0.8 + phaseSim * 0.2

-- | Find similar patterns in library
findSimilar :: ScalarPalette -> CoherencePattern -> Double -> [String]
findSimilar palette pattern' threshold =
  let entries = Map.toList (spLibrary palette)
      similarities = [(name, patternSimilarity pattern' (pePattern entry))
                     | (name, entry) <- entries]
      filtered = filter ((>= threshold) . snd) similarities
      sorted = sortBy (flip (comparing snd)) filtered
  in map fst sorted

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- | Empty pattern
emptyPattern :: CoherencePattern
emptyPattern = CoherencePattern [] [] 0 (0, 0, 0) 0

-- | Linear blend function
linearBlend :: Double -> Double -> Double -> Double
linearBlend t a b = a + (b - a) * t

-- | Golden ratio blend
goldenBlend :: Double -> Double -> Double -> Double
goldenBlend t a b =
  let goldenT = t * phi / (1 + t * (phi - 1))
  in a + (b - a) * goldenT

-- | Harmonic blend (geometric mean weighted)
harmonicBlend :: Double -> Double -> Double -> Double
harmonicBlend t a b
  | a <= 0 || b <= 0 = linearBlend t a b
  | otherwise = a ** (1 - t) * b ** t

-- | Max blend
maxBlend :: Double -> Double -> Double -> Double
maxBlend _t a b = max a b

-- | Min blend
minBlend :: Double -> Double -> Double -> Double
minBlend _t a b = min a b

-- | Blend harmonic lists
blendHarmonics :: [(Int, Int, Double)] -> [(Int, Int, Double)] -> Double -> [(Int, Int, Double)]
blendHarmonics h1 h2 factor =
  let scale1 = 1 - factor
      scale2 = factor
      scaled1 = [(l, m, a * scale1) | (l, m, a) <- h1]
      scaled2 = [(l, m, a * scale2) | (l, m, a) <- h2]
  in scaled1 ++ scaled2

-- | Pad values list
padValues :: [Double] -> Int -> [Double]
padValues vals targetLen
  | length vals >= targetLen = vals
  | otherwise = vals ++ replicate (targetLen - length vals) 0

-- | Apply undo action
applyUndo :: ScalarPalette -> HistoryEntry -> Int -> ScalarPalette
applyUndo palette (StoreAction name _) _ts =
  palette { spLibrary = Map.delete name (spLibrary palette) }
applyUndo palette (RecallAction _) _ts = palette
applyUndo palette (DeleteAction name pattern') ts =
  storePattern palette name pattern' [] ts
applyUndo palette (BlendAction _ _ _) _ts = palette

-- | Apply redo action
applyRedo :: ScalarPalette -> HistoryEntry -> Int -> ScalarPalette
applyRedo palette (StoreAction name pattern') ts =
  storePattern palette name pattern' [] ts
applyRedo palette (RecallAction _) _ts = palette
applyRedo palette (DeleteAction name _) ts =
  deletePattern palette name ts
applyRedo palette (BlendAction _ _ _) _ts = palette
