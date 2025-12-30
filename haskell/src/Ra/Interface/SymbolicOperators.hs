{-|
Module      : Ra.Interface.SymbolicOperators
Description : Mathematical extensions for coherence algebra
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Defines symbolic coherence and scalar logic operators for expressing,
composing, and evaluating coherence transformations, field contractions,
emergence gates, and resonance harmonics using an expressive symbolic DSL.

== Design Philosophy

This module supports symbolic reasoning and rule-based transformations:

* Formal rule encoding (e.g. coherence(θ) ∘ invert(φ))
* Transformation scripting for simulations
* Human-readable gate schematics (as glyphs or logic trees)
* Integration with symbolic music/geometry/memory structures

== Operator Composition

Operators are composable (monoidal where possible):

* Composition executes left-to-right (prefix logic)
* Support for coherence predicates, resonance filters, field algebra
* Extendable to visual glyphs, Ra.Music, and Ra.Appendage
-}
module Ra.Interface.SymbolicOperators
  ( -- * Core Operator Types
    CoherenceOp(..)
  , Axis(..)
  , AnkhVector(..)
  , SymbolicExpr(..)

    -- * Operator Application
  , applyOp
  , applyOps
  , evaluateExpr

    -- * Operator Rendering
  , renderOp
  , renderExpr
  , parseOp

    -- * Predefined Operators
  , coherenceFilter
  , phaseNormalizer
  , harmonicBoost
  , symmetryReset
  , inversionGate

    -- * Composition Helpers
  , compose
  , sequence'
  , parallel
  , conditional

    -- * Coherence Predicates
  , CoherencePredicate(..)
  , evalPredicate
  , aboveThreshold
  , belowThreshold
  , inBand
  , harmonicMatch

    -- * Field Algebra
  , FieldOp(..)
  , applyFieldOp
  , fieldContract
  , fieldExpand
  , fieldRotate

    -- * Resonance Filters
  , ResonanceFilter(..)
  , applyFilter
  , bandpass
  , highpass
  , lowpass
  , notch

    -- * Symbolic Serialization
  , encodeOps
  , decodeOps
  , opToGlyph
  , glyphToOp
  ) where

import Data.List (intercalate)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)

import Ra.Scalar
  ( EmergenceCondition(..)
  , Inversion(..)
  , WellDepth(..)
  , FluxCoherence(..)
  , TemporalWindow(..)
  , Coordinate(..)
  , ShellIndex(..)
  )
import Ra.Omega (OmegaFormat(..))
import Ra.Constants.Extended (phi, phiInverse)
import Ra.Constants (ankh, Ankh(..))

-- =============================================================================
-- Core Operator Types
-- =============================================================================

-- | Axis for angular operations
data Axis
  = Theta     -- ^ Semantic axis (27 Repitans)
  | Phi       -- ^ Access level axis (6 RACs)
  | Omega     -- ^ Harmonic depth axis (5 formats)
  | Radial    -- ^ Shell depth axis
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | Ankh symmetry vector for balance operations
data AnkhVector = AnkhVector
  { avTheta :: !Double    -- ^ θ-axis component
  , avPhi   :: !Double    -- ^ φ-axis component
  , avOmega :: !Double    -- ^ ω-axis component
  } deriving (Eq, Show)

-- | Zero vector (neutral symmetry)
zeroAnkh :: AnkhVector
zeroAnkh = AnkhVector 0 0 0

-- | Abstract operator over scalar coherence or emergence state
data CoherenceOp
  = PhaseShift !Double            -- ^ Shift φ-phase by amount
  | InvertAngle !Axis             -- ^ Invert over specified axis
  | BoostHarmonic !OmegaFormat    -- ^ Apply harmonic boost to format
  | GateThreshold !Double         -- ^ Minimum emergence threshold
  | SymmetryCancel !AnkhVector    -- ^ Nullify symmetry offset
  | ScaleCoherence !Double        -- ^ Multiply coherence by factor
  | ClampCoherence !Double !Double -- ^ Clamp to [min, max]
  | ShiftShell !Int               -- ^ Shift shell index
  | SetInversion !Inversion       -- ^ Set inversion state
  | TemporalPhase !Int            -- ^ Set φ^n temporal phase
  | NoOp                          -- ^ Identity operation
  | ComposeOp ![CoherenceOp]      -- ^ Compose multiple ops (left-to-right)
  deriving (Eq, Show)

-- | Symbolic expression for complex operations
data SymbolicExpr
  = OpExpr !CoherenceOp
  | AndExpr !SymbolicExpr !SymbolicExpr
  | OrExpr !SymbolicExpr !SymbolicExpr
  | NotExpr !SymbolicExpr
  | IfExpr !CoherencePredicate !SymbolicExpr !SymbolicExpr
  | LetExpr !String !SymbolicExpr !SymbolicExpr
  | VarExpr !String
  deriving (Eq, Show)

-- =============================================================================
-- Operator Application
-- =============================================================================

-- | Apply a single coherence operation to emergence condition
applyOp :: CoherenceOp -> EmergenceCondition -> EmergenceCondition
applyOp op ec = case op of
  PhaseShift delta ->
    let TW tw = ecTemporalPhase ec
    in ec { ecTemporalPhase = TW (tw + delta) }

  InvertAngle axis -> case axis of
    Theta -> ec  -- Theta inversion is semantic, not directly applicable here
    Phi -> ec    -- Phi inversion affects access level
    Omega -> ec  -- Omega inversion affects harmonic depth
    Radial -> ec { ecInversion = flipInversion (ecInversion ec) }

  BoostHarmonic fmt ->
    let coord = ecCoordinate ec
    in ec { ecCoordinate = coord { cOmega = fmt } }

  GateThreshold threshold ->
    let FC flux bands = ecFluxCoherence ec
        gatedFlux = if flux >= threshold then flux else 0.0
    in ec { ecFluxCoherence = FC gatedFlux bands }

  SymmetryCancel av ->
    let WD depth = ecPotential ec
        offset = avTheta av * 0.1 + avPhi av * 0.1 + avOmega av * 0.1
    in ec { ecPotential = WD (depth - offset) }

  ScaleCoherence factor ->
    let FC flux bands = ecFluxCoherence ec
    in ec { ecFluxCoherence = FC (flux * factor) bands }

  ClampCoherence minV maxV ->
    let FC flux bands = ecFluxCoherence ec
        clampedFlux = max minV (min maxV flux)
    in ec { ecFluxCoherence = FC clampedFlux bands }

  ShiftShell delta ->
    let coord = ecCoordinate ec
        SI s = cShell coord
        newShell = SI (max 0 (s + delta))
    in ec { ecCoordinate = coord { cShell = newShell } }

  SetInversion inv ->
    ec { ecInversion = inv }

  TemporalPhase n ->
    ec { ecTemporalPhase = TW (phi ** fromIntegral n) }

  NoOp -> ec

  ComposeOp ops -> foldl (flip applyOp) ec ops

-- | Flip inversion state
flipInversion :: Inversion -> Inversion
flipInversion Normal = Inverted
flipInversion Inverted = Normal

-- | Apply multiple operations in sequence
applyOps :: [CoherenceOp] -> EmergenceCondition -> EmergenceCondition
applyOps ops ec = foldl (flip applyOp) ec ops

-- | Evaluate symbolic expression with variable bindings
evaluateExpr :: Map String EmergenceCondition -> SymbolicExpr -> EmergenceCondition -> EmergenceCondition
evaluateExpr bindings expr ec = case expr of
  OpExpr op -> applyOp op ec

  AndExpr e1 e2 ->
    let ec1 = evaluateExpr bindings e1 ec
    in evaluateExpr bindings e2 ec1

  OrExpr e1 e2 ->
    let ec1 = evaluateExpr bindings e1 ec
        ec2 = evaluateExpr bindings e2 ec
        FC f1 _ = ecFluxCoherence ec1
        FC f2 _ = ecFluxCoherence ec2
    in if f1 >= f2 then ec1 else ec2

  NotExpr e ->
    let result = evaluateExpr bindings e ec
    in result { ecInversion = flipInversion (ecInversion result) }

  IfExpr pred' thenExpr elseExpr ->
    if evalPredicate pred' ec
    then evaluateExpr bindings thenExpr ec
    else evaluateExpr bindings elseExpr ec

  LetExpr name valExpr bodyExpr ->
    let val = evaluateExpr bindings valExpr ec
        newBindings = Map.insert name val bindings
    in evaluateExpr newBindings bodyExpr val

  VarExpr name ->
    Map.findWithDefault ec name bindings

-- =============================================================================
-- Operator Rendering
-- =============================================================================

-- | Render operator as symbolic DSL string
renderOp :: CoherenceOp -> String
renderOp op = case op of
  PhaseShift delta -> "φ-shift(" ++ show delta ++ ")"
  InvertAngle axis -> "invert(" ++ show axis ++ ")"
  BoostHarmonic fmt -> "boost(" ++ show fmt ++ ")"
  GateThreshold t -> "gate(≥" ++ show t ++ ")"
  SymmetryCancel av -> "cancel-sym(" ++ showAnkh av ++ ")"
  ScaleCoherence f -> "scale(" ++ show f ++ ")"
  ClampCoherence minV maxV -> "clamp[" ++ show minV ++ "," ++ show maxV ++ "]"
  ShiftShell d -> "shell-shift(" ++ show d ++ ")"
  SetInversion inv -> "set-inv(" ++ show inv ++ ")"
  TemporalPhase n -> "φ^" ++ show n
  NoOp -> "id"
  ComposeOp ops -> intercalate " ∘ " (map renderOp ops)

-- | Show Ankh vector compactly
showAnkh :: AnkhVector -> String
showAnkh (AnkhVector t p o) = "<" ++ show t ++ "," ++ show p ++ "," ++ show o ++ ">"

-- | Render symbolic expression
renderExpr :: SymbolicExpr -> String
renderExpr expr = case expr of
  OpExpr op -> renderOp op
  AndExpr e1 e2 -> "(" ++ renderExpr e1 ++ " ∧ " ++ renderExpr e2 ++ ")"
  OrExpr e1 e2 -> "(" ++ renderExpr e1 ++ " ∨ " ++ renderExpr e2 ++ ")"
  NotExpr e -> "¬(" ++ renderExpr e ++ ")"
  IfExpr p t e -> "if " ++ renderPredicate p ++ " then " ++ renderExpr t ++ " else " ++ renderExpr e
  LetExpr n v b -> "let " ++ n ++ " = " ++ renderExpr v ++ " in " ++ renderExpr b
  VarExpr n -> n

-- | Parse operator from string (simplified parser)
parseOp :: String -> Maybe CoherenceOp
parseOp s = case words s of
  ["id"] -> Just NoOp
  ["invert", "Theta"] -> Just (InvertAngle Theta)
  ["invert", "Phi"] -> Just (InvertAngle Phi)
  ["invert", "Omega"] -> Just (InvertAngle Omega)
  ["invert", "Radial"] -> Just (InvertAngle Radial)
  _ -> Nothing  -- Simplified: full parser would handle more cases

-- =============================================================================
-- Predefined Operators
-- =============================================================================

-- | Gate that only passes coherence above φ / ankh ≈ 0.318
coherenceFilter :: CoherenceOp
coherenceFilter = GateThreshold (phi / unAnkh ankh)

-- | Normalize phase to φ^0 = 1
phaseNormalizer :: CoherenceOp
phaseNormalizer = TemporalPhase 0

-- | Boost to Red harmonic format (highest precision)
harmonicBoost :: CoherenceOp
harmonicBoost = BoostHarmonic Red

-- | Reset symmetry to neutral
symmetryReset :: CoherenceOp
symmetryReset = SymmetryCancel zeroAnkh

-- | Gate based on inversion state
inversionGate :: Inversion -> CoherenceOp
inversionGate inv = ComposeOp
  [ SetInversion inv
  , GateThreshold phiInverse
  ]

-- =============================================================================
-- Composition Helpers
-- =============================================================================

-- | Compose two operators
compose :: CoherenceOp -> CoherenceOp -> CoherenceOp
compose op1 op2 = ComposeOp [op1, op2]

-- | Sequence multiple operators
sequence' :: [CoherenceOp] -> CoherenceOp
sequence' [] = NoOp
sequence' [op] = op
sequence' ops = ComposeOp ops

-- | Apply operators in parallel and combine results (takes max coherence)
parallel :: [CoherenceOp] -> CoherenceOp
parallel ops = ComposeOp ops  -- Simplified: real parallel would fork/join

-- | Conditional operator (gate-based)
conditional :: Double -> CoherenceOp -> CoherenceOp -> CoherenceOp
conditional threshold thenOp _elseOp =
  ComposeOp [GateThreshold threshold, thenOp]
  -- Note: elseOp would need runtime branching support

-- =============================================================================
-- Coherence Predicates
-- =============================================================================

-- | Predicate for coherence testing
data CoherencePredicate
  = AboveThreshold !Double
  | BelowThreshold !Double
  | InBand !Double !Double
  | HarmonicMatch !Int !Int    -- ^ (l, m) harmonic indices
  | IsInverted
  | IsNormal
  | AndPred !CoherencePredicate !CoherencePredicate
  | OrPred !CoherencePredicate !CoherencePredicate
  | NotPred !CoherencePredicate
  deriving (Eq, Show)

-- | Evaluate predicate on emergence condition
evalPredicate :: CoherencePredicate -> EmergenceCondition -> Bool
evalPredicate pred' ec = case pred' of
  AboveThreshold t ->
    let FC flux _ = ecFluxCoherence ec
    in flux >= t

  BelowThreshold t ->
    let FC flux _ = ecFluxCoherence ec
    in flux < t

  InBand lo hi ->
    let FC flux _ = ecFluxCoherence ec
    in flux >= lo && flux <= hi

  HarmonicMatch _ _ -> True  -- Simplified: would check harmonic signature

  IsInverted -> ecInversion ec == Inverted

  IsNormal -> ecInversion ec == Normal

  AndPred p1 p2 -> evalPredicate p1 ec && evalPredicate p2 ec

  OrPred p1 p2 -> evalPredicate p1 ec || evalPredicate p2 ec

  NotPred p -> not (evalPredicate p ec)

-- | Render predicate as string
renderPredicate :: CoherencePredicate -> String
renderPredicate pred' = case pred' of
  AboveThreshold t -> "coherence ≥ " ++ show t
  BelowThreshold t -> "coherence < " ++ show t
  InBand lo hi -> "coherence ∈ [" ++ show lo ++ "," ++ show hi ++ "]"
  HarmonicMatch l m -> "H_{" ++ show l ++ "," ++ show m ++ "}"
  IsInverted -> "inverted?"
  IsNormal -> "normal?"
  AndPred p1 p2 -> "(" ++ renderPredicate p1 ++ " ∧ " ++ renderPredicate p2 ++ ")"
  OrPred p1 p2 -> "(" ++ renderPredicate p1 ++ " ∨ " ++ renderPredicate p2 ++ ")"
  NotPred p -> "¬" ++ renderPredicate p

-- | Create above-threshold predicate
aboveThreshold :: Double -> CoherencePredicate
aboveThreshold = AboveThreshold

-- | Create below-threshold predicate
belowThreshold :: Double -> CoherencePredicate
belowThreshold = BelowThreshold

-- | Create in-band predicate
inBand :: Double -> Double -> CoherencePredicate
inBand = InBand

-- | Create harmonic match predicate
harmonicMatch :: Int -> Int -> CoherencePredicate
harmonicMatch = HarmonicMatch

-- =============================================================================
-- Field Algebra
-- =============================================================================

-- | Field-level operations
data FieldOp
  = Contract !Double        -- ^ Contract field by factor
  | Expand !Double          -- ^ Expand field by factor
  | Rotate !Axis !Double    -- ^ Rotate around axis by angle
  | Mirror !Axis            -- ^ Mirror across axis
  | Blend !Double !Double   -- ^ Blend two field states
  | Modulate !Double !Double -- ^ Frequency modulation
  deriving (Eq, Show)

-- | Apply field operation
applyFieldOp :: FieldOp -> EmergenceCondition -> EmergenceCondition
applyFieldOp fop ec = case fop of
  Contract factor ->
    let WD depth = ecPotential ec
    in ec { ecPotential = WD (depth * factor) }

  Expand factor ->
    let WD depth = ecPotential ec
    in ec { ecPotential = WD (depth / factor) }

  Rotate _ angle ->
    let TW tw = ecTemporalPhase ec
    in ec { ecTemporalPhase = TW (tw + angle) }

  Mirror axis ->
    applyOp (InvertAngle axis) ec

  Blend factor1 factor2 ->
    let WD depth = ecPotential ec
        FC flux bands = ecFluxCoherence ec
    in ec { ecPotential = WD (depth * factor1)
          , ecFluxCoherence = FC (flux * factor2) bands
          }

  Modulate freq amp ->
    let FC flux bands = ecFluxCoherence ec
        modulated = flux * (1.0 + amp * sin (freq * flux))
    in ec { ecFluxCoherence = FC modulated bands }

-- | Contract field by factor
fieldContract :: Double -> FieldOp
fieldContract = Contract

-- | Expand field by factor
fieldExpand :: Double -> FieldOp
fieldExpand = Expand

-- | Rotate field around axis
fieldRotate :: Axis -> Double -> FieldOp
fieldRotate = Rotate

-- =============================================================================
-- Resonance Filters
-- =============================================================================

-- | Resonance filter types
data ResonanceFilter
  = Bandpass !Double !Double   -- ^ Pass band [lo, hi]
  | Highpass !Double           -- ^ Pass above threshold
  | Lowpass !Double            -- ^ Pass below threshold
  | Notch !Double !Double      -- ^ Reject band [lo, hi]
  | Comb !Double !Int          -- ^ Comb filter (spacing, teeth)
  | Resonant !Double !Double   -- ^ Resonant peak (freq, Q)
  deriving (Eq, Show)

-- | Apply resonance filter
applyFilter :: ResonanceFilter -> EmergenceCondition -> EmergenceCondition
applyFilter filt ec = case filt of
  Bandpass lo hi ->
    let FC flux bands = ecFluxCoherence ec
        passed = if flux >= lo && flux <= hi then flux else 0.0
    in ec { ecFluxCoherence = FC passed bands }

  Highpass threshold ->
    let FC flux bands = ecFluxCoherence ec
        passed = if flux >= threshold then flux else 0.0
    in ec { ecFluxCoherence = FC passed bands }

  Lowpass threshold ->
    let FC flux bands = ecFluxCoherence ec
        passed = if flux <= threshold then flux else threshold
    in ec { ecFluxCoherence = FC passed bands }

  Notch lo hi ->
    let FC flux bands = ecFluxCoherence ec
        passed = if flux >= lo && flux <= hi then 0.0 else flux
    in ec { ecFluxCoherence = FC passed bands }

  Comb spacing teeth ->
    let FC flux bands = ecFluxCoherence ec
        -- Check if flux falls on comb tooth
        onTooth = any (\i -> abs (flux - fromIntegral i * spacing) < 0.01) [0..teeth]
        passed = if onTooth then flux else flux * 0.1
    in ec { ecFluxCoherence = FC passed bands }

  Resonant freq q ->
    let FC flux bands = ecFluxCoherence ec
        -- Resonant boost near frequency
        distance = abs (flux - freq)
        boost = q / (1.0 + distance * distance * q)
        resonated = flux * (1.0 + boost)
    in ec { ecFluxCoherence = FC resonated bands }

-- | Create bandpass filter
bandpass :: Double -> Double -> ResonanceFilter
bandpass = Bandpass

-- | Create highpass filter
highpass :: Double -> ResonanceFilter
highpass = Highpass

-- | Create lowpass filter
lowpass :: Double -> ResonanceFilter
lowpass = Lowpass

-- | Create notch filter
notch :: Double -> Double -> ResonanceFilter
notch = Notch

-- =============================================================================
-- Symbolic Serialization
-- =============================================================================

-- | Encode operators to string representation
encodeOps :: [CoherenceOp] -> String
encodeOps = intercalate "; " . map renderOp

-- | Decode operators from string (simplified)
decodeOps :: String -> [CoherenceOp]
decodeOps s = mapMaybe parseOp (splitOn "; " s)
  where
    splitOn :: String -> String -> [String]
    splitOn _ "" = []
    splitOn delim str = case breakOn delim str of
      (before, "") -> [before]
      (before, after) -> before : splitOn delim (drop (length delim) after)

    breakOn :: String -> String -> (String, String)
    breakOn _ "" = ("", "")
    breakOn delim str@(c:cs)
      | take (length delim) str == delim = ("", str)
      | otherwise = let (b, a) = breakOn delim cs in (c:b, a)

    mapMaybe :: (a -> Maybe b) -> [a] -> [b]
    mapMaybe _ [] = []
    mapMaybe f (x:xs) = case f x of
      Nothing -> mapMaybe f xs
      Just y -> y : mapMaybe f xs

-- | Convert operator to glyph representation
opToGlyph :: CoherenceOp -> String
opToGlyph op = case op of
  PhaseShift _ -> "⟲"      -- Rotation symbol
  InvertAngle Theta -> "⊖θ"
  InvertAngle Phi -> "⊖φ"
  InvertAngle Omega -> "⊖ω"
  InvertAngle Radial -> "⊖r"
  BoostHarmonic Red -> "△R"
  BoostHarmonic OmegaMajor -> "△Ω+"
  BoostHarmonic Green -> "△G"
  BoostHarmonic OmegaMinor -> "△Ω-"
  BoostHarmonic Blue -> "△B"
  GateThreshold _ -> "⊳"   -- Gate symbol
  SymmetryCancel _ -> "≎"  -- Symmetry cancel
  ScaleCoherence _ -> "×"
  ClampCoherence _ _ -> "⌈⌋"
  ShiftShell _ -> "↕"
  SetInversion Normal -> "◇"
  SetInversion Inverted -> "◆"
  TemporalPhase n -> "φ" ++ superscript n
  NoOp -> "·"
  ComposeOp ops -> intercalate "∘" (map opToGlyph ops)
  where
    superscript :: Int -> String
    superscript n
      | n == 0 = "⁰"
      | n == 1 = "¹"
      | n == 2 = "²"
      | n == 3 = "³"
      | n == 4 = "⁴"
      | n == 5 = "⁵"
      | n == 6 = "⁶"
      | n == 7 = "⁷"
      | n == 8 = "⁸"
      | n == 9 = "⁹"
      | otherwise = "^" ++ show n

-- | Convert glyph back to operator (simplified)
glyphToOp :: String -> Maybe CoherenceOp
glyphToOp glyph = case glyph of
  "⟲" -> Just (PhaseShift 0.0)  -- Default phase shift
  "⊖θ" -> Just (InvertAngle Theta)
  "⊖φ" -> Just (InvertAngle Phi)
  "⊖ω" -> Just (InvertAngle Omega)
  "⊖r" -> Just (InvertAngle Radial)
  "△R" -> Just (BoostHarmonic Red)
  "△G" -> Just (BoostHarmonic Green)
  "△B" -> Just (BoostHarmonic Blue)
  "⊳" -> Just (GateThreshold 0.5)  -- Default threshold
  "≎" -> Just (SymmetryCancel zeroAnkh)
  "·" -> Just NoOp
  "◇" -> Just (SetInversion Normal)
  "◆" -> Just (SetInversion Inverted)
  _ -> Nothing
