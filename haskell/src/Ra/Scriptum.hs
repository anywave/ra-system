{-|
Module      : Ra.Scriptum
Description : Ritual definition DSL for scalar interactions
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

A lightweight domain-specific language for scripting scalar interactions,
user rituals, fragment invitations, and coherence actions.

== Scriptum Syntax

Core commands:

* @summon@ - Invoke a fragment by ID or harmonic
* @amplify@ - Increase field potential
* @invert@ - Toggle inversion state
* @attune@ - Align to target frequency
* @ground@ - Reset to baseline
* @gate@ - Set consent gate level
* @wait@ - Pause for duration

== Example Script

@
summon \"heart.center\" harmonic(2,1)
amplify 0.8 duration(30s)
attune 528Hz
wait 60s
ground
@
-}
module Ra.Scriptum
  ( -- * AST Types
    ScriptumAST(..)
  , ScriptumCommand(..)
  , ScriptumExpr(..)
  , Duration(..)

    -- * Parsing
  , parseScriptum
  , parseCommand
  , parseExpr

    -- * Interpretation
  , ScriptumState(..)
  , ScriptumResult(..)
  , interpret
  , interpretCommand
  , runScript

    -- * Chamber Modification
  , ChamberMod(..)
  , applyMod
  , modToPotential
  , modToInversion

    -- * Built-in Commands
  , summonFragment
  , amplifyField
  , invertField
  , attuneFrequency
  , groundField
  , setGate
  , waitDuration

    -- * Script Building
  , script
  , (&>)
  , summon
  , amplify
  , invert
  , attune
  , ground
  , gate
  , wait
  ) where

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- AST Types
-- =============================================================================

-- | Complete Scriptum AST
data ScriptumAST = ScriptumAST
  { astCommands   :: ![ScriptumCommand]
  , astMetadata   :: ![(String, String)]  -- ^ Script metadata
  } deriving (Eq, Show)

-- | Individual command
data ScriptumCommand
  = Summon !String !(Maybe (Int, Int))      -- ^ Summon fragment with optional harmonic
  | Amplify !Double !(Maybe Duration)       -- ^ Amplify with optional duration
  | Invert                                  -- ^ Toggle inversion
  | Attune !Double                          -- ^ Attune to frequency (Hz)
  | Ground                                  -- ^ Reset to baseline
  | Gate !Int                               -- ^ Set consent gate level
  | Wait !Duration                          -- ^ Wait for duration
  | Sequence ![ScriptumCommand]             -- ^ Sequential execution
  | Parallel ![ScriptumCommand]             -- ^ Parallel execution
  | Repeat !Int !ScriptumCommand            -- ^ Repeat N times
  | Conditional !ScriptumExpr !ScriptumCommand !(Maybe ScriptumCommand)
  deriving (Eq, Show)

-- | Expression for conditionals
data ScriptumExpr
  = CoherenceAbove !Double
  | CoherenceBelow !Double
  | InversionState !Bool
  | GateLevel !Int
  | Always
  | Never
  | And !ScriptumExpr !ScriptumExpr
  | Or !ScriptumExpr !ScriptumExpr
  | Not !ScriptumExpr
  deriving (Eq, Show)

-- | Duration specification
data Duration
  = Seconds !Double
  | Beats !Int
  | PhiCycles !Int        -- ^ φ^n seconds
  deriving (Eq, Show)

-- =============================================================================
-- Parsing
-- =============================================================================

-- | Parse script from text
parseScriptum :: String -> Either String ScriptumAST
parseScriptum input =
  let linesRaw = lines input
      nonEmpty = filter (not . null . dropWhile (== ' ')) linesRaw
      parsed = mapM parseCommand nonEmpty
  in case parsed of
       Left err -> Left err
       Right cmds -> Right $ ScriptumAST cmds []

-- | Parse single command
parseCommand :: String -> Either String ScriptumCommand
parseCommand input =
  let tokens = words input
  in case tokens of
       ("summon":fragId:rest) ->
         Right $ Summon fragId (parseHarmonic rest)
       ("amplify":val:rest) ->
         case reads val of
           [(v, "")] -> Right $ Amplify v (parseDuration rest)
           _ -> Left $ "Invalid amplify value: " ++ val
       ["invert"] -> Right Invert
       ("attune":freq:_) ->
         case reads (filter (/= 'H') (filter (/= 'z') freq)) of
           [(f, "")] -> Right $ Attune f
           _ -> Left $ "Invalid frequency: " ++ freq
       ["ground"] -> Right Ground
       ("gate":level:_) ->
         case reads level of
           [(l, "")] -> Right $ Gate l
           _ -> Left $ "Invalid gate level: " ++ level
       ("wait":dur:_) ->
         case parseDurationStr dur of
           Just d -> Right $ Wait d
           Nothing -> Left $ "Invalid duration: " ++ dur
       ("repeat":n:rest) ->
         case reads n of
           [(count, "")] ->
             case parseCommand (unwords rest) of
               Right cmd -> Right $ Repeat count cmd
               Left err -> Left err
           _ -> Left $ "Invalid repeat count: " ++ n
       [] -> Left "Empty command"
       other -> Left $ "Unknown command: " ++ unwords other

-- Parse harmonic from tokens like "harmonic(2,1)"
parseHarmonic :: [String] -> Maybe (Int, Int)
parseHarmonic tokens = case tokens of
  (h:_) | take 9 h == "harmonic(" ->
    let inner = takeWhile (/= ')') (drop 9 h)
        parts = break (== ',') inner
    in case (reads (fst parts), reads (drop 1 (snd parts))) of
         ([(l, "")], [(m, "")]) -> Just (l, m)
         _ -> Nothing
  _ -> Nothing

-- Parse duration from tokens
parseDuration :: [String] -> Maybe Duration
parseDuration tokens = case tokens of
  (d:_) | take 9 d == "duration(" ->
    parseDurationStr (takeWhile (/= ')') (drop 9 d))
  _ -> Nothing

-- Parse duration string like "30s" or "5phi"
parseDurationStr :: String -> Maybe Duration
parseDurationStr s
  | last s == 's' = case reads (init s) of
      [(v, "")] -> Just (Seconds v)
      _ -> Nothing
  | take 3 (reverse s) == "ihp" = case reads (take (length s - 3) s) of
      [(n, "")] -> Just (PhiCycles n)
      _ -> Nothing
  | last s == 'b' = case reads (init s) of
      [(n, "")] -> Just (Beats n)
      _ -> Nothing
  | otherwise = case reads s of
      [(v, "")] -> Just (Seconds v)
      _ -> Nothing

-- | Parse expression
parseExpr :: String -> Either String ScriptumExpr
parseExpr input =
  let trimmed = dropWhile (== ' ') input
  in case words trimmed of
       ("coherence":">":val:_) ->
         case reads val of
           [(v, "")] -> Right $ CoherenceAbove v
           _ -> Left "Invalid coherence value"
       ("coherence":"<":val:_) ->
         case reads val of
           [(v, "")] -> Right $ CoherenceBelow v
           _ -> Left "Invalid coherence value"
       ["inverted"] -> Right $ InversionState True
       ["not", "inverted"] -> Right $ InversionState False
       ["always"] -> Right Always
       ["never"] -> Right Never
       _ -> Left $ "Unknown expression: " ++ input

-- =============================================================================
-- Interpretation
-- =============================================================================

-- | Interpreter state
data ScriptumState = ScriptumState
  { ssCoherence   :: !Double
  , ssPotential   :: !Double
  , ssInversion   :: !Bool
  , ssFrequency   :: !Double
  , ssGateLevel   :: !Int
  , ssFragments   :: ![String]      -- ^ Summoned fragments
  , ssElapsed     :: !Double        -- ^ Total elapsed time
  } deriving (Eq, Show)

-- | Interpretation result
data ScriptumResult = ScriptumResult
  { srState       :: !ScriptumState
  , srMods        :: ![ChamberMod]
  , srMessages    :: ![String]
  , srSuccess     :: !Bool
  } deriving (Eq, Show)

-- | Interpret AST
interpret :: ScriptumAST -> ScriptumState -> ScriptumResult
interpret ast state =
  let initialResult = ScriptumResult state [] [] True
  in foldl interpretStep initialResult (astCommands ast)

-- Interpret single step
interpretStep :: ScriptumResult -> ScriptumCommand -> ScriptumResult
interpretStep result cmd =
  if not (srSuccess result)
  then result
  else interpretCommand cmd (srState result) result

-- | Interpret single command
interpretCommand :: ScriptumCommand -> ScriptumState -> ScriptumResult -> ScriptumResult
interpretCommand cmd state result = case cmd of
  Summon fragId harmonic ->
    let newState = state { ssFragments = fragId : ssFragments state }
        mod' = case harmonic of
          Just (l, m) -> ModHarmonic l m
          Nothing -> ModFragment fragId
        msg = "Summoned: " ++ fragId
    in result { srState = newState, srMods = mod' : srMods result, srMessages = msg : srMessages result }

  Amplify amount duration ->
    let newPotential = min 1.0 (ssPotential state + amount * 0.2)
        waitTime = durationToSeconds duration
        newState = state { ssPotential = newPotential, ssElapsed = ssElapsed state + waitTime }
        msg = "Amplified to " ++ show newPotential
    in result { srState = newState, srMods = ModPotential newPotential : srMods result, srMessages = msg : srMessages result }

  Invert ->
    let newState = state { ssInversion = not (ssInversion state) }
        msg = "Inversion: " ++ show (ssInversion newState)
    in result { srState = newState, srMods = ModInversion (ssInversion newState) : srMods result, srMessages = msg : srMessages result }

  Attune freq ->
    let newState = state { ssFrequency = freq }
        msg = "Attuned to " ++ show freq ++ "Hz"
    in result { srState = newState, srMods = ModFrequency freq : srMods result, srMessages = msg : srMessages result }

  Ground ->
    let newState = state { ssPotential = 0.5, ssInversion = False, ssFrequency = 7.83 }
        msg = "Grounded"
    in result { srState = newState, srMods = ModGround : srMods result, srMessages = msg : srMessages result }

  Gate level ->
    let newState = state { ssGateLevel = level }
        msg = "Gate level: " ++ show level
    in result { srState = newState, srMods = ModGate level : srMods result, srMessages = msg : srMessages result }

  Wait duration ->
    let waitTime = durationToSeconds (Just duration)
        newState = state { ssElapsed = ssElapsed state + waitTime }
        msg = "Waited " ++ show waitTime ++ "s"
    in result { srState = newState, srMessages = msg : srMessages result }

  Sequence cmds ->
    foldl interpretStep result cmds

  Parallel cmds ->
    -- Parallel just runs all (simplified - real impl would be concurrent)
    foldl interpretStep result cmds

  Repeat n cmd' ->
    foldl interpretStep result (replicate n cmd')

  Conditional expr thenCmd elseCmd ->
    let condResult = evalExpr expr state
    in if condResult
       then interpretCommand thenCmd state result
       else case elseCmd of
              Just elseBranch -> interpretCommand elseBranch state result
              Nothing -> result

-- Evaluate expression
evalExpr :: ScriptumExpr -> ScriptumState -> Bool
evalExpr expr state = case expr of
  CoherenceAbove v -> ssCoherence state > v
  CoherenceBelow v -> ssCoherence state < v
  InversionState b -> ssInversion state == b
  GateLevel l -> ssGateLevel state >= l
  Always -> True
  Never -> False
  And e1 e2 -> evalExpr e1 state && evalExpr e2 state
  Or e1 e2 -> evalExpr e1 state || evalExpr e2 state
  Not e -> not (evalExpr e state)

-- Convert duration to seconds
durationToSeconds :: Maybe Duration -> Double
durationToSeconds md = case md of
  Nothing -> 0.0
  Just (Seconds s) -> s
  Just (Beats b) -> fromIntegral b * 0.5  -- Assume 120 BPM
  Just (PhiCycles n) -> phi ** fromIntegral n

-- | Run script from text
runScript :: String -> ScriptumState -> Either String ScriptumResult
runScript text state = case parseScriptum text of
  Left err -> Left err
  Right ast -> Right $ interpret ast state

-- =============================================================================
-- Chamber Modification
-- =============================================================================

-- | Chamber modification
data ChamberMod
  = ModPotential !Double
  | ModInversion !Bool
  | ModFrequency !Double
  | ModHarmonic !Int !Int
  | ModFragment !String
  | ModGate !Int
  | ModGround
  deriving (Eq, Show)

-- | Apply modification (returns new potential)
applyMod :: ChamberMod -> Double -> Double
applyMod mod' current = case mod' of
  ModPotential p -> p
  ModGround -> 0.5
  _ -> current

-- | Get potential from mod
modToPotential :: ChamberMod -> Maybe Double
modToPotential mod' = case mod' of
  ModPotential p -> Just p
  ModGround -> Just 0.5
  _ -> Nothing

-- | Get inversion from mod
modToInversion :: ChamberMod -> Maybe Bool
modToInversion mod' = case mod' of
  ModInversion b -> Just b
  ModGround -> Just False
  _ -> Nothing

-- =============================================================================
-- Built-in Commands
-- =============================================================================

-- | Summon fragment
summonFragment :: String -> Maybe (Int, Int) -> ScriptumCommand
summonFragment = Summon

-- | Amplify field
amplifyField :: Double -> Maybe Duration -> ScriptumCommand
amplifyField = Amplify

-- | Invert field
invertField :: ScriptumCommand
invertField = Invert

-- | Attune to frequency
attuneFrequency :: Double -> ScriptumCommand
attuneFrequency = Attune

-- | Ground field
groundField :: ScriptumCommand
groundField = Ground

-- | Set gate level
setGate :: Int -> ScriptumCommand
setGate = Gate

-- | Wait for duration
waitDuration :: Duration -> ScriptumCommand
waitDuration = Wait

-- =============================================================================
-- Script Building (EDSL)
-- =============================================================================

-- | Create script from commands
script :: [ScriptumCommand] -> ScriptumAST
script cmds = ScriptumAST cmds []

-- | Sequence operator
(&>) :: ScriptumCommand -> ScriptumCommand -> ScriptumCommand
cmd1 &> cmd2 = Sequence [cmd1, cmd2]

infixr 5 &>

-- | Shorthand for Summon
summon :: String -> ScriptumCommand
summon fragId = Summon fragId Nothing

-- | Shorthand for Amplify
amplify :: Double -> ScriptumCommand
amplify v = Amplify v Nothing

-- | Shorthand for Invert
invert :: ScriptumCommand
invert = Invert

-- | Shorthand for Attune
attune :: Double -> ScriptumCommand
attune = Attune

-- | Shorthand for Ground
ground :: ScriptumCommand
ground = Ground

-- | Shorthand for Gate
gate :: Int -> ScriptumCommand
gate = Gate

-- | Shorthand for Wait
wait :: Double -> ScriptumCommand
wait s = Wait (Seconds s)
