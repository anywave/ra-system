{-|
Module      : Ra.AetherSensor
Description : Psychic/intent input interface for scalar resonance
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Implements a speculative input modality using psychotronic radionic patterns
or intention signatures as scalar resonance seeds. Based on radionics rate
principles and dowsing research.

== Aether Sensor Theory

=== Intent as Input

Intentions can be encoded as:

* Numeric rates (traditional radionics)
* Symbolic patterns (sacred geometry)
* Geometric forms (platonic solids)
* Semantic vectors (natural language)

=== Resonance Seeding

Aether input seeds the scalar field by:

* Modifying base potential
* Shifting harmonic alignment
* Adjusting coherence phase
* Creating attractor patterns

=== Safety Considerations

All aether input is validated:

* Rate range checking
* Chaos detection
* Consent verification
* Coherence gating
-}
module Ra.AetherSensor
  ( -- * Core Types
    AetherInput(..)
  , InputForm(..)
  , IntentSignature(..)
  , mkAetherInput

    -- * Input Validation
  , ValidationResult(..)
  , validateInput
  , isValid
  , validationMessage

    -- * Consent Binding
  , ConsentBind(..)
  , bindToConsent
  , checkConsentBind
  , unbind

    -- * Chamber Binding
  , ChamberBind(..)
  , bindToChamber
  , chamberSignature

    -- * Scalar Generation
  , ScalarSeed(..)
  , generateSeed
  , seedToComponent
  , seedStrength

    -- * Rate Processing
  , RadionicsRate(..)
  , parseRate
  , rateToHarmonic
  , rateStrength

    -- * Safety Layer
  , SafetyCheck(..)
  , checkSafety
  , rejectChaotic
  , sanitizeInput

    -- * Intent Resolution
  , ResolvedIntent(..)
  , resolveIntent
  , intentConfidence
  , intentHarmonic
  ) where

import Ra.Constants.Extended
  ( phi )

-- =============================================================================
-- Core Types
-- =============================================================================

-- | Aether input from intention
data AetherInput = AetherInput
  { aiForm        :: !InputForm
  , aiSignature   :: !IntentSignature
  , aiStrength    :: !Double          -- ^ Input strength [0,1]
  , aiTimestamp   :: !Double          -- ^ When received
  , aiUserId      :: !String          -- ^ Source user
  , aiValidated   :: !Bool            -- ^ Has been validated
  } deriving (Eq, Show)

-- | Form of input
data InputForm
  = NumericRate !String       -- ^ Traditional rate (e.g., "568.12")
  | SymbolicPattern !String   -- ^ Pattern name (e.g., "flower_of_life")
  | GeometricForm !String     -- ^ Platonic solid (e.g., "tetrahedron")
  | SemanticVector ![String]  -- ^ Natural language words
  | CompositeForm ![(InputForm, Double)]  -- ^ Weighted combination
  deriving (Eq, Show)

-- | Intent signature
data IntentSignature = IntentSignature
  { isCategory    :: !String        -- ^ Intent category
  , isPolarity    :: !Double        -- ^ Positive/negative [-1, 1]
  , isComplexity  :: !Double        -- ^ Signal complexity [0, 1]
  , isCoherence   :: !Double        -- ^ Internal coherence [0, 1]
  } deriving (Eq, Show)

-- | Create aether input
mkAetherInput :: InputForm -> String -> Double -> AetherInput
mkAetherInput form userId timestamp =
  let sig = signatureFromForm form
  in AetherInput
      { aiForm = form
      , aiSignature = sig
      , aiStrength = computeStrength form
      , aiTimestamp = timestamp
      , aiUserId = userId
      , aiValidated = False
      }

-- Derive signature from form
signatureFromForm :: InputForm -> IntentSignature
signatureFromForm form = case form of
  NumericRate rate -> IntentSignature
    { isCategory = "radionic"
    , isPolarity = if head rate == '-' then (-1.0) else 1.0
    , isComplexity = fromIntegral (length rate) / 10.0
    , isCoherence = 0.8
    }
  SymbolicPattern name -> IntentSignature
    { isCategory = "symbolic"
    , isPolarity = 1.0
    , isComplexity = 0.6
    , isCoherence = if name `elem` sacredPatterns then 0.9 else 0.5
    }
  GeometricForm _ -> IntentSignature
    { isCategory = "geometric"
    , isPolarity = 1.0
    , isComplexity = 0.5
    , isCoherence = 0.85
    }
  SemanticVector wordList -> IntentSignature
    { isCategory = "semantic"
    , isPolarity = semanticPolarity wordList
    , isComplexity = fromIntegral (length wordList) / 10.0
    , isCoherence = 0.6
    }
  CompositeForm components -> IntentSignature
    { isCategory = "composite"
    , isPolarity = sum [w * isPolarity (signatureFromForm f) | (f, w) <- components] /
                   sum [w | (_, w) <- components]
    , isComplexity = 0.8
    , isCoherence = 0.7
    }

-- Compute input strength
computeStrength :: InputForm -> Double
computeStrength form = case form of
  NumericRate _ -> 0.7
  SymbolicPattern _ -> 0.8
  GeometricForm _ -> 0.75
  SemanticVector _ -> 0.6
  CompositeForm _ -> 0.85

-- Sacred patterns
sacredPatterns :: [String]
sacredPatterns = ["flower_of_life", "metatron", "sri_yantra", "vesica_piscis", "torus"]

-- Semantic polarity
semanticPolarity :: [String] -> Double
semanticPolarity wordList =
  let positive = ["love", "heal", "peace", "joy", "light", "grow", "create"]
      negative = ["fear", "hate", "pain", "dark", "destroy", "block"]
      posCount = length $ filter (`elem` positive) wordList
      negCount = length $ filter (`elem` negative) wordList
      total = posCount + negCount
  in if total == 0 then 0.0
     else fromIntegral (posCount - negCount) / fromIntegral total

-- =============================================================================
-- Input Validation
-- =============================================================================

-- | Validation result
data ValidationResult = ValidationResult
  { vrValid     :: !Bool
  , vrMessage   :: !String
  , vrCorrected :: !(Maybe AetherInput)
  , vrWarnings  :: ![String]
  } deriving (Eq, Show)

-- | Validate aether input
validateInput :: AetherInput -> ValidationResult
validateInput input =
  let sig = aiSignature input
      form = aiForm input

      -- Check coherence threshold
      coherenceOk = isCoherence sig >= 0.3

      -- Check complexity bounds
      complexityOk = isComplexity sig <= 1.0

      -- Check for chaotic patterns
      chaoticCheck = not (isChaotic form)

      -- Check strength bounds
      strengthOk = aiStrength input >= 0.0 && aiStrength input <= 1.0

      allChecks = coherenceOk && complexityOk && chaoticCheck && strengthOk

      warnings = concat
        [ ["Low coherence" | not coherenceOk]
        , ["High complexity" | isComplexity sig > 0.8]
        , ["Chaotic pattern detected" | not chaoticCheck]
        ]

      corrected = if allChecks
                  then Nothing
                  else Just $ sanitizeInput input
  in ValidationResult
      { vrValid = allChecks
      , vrMessage = if allChecks then "Valid" else "Validation failed"
      , vrCorrected = corrected
      , vrWarnings = warnings
      }

-- | Check if valid
isValid :: ValidationResult -> Bool
isValid = vrValid

-- | Get validation message
validationMessage :: ValidationResult -> String
validationMessage = vrMessage

-- Check for chaotic patterns
isChaotic :: InputForm -> Bool
isChaotic form = case form of
  NumericRate rate -> length rate > 20 || any (`elem` ("!@#$%^&*" :: String)) rate
  SymbolicPattern name -> "chaos" `elem` words name || "random" `elem` words name
  SemanticVector words' -> length words' > 50 || "chaos" `elem` words'
  CompositeForm components -> length components > 10
  _ -> False

-- =============================================================================
-- Consent Binding
-- =============================================================================

-- | Consent binding
data ConsentBind = ConsentBind
  { cbInput     :: !AetherInput
  , cbUserId    :: !String
  , cbLevel     :: !Int             -- ^ Consent level (1-6)
  , cbBound     :: !Bool
  , cbTimestamp :: !Double
  } deriving (Eq, Show)

-- | Bind input to user consent
bindToConsent :: AetherInput -> String -> Int -> ConsentBind
bindToConsent input userId level = ConsentBind
  { cbInput = input
  , cbUserId = userId
  , cbLevel = max 1 (min 6 level)
  , cbBound = level >= 3  -- Minimum level 3 for binding
  , cbTimestamp = aiTimestamp input
  }

-- | Check consent binding
checkConsentBind :: ConsentBind -> Bool
checkConsentBind = cbBound

-- | Unbind from consent
unbind :: ConsentBind -> ConsentBind
unbind cb = cb { cbBound = False }

-- =============================================================================
-- Chamber Binding
-- =============================================================================

-- | Chamber binding
data ChamberBind = ChamberBind
  { chbInput       :: !AetherInput
  , chbChamberId   :: !String
  , chbSignature   :: !String       -- ^ Chamber signature
  , chbCompatible  :: !Bool
  } deriving (Eq, Show)

-- | Bind input to chamber
bindToChamber :: AetherInput -> String -> String -> ChamberBind
bindToChamber input chamberId signature =
  let compatible = checkCompatibility input signature
  in ChamberBind
      { chbInput = input
      , chbChamberId = chamberId
      , chbSignature = signature
      , chbCompatible = compatible
      }

-- | Get chamber signature
chamberSignature :: ChamberBind -> String
chamberSignature = chbSignature

-- Check compatibility
checkCompatibility :: AetherInput -> String -> Bool
checkCompatibility input signature =
  let sig = aiSignature input
  in isCoherence sig > 0.5 && not (null signature)

-- =============================================================================
-- Scalar Generation
-- =============================================================================

-- | Scalar field seed
data ScalarSeed = ScalarSeed
  { ssPotential   :: !Double        -- ^ Potential contribution
  , ssHarmonic    :: !(Int, Int)    -- ^ Harmonic mode (l, m)
  , ssPhase       :: !Double        -- ^ Phase offset
  , ssAmplitude   :: !Double        -- ^ Amplitude
  , ssCoherence   :: !Double        -- ^ Coherence factor
  } deriving (Eq, Show)

-- | Generate scalar seed from input
generateSeed :: AetherInput -> ScalarSeed
generateSeed input =
  let sig = aiSignature input
      form = aiForm input

      -- Potential from strength and coherence
      potential = aiStrength input * isCoherence sig * phi

      -- Harmonic from form
      harmonic = harmonicFromForm form

      -- Phase from polarity
      phase = if isPolarity sig >= 0 then 0 else pi

      -- Amplitude from strength
      amplitude = aiStrength input * 0.8

      -- Coherence directly from signature
      coherence = isCoherence sig
  in ScalarSeed
      { ssPotential = potential
      , ssHarmonic = harmonic
      , ssPhase = phase
      , ssAmplitude = amplitude
      , ssCoherence = coherence
      }

-- Harmonic from form type
harmonicFromForm :: InputForm -> (Int, Int)
harmonicFromForm form = case form of
  NumericRate rate -> rateToHarmonic' rate
  SymbolicPattern name -> patternHarmonic name
  GeometricForm name -> geometryHarmonic name
  SemanticVector _ -> (1, 1)  -- Default
  CompositeForm _ -> (2, 0)   -- Default composite

-- | Convert seed to scalar component values
seedToComponent :: ScalarSeed -> (Double, Double, Double)
seedToComponent seed = (ssPotential seed, ssAmplitude seed, ssCoherence seed)

-- | Get seed strength
seedStrength :: ScalarSeed -> Double
seedStrength seed = ssPotential seed * ssAmplitude seed * ssCoherence seed

-- Pattern to harmonic
patternHarmonic :: String -> (Int, Int)
patternHarmonic name = case name of
  "flower_of_life" -> (6, 0)
  "metatron" -> (4, 4)
  "sri_yantra" -> (3, 3)
  "vesica_piscis" -> (2, 0)
  "torus" -> (1, 1)
  _ -> (0, 0)

-- Geometry to harmonic
geometryHarmonic :: String -> (Int, Int)
geometryHarmonic name = case name of
  "tetrahedron" -> (3, 0)
  "cube" -> (4, 0)
  "octahedron" -> (4, 4)
  "dodecahedron" -> (5, 5)
  "icosahedron" -> (5, 0)
  _ -> (0, 0)

-- =============================================================================
-- Rate Processing
-- =============================================================================

-- | Radionics rate
data RadionicsRate = RadionicsRate
  { rrValue     :: !Double
  , rrHarmonic  :: !(Int, Int)
  , rrStrength  :: !Double
  } deriving (Eq, Show)

-- | Parse radionics rate string
parseRate :: String -> Maybe RadionicsRate
parseRate s =
  let cleaned = filter (\c -> c `elem` ("0123456789.-" :: String)) s
  in case reads cleaned of
       [(val, "")] -> Just $ RadionicsRate
         { rrValue = val
         , rrHarmonic = rateToHarmonic' s
         , rrStrength = rateStrength' val
         }
       _ -> Nothing

-- | Convert rate to harmonic
rateToHarmonic :: RadionicsRate -> (Int, Int)
rateToHarmonic = rrHarmonic

-- Internal rate to harmonic
rateToHarmonic' :: String -> (Int, Int)
rateToHarmonic' s = case reads (filter (`elem` ("0123456789.-" :: String)) s) :: [(Double, String)] of
  [(val, "")] ->
    let l = floor (val / 100 :: Double) `mod` (10 :: Int)
        m = floor (val / 10 :: Double) `mod` (10 :: Int) - 5
    in (l, m)
  _ -> (0, 0)

-- | Get rate strength
rateStrength :: RadionicsRate -> Double
rateStrength = rrStrength

-- Internal strength calculation
rateStrength' :: Double -> Double
rateStrength' val = clamp01 (val / 1000 * phi)

-- =============================================================================
-- Safety Layer
-- =============================================================================

-- | Safety check result
data SafetyCheck = SafetyCheck
  { scPassed     :: !Bool
  , scReason     :: !String
  , scSeverity   :: !Int            -- ^ 0=none, 1=warning, 2=block
  } deriving (Eq, Show)

-- | Check input safety
checkSafety :: AetherInput -> SafetyCheck
checkSafety input =
  let sig = aiSignature input
      form = aiForm input

      -- Coherence check
      coherenceIssue = isCoherence sig < 0.2

      -- Polarity check (extreme negative)
      polarityIssue = isPolarity sig < -0.8

      -- Chaos check
      chaosIssue = isChaotic form

      -- Complexity check
      complexityIssue = isComplexity sig > 0.95

      anyIssue = coherenceIssue || polarityIssue || chaosIssue || complexityIssue
      severity = if chaosIssue || polarityIssue then 2
                 else if coherenceIssue || complexityIssue then 1
                 else 0
      reason = if chaosIssue then "Chaotic pattern"
               else if polarityIssue then "Extreme negative polarity"
               else if coherenceIssue then "Low coherence"
               else if complexityIssue then "Excessive complexity"
               else "Passed"
  in SafetyCheck
      { scPassed = not anyIssue
      , scReason = reason
      , scSeverity = severity
      }

-- | Reject chaotic inputs
rejectChaotic :: AetherInput -> Either String AetherInput
rejectChaotic input =
  if isChaotic (aiForm input)
  then Left "Chaotic pattern rejected"
  else Right input

-- | Sanitize input
sanitizeInput :: AetherInput -> AetherInput
sanitizeInput input =
  let sig = aiSignature input
      -- Boost low coherence
      newCoherence = max 0.3 (isCoherence sig)
      -- Clamp polarity
      newPolarity = max (-0.5) (min 0.5 (isPolarity sig))
      -- Reduce complexity
      newComplexity = min 0.8 (isComplexity sig)

      newSig = sig
        { isCoherence = newCoherence
        , isPolarity = newPolarity
        , isComplexity = newComplexity
        }
  in input
      { aiSignature = newSig
      , aiValidated = True
      }

-- =============================================================================
-- Intent Resolution
-- =============================================================================

-- | Resolved intent
data ResolvedIntent = ResolvedIntent
  { riCategory    :: !String
  , riHarmonic    :: !(Int, Int)
  , riConfidence  :: !Double
  , riAction      :: !String
  } deriving (Eq, Show)

-- | Resolve intent from input
resolveIntent :: AetherInput -> ResolvedIntent
resolveIntent input =
  let sig = aiSignature input
      seed = generateSeed input

      category = isCategory sig
      harmonic = ssHarmonic seed

      -- Confidence from coherence and validation
      confidence = isCoherence sig * (if aiValidated input then 1.0 else 0.7)

      -- Action from category
      action = case category of
        "radionic" -> "modulate_field"
        "symbolic" -> "invoke_pattern"
        "geometric" -> "structure_field"
        "semantic" -> "direct_intent"
        "composite" -> "complex_action"
        _ -> "unknown"
  in ResolvedIntent
      { riCategory = category
      , riHarmonic = harmonic
      , riConfidence = confidence
      , riAction = action
      }

-- | Get intent confidence
intentConfidence :: ResolvedIntent -> Double
intentConfidence = riConfidence

-- | Get intent harmonic
intentHarmonic :: ResolvedIntent -> (Int, Int)
intentHarmonic = riHarmonic

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- | Clamp value to [0, 1]
clamp01 :: Double -> Double
clamp01 x = max 0.0 (min 1.0 x)
