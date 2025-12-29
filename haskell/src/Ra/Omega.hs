{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-|
Module      : Ra.Omega
Description : OmegaFormat enum + conversion functions
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Five-level Omega format system for frequency/precision tiers.

Hierarchy: Red > Omega Major > Green > Omega Minor > Blue

Conversions use the Omega Ratio (Q-Ratio): Ω = 1.005662978

@
Green → Omega_Major:  x / Ω
Green → Omega_Minor:  x × Ω
@
-}
module Ra.Omega
    ( -- * OmegaFormat Type
      OmegaFormat(..)
      -- * Omega Ratio
    , omega
    , omegaReciprocal
      -- * Format Conversions
    , convertOmega
    , greenToOmegaMajor
    , omegaMajorToGreen
    , greenToOmegaMinor
    , omegaMinorToGreen
    , omegaMajorToRed
    , redToOmegaMajor
    , omegaMajorToBlue
    , blueToOmegaMajor
    , redToGreen
    , greenToRed
    , redToBlue
    , blueToRed
      -- * Roundtrip Verification
    , verifyOmegaRoundtrip
    , roundtripTolerance
    ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)

-- | The five Omega format levels (coherence depth tiers)
-- Index 0 = Red (highest precision), Index 4 = Blue
data OmegaFormat = Red | OmegaMajor | Green | OmegaMinor | Blue
    deriving (Eq, Ord, Enum, Bounded, Show, Read, Generic, NFData)

-- | Omega Ratio (Q-Ratio): 1.005662978
omega :: Double
omega = 1.005662978

-- | Reciprocal of Omega Ratio
omegaReciprocal :: Double
omegaReciprocal = 1.0 / omega  -- 0.994368911

-- | Convert a value between two Omega formats
convertOmega :: OmegaFormat -> OmegaFormat -> Double -> Double
convertOmega from to x = x * conversionFactor from to

-- | Get conversion factor between formats
conversionFactor :: OmegaFormat -> OmegaFormat -> Double
conversionFactor from to = case (from, to) of
    -- Identity
    (f, t) | f == t -> 1.0

    -- From Green (reference)
    (Green, OmegaMajor) -> 0.994368911     -- 1/Ω
    (Green, OmegaMinor) -> 1.005662978     -- Ω
    (Green, Red)        -> 0.999648641
    (Green, Blue)       -> 1.000351482

    -- To Green (reference)
    (OmegaMajor, Green) -> 1.005662978     -- Ω
    (OmegaMinor, Green) -> 0.994368911     -- 1/Ω
    (Red, Green)        -> 1.000351482
    (Blue, Green)       -> 0.999648641

    -- Omega Major conversions
    (OmegaMajor, Red)        -> 1.005309630
    (Red, OmegaMajor)        -> 0.994718414
    (OmegaMajor, Blue)       -> 1.006016451
    (Blue, OmegaMajor)       -> 0.994019530
    (OmegaMajor, OmegaMinor) -> 1.011358026
    (OmegaMinor, OmegaMajor) -> 0.988769530

    -- Red/Blue conversions
    (Red, Blue)  -> 1.000703088
    (Blue, Red)  -> 0.999297406

    -- Red/Omega Minor
    (Red, OmegaMinor)  -> 1.006016451
    (OmegaMinor, Red)  -> 0.994019530

    -- Blue/Omega Minor
    (Blue, OmegaMinor)  -> 1.005309630
    (OmegaMinor, Blue)  -> 0.994718414

    -- Catch-all (should never be reached if all pairs are covered)
    _ -> error "conversionFactor: unexpected format pair"

-- | Green to Omega Major: x / Ω
greenToOmegaMajor :: Double -> Double
greenToOmegaMajor = convertOmega Green OmegaMajor

-- | Omega Major to Green: x × Ω
omegaMajorToGreen :: Double -> Double
omegaMajorToGreen = convertOmega OmegaMajor Green

-- | Green to Omega Minor: x × Ω
greenToOmegaMinor :: Double -> Double
greenToOmegaMinor = convertOmega Green OmegaMinor

-- | Omega Minor to Green: x / Ω
omegaMinorToGreen :: Double -> Double
omegaMinorToGreen = convertOmega OmegaMinor Green

-- | Omega Major to Red
omegaMajorToRed :: Double -> Double
omegaMajorToRed = convertOmega OmegaMajor Red

-- | Red to Omega Major
redToOmegaMajor :: Double -> Double
redToOmegaMajor = convertOmega Red OmegaMajor

-- | Omega Major to Blue
omegaMajorToBlue :: Double -> Double
omegaMajorToBlue = convertOmega OmegaMajor Blue

-- | Blue to Omega Major
blueToOmegaMajor :: Double -> Double
blueToOmegaMajor = convertOmega Blue OmegaMajor

-- | Red to Green
redToGreen :: Double -> Double
redToGreen = convertOmega Red Green

-- | Green to Red
greenToRed :: Double -> Double
greenToRed = convertOmega Green Red

-- | Red to Blue
redToBlue :: Double -> Double
redToBlue = convertOmega Red Blue

-- | Blue to Red
blueToRed :: Double -> Double
blueToRed = convertOmega Blue Red

-- | Tolerance for roundtrip conversion verification
roundtripTolerance :: Double
roundtripTolerance = 1e-10

-- | Verify Invariant C1: roundtrip conversions preserve value
-- convertOmega f t . convertOmega t f ≈ id
verifyOmegaRoundtrip :: OmegaFormat -> OmegaFormat -> Double -> Bool
verifyOmegaRoundtrip from to x =
    let roundtrip = convertOmega to from (convertOmega from to x)
    in abs (roundtrip - x) < roundtripTolerance
