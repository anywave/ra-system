{-|
Module      : Ra.Constants
Description : Typed Ra System constants with newtype wrappers
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Fundamental constants from "The Rods of Amon Ra" by Wesley H. Bateman.
All values preserved exactly as stated in source material.
-}
module Ra.Constants
    ( -- * Primary Constants
      Ankh(..)
    , ankh
    , RedPi(..)
    , redPi
    , GreenPi(..)
    , greenPi
    , BluePi(..)
    , bluePi
    , GreenPhi(..)
    , greenPhi
    , RedPhi(..)
    , redPhi
    , Hunab(..)
    , hunab
    , HBar(..)
    , hBar
    , OmegaRatio(..)
    , omegaRatio
    , FineStructure(..)
    , fineStructure
    , Balmer(..)
    , balmer
    , Rydberg(..)
    , rydberg
    , SpeedOfLight(..)
    , speedOfLight
      -- * Derived Constants
    , divisor81
    , key2
      -- * Invariant Checks
    , verifyAnkhComposition
    , verifyRac1Derivation
    ) where

-- | Master harmonic constant - "Number for Life"
-- Ankh = Red Pi × Green Phi = 3.141592592 × 1.62
newtype Ankh = Ankh { unAnkh :: Double }
    deriving (Eq, Ord, Show)

-- | The canonical Ankh value
ankh :: Ankh
ankh = Ankh 5.08938

-- | Red Trac Pi - Primary Pi value in Ra System
-- Derived from 10.602875 / 13.5 × 4
newtype RedPi = RedPi { unRedPi :: Double }
    deriving (Eq, Ord, Show)

redPi :: RedPi
redPi = RedPi 3.141592592

-- | Green Trac Pi - Secondary Pi value
-- √9.876543210
newtype GreenPi = GreenPi { unGreenPi :: Double }
    deriving (Eq, Ord, Show)

greenPi :: GreenPi
greenPi = GreenPi 3.142696806

-- | Blue Trac Pi - Tertiary Pi value
-- 1/0.318086250
newtype BluePi = BluePi { unBluePi :: Double }
    deriving (Eq, Ord, Show)

bluePi :: BluePi
bluePi = BluePi 3.143801408

-- | Green Phi - Threshold harmonic for life forms
-- Golden ratio approximation (truncated)
newtype GreenPhi = GreenPhi { unGreenPhi :: Double }
    deriving (Eq, Ord, Show)

greenPhi :: GreenPhi
greenPhi = GreenPhi 1.62

-- | Red Phi - Full precision golden ratio in Ra System
newtype RedPhi = RedPhi { unRedPhi :: Double }
    deriving (Eq, Ord, Show)

redPhi :: RedPhi
redPhi = RedPhi 1.619430799

-- | Hunab - Harmonic scalar unit (Mayan God of Measure)
-- 10.602875 Hz / 10, same as Key 1
newtype Hunab = Hunab { unHunab :: Double }
    deriving (Eq, Ord, Show)

hunab :: Hunab
hunab = Hunab 1.0602875

-- | H-Bar (Omega Major) - Reduced Planck constant in Ra System
-- Hunab / Omega Ratio = 1.0602875 / 1.005662978
newtype HBar = HBar { unHBar :: Double }
    deriving (Eq, Ord, Show)

hBar :: HBar
hBar = HBar 1.0546875

-- | Omega Ratio - Q-Ratio for format conversions
-- Green ÷ Omega = Omega Major, Green × Omega = Omega Minor
newtype OmegaRatio = OmegaRatio { unOmegaRatio :: Double }
    deriving (Eq, Ord, Show)

omegaRatio :: OmegaRatio
omegaRatio = OmegaRatio 1.005662978

-- | Fine Structure Constant (Ra)
-- Repitan(1)² = (1/27)² = 0.037² = 0.0013717421
-- Scaled: 0.0001371742
newtype FineStructure = FineStructure { unFineStructure :: Double }
    deriving (Eq, Ord, Show)

fineStructure :: FineStructure
fineStructure = FineStructure 0.0013717421

-- | Balmer Constant (Green format)
-- 729/2 = 27 × 13.5 = 364.5
newtype Balmer = Balmer { unBalmer :: Double }
    deriving (Eq, Ord, Show)

balmer :: Balmer
balmer = Balmer 364.5

-- | Rydberg Constant (Omega Major format)
newtype Rydberg = Rydberg { unRydberg :: Double }
    deriving (Eq, Ord, Show)

rydberg :: Rydberg
rydberg = Rydberg 0.91125

-- | Speed of Light in Ra System (Omega Major)
-- 300,000 Omega Major kilorams per natural second
newtype SpeedOfLight = SpeedOfLight { unSpeedOfLight :: Double }
    deriving (Eq, Ord, Show)

speedOfLight :: SpeedOfLight
speedOfLight = SpeedOfLight 300000

-- | 81 Divisor - Elemental divisor (3⁴)
-- Relates Green Phi to basic elements
divisor81 :: Int
divisor81 = 81

-- | Key 2 (Red √2) - Diagonal calculations
-- Close to √2 (1.41421356)
key2 :: Double
key2 = 1.41371666

-- | Verify Invariant I1: Ankh = π_red × φ_green
-- Returns True if composition holds within tolerance
verifyAnkhComposition :: Double -> Bool
verifyAnkhComposition tolerance =
    abs (unAnkh ankh - unRedPi redPi * unGreenPhi greenPhi) < tolerance

-- | Verify Invariant I2: RAC₁ = Ankh / 8
-- Takes RAC1 value and returns True if derivation holds
verifyRac1Derivation :: Double -> Double -> Bool
verifyRac1Derivation rac1Value tolerance =
    abs (rac1Value - unAnkh ankh / 8) < tolerance
