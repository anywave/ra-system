{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-|
Module      : Ra.Rac
Description : RacLevel ADT (RAC1..RAC6) with Enum, Bounded
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Resonant Access Constants (RACs) represent access sensitivity levels.
RAC1 is the highest (least restrictive), RAC6 is the lowest (most restrictive).

Invariant: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0
-}
module Ra.Rac
    ( -- * RacLevel Type
      RacLevel(..)
    , RacValue(..)
      -- * RAC Values
    , racValue
    , racValueMeters
    , racValueNormalized
      -- * Smart Constructors
    , mkRacValue
    , deriveRac1
      -- * All RACs
    , allRacLevels
    , racByLevel
      -- * Derived Values
    , pyramidDivision
      -- * Invariant Checks
    , verifyRacOrdering
    , isValidRacValue
    ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)
import Ra.Constants (Ankh(..), ankh)

-- | The six Resonant Access Constant levels
-- RAC1 is highest access (least restricted), RAC6 is lowest (most restricted)
data RacLevel = RAC1 | RAC2 | RAC3 | RAC4 | RAC5 | RAC6
    deriving (Eq, Ord, Enum, Bounded, Show, Read, Generic, NFData)

-- | RAC value in Red Rams (must be 0 < x < 1)
newtype RacValue = RacValue { unRacValue :: Double }
    deriving (Eq, Ord, Show)

-- | Get the RAC value in Red Rams for a given level
racValue :: RacLevel -> RacValue
racValue RAC1 = RacValue 0.6361725      -- Ankh / 8
racValue RAC2 = RacValue 0.628318519    -- 2π/10 approximation
racValue RAC3 = RacValue 0.57255525     -- φ × Hunab × 1/3
racValue RAC4 = RacValue 0.523598765    -- π/6 approximation
racValue RAC5 = RacValue 0.4580442      -- Ankh × 9 / 100
racValue RAC6 = RacValue 0.3998594565   -- RAC lattice terminus

-- | Get the RAC value in meters for a given level
racValueMeters :: RacLevel -> Double
racValueMeters RAC1 = 0.639591666
racValueMeters RAC2 = 0.631695473
racValueMeters RAC3 = 0.5756325
racValueMeters RAC4 = 0.526412894
racValueMeters RAC5 = 0.460506
racValueMeters RAC6 = 0.4020085371

-- | Get the RAC value normalized to RAC1 (for threshold calculations)
-- RAC1 normalized = 1.0
racValueNormalized :: RacLevel -> Double
racValueNormalized level =
    unRacValue (racValue level) / unRacValue (racValue RAC1)

-- | All RAC levels in order (RAC1 to RAC6)
allRacLevels :: [RacLevel]
allRacLevels = [minBound .. maxBound]

-- | Get RacLevel by numeric level (1-6), returns Nothing if invalid
racByLevel :: Int -> Maybe RacLevel
racByLevel 1 = Just RAC1
racByLevel 2 = Just RAC2
racByLevel 3 = Just RAC3
racByLevel 4 = Just RAC4
racByLevel 5 = Just RAC5
racByLevel 6 = Just RAC6
racByLevel _ = Nothing

-- | Pyramid base divided by each RAC yields key numbers
pyramidDivision :: RacLevel -> Double
pyramidDivision RAC1 = 360.0      -- Circle degrees
pyramidDivision RAC2 = 364.5      -- Balmer constant
pyramidDivision RAC3 = 400.0
pyramidDivision RAC4 = 437.4      -- 27 × φ_green
pyramidDivision RAC5 = 500.0
pyramidDivision RAC6 = 572.756493 -- 1.125 × Green Ankh

-- | Verify Invariant O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0
verifyRacOrdering :: Bool
verifyRacOrdering =
    let values = map (unRacValue . racValue) allRacLevels
        pairs = zip values (tail values)
    in all (\(a, b) -> a > b) pairs && all (> 0) values

-- | Check if a value is valid for a RAC (0 < x < 1)
isValidRacValue :: Double -> Bool
isValidRacValue x = x > 0 && x < 1

-- | Smart constructor for RacValue with validation
mkRacValue :: Double -> Maybe RacValue
mkRacValue x
    | isValidRacValue x = Just (RacValue x)
    | otherwise = Nothing

-- | Derive RAC1 from Ankh (Invariant I2: RAC1 = Ankh / 8)
deriveRac1 :: Ankh -> RacValue
deriveRac1 (Ankh a) = RacValue (a / 8)
