{-|
Module      : Ra.Repitans
Description : Repitan type with smart constructor (1-27)
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

The 27 Repitans represent fractional parts of 27, used for semantic
sector mapping in the theta dimension.

@
Repitan(n) = n / 27,  for n ∈ {1, 2, ..., 27}
@
-}
module Ra.Repitans
    ( -- * Repitan Type
      Repitan
    , repitan
    , unsafeRepitan
    , repitanIndex
    , repitanValue
      -- * All Repitans
    , allRepitans
    , repitanByIndex
      -- * Special Repitans
    , firstRepitan
    , ninthRepitan
    , eighteenthRepitan
    , unityRepitan
      -- * Operations
    , repitanToTheta
    , thetaToRepitan
    , nextRepitan
    , prevRepitan
      -- * Predicates
    , isValidRepitanIndex
    ) where

import Data.Maybe (fromMaybe)

-- | Repitan represents a fractional part of 27.
-- The index is guaranteed to be in range [1, 27].
newtype Repitan = MkRepitan { repitanIndex :: Int }
    deriving (Eq, Ord)

instance Show Repitan where
    show r = "Repitan " ++ show (repitanIndex r) ++ " (" ++ show (repitanValue r) ++ ")"

-- | Smart constructor for Repitan. Returns Nothing if index out of range.
repitan :: Int -> Maybe Repitan
repitan n
    | isValidRepitanIndex n = Just (MkRepitan n)
    | otherwise = Nothing

-- | Unsafe constructor. Throws error if index invalid.
-- Only use when index is statically known to be valid.
unsafeRepitan :: Int -> Repitan
unsafeRepitan n
    | isValidRepitanIndex n = MkRepitan n
    | otherwise = error $ "unsafeRepitan: index " ++ show n ++ " out of range [1,27]"

-- | Check if an index is valid for a Repitan (1-27)
isValidRepitanIndex :: Int -> Bool
isValidRepitanIndex n = n >= 1 && n <= 27

-- | Get the value of a Repitan: n/27
repitanValue :: Repitan -> Double
repitanValue (MkRepitan n) = fromIntegral n / 27.0

-- | All 27 Repitans in order
allRepitans :: [Repitan]
allRepitans = map MkRepitan [1..27]

-- | Get Repitan by index (1-based), returns Nothing if invalid
repitanByIndex :: Int -> Maybe Repitan
repitanByIndex = repitan

-- | First Repitan: 1/27 = 0.037037...
-- This is the Fine Structure root (squared = 0.0013717421)
firstRepitan :: Repitan
firstRepitan = MkRepitan 1

-- | Ninth Repitan: 9/27 = 1/3 = 0.333...
ninthRepitan :: Repitan
ninthRepitan = MkRepitan 9

-- | Eighteenth Repitan: 18/27 = 2/3 = 0.666...
-- Related to Planck's Constant
eighteenthRepitan :: Repitan
eighteenthRepitan = MkRepitan 18

-- | Unity Repitan: 27/27 = 1.0
unityRepitan :: Repitan
unityRepitan = MkRepitan 27

-- | Convert Repitan to theta angle in degrees (0-360)
repitanToTheta :: Repitan -> Double
repitanToTheta r = repitanValue r * 360.0

-- | Convert theta angle (degrees) to nearest Repitan
thetaToRepitan :: Double -> Repitan
thetaToRepitan theta =
    let normalized = theta - fromIntegral (floor (theta / 360.0) :: Int) * 360.0
        index = max 1 $ min 27 $ ceiling (normalized * 27.0 / 360.0)
    in MkRepitan index

-- | Get next Repitan (wraps from 27 to 1)
nextRepitan :: Repitan -> Repitan
nextRepitan (MkRepitan 27) = MkRepitan 1
nextRepitan (MkRepitan n) = MkRepitan (n + 1)

-- | Get previous Repitan (wraps from 1 to 27)
prevRepitan :: Repitan -> Repitan
prevRepitan (MkRepitan 1) = MkRepitan 27
prevRepitan (MkRepitan n) = MkRepitan (n - 1)
