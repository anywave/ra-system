{-|
Module      : Ra
Description : Ra System mathematical framework
Copyright   : (c) Anywave, 2025
License     : Apache-2.0
Maintainer  : alex@anywavecreations.com
Stability   : experimental

Re-exports all Ra System modules for convenient access.

@
import Ra

-- Use typed constants
let a = ankh          -- 5.08938
let r = repitan 9     -- Just (Repitan 9) = 1/3
let rac = RAC1        -- Highest access level

-- Convert between Omega formats
let green = 1.62
let omegaMajor = greenToOmegaMajor green  -- 1.62 / 1.005662978
@
-}
module Ra
    ( -- * Re-exports
      module Ra.Constants
    , module Ra.Repitans
    , module Ra.Rac
    , module Ra.Omega
    , module Ra.Spherical
    , module Ra.Gates
    ) where

import Ra.Constants
import Ra.Repitans
import Ra.Rac
import Ra.Omega
import Ra.Spherical
import Ra.Gates
