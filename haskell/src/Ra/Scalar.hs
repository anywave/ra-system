{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Ra.Scalar
  ( -- * Core Angular Types
    Radian(..), mkRadian
  , ThetaSector(..)
  , PhiLevel(..)
  , ShellIndex(..), mkShellIndex
  , Inversion(..)
  , TemporalWindow(..)
  , RepitanIndex(..), mkRepitanIndex, repitanToIndex

    -- * Harmonic + Radial Structures
  , SphericalHarmonic(..), mkSphericalHarmonic
  , RadialType(..)
  , RadialProfile(..)
  , WellDepth(..)
  , ScalarComponent(..)
  , ScalarField(..)
  , BowlField(..)
  , FluxCoherence(..)

    -- * Coordinates + Emergence
  , Coordinate(..)
  , EmergenceCondition(..)
  , EmergenceResult(..)

    -- * Computation interfaces
  , sphericalY
  , evalRadialProfile
  , wellDepth
  , fluxIntegral
  , goldenWindow
  ) where

import GHC.Generics (Generic)
import Control.DeepSeq (NFData)

-- Import from real Ra modules
import Ra.Constants (Ankh(..), ankh, GreenPhi(..), greenPhi)
import Ra.Repitans (Repitan, repitanIndex, repitanValue, repitan)
import Ra.Rac (RacLevel(..))
import Ra.Omega (OmegaFormat(..))

--------------------------------------------------------------------------------
-- Re-export RepitanIndex as local alias (for Scalar module use)
--------------------------------------------------------------------------------

-- | Repitan index (1-27) - alias for use in FluxCoherence
newtype RepitanIndex = RI { unRI :: Int }
  deriving (Eq, Ord, Show, Generic, NFData)

mkRepitanIndex :: Int -> Maybe RepitanIndex
mkRepitanIndex n
  | n >= 1 && n <= 27 = Just (RI n)
  | otherwise = Nothing

-- | Convert Repitan to RepitanIndex
repitanToIndex :: Repitan -> RepitanIndex
repitanToIndex = RI . repitanIndex

--------------------------------------------------------------------------------
-- Angular / Dimensional Types
--------------------------------------------------------------------------------

-- | Radian constrained to [0, 2π)
newtype Radian = Radian { unRadian :: Double }
  deriving (Eq, Ord, Show, Generic, NFData)

-- | Smart constructor: normalizes to [0, 2π)
mkRadian :: Double -> Radian
mkRadian x = Radian (x `mod'` (2 * pi))
  where
    mod' a b = a - b * fromIntegral (floor (a / b) :: Int)

-- | Theta sector mapped to 27 Repitans
newtype ThetaSector = TS { unTS :: RepitanIndex }
  deriving (Eq, Ord, Show, Generic, NFData)

-- | Phi level mapped to 6 RACs
newtype PhiLevel = PL { unPL :: RacLevel }
  deriving (Eq, Ord, Show, Generic, NFData)

-- | Radial shell index (positive integer)
newtype ShellIndex = SI { unSI :: Int }
  deriving (Eq, Ord, Show, Generic, NFData)

mkShellIndex :: Int -> Maybe ShellIndex
mkShellIndex n
  | n >= 0 = Just (SI n)
  | otherwise = Nothing

-- | Inversion flag for shadow fragments
data Inversion
  = Normal
  | Inverted
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | Golden-ratio temporal window for phase gating
-- Value represents φ^n where φ = 1.62
newtype TemporalWindow = TW { unTW :: Double }
  deriving (Eq, Ord, Show, Num, Fractional, Generic, NFData)

-- | Generate temporal window at golden ratio power
goldenWindow :: Int -> TemporalWindow
goldenWindow n = TW (phi ** fromIntegral n)
  where phi = unGreenPhi greenPhi  -- 1.62

--------------------------------------------------------------------------------
-- Harmonic Structures
--------------------------------------------------------------------------------

-- | Spherical harmonic Y_l^m(θ, φ)
data SphericalHarmonic = SH
  { shL     :: !Int     -- ^ degree (l ≥ 0)
  , shM     :: !Int     -- ^ order  (-l ≤ m ≤ l)
  , shTheta :: !Radian  -- ^ polar angle
  , shPhi   :: !Radian  -- ^ azimuthal angle
  } deriving (Eq, Show, Generic, NFData)

-- | Smart constructor enforcing |m| ≤ l
mkSphericalHarmonic :: Int -> Int -> Radian -> Radian -> Maybe SphericalHarmonic
mkSphericalHarmonic l m theta phi
  | l >= 0 && abs m <= l = Just (SH l m theta phi)
  | otherwise = Nothing

-- | Radial profile type (for serializable representation)
data RadialType
  = InverseSquare   -- ^ 1 / (1 + r)²
  | Exponential     -- ^ exp(-αr)
  | Gaussian        -- ^ exp(-r²/2σ²)
  | AnkhModulated   -- ^ Ankh / (1 + r)²
  deriving (Eq, Ord, Show, Enum, Bounded, Generic, NFData)

-- | Radial profile with coefficients (serializable, unlike function type)
data RadialProfile = RP
  { rpType   :: !RadialType
  , rpScale  :: !Double       -- ^ scaling factor
  , rpDecay  :: !Double       -- ^ decay rate (α or σ)
  } deriving (Eq, Show, Generic, NFData)

-- | Evaluate radial profile at shell depth
evalRadialProfile :: RadialProfile -> ShellIndex -> Double
evalRadialProfile (RP typ scale decay) (SI s) =
  let r = fromIntegral s
  in case typ of
    InverseSquare  -> scale / (1 + r) ** 2
    Exponential    -> scale * exp (-decay * r)
    Gaussian       -> scale * exp (-(r ** 2) / (2 * decay ** 2))
    AnkhModulated  -> (unAnkh ankh * scale) / (1 + r) ** 2

-- | Potential well depth (scalar potential minimum)
newtype WellDepth = WD { unWD :: Double }
  deriving (Eq, Ord, Show, Num, Fractional, Generic, NFData)

-- | Single component of a scalar field
data ScalarComponent = SC
  { scL       :: !Int           -- ^ harmonic degree
  , scM       :: !Int           -- ^ harmonic order
  , scProfile :: !RadialProfile -- ^ radial behavior
  , scWeight  :: !Double        -- ^ coefficient weight
  } deriving (Eq, Show, Generic, NFData)

-- | Scalar field as weighted sum of components
newtype ScalarField = ScalarField { unField :: [ScalarComponent] }
  deriving (Eq, Show, Generic, NFData)

-- | Parametric toroidal bowl surface (Primer Field inspired)
data BowlField = Bowl
  { bfCurvature   :: !Double   -- ^ surface curvature parameter
  , bfFluxDensity :: !Double   -- ^ field strength
  , bfShells      :: !Int      -- ^ number of nested shells
  } deriving (Eq, Show, Generic, NFData)

-- | Flux coherence: shell-integrated coherence measure
data FluxCoherence = FC
  { fcValue   :: !Double           -- ^ integrated flux
  , fcBand    :: ![RepitanIndex]   -- ^ contributing Repitan bands
  } deriving (Eq, Show, Generic, NFData)

--------------------------------------------------------------------------------
-- Composite Coordinate & Emergence
--------------------------------------------------------------------------------

-- | Full 4D coordinate in Ra system
data Coordinate = Coord
  { cTheta :: !ThetaSector   -- ^ θ: semantic sector (1-27 Repitans)
  , cPhi   :: !PhiLevel      -- ^ φ: access level (RAC 1-6)
  , cOmega :: !OmegaFormat   -- ^ h: harmonic depth (5 Omega formats)
  , cShell :: !ShellIndex    -- ^ r: radial shell
  } deriving (Eq, Show, Generic, NFData)

-- | Conditions required for fragment emergence
data EmergenceCondition = EC
  { ecCoordinate    :: !Coordinate
  , ecPotential     :: !WellDepth
  , ecFluxCoherence :: !FluxCoherence
  , ecInversion     :: !Inversion
  , ecTemporalPhase :: !TemporalWindow
  } deriving (Eq, Show, Generic, NFData)

-- | Result of emergence attempt
data EmergenceResult fragment
  = FullEmergence fragment              -- ^ Complete phase lock
  | PartialEmergence !Double fragment   -- ^ α ∈ (0,1), partial coherence
  | Dark                                -- ^ Phase mismatch, no emergence
  | ShadowEmergence fragment            -- ^ Inverted/mirror emergence
  deriving (Eq, Show, Generic, NFData)

--------------------------------------------------------------------------------
-- Computational Interfaces
--------------------------------------------------------------------------------

-- | Spherical harmonic Y_l^m(θ, φ)
-- Placeholder: returns 0.0 until Legendre polynomial implementation
sphericalY :: Int -> Int -> Radian -> Radian -> Double
sphericalY l m (Radian theta) (Radian phi)
  | l < 0 || abs m > l = 0.0
  | otherwise = 0.0  -- TODO: implement via associated Legendre polynomials
  -- Real implementation:
  -- Y_l^m = normalization * P_l^m(cos θ) * exp(i m φ)
  -- For real spherical harmonics, split into cos(mφ) and sin(mφ) parts

-- | Compute scalar potential at coordinate
wellDepth :: ScalarField -> Coordinate -> WellDepth
wellDepth (ScalarField components) coord =
  let shellIdx = cShell coord
      -- Sum weighted radial profiles (angular part placeholder)
      total = sum [ scWeight sc * evalRadialProfile (scProfile sc) shellIdx
                  | sc <- components ]
  in WD total

-- | Compute flux coherence for a shell
fluxIntegral :: ScalarField -> ShellIndex -> FluxCoherence
fluxIntegral (ScalarField components) shell@(SI s) =
  let -- Integrate over shell (placeholder: sum of weighted profiles)
      flux = sum [ scWeight sc * evalRadialProfile (scProfile sc) shell
                 | sc <- components ]
      -- All Repitan bands contribute (placeholder)
      bands = [RI n | n <- [1..27]]
  in FC flux bands

--------------------------------------------------------------------------------
-- Constants (re-exported from Ra.Constants for convenience)
--------------------------------------------------------------------------------

-- | Green Phi value as Double (convenience accessor)
phi_green_value :: Double
phi_green_value = unGreenPhi greenPhi  -- 1.62

-- | Pi variants (local copies for scalar calculations)
pi_red :: Double
pi_red = 3.141592592
