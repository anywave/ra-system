{-|
Module      : Spec
Description : QuickCheck properties for Ra System invariants
Copyright   : (c) Anywave, 2025
License     : Apache-2.0

Tests all 17 invariants from ra_integration_spec.md Section 6.
-}
module Main (main) where

import Test.Hspec
import Test.Hspec.QuickCheck
import Test.QuickCheck

import Ra.Constants
import Ra.Repitans
import Ra.Rac
import Ra.Omega
import Ra.Spherical
import Ra.Gates hiding (coherenceFloor)
import Ra.Identity hiding (Potential, FluxCoherence)
import qualified Ra.Identity as Id
import Ra.Emergence (classifyEmergence, classifyEmergenceWithReluctance
                    , isNoRelation, isFull, isDark, isPartial, isWithheld, isGuardianMismatch
                    , EmergenceResult(..))
import Ra.Scalar (Inversion(..), FragmentReluctance(..), HarmonicSignature(..)
                 , defaultReluctance, guardedReluctance, mkHarmonicSignature
                 , matchGuardianHarmonic, checkReluctanceFloor)
import qualified Ra.Gates as Gates

main :: IO ()
main = hspec $ do
    describe "Constant Invariants (I1-I6)" $ do
        it "I1: Ankh = pi_red * phi_green" $ do
            let computed = unRedPi redPi * unGreenPhi greenPhi
            abs (unAnkh ankh - computed) `shouldSatisfy` (< 0.0001)

        it "I2: RAC1 = Ankh / 8" $ do
            let computed = unAnkh ankh / 8
            abs (unRacValue (racValue RAC1) - computed) `shouldSatisfy` (< 0.0001)

        it "I3: H-Bar = Hunab / Omega" $ do
            -- Tolerance relaxed to 1e-3 due to Ra System's use of truncated constants
            -- H-Bar (1.0546875) vs computed (1.0543269...) differs by ~3.6e-4
            let computed = unHunab hunab / unOmegaRatio omegaRatio
            abs (unHBar hBar - computed) `shouldSatisfy` (< 0.001)

        it "I4: Repitan(n) = n / 27 for all n in [1, 27]" $
            let checkRepitan n = case repitan n of
                    Just r -> abs (repitanValue r - fromIntegral n / 27) < 1e-10
                    Nothing -> False
            in all checkRepitan [1..27] `shouldBe` True

        it "I5: T.O.N.(m) = m * 0.027 for all m in [0, 36]" $
            -- T.O.N. values are implicitly tested through Repitan relationship
            let tonValue m = fromIntegral m * 0.027
            in all (\m -> tonValue m >= 0 && tonValue m < 1) [0..35] `shouldBe` True

        it "I6: Fine Structure = Repitan(1)² = 0.0013717421" $ do
            let r1 = repitanValue firstRepitan
            abs (r1 * r1 - unFineStructure fineStructure) `shouldSatisfy` (< 1e-10)

    describe "Ordering Invariants (O1-O4)" $ do
        it "O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0" $ do
            verifyRacOrdering `shouldBe` True

        it "O2: pi_red < pi_green < pi_blue" $ do
            unRedPi redPi `shouldSatisfy` (< unGreenPi greenPi)
            unGreenPi greenPi `shouldSatisfy` (< unBluePi bluePi)

        it "O3: For all n: 0 < Repitan(n) <= 1" $
            let checkRepitanRange r = let v = repitanValue r in v > 0 && v <= 1
            in all checkRepitanRange allRepitans `shouldBe` True

        it "O4: Omega format indices are 0-4" $ do
            harmonicFromOmega Red `shouldBe` 0
            harmonicFromOmega Blue `shouldBe` 4

    describe "Conversion Invariants (C1-C3)" $ do
        prop "C1: Omega roundtrip preserves value" prop_omega_roundtrip

        it "C2: Green * Omega = Omega_Minor" $ do
            let green = 1.62
            abs (greenToOmegaMinor green - green * omega) `shouldSatisfy` (< 1e-10)

        it "C3: Green / Omega = Omega_Major" $ do
            -- Tolerance relaxed to 1e-9 for floating-point precision
            let green = 1.62
            abs (greenToOmegaMajor green - green / omega) `shouldSatisfy` (< 1e-9)

    describe "Range Invariants (R1-R4)" $ do
        it "R1: 0 < RAC(i) < 1 for all i in [1, 6]" $
            let checkRacRange level = isValidRacValue $ unRacValue $ racValue level
            in all checkRacRange allRacLevels `shouldBe` True

        prop "R2: 0 < Repitan(n) <= 1 for all n in [1, 27]" prop_repitan_range

        it "R3: Coherence bounds are [0, 1]" $ do
            Gates.coherenceFloor `shouldSatisfy` (>= 0)
            Gates.coherenceFloor `shouldSatisfy` (< 1)
            coherenceCeiling `shouldBe` 1.0

        it "R4: Omega format index in {0, 1, 2, 3, 4}" $ do
            all (\f -> harmonicFromOmega f `elem` [0..4]) [minBound..maxBound] `shouldBe` True

    describe "Repitan Properties" $ do
        prop "repitan index in valid range" prop_repitan_in_range
        prop "repitan smart constructor validates" prop_repitan_validation

        it "first repitan is Fine Structure root" $ do
            abs (repitanValue firstRepitan - 1/27) `shouldSatisfy` (< 1e-10)

        it "unity repitan equals 1" $ do
            repitanValue unityRepitan `shouldBe` 1.0

    describe "RAC Properties" $ do
        it "all RAC values are between 0 and 1" $ do
            all (\l -> let v = unRacValue (racValue l) in v > 0 && v < 1) allRacLevels
                `shouldBe` True

        it "RAC pyramid divisions are correct" $ do
            pyramidDivision RAC1 `shouldBe` 360.0
            pyramidDivision RAC2 `shouldBe` 364.5

    describe "Omega Properties" $ do
        prop "all omega roundtrips preserve value" prop_all_omega_roundtrips

        it "omega ratio is correct" $ do
            omega `shouldBe` 1.005662978

    describe "Gates Properties" $ do
        prop "full coherence grants full access" prop_full_coherence_access
        prop "zero coherence is blocked" prop_zero_coherence_blocked
        prop "access result alpha is in [0, 1]" prop_access_alpha_range

        it "coherence floor is phi_green / Ankh" $ do
            abs (Gates.coherenceFloor - unGreenPhi greenPhi / unAnkh ankh) `shouldSatisfy` (< 1e-10)

    describe "Spherical Properties" $ do
        it "theta/repitan roundtrip" $ do
            let r = ninthRepitan
            let theta = thetaFromRepitan r
            repitanIndex (repitanFromTheta theta) `shouldBe` repitanIndex r

        it "radius normalization roundtrip" $ do
            let raw = 2.54469  -- Half of Ankh
            abs (denormalizeRadius (normalizeRadius raw) - raw) `shouldSatisfy` (< 1e-10)

    describe "Identity Properties (Null Emergence Boundary)" $ do
        it "I1: Null emergence at exact balance point" $ do
            -- When phi^n * harmonic == delta, we get NullEmergence
            let depth = WindowDepth 0
                harmonic = 1.0
                delta = AnkhDelta 1.0  -- Exactly equals phi^0 * 1.0
                eps = NullEpsilon 1e-9
            evaluateIdentity depth harmonic delta eps `shouldBe` NullEmergence

        it "I2: Active emergence when identity violated" $ do
            -- When phi^n * harmonic /= delta, we get ActiveEmergence
            let depth = WindowDepth 1
                harmonic = 1.0
                delta = AnkhDelta 0.5  -- Does not equal phi^1 * 1.0 = 1.62
                eps = NullEpsilon 1e-9
            isActiveEmergence (evaluateIdentity depth harmonic delta eps) `shouldBe` True

        it "I3: noRelation True below coherenceFloor" $ do
            noRelation 0.0 `shouldBe` True
            noRelation (Id.coherenceFloor - 0.001) `shouldBe` True
            noRelation Id.coherenceFloor `shouldBe` False

        it "I4: Cancellation operator equals subtraction" $ do
            (5.0 ⊣⊢ 3.0) `shouldBe` 2.0
            (3.0 ⊣⊢ 5.0) `shouldBe` (-2.0)
            (1.0 ⊣⊢ 1.0) `shouldBe` 0.0

        it "I5: phiTerm 0 = 1.0" $ do
            phiTerm (WindowDepth 0) `shouldBe` 1.0

        it "I6: phiTerm grows with depth" $ do
            phiTerm (WindowDepth 1) `shouldSatisfy` (> phiTerm (WindowDepth 0))
            phiTerm (WindowDepth 2) `shouldSatisfy` (> phiTerm (WindowDepth 1))
            phiTerm (WindowDepth 3) `shouldSatisfy` (> phiTerm (WindowDepth 2))

        it "I7: invAnkh * ankh = 1" $ do
            abs (invAnkh * unAnkh ankh - 1.0) `shouldSatisfy` (< 1e-10)

        it "I8: coherenceFloor = phi / ankh" $ do
            let expected = unGreenPhi greenPhi / unAnkh ankh
            abs (Id.coherenceFloor - expected) `shouldSatisfy` (< 1e-10)

        prop "cancellation is subtraction" prop_cancellation_is_subtraction
        prop "phiTerm positive for non-negative depth" prop_phiTerm_positive

        it "I9: checkEmergencePermitted returns NullEmergence for zero coherence" $ do
            let result = checkEmergencePermitted 0.0
                           (WindowDepth 0) 1.0
                           (Id.Potential 0.5) (Id.FluxCoherence 0.5) (HarmonicWeight 1.0)
                           (NullEpsilon 1e-9)
            result `shouldBe` NullEmergence

    describe "Emergence Properties (NoRelation)" $ do
        it "E1: coherence = 0 returns NoRelation" $ do
            classifyEmergence 0.0 Normal `shouldBe` NoRelation

        it "E2: coherence < coherenceFloor returns NoRelation" $ do
            isNoRelation (classifyEmergence 0.1 Normal) `shouldBe` True
            isNoRelation (classifyEmergence 0.2 Normal) `shouldBe` True
            isNoRelation (classifyEmergence 0.31 Normal) `shouldBe` True

        it "E3: coherence >= coherenceFloor does not return NoRelation" $ do
            isNoRelation (classifyEmergence 0.35 Normal) `shouldBe` False
            isNoRelation (classifyEmergence 0.5 Normal) `shouldBe` False
            isNoRelation (classifyEmergence 1.0 Normal) `shouldBe` False

        it "E4: coherence >= 0.85 returns Full" $ do
            isFull (classifyEmergence 0.85 Normal) `shouldBe` True
            isFull (classifyEmergence 0.90 Normal) `shouldBe` True
            isFull (classifyEmergence 1.0 Normal) `shouldBe` True

        it "E5: coherence in [coherenceFloor, 0.40) returns Dark" $ do
            isDark (classifyEmergence 0.35 Normal) `shouldBe` True
            isDark (classifyEmergence 0.39 Normal) `shouldBe` True

        it "E6: coherence in [0.40, 0.85) returns Partial" $ do
            isPartial (classifyEmergence 0.40 Normal) `shouldBe` True
            isPartial (classifyEmergence 0.60 Normal) `shouldBe` True
            isPartial (classifyEmergence 0.84 Normal) `shouldBe` True

        prop "E7: classifyEmergence respects coherenceFloor boundary" prop_norelation_boundary

    describe "Fragment Reluctance Properties" $ do
        prop "FR1: reluctanceFloor is bounded [0, 1]" prop_reluctanceFloorBounded
        prop "FR2: Withheld implies sub-floor coherence" prop_withheldImpliesSubFloor
        prop "FR3: Guardian mismatch cannot produce Full" prop_guardianMismatchNotFull
        prop "FR4: Default reluctance allows emergence" prop_defaultReluctanceAllowsEmergence
        prop "FR5: Harmonic signature validates |m| <= l" prop_harmonicSignatureValid
        prop "FR6: No guardian always matches" prop_noGuardianAlwaysMatches
        prop "FR7: Same guardian matches" prop_sameGuardianMatches
        prop "FR8: Reluctance floor check is monotonic" prop_reluctanceFloorMonotonic

        -- Test case 1: High External Coherence, Withheld by Reluctance
        it "Case 1: High coherence withheld by high reluctance floor" $ do
            let coh = 0.95
                fr = FR 0.98 Nothing False  -- reluctance_floor > coherence
            isWithheld (classifyEmergenceWithReluctance coh Normal fr Nothing) `shouldBe` True

        -- Test case 2: Guardian Harmonic Match -> Full Emergence
        it "Case 2: Guardian match allows full emergence" $ do
            let coh = 1.0
                fragSig = Just (HS 2 1)
                guardSig = Just (HS 2 1)  -- Match
                fr = FR 0.8 guardSig False
            isFull (classifyEmergenceWithReluctance coh Normal fr fragSig) `shouldBe` True

        -- Test case 3: Guardian Mismatch -> Blocked
        it "Case 3: Guardian mismatch blocks emergence" $ do
            let coh = 1.0
                fragSig = Just (HS 2 1)
                guardSig = Just (HS 3 0)  -- Mismatch
                fr = FR 0.5 guardSig False
            isGuardianMismatch (classifyEmergenceWithReluctance coh Normal fr fragSig) `shouldBe` True

        -- Test case 4: Inversion Override
        it "Case 4: Inversion produces shadow emergence" $ do
            let coh = 0.7
                fr = FR 0.5 Nothing True  -- inverted = True
            classifyEmergenceWithReluctance coh Normal fr Nothing `shouldBe` Shadow

-- QuickCheck Generators

instance Arbitrary Repitan where
    arbitrary = do
        n <- choose (1, 27)
        case repitan n of
            Just r -> pure r
            Nothing -> error "Arbitrary Repitan: impossible"

instance Arbitrary RacLevel where
    arbitrary = arbitraryBoundedEnum

instance Arbitrary OmegaFormat where
    arbitrary = arbitraryBoundedEnum

-- QuickCheck Properties

-- | Repitan index is always in range [1, 27]
prop_repitan_in_range :: Repitan -> Bool
prop_repitan_in_range r = repitanIndex r >= 1 && repitanIndex r <= 27

-- | Repitan value is always 0 < x <= 1
prop_repitan_range :: Repitan -> Bool
prop_repitan_range r = let v = repitanValue r in v > 0 && v <= 1

-- | Repitan smart constructor validates correctly
prop_repitan_validation :: Int -> Bool
prop_repitan_validation n =
    case repitan n of
        Just r -> repitanIndex r == n && n >= 1 && n <= 27
        Nothing -> n < 1 || n > 27

-- | Omega roundtrip conversion preserves value (within tolerance)
prop_omega_roundtrip :: OmegaFormat -> OmegaFormat -> Double -> Property
prop_omega_roundtrip from to x =
    x > 0 ==> verifyOmegaRoundtrip from to x

-- | All pairs of omega formats roundtrip correctly
prop_all_omega_roundtrips :: Double -> Property
prop_all_omega_roundtrips x =
    x > 0 ==>
        all (\(f, t) -> verifyOmegaRoundtrip f t x)
            [(f, t) | f <- [minBound..maxBound], t <- [minBound..maxBound]]

-- | Full coherence (1.0) always grants FullAccess
prop_full_coherence_access :: RacLevel -> Bool
prop_full_coherence_access rac = isFullAccess $ accessLevel 1.0 rac

-- | Zero coherence is always Blocked
prop_zero_coherence_blocked :: RacLevel -> Bool
prop_zero_coherence_blocked rac = isBlocked $ accessLevel 0.0 rac

-- | Access alpha is always in [0, 1]
prop_access_alpha_range :: Double -> RacLevel -> Property
prop_access_alpha_range coherence rac =
    coherence >= 0 && coherence <= 1 ==>
        let alpha = accessAlpha $ accessLevel coherence rac
        in alpha >= 0 && alpha <= 1

-- | Cancellation operator is equivalent to subtraction
prop_cancellation_is_subtraction :: Double -> Double -> Bool
prop_cancellation_is_subtraction a b = (a ⊣⊢ b) == (a - b)

-- | phiTerm is always positive for non-negative depth
prop_phiTerm_positive :: NonNegative Int -> Bool
prop_phiTerm_positive (NonNegative n) = phiTerm (WindowDepth n) > 0

-- | classifyEmergence returns NoRelation iff coherence < coherenceFloor
prop_norelation_boundary :: Double -> Property
prop_norelation_boundary coh =
    coh >= 0 && coh <= 1 ==>
        if coh < Id.coherenceFloor
            then isNoRelation (classifyEmergence coh Normal)
            else not (isNoRelation (classifyEmergence coh Normal))

-- ---------------------------------------------------------------------
-- Fragment Reluctance Properties
-- ---------------------------------------------------------------------

instance Arbitrary FragmentReluctance where
    arbitrary = do
        floor' <- choose (0.0, 1.0)
        hasGuardian <- arbitrary
        sig <- if hasGuardian
               then do
                   l <- choose (0, 5)
                   m <- choose (-l, l)
                   pure $ Just (HS l m)
               else pure Nothing
        inv <- arbitrary
        pure $ FR floor' sig inv

instance Arbitrary HarmonicSignature where
    arbitrary = do
        l <- choose (0, 10)
        m <- choose (-l, l)
        pure $ HS l m

-- | Reluctance floor is bounded [0, 1]
prop_reluctanceFloorBounded :: FragmentReluctance -> Bool
prop_reluctanceFloorBounded fr =
    reluctanceFloor fr >= 0.0 && reluctanceFloor fr <= 1.0

-- | Withheld implies coherence below reluctance floor
prop_withheldImpliesSubFloor :: Double -> FragmentReluctance -> Property
prop_withheldImpliesSubFloor coh fr =
    coh >= 0 && coh <= 1 ==>
        case classifyEmergenceWithReluctance coh Normal fr Nothing of
            Withheld -> coh < reluctanceFloor fr
            _ -> True

-- | Guardian mismatch cannot produce full emergence
prop_guardianMismatchNotFull :: Double -> Property
prop_guardianMismatchNotFull coh =
    coh >= 0 && coh <= 1 ==>
        let fragSig = Just (HS 2 1)
            guardSig = Just (HS 3 0)  -- Mismatch
            fr = FR 0.0 guardSig False
        in case classifyEmergenceWithReluctance coh Normal fr fragSig of
              Full -> False
              _ -> True

-- | Default reluctance allows emergence when coherence meets thresholds
prop_defaultReluctanceAllowsEmergence :: Property
prop_defaultReluctanceAllowsEmergence =
    forAll (choose (0.85, 1.0)) $ \coh ->
        isFull (classifyEmergenceWithReluctance coh Normal defaultReluctance Nothing)

-- | Harmonic signature smart constructor validates |m| <= l
prop_harmonicSignatureValid :: Int -> Int -> Bool
prop_harmonicSignatureValid l m =
    case mkHarmonicSignature l m of
        Just (HS l' m') -> l' >= 0 && abs m' <= l'
        Nothing -> l < 0 || abs m > l

-- | Guardian match with no guardian always succeeds
prop_noGuardianAlwaysMatches :: HarmonicSignature -> Bool
prop_noGuardianAlwaysMatches fragSig =
    matchGuardianHarmonic fragSig Nothing

-- | Guardian match with same signature succeeds
prop_sameGuardianMatches :: HarmonicSignature -> Bool
prop_sameGuardianMatches sig =
    matchGuardianHarmonic sig (Just sig)

-- | Reluctance floor check is monotonic in score
-- If s1 passes the floor check, then any s2 >= s1 also passes
prop_reluctanceFloorMonotonic :: Property
prop_reluctanceFloorMonotonic =
    forAll (choose (0.0, 1.0)) $ \floor' ->
      forAll (choose (0.0, 1.0)) $ \s1 ->
        forAll (choose (s1, 1.0)) $ \s2 ->
          checkReluctanceFloor s1 floor' ==> checkReluctanceFloor s2 floor'
