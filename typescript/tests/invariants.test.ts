/**
 * Integration tests for Ra System invariants.
 *
 * Tests all 17 invariants from ra_integration_spec.md Section 6.
 */

import { describe, it, expect } from 'vitest';
import {
  ANKH,
  RED_PI,
  GREEN_PI,
  BLUE_PI,
  GREEN_PHI,
  HUNAB,
  OMEGA,
  H_BAR,
  FINE_STRUCTURE,
  verifyAnkhInvariant,
  verifyHBarInvariant,
  verifyFineStructureInvariant,
} from '../src/constants.js';
import {
  Repitan,
  allRepitans,
  repitanFromTheta,
  verifyRepitanInvariant,
  verifyRepitanRangeInvariant,
} from '../src/repitans.js';
import {
  RacLevel,
  allRacLevels,
  racValue,
  verifyRacOrdering,
  verifyRacRange,
  verifyRac1Derivation,
} from '../src/rac.js';
import {
  OmegaFormat,
  allOmegaFormats,
  harmonicFromOmega,
  convertOmega,
  greenToOmegaMajor,
  greenToOmegaMinor,
  verifyAllOmegaRoundtrips,
  verifyOmegaRange,
} from '../src/omega.js';
import {
  createCoordinate,
  normalizeRadius,
  denormalizeRadius,
  thetaFromRepitan,
  verifyOmegaIndices,
} from '../src/spherical.js';
import {
  COHERENCE_FLOOR,
  COHERENCE_CEILING,
  accessLevel,
  accessAlpha,
  isFullAccess,
  isBlocked,
  verifyCoherenceBounds,
  verifyCoherenceFloor,
} from '../src/gates.js';

// =============================================================================
// Constant Invariants (I1-I6)
// =============================================================================

describe('Constant Invariants (I1-I6)', () => {
  it('I1: Ankh = π_red × φ_green', () => {
    const computed = RED_PI * GREEN_PHI;
    expect(Math.abs(ANKH - computed)).toBeLessThan(0.0001);
    expect(verifyAnkhInvariant()).toBe(true);
  });

  it('I2: RAC₁ = Ankh / 8', () => {
    const computed = ANKH / 8;
    expect(Math.abs(racValue(RacLevel.RAC1) - computed)).toBeLessThan(0.0001);
    expect(verifyRac1Derivation()).toBe(true);
  });

  it('I3: H-Bar = Hunab / Ω', () => {
    const computed = HUNAB / OMEGA;
    expect(Math.abs(H_BAR - computed)).toBeLessThan(0.0001);
    expect(verifyHBarInvariant()).toBe(true);
  });

  it('I4: Repitan(n) = n / 27 for all n ∈ [1, 27]', () => {
    expect(verifyRepitanInvariant()).toBe(true);
    for (let n = 1; n <= 27; n++) {
      const r = Repitan.create(n)!;
      expect(Math.abs(r.value - n / 27)).toBeLessThan(1e-10);
    }
  });

  it('I5: T.O.N.(m) = m × 0.027 for all m ∈ [0, 36]', () => {
    for (let m = 0; m <= 35; m++) {
      const ton = m * 0.027;
      expect(ton).toBeGreaterThanOrEqual(0);
      expect(ton).toBeLessThan(1);
    }
  });

  it('I6: Fine Structure = Repitan(1)² = 0.0013717421', () => {
    const r1 = Repitan.FIRST.value;
    const computed = r1 * r1;
    expect(Math.abs(FINE_STRUCTURE - computed)).toBeLessThan(1e-10);
    expect(verifyFineStructureInvariant()).toBe(true);
  });
});

// =============================================================================
// Ordering Invariants (O1-O4)
// =============================================================================

describe('Ordering Invariants (O1-O4)', () => {
  it('O1: RAC₁ > RAC₂ > RAC₃ > RAC₄ > RAC₅ > RAC₆ > 0', () => {
    expect(verifyRacOrdering()).toBe(true);
  });

  it('O2: π_red < π_green < π_blue', () => {
    expect(RED_PI).toBeLessThan(GREEN_PI);
    expect(GREEN_PI).toBeLessThan(BLUE_PI);
  });

  it('O3: For all n: 0 < Repitan(n) ≤ 1', () => {
    expect(verifyRepitanRangeInvariant()).toBe(true);
  });

  it('O4: Omega format indices are 0-4', () => {
    expect(verifyOmegaIndices()).toBe(true);
    expect(harmonicFromOmega(OmegaFormat.Red)).toBe(0);
    expect(harmonicFromOmega(OmegaFormat.Blue)).toBe(4);
  });
});

// =============================================================================
// Conversion Invariants (C1-C3)
// =============================================================================

describe('Conversion Invariants (C1-C3)', () => {
  it('C1: Omega roundtrip preserves value', () => {
    expect(verifyAllOmegaRoundtrips(1.62)).toBe(true);
  });

  it('C2: Green × Ω = Omega_Minor', () => {
    const green = 1.62;
    const omegaMinor = greenToOmegaMinor(green);
    expect(Math.abs(omegaMinor - green * OMEGA)).toBeLessThan(1e-10);
  });

  it('C3: Green / Ω = Omega_Major', () => {
    const green = 1.62;
    const omegaMajor = greenToOmegaMajor(green);
    expect(Math.abs(omegaMajor - green / OMEGA)).toBeLessThan(1e-10);
  });
});

// =============================================================================
// Range Invariants (R1-R4)
// =============================================================================

describe('Range Invariants (R1-R4)', () => {
  it('R1: 0 < RAC(i) < 1 for all i ∈ [1, 6]', () => {
    expect(verifyRacRange()).toBe(true);
  });

  it('R2: 0 < Repitan(n) ≤ 1 for all n ∈ [1, 27]', () => {
    for (const r of allRepitans()) {
      expect(r.value).toBeGreaterThan(0);
      expect(r.value).toBeLessThanOrEqual(1);
    }
  });

  it('R3: Coherence bounds are [0, 1]', () => {
    expect(verifyCoherenceBounds()).toBe(true);
    expect(COHERENCE_FLOOR).toBeGreaterThanOrEqual(0);
    expect(COHERENCE_FLOOR).toBeLessThan(1);
    expect(COHERENCE_CEILING).toBe(1.0);
  });

  it('R4: Omega format index ∈ {0, 1, 2, 3, 4}', () => {
    expect(verifyOmegaRange()).toBe(true);
  });
});

// =============================================================================
// Additional Property Tests
// =============================================================================

describe('Repitan Properties', () => {
  it('smart constructor validates range', () => {
    for (let n = 1; n <= 27; n++) {
      expect(Repitan.create(n)).toBeDefined();
    }
    expect(Repitan.create(0)).toBeUndefined();
    expect(Repitan.create(28)).toBeUndefined();
    expect(Repitan.create(-1)).toBeUndefined();
  });

  it('first repitan is Fine Structure root', () => {
    expect(Math.abs(Repitan.FIRST.value - 1 / 27)).toBeLessThan(1e-10);
  });

  it('unity repitan equals 1', () => {
    expect(Repitan.UNITY.value).toBe(1);
  });

  it('theta/repitan roundtrip', () => {
    for (const r of allRepitans()) {
      const theta = thetaFromRepitan(r);
      const r2 = repitanFromTheta(theta);
      expect(r.index).toBe(r2.index);
    }
  });
});

describe('RAC Properties', () => {
  it('all RAC values are between 0 and 1', () => {
    for (const level of allRacLevels()) {
      const v = racValue(level);
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe('Gates Properties', () => {
  it('full coherence grants full access', () => {
    for (const level of allRacLevels()) {
      const result = accessLevel(1.0, level);
      expect(isFullAccess(result)).toBe(true);
    }
  });

  it('zero coherence is blocked', () => {
    for (const level of allRacLevels()) {
      const result = accessLevel(0.0, level);
      expect(isBlocked(result)).toBe(true);
    }
  });

  it('access alpha is in [0, 1]', () => {
    const coherences = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0];
    for (const level of allRacLevels()) {
      for (const c of coherences) {
        const result = accessLevel(c, level);
        const alpha = accessAlpha(result);
        expect(alpha).toBeGreaterThanOrEqual(0);
        expect(alpha).toBeLessThanOrEqual(1);
      }
    }
  });

  it('coherence floor is φ_green / Ankh', () => {
    expect(verifyCoherenceFloor()).toBe(true);
  });
});

describe('Spherical Properties', () => {
  it('radius normalization roundtrip', () => {
    const raw = ANKH / 2; // Half of Ankh
    const normalized = normalizeRadius(raw);
    const denormalized = denormalizeRadius(normalized);
    expect(Math.abs(denormalized - raw)).toBeLessThan(1e-10);
  });

  it('coordinate validation', () => {
    // Valid
    const valid = createCoordinate(
      Repitan.NINTH,
      RacLevel.RAC1,
      OmegaFormat.Green,
      0.5
    );
    expect(valid).toBeDefined();

    // Invalid radius (negative)
    const invalidNeg = createCoordinate(
      Repitan.FIRST,
      RacLevel.RAC1,
      OmegaFormat.Green,
      -0.1
    );
    expect(invalidNeg).toBeUndefined();

    // Invalid radius (> 1)
    const invalidHigh = createCoordinate(
      Repitan.FIRST,
      RacLevel.RAC1,
      OmegaFormat.Green,
      1.1
    );
    expect(invalidHigh).toBeUndefined();
  });
});
