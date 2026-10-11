/**
 * Ra System fundamental constants with type-safe wrappers.
 *
 * Values are Wesley H. Bateman's, from "The Rods of Amon Ra" (1992-1997), as
 * printed and as checked page by page by the lattice's TR node (Aug-Sep 2026).
 * They are the author's own units and ratios, not physics claims. Citations
 * name the lattice post (TR-###) and the archive page or printed folio; the
 * Python package (python/ra_system/constants.py) carries the full notes.
 */

/** Colour ratio r = 60000 sqrt2 / 84823 = 1.000351482 (TR-291, TR-300; vol2a g00215_src01_p0031). Applies as r^d. */
export const COLOUR_RATIO = (60000 * Math.SQRT2) / 84823;

/** Omega Ratio (Q-Ratio) = 32 sqrt2 / 45 = 1.005662978 (TR-304; vol2a prologue, TR-302) */
export const OMEGA = (32 * Math.SQRT2) / 45;

/** Outer ladder rung Omega / r = 84823/84375 = 1.005309630 (TR-308, TR-309, TR-312) */
export const OMEGA_OVER_R = 84823 / 84375;

// Pi variants (chromatic)
/** Red Pi = 84823/27000 = 3.141592592... Not pi: pi rounded at a base-27 scale (TR-291, TR-349) */
export const RED_PI = 84823 / 27000;
/** Green Pi = (20/9) sqrt2 = 3.142696805 (TR-300; named at volume_1 p.0129, TR-301) */
export const GREEN_PI = (20 / 9) * Math.SQRT2;
/** Blue Pi = Green Pi x r = 3.143801406 (TR-308; volume_1 p.0122) */
export const BLUE_PI = GREEN_PI * COLOUR_RATIO;

// Phi variants (chromatic): Bateman's erratum, supplement g00857_src03_p0002
// (FP-1399, TR-312, TR-316). It withdraws the Vol 2/3 Red Phi 1.619430799.
/** Omega Major Phi = 1.62, also the Ankh wheel diameter (volume_1 p.0168, TR-298) */
export const OMEGA_MAJOR_PHI = 1.62;
/** Red Phi = 1.62 x 84823/84375 = 1.6286016 (supplement p.0002; TR-312, TR-316) */
export const RED_PHI = 1.6286016;
/** Green Phi = 1.629174025 (supplement p.0002; TR-312) */
export const GREEN_PHI = 1.629174025;
/** Blue Phi = 1.629746650 (supplement p.0002; TR-312) */
export const BLUE_PHI = 1.62974665;

/** Hunab (Red) = 84823/80000 = 1.0602875 (TR-291; volume_1 p.0150; FP-1399) */
export const HUNAB = 84823 / 80000;

/** Green Hunab = (3/4) sqrt2 = Hunab x r exactly (TR-320, FP-1430, TR-327) */
export const GREEN_HUNAB = 0.75 * Math.SQRT2;

/** H-Bar = 135/128 = 1.0546875 = Green Hunab / Omega (TR-304: vol2a g00194_src01_p0010). Not the physical reduced Planck constant. */
export const H_BAR = 135 / 128;

/** Ankh = 3 x 84823/50000 = 5.08938 = Red Pi x 1.62 (TR-329; volume_1 p.0168, TR-298) */
export const ANKH = (3 * 84823) / 50000;

/** Bateman's f.s.Delta unit = 1/729 = Repitan(1)^2 (TR-323, TR-331). Not the physical fine-structure constant. */
export const FINE_STRUCTURE = 1 / 729;

/** Bateman's Ra Rydberg = 800/729 (volume_1 p.0126 "Green Rydberg"; TR-320, TR-322). Not CODATA; do not normalise (TR-283). */
export const RA_RYDBERG = 800 / 729;

/** Balmer constant = 729/2 = 364.5 (TR-288, TR-331) */
export const BALMER_CONSTANT = 729 / 2;

/**
 * Typed wrapper for Ankh values
 */
export class AnkhValue {
  readonly value: number;

  constructor(value: number = ANKH) {
    this.value = value;
  }

  /** Derive RAC1 from Ankh (Invariant I2: RAC1 = Ankh / 8) */
  deriveRac1(): number {
    return this.value / 8;
  }
}

/**
 * Typed wrapper for Omega Ratio
 */
export class OmegaRatio {
  readonly value: number;

  constructor(value: number = OMEGA) {
    this.value = value;
  }

  /** Get the reciprocal */
  reciprocal(): number {
    return 1 / this.value;
  }
}

/**
 * Verify Invariant I1: Ankh = π_red × 1.62 (Omega Major Phi, the Ankh wheel diameter)
 */
export function verifyAnkhInvariant(): boolean {
  const computed = RED_PI * OMEGA_MAJOR_PHI;
  return Math.abs(ANKH - computed) < 1e-12;
}

/**
 * Verify Invariant I2: RAC1 = Ankh / 8
 */
export function verifyRac1Invariant(rac1Value: number): boolean {
  const computed = ANKH / 8;
  return Math.abs(rac1Value - computed) < 0.0001;
}

/**
 * Verify Invariant I3: H-Bar = Green Hunab / Ω = 135/128 (TR-304)
 */
export function verifyHBarInvariant(): boolean {
  const computed = GREEN_HUNAB / OMEGA;
  return Math.abs(H_BAR - computed) < 1e-12;
}

/**
 * Verify Invariant I6: Fine Structure = Repitan(1)² = (1/27)²
 */
export function verifyFineStructureInvariant(): boolean {
  const r1 = 1 / 27;
  return Math.abs(FINE_STRUCTURE - r1 * r1) < 1e-10;
}
