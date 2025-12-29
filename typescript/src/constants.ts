/**
 * Ra System fundamental constants with type-safe wrappers.
 *
 * All constants are derived from "The Rods of Amon Ra" by Wesley H. Bateman.
 */

/** Ankh: Master harmonic constant = 5.08938 */
export const ANKH = 5.08938;

/** Hunab: Natural constant = 1.05946 (12th root of 2) */
export const HUNAB = 1.05946;

/** H-Bar: Hunab / Omega = 1.05346545 */
export const H_BAR = 1.05346545;

/** Omega Ratio (Q-Ratio): 1.005662978 */
export const OMEGA = 1.005662978;

/** Fine Structure: Repitan(1)² = 0.0013717421 */
export const FINE_STRUCTURE = 0.0013717421;

// Pi variants (chromatic)
/** Red Pi: Standard π = 3.14159265 */
export const RED_PI = 3.14159265;
/** Green Pi: 3.14754099 */
export const GREEN_PI = 3.14754099;
/** Blue Pi: 3.15349386 */
export const BLUE_PI = 3.15349386;

// Phi variants (chromatic)
/** Red Phi: 1.614 */
export const RED_PHI = 1.614;
/** Green Phi: φ = 1.62 */
export const GREEN_PHI = 1.62;
/** Blue Phi: 1.626 */
export const BLUE_PHI = 1.626;

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
 * Verify Invariant I1: Ankh = π_red × φ_green
 */
export function verifyAnkhInvariant(): boolean {
  const computed = RED_PI * GREEN_PHI;
  return Math.abs(ANKH - computed) < 0.0001;
}

/**
 * Verify Invariant I2: RAC1 = Ankh / 8
 */
export function verifyRac1Invariant(rac1Value: number): boolean {
  const computed = ANKH / 8;
  return Math.abs(rac1Value - computed) < 0.0001;
}

/**
 * Verify Invariant I3: H-Bar = Hunab / Ω
 */
export function verifyHBarInvariant(): boolean {
  const computed = HUNAB / OMEGA;
  return Math.abs(H_BAR - computed) < 0.0001;
}

/**
 * Verify Invariant I6: Fine Structure = Repitan(1)² = (1/27)²
 */
export function verifyFineStructureInvariant(): boolean {
  const r1 = 1 / 27;
  return Math.abs(FINE_STRUCTURE - r1 * r1) < 1e-10;
}
