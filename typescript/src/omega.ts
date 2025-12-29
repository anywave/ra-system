/**
 * OmegaFormat enum with conversion functions.
 *
 * Five-level Omega format system for frequency/precision tiers.
 * Hierarchy: Red > Omega Major > Green > Omega Minor > Blue
 *
 * Conversions use the Omega Ratio (Q-Ratio): Ω = 1.005662978
 */

import { OMEGA } from './constants.js';

/**
 * The five Omega format levels (coherence depth tiers)
 * Index 0 = Red (highest precision), Index 4 = Blue
 */
export enum OmegaFormat {
  Red = 0,
  OmegaMajor = 1,
  Green = 2,
  OmegaMinor = 3,
  Blue = 4,
}

/**
 * Get all Omega formats in order
 */
export function allOmegaFormats(): OmegaFormat[] {
  return [
    OmegaFormat.Red,
    OmegaFormat.OmegaMajor,
    OmegaFormat.Green,
    OmegaFormat.OmegaMinor,
    OmegaFormat.Blue,
  ];
}

/**
 * Get format from harmonic index (0-4)
 */
export function omegaFromHarmonic(h: number): OmegaFormat | undefined {
  if (h >= 0 && h <= 4 && Number.isInteger(h)) {
    return h as OmegaFormat;
  }
  return undefined;
}

/**
 * Get harmonic index (0-4) from format
 */
export function harmonicFromOmega(fmt: OmegaFormat): number {
  return fmt;
}

/** Conversion factors between formats */
const CONVERSION_FACTORS: Record<OmegaFormat, Record<OmegaFormat, number>> = {
  [OmegaFormat.Red]: {
    [OmegaFormat.Red]: 1.0,
    [OmegaFormat.OmegaMajor]: 0.994718414,
    [OmegaFormat.Green]: 1.000351482,
    [OmegaFormat.OmegaMinor]: 1.006016451,
    [OmegaFormat.Blue]: 1.000703088,
  },
  [OmegaFormat.OmegaMajor]: {
    [OmegaFormat.Red]: 1.005309630,
    [OmegaFormat.OmegaMajor]: 1.0,
    [OmegaFormat.Green]: 1.005662978,     // Ω
    [OmegaFormat.OmegaMinor]: 1.011358026,
    [OmegaFormat.Blue]: 1.006016451,
  },
  [OmegaFormat.Green]: {
    [OmegaFormat.Red]: 0.999648641,
    [OmegaFormat.OmegaMajor]: 0.994368911,     // 1/Ω
    [OmegaFormat.Green]: 1.0,
    [OmegaFormat.OmegaMinor]: 1.005662978,     // Ω
    [OmegaFormat.Blue]: 1.000351482,
  },
  [OmegaFormat.OmegaMinor]: {
    [OmegaFormat.Red]: 0.994019530,
    [OmegaFormat.OmegaMajor]: 0.988769530,
    [OmegaFormat.Green]: 0.994368911,     // 1/Ω
    [OmegaFormat.OmegaMinor]: 1.0,
    [OmegaFormat.Blue]: 0.994718414,
  },
  [OmegaFormat.Blue]: {
    [OmegaFormat.Red]: 0.999297406,
    [OmegaFormat.OmegaMajor]: 0.994019530,
    [OmegaFormat.Green]: 0.999648641,
    [OmegaFormat.OmegaMinor]: 1.005309630,
    [OmegaFormat.Blue]: 1.0,
  },
};

/**
 * Convert a value between two Omega formats
 */
export function convertOmega(from: OmegaFormat, to: OmegaFormat, x: number): number {
  return x * CONVERSION_FACTORS[from][to];
}

/** Green to Omega Major: x / Ω */
export function greenToOmegaMajor(x: number): number {
  return convertOmega(OmegaFormat.Green, OmegaFormat.OmegaMajor, x);
}

/** Omega Major to Green: x × Ω */
export function omegaMajorToGreen(x: number): number {
  return convertOmega(OmegaFormat.OmegaMajor, OmegaFormat.Green, x);
}

/** Green to Omega Minor: x × Ω */
export function greenToOmegaMinor(x: number): number {
  return convertOmega(OmegaFormat.Green, OmegaFormat.OmegaMinor, x);
}

/** Omega Minor to Green: x / Ω */
export function omegaMinorToGreen(x: number): number {
  return convertOmega(OmegaFormat.OmegaMinor, OmegaFormat.Green, x);
}

/** Red to Blue */
export function redToBlue(x: number): number {
  return convertOmega(OmegaFormat.Red, OmegaFormat.Blue, x);
}

/** Blue to Red */
export function blueToRed(x: number): number {
  return convertOmega(OmegaFormat.Blue, OmegaFormat.Red, x);
}

/** Tolerance for roundtrip conversion verification */
export const ROUNDTRIP_TOLERANCE = 1e-10;

/**
 * Verify Invariant C1: roundtrip conversions preserve value
 */
export function verifyOmegaRoundtrip(from: OmegaFormat, to: OmegaFormat, x: number): boolean {
  const roundtrip = convertOmega(to, from, convertOmega(from, to, x));
  return Math.abs(roundtrip - x) < ROUNDTRIP_TOLERANCE;
}

/**
 * Verify all omega roundtrips for a value
 */
export function verifyAllOmegaRoundtrips(x: number): boolean {
  return allOmegaFormats().every((from) =>
    allOmegaFormats().every((to) => verifyOmegaRoundtrip(from, to, x))
  );
}

/**
 * Verify Invariant R4: Omega format index ∈ {0, 1, 2, 3, 4}
 */
export function verifyOmegaRange(): boolean {
  return allOmegaFormats().every((fmt) => harmonicFromOmega(fmt) <= 4);
}
