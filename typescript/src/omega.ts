/**
 * OmegaFormat enum with conversion functions.
 *
 * Bateman's five "windows" for one quantity (vol2a prologue; TR-302, TR-312).
 * In magnitude a value runs Omega Major < Red < Green < Blue < Omega Minor, with
 * steps Omega/r (= 84823/84375), r, r, Omega/r (TR-300, TR-304, TR-312). The enum
 * order follows the printed column layout, not the magnitude - do not sort on it.
 *
 * Each factor applies to the power d, the number of lengths in play: 1 for a
 * length, 2 for an area, 3 for a volume (volume_1 p.0117; TR-294, TR-301, TR-309).
 */

import { COLOUR_RATIO, OMEGA, OMEGA_OVER_R } from './constants.js';

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

/** Position of each window on the ladder, relative to Omega Major (TR-312) */
const LADDER_POSITION: Record<OmegaFormat, number> = {
  [OmegaFormat.OmegaMajor]: 1.0,
  [OmegaFormat.Red]: OMEGA_OVER_R,
  [OmegaFormat.Green]: OMEGA,
  [OmegaFormat.Blue]: OMEGA * COLOUR_RATIO,
  [OmegaFormat.OmegaMinor]: OMEGA * OMEGA,
};

/**
 * Convert a value between two Omega formats.
 *
 * @param degree number of lengths in play: 1 length, 2 area, 3 volume, 0 none (TR-294, TR-301)
 */
export function convertOmega(from: OmegaFormat, to: OmegaFormat, x: number, degree = 1): number {
  return x * (LADDER_POSITION[to] / LADDER_POSITION[from]) ** degree;
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

/** Tolerance for roundtrip conversion verification (the ladder is exact up to float error) */
export const ROUNDTRIP_TOLERANCE = 1e-9;

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
