/**
 * θ/φ/r coordinate functions for Ra System dimensional mapping.
 *
 * Coordinate transforms:
 * - θ (theta): Semantic sector ← 27 Repitans
 * - φ (phi): Access sensitivity ← 6 RACs
 * - h (harmonic): Coherence depth ← 5 Omega formats
 * - r (radius): Emergence intensity ← Ankh-normalized scalar
 */

import { ANKH } from './constants.js';
import { Repitan, repitanFromTheta } from './repitans.js';
import { RacLevel } from './rac.js';
import { OmegaFormat, harmonicFromOmega, omegaFromHarmonic } from './omega.js';

/**
 * A complete Ra coordinate in 4-dimensional space
 */
export interface RaCoordinate {
  /** Semantic sector (1-27) */
  theta: Repitan;
  /** Access sensitivity (RAC1-RAC6) */
  phi: RacLevel;
  /** Coherence depth (Red-Blue) */
  harmonic: OmegaFormat;
  /** Ankh-normalized intensity [0,1] */
  radius: number;
}

/**
 * Create a validated RaCoordinate
 * @returns RaCoordinate or undefined if radius is invalid
 */
export function createCoordinate(
  theta: Repitan,
  phi: RacLevel,
  harmonic: OmegaFormat,
  radius: number
): RaCoordinate | undefined {
  if (radius >= 0 && radius <= 1) {
    return { theta, phi, harmonic, radius };
  }
  return undefined;
}

/**
 * Check if a coordinate is valid
 */
export function isCoordinateValid(coord: RaCoordinate): boolean {
  return coord.radius >= 0 && coord.radius <= 1;
}

/**
 * Convert Repitan to theta angle in degrees (0-360)
 */
export function thetaFromRepitan(r: Repitan): number {
  return r.theta;
}

/** Phi values for each RAC level (0-255 encoded) */
const PHI_VALUES: Record<RacLevel, number> = {
  [RacLevel.RAC1]: 0,    // Least restrictive
  [RacLevel.RAC2]: 43,
  [RacLevel.RAC3]: 85,
  [RacLevel.RAC4]: 128,
  [RacLevel.RAC5]: 170,
  [RacLevel.RAC6]: 255,  // Most restrictive
};

/**
 * Convert RacLevel to phi value (0-255 encoded)
 */
export function phiFromRac(rac: RacLevel): number {
  return PHI_VALUES[rac];
}

/**
 * Convert phi value (0-255) to RacLevel
 */
export function racFromPhi(phi: number): RacLevel {
  if (phi < 22) return RacLevel.RAC1;
  if (phi < 64) return RacLevel.RAC2;
  if (phi < 107) return RacLevel.RAC3;
  if (phi < 149) return RacLevel.RAC4;
  if (phi < 213) return RacLevel.RAC5;
  return RacLevel.RAC6;
}

/**
 * Normalize a raw radius value to [0, 1] using Ankh
 * r_normalized = r_raw / Ankh
 */
export function normalizeRadius(r: number): number {
  return Math.max(0, Math.min(1, r / ANKH));
}

/**
 * Denormalize a radius value from [0, 1] to raw scale
 * r_raw = r_normalized × Ankh
 */
export function denormalizeRadius(r: number): number {
  return r * ANKH;
}

/**
 * Calculate weighted distance between two coordinates
 * Returns value in [0, 1] where 0 = identical, 1 = maximally different
 */
export function coordinateDistance(c1: RaCoordinate, c2: RaCoordinate): number {
  const thetaDist = c1.theta.distance(c2.theta) / 13.5;
  const phiDist = Math.abs(phiFromRac(c1.phi) - phiFromRac(c2.phi)) / 255;
  const hDist = Math.abs(harmonicFromOmega(c1.harmonic) - harmonicFromOmega(c2.harmonic)) / 4;
  const rDist = Math.abs(c1.radius - c2.radius);

  // Weighted average (from spec: w_θ=0.3, w_φ=0.4, w_h=0.2, w_r=0.1)
  return 0.3 * thetaDist + 0.4 * phiDist + 0.2 * hDist + 0.1 * rDist;
}

/**
 * Verify Invariant O4: Omega format indices are 0-4
 */
export function verifyOmegaIndices(): boolean {
  return harmonicFromOmega(OmegaFormat.Red) === 0 &&
         harmonicFromOmega(OmegaFormat.Blue) === 4;
}
