/**
 * AccessResult type + gating logic from spec Section 4.
 *
 * Access gating determines whether a fragment/signal can emerge based on
 * coherence and consent levels.
 *
 * From Section 4 of ra_integration_spec.md:
 *
 * ```
 * AccessLevel(user_coherence, fragment_rac) → {FullAccess, PartialAccess(α), Blocked}
 *
 * threshold(R_f) = RAC(R_f) / RAC₁
 * C_floor = φ_green / Ankh ≈ 0.3183
 * C_ceiling = 1.0
 * ```
 */

import { ANKH, GREEN_PHI } from './constants.js';
import { RacLevel, racValueNormalized } from './rac.js';
import { Repitan } from './repitans.js';

/** Coherence floor: φ_green / Ankh ≈ 0.3183 */
export const COHERENCE_FLOOR = GREEN_PHI / ANKH;

/** Coherence ceiling: 1.0 */
export const COHERENCE_CEILING = 1.0;

/**
 * Result of access gating check
 */
export type AccessResult =
  | { type: 'FullAccess' }
  | { type: 'PartialAccess'; alpha: number }
  | { type: 'Blocked' };

/** Create FullAccess result */
export function fullAccess(): AccessResult {
  return { type: 'FullAccess' };
}

/** Create PartialAccess result */
export function partialAccess(alpha: number): AccessResult {
  return { type: 'PartialAccess', alpha: Math.max(0, Math.min(1, alpha)) };
}

/** Create Blocked result */
export function blocked(): AccessResult {
  return { type: 'Blocked' };
}

/** Check if result is FullAccess */
export function isFullAccess(result: AccessResult): boolean {
  return result.type === 'FullAccess';
}

/** Check if result is PartialAccess */
export function isPartialAccess(result: AccessResult): boolean {
  return result.type === 'PartialAccess';
}

/** Check if result is Blocked */
export function isBlocked(result: AccessResult): boolean {
  return result.type === 'Blocked';
}

/** Extract alpha value: 1.0 for FullAccess, α for PartialAccess, 0.0 for Blocked */
export function accessAlpha(result: AccessResult): number {
  switch (result.type) {
    case 'FullAccess':
      return 1.0;
    case 'PartialAccess':
      return result.alpha;
    case 'Blocked':
      return 0.0;
  }
}

/**
 * Get threshold for a RAC level (normalized to RAC1)
 */
export function racThreshold(rac: RacLevel): number {
  return racValueNormalized(rac);
}

/**
 * Core gating function from spec Section 4.1
 * Determines access level based on user coherence and fragment RAC requirement
 */
export function accessLevel(userCoherence: number, fragmentRac: RacLevel): AccessResult {
  const threshold = racThreshold(fragmentRac);

  if (userCoherence >= threshold) {
    return fullAccess();
  } else if (userCoherence >= COHERENCE_FLOOR) {
    const alpha = (userCoherence - COHERENCE_FLOOR) / (threshold - COHERENCE_FLOOR);
    return partialAccess(alpha);
  } else {
    return blocked();
  }
}

/**
 * Simple check if access is allowed (not Blocked)
 */
export function canAccess(coherence: number, rac: RacLevel): boolean {
  return !isBlocked(accessLevel(coherence, rac));
}

/**
 * Calculate effective coherence given access result
 */
export function effectiveCoherence(result: AccessResult): number {
  return accessAlpha(result);
}

/**
 * Calculate partial emergence within a Repitan band
 * From spec Section 4.4
 */
export function partialEmergence(currentBand: Repitan, alpha: number): number {
  const bandLow = currentBand.value;
  const bandHigh = currentBand.next().value;
  return bandLow + alpha * (bandHigh - bandLow);
}

/**
 * Weights for resonance score calculation
 */
export interface ResonanceWeights {
  /** θ alignment weight */
  theta: number;
  /** φ access weight */
  phi: number;
  /** h harmonic match weight */
  harmonic: number;
  /** r intensity weight */
  radius: number;
}

/**
 * Default weights from spec Section 5.3
 * w_θ = 0.3, w_φ = 0.4, w_h = 0.2, w_r = 0.1
 */
export const DEFAULT_WEIGHTS: ResonanceWeights = {
  theta: 0.3,
  phi: 0.4,
  harmonic: 0.2,
  radius: 0.1,
};

/**
 * Calculate composite resonance score
 * resonance = w_θ × θ_match + w_φ × φ_access + w_h × h_match + w_r × r_intensity
 */
export function resonanceScore(
  weights: ResonanceWeights,
  thetaMatch: number,
  phiAccess: number,
  harmonicMatch: number,
  intensity: number
): number {
  return (
    weights.theta * thetaMatch +
    weights.phi * phiAccess +
    weights.harmonic * harmonicMatch +
    weights.radius * intensity
  );
}

/**
 * Verify Invariant R3: Coherence bounds are [0, 1]
 */
export function verifyCoherenceBounds(): boolean {
  return COHERENCE_FLOOR >= 0 && COHERENCE_FLOOR < 1 && COHERENCE_CEILING === 1.0;
}

/**
 * Verify coherence floor calculation: φ_green / Ankh
 */
export function verifyCoherenceFloor(): boolean {
  return Math.abs(COHERENCE_FLOOR - GREEN_PHI / ANKH) < 1e-10;
}
