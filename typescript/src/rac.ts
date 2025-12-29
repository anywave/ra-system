/**
 * RacLevel enum (RAC1..RAC6) with validation.
 *
 * Resonant Access Constants (RACs) represent access sensitivity levels.
 * RAC1 is the highest (least restrictive), RAC6 is the lowest (most restrictive).
 *
 * Invariant O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0
 */

import { ANKH } from './constants.js';

/**
 * The six Resonant Access Constant levels
 */
export enum RacLevel {
  RAC1 = 1,
  RAC2 = 2,
  RAC3 = 3,
  RAC4 = 4,
  RAC5 = 5,
  RAC6 = 6,
}

/** RAC values in Red Rams */
const RAC_VALUES: Record<RacLevel, number> = {
  [RacLevel.RAC1]: 0.6361725,      // Ankh / 8
  [RacLevel.RAC2]: 0.628318519,    // 2π/10 approximation
  [RacLevel.RAC3]: 0.57255525,     // φ × Hunab × 1/3
  [RacLevel.RAC4]: 0.523598765,    // π/6 approximation
  [RacLevel.RAC5]: 0.4580442,      // Ankh × 9 / 100
  [RacLevel.RAC6]: 0.3998594565,   // RAC lattice terminus
};

/** RAC values in meters */
const RAC_VALUES_METERS: Record<RacLevel, number> = {
  [RacLevel.RAC1]: 0.639591666,
  [RacLevel.RAC2]: 0.631695473,
  [RacLevel.RAC3]: 0.5756325,
  [RacLevel.RAC4]: 0.526412894,
  [RacLevel.RAC5]: 0.460506,
  [RacLevel.RAC6]: 0.4020085371,
};

/** Pyramid divisions for each RAC */
const PYRAMID_DIVISIONS: Record<RacLevel, number> = {
  [RacLevel.RAC1]: 360.0,      // Circle degrees
  [RacLevel.RAC2]: 364.5,      // Balmer constant
  [RacLevel.RAC3]: 400.0,
  [RacLevel.RAC4]: 437.4,      // 27 × φ_green
  [RacLevel.RAC5]: 500.0,
  [RacLevel.RAC6]: 572.756493, // 1.125 × Green Ankh
};

/**
 * Get all RAC levels in order
 */
export function allRacLevels(): RacLevel[] {
  return [
    RacLevel.RAC1,
    RacLevel.RAC2,
    RacLevel.RAC3,
    RacLevel.RAC4,
    RacLevel.RAC5,
    RacLevel.RAC6,
  ];
}

/**
 * Get RacLevel from numeric level (1-6)
 */
export function racFromLevel(n: number): RacLevel | undefined {
  if (n >= 1 && n <= 6 && Number.isInteger(n)) {
    return n as RacLevel;
  }
  return undefined;
}

/**
 * Get the RAC value in Red Rams for a given level
 */
export function racValue(level: RacLevel): number {
  return RAC_VALUES[level];
}

/**
 * Get the RAC value in meters for a given level
 */
export function racValueMeters(level: RacLevel): number {
  return RAC_VALUES_METERS[level];
}

/**
 * Get the RAC value normalized to RAC1 (for threshold calculations)
 * RAC1 normalized = 1.0
 */
export function racValueNormalized(level: RacLevel): number {
  return racValue(level) / racValue(RacLevel.RAC1);
}

/**
 * Get pyramid division for a RAC level
 */
export function pyramidDivision(level: RacLevel): number {
  return PYRAMID_DIVISIONS[level];
}

/**
 * Check if a value is valid for a RAC (0 < x < 1)
 */
export function isValidRacValue(x: number): boolean {
  return x > 0 && x < 1;
}

/**
 * Verify Invariant O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0
 */
export function verifyRacOrdering(): boolean {
  const levels = allRacLevels();
  const values = levels.map(racValue);

  // Check descending order
  for (let i = 0; i < values.length - 1; i++) {
    if (values[i] <= values[i + 1]) return false;
  }

  // Check all positive
  return values.every((v) => v > 0);
}

/**
 * Verify Invariant R1: 0 < RAC(i) < 1 for all i ∈ [1, 6]
 */
export function verifyRacRange(): boolean {
  return allRacLevels().every((level) => isValidRacValue(racValue(level)));
}

/**
 * Verify Invariant I2: RAC1 = Ankh / 8
 */
export function verifyRac1Derivation(): boolean {
  const computed = ANKH / 8;
  return Math.abs(racValue(RacLevel.RAC1) - computed) < 0.0001;
}
