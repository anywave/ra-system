/**
 * Ra System mathematical constants - TypeScript bindings
 *
 * This module provides type-safe access to the Ra System constants from
 * "The Rods of Amon Ra" by Wesley H. Bateman.
 *
 * @example
 * ```typescript
 * import { ANKH, Repitan, RacLevel, OmegaFormat, accessLevel } from '@anywave/ra-system';
 *
 * // Check access at 80% coherence for RAC1
 * const result = accessLevel(0.8, RacLevel.RAC1);
 * console.log(result.isFullAccess()); // true
 *
 * // Create a validated Repitan
 * const r = Repitan.create(9);
 * console.log(r?.value); // 0.333...
 * ```
 *
 * @packageDocumentation
 */

export * from './constants.js';
export * from './repitans.js';
export * from './rac.js';
export * from './omega.js';
export * from './spherical.js';
export * from './gates.js';
