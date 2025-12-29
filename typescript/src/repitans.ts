/**
 * Repitan type with smart constructor validating range [1, 27].
 *
 * Repitans represent the 27 semantic sectors of the Ra System.
 * Each Repitan(n) = n/27 for n ∈ [1, 27].
 */

/**
 * Check if a value is a valid Repitan index
 */
export function isValidRepitanIndex(n: number): boolean {
  return Number.isInteger(n) && n >= 1 && n <= 27;
}

/**
 * A validated Repitan (semantic sector index)
 *
 * Invariants:
 * - Index is in range [1, 27]
 * - Value = index / 27
 * - O3: 0 < Repitan(n) ≤ 1 for all n
 */
export class Repitan {
  private readonly _index: number;

  private constructor(index: number) {
    this._index = index;
  }

  /**
   * Create a new Repitan with validation
   * @returns Repitan or undefined if invalid
   */
  static create(n: number): Repitan | undefined {
    if (isValidRepitanIndex(n)) {
      return new Repitan(n);
    }
    return undefined;
  }

  /** Get the index (1-27) */
  get index(): number {
    return this._index;
  }

  /** Get the Repitan value (n/27) - Invariant I4 */
  get value(): number {
    return this._index / 27;
  }

  /** Get theta angle in degrees (0-360) */
  get theta(): number {
    return this.value * 360;
  }

  /** Get theta angle in radians */
  get thetaRadians(): number {
    return this.value * 2 * Math.PI;
  }

  /** Get the next Repitan (wraps from 27 to 1) */
  next(): Repitan {
    return new Repitan(this._index === 27 ? 1 : this._index + 1);
  }

  /** Get the previous Repitan (wraps from 1 to 27) */
  prev(): Repitan {
    return new Repitan(this._index === 1 ? 27 : this._index - 1);
  }

  /** Calculate angular distance to another Repitan (max 13) */
  distance(other: Repitan): number {
    const d = Math.abs(this._index - other._index);
    return Math.min(d, 27 - d);
  }

  /** First Repitan: n=1, value=1/27 (Fine Structure root) */
  static readonly FIRST = new Repitan(1);

  /** Ninth Repitan: n=9, value=9/27=1/3 */
  static readonly NINTH = new Repitan(9);

  /** Unity Repitan: n=27, value=27/27=1 */
  static readonly UNITY = new Repitan(27);
}

/**
 * Convert theta angle (degrees) to nearest Repitan
 */
export function repitanFromTheta(theta: number): Repitan {
  const normalized = ((theta % 360) + 360) % 360 / 360;
  let n = Math.round(normalized * 27);
  n = Math.max(1, Math.min(27, n === 0 ? 27 : n));
  return Repitan.create(n)!;
}

/**
 * Get all 27 Repitans
 */
export function allRepitans(): Repitan[] {
  return Array.from({ length: 27 }, (_, i) => Repitan.create(i + 1)!);
}

/**
 * Verify Invariant I4: Repitan(n) = n/27 for all n ∈ [1, 27]
 */
export function verifyRepitanInvariant(): boolean {
  return allRepitans().every((r) => Math.abs(r.value - r.index / 27) < 1e-10);
}

/**
 * Verify Invariant O3: For all n: 0 < Repitan(n) ≤ 1
 */
export function verifyRepitanRangeInvariant(): boolean {
  return allRepitans().every((r) => r.value > 0 && r.value <= 1);
}
