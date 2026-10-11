/**
 * Regression tests for the TR (Trellis) Bateman extractions, Aug-Sep 2026.
 * Bateman's values as printed and verified, not physics claims. See
 * python/tests/test_tr_extractions.py for the full set and citations.
 */

import { describe, it, expect } from 'vitest';
import {
  ANKH,
  BLUE_PHI,
  BLUE_PI,
  COLOUR_RATIO,
  FINE_STRUCTURE,
  GREEN_HUNAB,
  GREEN_PHI,
  GREEN_PI,
  H_BAR,
  HUNAB,
  OMEGA,
  OMEGA_MAJOR_PHI,
  OMEGA_OVER_R,
  RA_RYDBERG,
  RED_PHI,
  RED_PI,
} from '../src/constants.js';
import { COHERENCE_FLOOR } from '../src/gates.js';

const close = (a: number, b: number, tol: number): boolean => Math.abs(a - b) < tol;

describe('TR extractions', () => {
  it('Red Pi is 84823/27000 (TR-291, TR-349)', () => {
    expect(close(RED_PI, 84823 / 27000, 1e-15)).toBe(true);
  });

  it('Green and Blue Pi follow the r-ladder (TR-300, TR-308)', () => {
    expect(close(GREEN_PI, (20 / 9) * Math.SQRT2, 1e-15)).toBe(true);
    expect(close(BLUE_PI, 3.143801406, 2e-9)).toBe(true);
    expect(close(GREEN_PI / RED_PI, COLOUR_RATIO, 1e-14)).toBe(true);
  });

  it('Hunab and H-Bar (TR-291, TR-304)', () => {
    expect(close(HUNAB, 84823 / 80000, 1e-15)).toBe(true);
    expect(close(GREEN_HUNAB, 0.75 * Math.SQRT2, 1e-15)).toBe(true);
    expect(H_BAR).toBe(135 / 128);
    expect(close(GREEN_HUNAB / OMEGA, H_BAR, 1e-12)).toBe(true);
  });

  it('Omega, Rydberg, f.s.Delta (TR-304, TR-320, TR-323)', () => {
    expect(close(OMEGA, (32 * Math.SQRT2) / 45, 1e-15)).toBe(true);
    expect(close(OMEGA_OVER_R, 84823 / 84375, 1e-15)).toBe(true);
    expect(close(RA_RYDBERG, 800 / 729, 1e-15)).toBe(true);
    expect(close(FINE_STRUCTURE, 1 / 729, 1e-16)).toBe(true);
  });

  it('Phi follows the supplement erratum (FP-1399, TR-312, TR-316)', () => {
    expect(OMEGA_MAJOR_PHI).toBe(1.62);
    expect(RED_PHI).toBe(1.6286016);
    expect(GREEN_PHI).toBe(1.629174025);
    expect(BLUE_PHI).toBe(1.62974665);
    expect(close(OMEGA_MAJOR_PHI * OMEGA_OVER_R, RED_PHI, 1e-12)).toBe(true);
  });

  it('Ankh = Red Pi x 1.62 and the gate floor is unchanged (TR-298, TR-329)', () => {
    expect(close(ANKH, (3 * 84823) / 50000, 1e-15)).toBe(true);
    expect(close(RED_PI * OMEGA_MAJOR_PHI, ANKH, 1e-12)).toBe(true);
    expect(close(COHERENCE_FLOOR, 1.62 / 5.08938, 1e-15)).toBe(true);
  });
});
