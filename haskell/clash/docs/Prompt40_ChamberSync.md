# Prompt 40: Multi-Chamber Synchronization

## Purpose

Synchronizes multiple chamber nodes to maintain phase coherence across a distributed field synthesis network. Tracks sync state, phase drift, and quality metrics.

## Module

`RaChamberSync.hs`

## Input Format

```haskell
data ChamberState = Idle | Spinning | Stabilizing | Emanating

-- N-element vector of chamber states
chamberStates :: Vec n ChamberState
```

### JSON Simulation Input

```json
{
  "chambers": ["Idle", "Spinning", "Stabilizing", "Emanating"]
}
```

## Output Format

```haskell
data SyncState = Desync | Aligning | Locked | Drifting

data SyncOutput = SyncOutput
  { syncState   :: SyncState     -- Current sync status
  , syncPulse   :: Bool          -- Alignment trigger pulse
  , syncQuality :: Unsigned 8    -- Quality 0-255 (255 = perfect)
  , phaseDrift  :: Unsigned 8    -- Cycles since last full sync
  }
```

### JSON Simulation Output

```json
{
  "syncState": "Locked",
  "syncPulse": false,
  "syncQuality": 255,
  "phaseDrift": 0
}
```

## Sync States

| State | Description | Trigger Condition |
|-------|-------------|-------------------|
| Desync | Chambers out of phase | No majority agreement |
| Aligning | Active phase correction | Majority agree, not all |
| Locked | All chambers synchronized | All same state, drift < 8 |
| Drifting | Minor drift detected | All same state, drift >= 8 |

## Trigger Logic

### Sync State Determination

```
if all_same_state AND drift < 8:
    syncState = Locked
elif all_same_state AND drift >= 8:
    syncState = Drifting
elif majority_agree:
    syncState = Aligning
else:
    syncState = Desync
```

### Phase Drift Tracking

```
drift = cycles since last Locked state
maxDrift = 32 cycles before quality degrades
```

### Quality Calculation

```
if Locked:    quality = 255
if Drifting:  quality = 255 - (drift * 4)
if Aligning:  quality = 128
if Desync:    quality = 0
```

## Example Scenarios

| Chambers | syncState | syncQuality | phaseDrift | Notes |
|----------|-----------|-------------|------------|-------|
| [E,E,E,E] | Locked | 255 | 0 | All emanating |
| [S,S,S,I] | Aligning | 128 | - | 3/4 agreement |
| [I,S,E,I] | Desync | 0 | - | No majority |
| [E,E,E,E] (d=10) | Drifting | 215 | 10 | Drift detected |

## Tokenomics

| Sync State | Token Cost | Reason |
|------------|------------|--------|
| Locked | 0.5 units | Minimal processing |
| Drifting | 1.0 units | Drift compensation |
| Aligning | 1.5 units | Active correction |
| Desync | 2.0 units | Full resync attempt |

## Hardware Notes

### FPGA Synthesis

- **Logic Cells**: ~80 LUTs (scales with N)
- **Registers**: 8 × N (chamber states) + 16 (output)
- **Clock**: Single cycle combinational
- **Latency**: 1 clock cycle

### Resource Usage

```
Module: chamberSyncTop
- Comparators: N × (N-1) / 2 (pairwise comparison)
- Counter: 8-bit (drift tracking)
- MUX: 4-way (state selection)
```

### VCD Waveform Generation

```bash
clash --vcd RaChamberSync.hs
gtkwave testBench.vcd
```

## Dashboard Panel

**Location**: Phase 1 (integrated with Chamber State Monitor)

**Features**:
- Multi-chamber state visualization
- Sync state indicator
- Quality bar (0-100%)
- Drift counter display

## Testbench

```haskell
testInputs :: Vec 5 (Vec 4 ChamberState)
testInputs =
  (Idle :> Idle :> Idle :> Idle :> Nil) :>         -- Locked
  (Spinning :> Spinning :> Idle :> Spinning :> Nil) :>  -- Aligning
  (Idle :> Spinning :> Stabilizing :> Emanating :> Nil) :>  -- Desync
  (Emanating :> Emanating :> Emanating :> Emanating :> Nil) :>  -- Locked
  (Stabilizing :> Stabilizing :> Spinning :> Stabilizing :> Nil) :>  -- Aligning
  Nil
```

## Integration

- **Upstream**: RaFieldSynthesisNode (chamber states)
- **Downstream**: RaVisualizerShell (visual feedback), timing system
- **Related**: RaChamberMorphology (form changes), RaConsentTransformer (quorum)

## Distributed Systems Context

Multi-chamber synchronization is essential for:

1. **Coherent field synthesis**: All chambers must be in phase
2. **Visual consistency**: Users see unified field state
3. **Energy efficiency**: Locked state uses minimal resources
4. **Fault tolerance**: Drifting state allows graceful degradation

The sync pulse output can trigger re-alignment procedures when drift exceeds acceptable thresholds.
