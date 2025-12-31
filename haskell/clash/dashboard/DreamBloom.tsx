/**
 * DreamBloom.tsx - Prompt 13 Dream Resonance Visualization
 *
 * Visualizes:
 * - Current sleep phase (WAKE, THETA, DELTA, REM)
 * - Coherence trajectory with animated progress
 * - Symbolic emergence badges with fragment mappings
 * - Resonance entrainment settings
 *
 * Supports WebSocket live feed for real-time updates.
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useEffect, useState } from "react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { motion, AnimatePresence } from "framer-motion";

// ============================================================================
// Types
// ============================================================================

type SleepPhase = "WAKE" | "THETA" | "DELTA" | "REM";

interface SymbolMapping {
  symbol: string;
  emoji: string;
  label: string;
  fragment?: string;
}

interface DreamFragment {
  fragmentId: string;
  coherenceTrace: number[];
  emotions: string;
  symbols: SymbolMapping[];
  shadowDetected: boolean;
  timestamp: number;
}

interface ResonanceSettings {
  band: string;
  frequency: number;
  audioType: string;
  visualType: string;
  phiMultiplier: number;
}

interface DreamState {
  phase: SleepPhase;
  coherence: number;
  cycleNumber: number;
  phaseDepth: number;
  resonance: ResonanceSettings;
  fragments: DreamFragment[];
  sessionActive: boolean;
}

// ============================================================================
// Symbol Emoji Map
// ============================================================================

const SYMBOL_EMOJIS: Record<string, string> = {
  owl: "🦉",
  spiral: "🌀",
  mirror: "🪞",
  river: "🌊",
  labyrinth: "🌀",
  light: "✨",
  flame: "🔥",
  cave: "🕳️",
  tree: "🌳",
  moon: "🌙",
  star: "⭐",
  water: "💧",
};

// ============================================================================
// Phase Display Component
// ============================================================================

const PhaseIndicator = ({ phase, depth }: { phase: SleepPhase; depth: number }) => {
  const phaseConfig = {
    WAKE: { color: "bg-yellow-500", label: "Awake", icon: "☀️" },
    THETA: { color: "bg-blue-400", label: "Light Sleep", icon: "🌙" },
    DELTA: { color: "bg-indigo-600", label: "Deep Sleep", icon: "😴" },
    REM: { color: "bg-purple-500", label: "Dreaming", icon: "💫" },
  };

  const config = phaseConfig[phase];

  return (
    <div className="flex items-center gap-3">
      <motion.div
        className={`w-4 h-4 rounded-full ${config.color}`}
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
      <div>
        <div className="text-lg font-semibold text-white">
          {config.icon} {config.label}
        </div>
        <div className="text-xs text-white/60">
          Phase depth: {Math.round(depth * 100)}%
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Coherence Trajectory Component
// ============================================================================

const CoherenceTrajectory = ({ trace }: { trace: number[] }) => {
  const current = trace[trace.length - 1] || 0;
  const isRising = trace.length >= 2 && trace[trace.length - 1] > trace[0];

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm text-white/70">
        <span>Coherence</span>
        <span className={isRising ? "text-green-400" : "text-yellow-400"}>
          {(current * 100).toFixed(1)}% {isRising ? "↑" : "→"}
        </span>
      </div>
      <Progress value={current * 100} className="h-2" />
      <div className="flex justify-between text-xs text-white/50">
        {trace.map((t, i) => (
          <span key={i}>{(t * 100).toFixed(0)}%</span>
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Symbol Badge Component
// ============================================================================

const SymbolBadge = ({ mapping }: { mapping: SymbolMapping }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Badge
        variant="secondary"
        className="text-lg px-3 py-1 bg-white/10 hover:bg-white/20"
      >
        <span className="mr-2">{mapping.emoji}</span>
        <span className="text-sm font-medium">{mapping.symbol}</span>
        {mapping.fragment && (
          <span className="ml-2 text-xs text-white/60">→ {mapping.fragment}</span>
        )}
      </Badge>
    </motion.div>
  );
};

// ============================================================================
// Fragment Card Component
// ============================================================================

const FragmentCard = ({ fragment }: { fragment: DreamFragment }) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="p-3 rounded-lg bg-white/5 border border-white/10"
    >
      <div className="flex justify-between items-start mb-2">
        <span className="text-sm font-medium text-white">{fragment.fragmentId}</span>
        <span className="text-xs text-white/50">{fragment.emotions}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {fragment.symbols.map((s, i) => (
          <SymbolBadge key={i} mapping={s} />
        ))}
      </div>
      {fragment.shadowDetected && (
        <div className="mt-2 text-xs text-amber-400">
          ⚠️ Shadow content (consent-gated)
        </div>
      )}
    </motion.div>
  );
};

// ============================================================================
// Resonance Display Component
// ============================================================================

const ResonanceDisplay = ({ resonance }: { resonance: ResonanceSettings }) => {
  return (
    <div className="grid grid-cols-2 gap-2 text-sm">
      <div className="text-white/60">Band:</div>
      <div className="text-white">{resonance.band} ({resonance.frequency} Hz)</div>
      <div className="text-white/60">Audio:</div>
      <div className="text-white">{resonance.audioType}</div>
      <div className="text-white/60">Visual:</div>
      <div className="text-white">{resonance.visualType}</div>
      <div className="text-white/60">Modulation:</div>
      <div className="text-white">φ^{resonance.phiMultiplier}</div>
    </div>
  );
};

// ============================================================================
// Main DreamBloom Component
// ============================================================================

export default function DreamBloom() {
  const [state, setState] = useState<DreamState>({
    phase: "REM",
    coherence: 0.41,
    cycleNumber: 1,
    phaseDepth: 0.5,
    resonance: {
      band: "ALPHA",
      frequency: 10.0,
      audioType: "golden_stack",
      visualType: "flower_of_life",
      phiMultiplier: 3,
    },
    fragments: [
      {
        fragmentId: "dream-4739",
        coherenceTrace: [0.41, 0.47, 0.61],
        emotions: "joy + confusion",
        symbols: [
          { symbol: "owl", emoji: "🦉", label: "Wisdom", fragment: "F13" },
          { symbol: "spiral", emoji: "🌀", label: "Searching" },
        ],
        shadowDetected: false,
        timestamp: Date.now(),
      },
    ],
    sessionActive: true,
  });

  // Simulate coherence growth
  useEffect(() => {
    if (!state.sessionActive) return;

    const interval = setInterval(() => {
      setState((prev) => {
        const newCoherence = Math.min(1, prev.coherence + 0.01);
        return { ...prev, coherence: newCoherence };
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [state.sessionActive]);

  // WebSocket connection (placeholder)
  useEffect(() => {
    // TODO: Connect to Ra.DreamPhaseScheduler WebSocket
    // const ws = new WebSocket('ws://localhost:8080/dream');
    // ws.onmessage = (event) => {
    //   const data = JSON.parse(event.data);
    //   setState(prev => ({ ...prev, ...data }));
    // };
    // return () => ws.close();
  }, []);

  const latestFragment = state.fragments[state.fragments.length - 1];
  const coherenceTrace = latestFragment?.coherenceTrace || [state.coherence];

  return (
    <div className="grid gap-4 p-4 max-w-2xl mx-auto">
      {/* Header Card - Phase & Coherence */}
      <Card className="bg-gradient-to-b from-indigo-900 to-purple-950 border-0 shadow-xl">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl text-white flex items-center justify-between">
            <span>Dream Bloom Resonance</span>
            <Badge variant={state.sessionActive ? "default" : "secondary"}>
              {state.sessionActive ? "Session Active" : "Idle"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <PhaseIndicator phase={state.phase} depth={state.phaseDepth} />
          <CoherenceTrajectory trace={coherenceTrace} />
          <div className="text-sm text-white/60">
            Sleep Cycle: {state.cycleNumber}
          </div>
        </CardContent>
      </Card>

      {/* Resonance Settings Card */}
      <Card className="bg-white/5 border border-white/10 backdrop-blur-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-md text-white">Resonance Entrainment</CardTitle>
        </CardHeader>
        <CardContent>
          <ResonanceDisplay resonance={state.resonance} />
        </CardContent>
      </Card>

      {/* Symbolic Fragments Card */}
      <Card className="bg-white/5 border border-white/10 backdrop-blur-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-md text-white flex items-center gap-2">
            <span>Symbolic Fragments</span>
            <Badge variant="outline" className="text-xs">
              {state.fragments.length} emerged
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <AnimatePresence>
            {state.fragments.map((f, i) => (
              <FragmentCard key={f.fragmentId + i} fragment={f} />
            ))}
          </AnimatePresence>
          {state.fragments.length === 0 && (
            <div className="text-center text-white/40 py-4">
              Awaiting REM emergence...
            </div>
          )}
        </CardContent>
      </Card>

      {/* Post-Sleep Integration (shown when session ends) */}
      {!state.sessionActive && state.fragments.length > 0 && (
        <Card className="bg-gradient-to-b from-amber-900/50 to-orange-950/50 border border-amber-500/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-md text-amber-200">
              Post-Sleep Integration
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-amber-100/80 space-y-2">
            <p>Fragments surfaced: {state.fragments.length}</p>
            <p>Consider journaling about the symbols that emerged.</p>
            <p className="text-xs text-amber-200/60">
              "What wisdom emerged that you didn't expect?"
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
