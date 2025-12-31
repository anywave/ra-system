/**
 * Ra System - Scalar Resonance Biofeedback Dashboard
 * Prompt 10: Real-time visualization of coherence, chakra drift, and harmonic output
 */

import React, { useState, useEffect, useCallback } from 'react';

// Types matching Clash module
interface RGB {
  red: number;
  green: number;
  blue: number;
}

interface BiometricInput {
  hrvMs: number;
  eegAlphaUv: number;
  gsrUs: number;
  breathCpm: number;
}

interface CoherenceState {
  coherenceLevel: number;
  emotionalTension: number;
  chakraDrift: number[];
}

interface AudioOutput {
  primaryHz: number;
  secondaryHz: number;
  carrierHz: number;
}

interface VisualOutput {
  rgb: RGB;
  pattern: 'BREATH' | 'WAVE' | 'PULSE' | 'STATIC';
  intensity: number;
}

interface TactileOutput {
  pulseFreq: number;
  pulseDuty: number;
  pulseIntensity: number;
  active: boolean;
}

interface HarmonicOutput {
  audio: AudioOutput;
  visual: VisualOutput;
  tactile: TactileOutput;
  phiSync: boolean;
}

interface FeedbackState {
  prevCoherence: number;
  coherenceDelta: number;
  adaptationMode: 'REINFORCE' | 'ADJUST' | 'PEAK_PULSE' | 'STABILIZE' | 'EMERGENCY_STAB';
  cycleCount: number;
  dorTimer: number;
}

interface ResonanceData {
  currentPhase: 'IDLE' | 'BASELINE' | 'ALIGNMENT' | 'ENTRAINMENT' | 'INTEGRATION' | 'COMPLETE';
  coherence: CoherenceState;
  harmonic: HarmonicOutput;
  feedback: FeedbackState;
  safetyAlert: boolean;
  timestamp: number;
}

// Chakra configuration
const CHAKRAS = [
  { name: 'Root', color: '#FF0000', frequency: 396 },
  { name: 'Sacral', color: '#FF7F00', frequency: 417 },
  { name: 'Solar', color: '#FFFF00', frequency: 528 },
  { name: 'Heart', color: '#00FF00', frequency: 639 },
  { name: 'Throat', color: '#007FFF', frequency: 741 },
  { name: 'Third Eye', color: '#4B0082', frequency: 852 },
  { name: 'Crown', color: '#9400D3', frequency: 963 },
];

// Coherence bar component
const CoherenceBar: React.FC<{ value: number; label: string; color?: string }> = ({
  value,
  label,
  color = '#4CAF50'
}) => {
  const percentage = (value / 255) * 100;
  return (
    <div className="coherence-bar">
      <div className="bar-label">{label}</div>
      <div className="bar-container">
        <div
          className="bar-fill"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
      <div className="bar-value">{(value / 255).toFixed(3)}</div>
    </div>
  );
};

// Chakra drift visualization
const ChakraDriftPanel: React.FC<{ drift: number[]; dominantIndex: number }> = ({
  drift,
  dominantIndex
}) => (
  <div className="chakra-panel">
    <h3>Chakra Drift</h3>
    {CHAKRAS.map((chakra, idx) => {
      const driftValue = drift[idx] || 0;
      const normalizedDrift = Math.min(1, Math.max(0, (driftValue + 128) / 256));
      const isDominant = idx === dominantIndex;
      return (
        <div key={chakra.name} className={`chakra-row ${isDominant ? 'dominant' : ''}`}>
          <span className="chakra-name" style={{ color: chakra.color }}>
            {chakra.name}
          </span>
          <div className="drift-bar">
            <div
              className="drift-fill"
              style={{
                width: `${normalizedDrift * 100}%`,
                backgroundColor: chakra.color,
                opacity: isDominant ? 1 : 0.6
              }}
            />
          </div>
          <span className="drift-value">{driftValue.toFixed(0)}</span>
          {isDominant && <span className="dominant-marker">▶</span>}
        </div>
      );
    })}
  </div>
);

// Session phase indicator
const SessionPhaseIndicator: React.FC<{ phase: ResonanceData['currentPhase'] }> = ({ phase }) => {
  const phases = ['IDLE', 'BASELINE', 'ALIGNMENT', 'ENTRAINMENT', 'INTEGRATION', 'COMPLETE'];
  const currentIndex = phases.indexOf(phase);

  return (
    <div className="phase-indicator">
      <h3>Session Phase</h3>
      <div className="phase-timeline">
        {phases.map((p, idx) => (
          <div
            key={p}
            className={`phase-node ${idx === currentIndex ? 'active' : ''} ${idx < currentIndex ? 'completed' : ''}`}
          >
            <div className="phase-dot" />
            <span className="phase-label">{p}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Harmonic output display
const HarmonicDisplay: React.FC<{ harmonic: HarmonicOutput }> = ({ harmonic }) => {
  const rgbString = `rgb(${harmonic.visual.rgb.red}, ${harmonic.visual.rgb.green}, ${harmonic.visual.rgb.blue})`;

  return (
    <div className="harmonic-panel">
      <h3>Harmonic Output</h3>
      <div className="harmonic-grid">
        <div className="audio-section">
          <h4>Audio</h4>
          <div className="freq-row">
            <span>Primary:</span>
            <strong>{harmonic.audio.primaryHz} Hz</strong>
          </div>
          <div className="freq-row">
            <span>Secondary:</span>
            <strong>{harmonic.audio.secondaryHz} Hz</strong>
          </div>
          <div className="freq-row">
            <span>Carrier:</span>
            <strong>{(harmonic.audio.carrierHz / 10).toFixed(1)} Hz</strong>
          </div>
        </div>

        <div className="visual-section">
          <h4>Visual</h4>
          <div
            className={`color-preview pattern-${harmonic.visual.pattern.toLowerCase()}`}
            style={{
              backgroundColor: rgbString,
              opacity: harmonic.visual.intensity / 255
            }}
          />
          <div className="pattern-label">{harmonic.visual.pattern}</div>
        </div>

        <div className="tactile-section">
          <h4>Tactile</h4>
          <div className={`tactile-status ${harmonic.tactile.active ? 'active' : 'inactive'}`}>
            {harmonic.tactile.active ? 'ACTIVE' : 'OFF'}
          </div>
          {harmonic.tactile.active && (
            <div className="tactile-params">
              <span>{harmonic.tactile.pulseFreq} Hz</span>
              <span>{((harmonic.tactile.pulseIntensity / 255) * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      </div>

      <div className={`phi-sync ${harmonic.phiSync ? 'active' : ''}`}>
        Φ Sync: {harmonic.phiSync ? 'LOCKED' : 'OFF'}
      </div>
    </div>
  );
};

// Feedback mode indicator
const FeedbackModePanel: React.FC<{ feedback: FeedbackState; safetyAlert: boolean }> = ({
  feedback,
  safetyAlert
}) => {
  const modeColors: Record<string, string> = {
    REINFORCE: '#4CAF50',
    ADJUST: '#FFC107',
    PEAK_PULSE: '#9C27B0',
    STABILIZE: '#2196F3',
    EMERGENCY_STAB: '#F44336'
  };

  return (
    <div className="feedback-panel">
      <h3>Feedback Loop</h3>
      <div
        className="mode-badge"
        style={{ backgroundColor: modeColors[feedback.adaptationMode] || '#666' }}
      >
        {feedback.adaptationMode.replace('_', ' ')}
      </div>

      <div className="feedback-stats">
        <div className="stat">
          <span>Delta:</span>
          <strong className={feedback.coherenceDelta >= 0 ? 'positive' : 'negative'}>
            {feedback.coherenceDelta > 0 ? '+' : ''}{feedback.coherenceDelta}
          </strong>
        </div>
        <div className="stat">
          <span>Cycles:</span>
          <strong>{feedback.cycleCount}</strong>
        </div>
        <div className="stat">
          <span>DOR Timer:</span>
          <strong className={feedback.dorTimer > 25 ? 'warning' : ''}>
            {feedback.dorTimer}s / 30s
          </strong>
        </div>
      </div>

      {safetyAlert && (
        <div className="safety-alert">
          ⚠️ SAFETY LIMIT TRIGGERED
        </div>
      )}
    </div>
  );
};

// Main dashboard component
export const ScalarResonanceDashboard: React.FC<{
  websocketUrl?: string;
  onDataUpdate?: (data: ResonanceData) => void;
}> = ({ websocketUrl = 'ws://localhost:8080/resonance', onDataUpdate }) => {
  const [data, setData] = useState<ResonanceData | null>(null);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<number[]>([]);

  // Find dominant chakra (max drift magnitude)
  const dominantChakra = data?.coherence.chakraDrift.reduce(
    (maxIdx, val, idx, arr) => Math.abs(val) > Math.abs(arr[maxIdx]) ? idx : maxIdx,
    0
  ) ?? 0;

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(websocketUrl);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      try {
        const newData = JSON.parse(event.data) as ResonanceData;
        setData(newData);
        setHistory(prev => [...prev.slice(-100), newData.coherence.coherenceLevel]);
        onDataUpdate?.(newData);
      } catch (e) {
        console.error('Failed to parse resonance data:', e);
      }
    };

    return () => ws.close();
  }, [websocketUrl, onDataUpdate]);

  if (!data) {
    return (
      <div className="dashboard loading">
        <div className="connection-status">
          {connected ? 'Waiting for data...' : 'Connecting...'}
        </div>
      </div>
    );
  }

  return (
    <div className="scalar-resonance-dashboard">
      <header className="dashboard-header">
        <h1>Scalar Resonance Biofeedback</h1>
        <div className={`connection-indicator ${connected ? 'connected' : 'disconnected'}`}>
          {connected ? '● LIVE' : '○ OFFLINE'}
        </div>
      </header>

      <SessionPhaseIndicator phase={data.currentPhase} />

      <div className="main-panels">
        <div className="coherence-section">
          <h3>Coherence Metrics</h3>
          <CoherenceBar
            value={data.coherence.coherenceLevel}
            label="Coherence"
            color="#4CAF50"
          />
          <CoherenceBar
            value={data.coherence.emotionalTension}
            label="Tension"
            color="#FF5722"
          />

          <div className="coherence-history">
            <svg viewBox="0 0 100 40" preserveAspectRatio="none">
              <polyline
                fill="none"
                stroke="#4CAF50"
                strokeWidth="1"
                points={history.map((v, i) => `${i},${40 - (v / 255) * 40}`).join(' ')}
              />
            </svg>
          </div>
        </div>

        <ChakraDriftPanel
          drift={data.coherence.chakraDrift}
          dominantIndex={dominantChakra}
        />

        <HarmonicDisplay harmonic={data.harmonic} />

        <FeedbackModePanel
          feedback={data.feedback}
          safetyAlert={data.safetyAlert}
        />
      </div>

      <footer className="dashboard-footer">
        <span>Cycle #{data.feedback.cycleCount}</span>
        <span>Target: {CHAKRAS[dominantChakra].name} ({CHAKRAS[dominantChakra].frequency} Hz)</span>
        <span>{new Date(data.timestamp).toLocaleTimeString()}</span>
      </footer>
    </div>
  );
};

// CSS styles (can be extracted to separate file)
export const dashboardStyles = `
.scalar-resonance-dashboard {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: #1a1a2e;
  color: #eee;
  padding: 20px;
  min-height: 100vh;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.dashboard-header h1 {
  margin: 0;
  font-size: 1.5rem;
}

.connection-indicator {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.8rem;
}

.connection-indicator.connected {
  background: #1b5e20;
  color: #a5d6a7;
}

.connection-indicator.disconnected {
  background: #b71c1c;
  color: #ef9a9a;
}

.phase-indicator {
  margin-bottom: 20px;
}

.phase-timeline {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
}

.phase-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  opacity: 0.4;
}

.phase-node.active {
  opacity: 1;
}

.phase-node.completed {
  opacity: 0.7;
}

.phase-dot {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #444;
  margin-bottom: 4px;
}

.phase-node.active .phase-dot {
  background: #4CAF50;
  box-shadow: 0 0 10px #4CAF50;
}

.phase-node.completed .phase-dot {
  background: #2196F3;
}

.main-panels {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.coherence-section, .chakra-panel, .harmonic-panel, .feedback-panel {
  background: #16213e;
  border-radius: 8px;
  padding: 16px;
}

.coherence-bar {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}

.bar-label {
  width: 80px;
  font-size: 0.85rem;
}

.bar-container {
  flex: 1;
  height: 20px;
  background: #0f3460;
  border-radius: 4px;
  overflow: hidden;
  margin: 0 12px;
}

.bar-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.bar-value {
  width: 60px;
  text-align: right;
  font-family: monospace;
}

.coherence-history {
  height: 60px;
  margin-top: 16px;
  background: #0f3460;
  border-radius: 4px;
}

.coherence-history svg {
  width: 100%;
  height: 100%;
}

.chakra-row {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
  padding: 4px;
  border-radius: 4px;
}

.chakra-row.dominant {
  background: rgba(255, 255, 255, 0.1);
}

.chakra-name {
  width: 80px;
  font-size: 0.85rem;
}

.drift-bar {
  flex: 1;
  height: 12px;
  background: #0f3460;
  border-radius: 3px;
  overflow: hidden;
  margin: 0 8px;
}

.drift-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.drift-value {
  width: 40px;
  text-align: right;
  font-family: monospace;
  font-size: 0.8rem;
}

.dominant-marker {
  margin-left: 4px;
  color: #FFC107;
}

.harmonic-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.freq-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 0.85rem;
}

.color-preview {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  margin: 8px auto;
}

.pattern-breath { animation: breathe 4s ease-in-out infinite; }
.pattern-wave { animation: wave 2s ease-in-out infinite; }
.pattern-pulse { animation: pulse 0.5s ease-in-out infinite; }

@keyframes breathe {
  0%, 100% { transform: scale(0.9); opacity: 0.6; }
  50% { transform: scale(1.1); opacity: 1; }
}

@keyframes wave {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.2); }
}

.tactile-status {
  text-align: center;
  padding: 8px;
  border-radius: 4px;
  font-weight: bold;
}

.tactile-status.active { background: #4CAF50; }
.tactile-status.inactive { background: #666; }

.phi-sync {
  margin-top: 12px;
  text-align: center;
  padding: 8px;
  border-radius: 4px;
  background: #333;
}

.phi-sync.active {
  background: #9C27B0;
  color: #fff;
}

.mode-badge {
  text-align: center;
  padding: 12px;
  border-radius: 8px;
  font-weight: bold;
  font-size: 1.1rem;
  margin-bottom: 12px;
}

.feedback-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px;
  background: #0f3460;
  border-radius: 4px;
}

.stat strong.positive { color: #4CAF50; }
.stat strong.negative { color: #F44336; }
.stat strong.warning { color: #FFC107; }

.safety-alert {
  margin-top: 12px;
  padding: 12px;
  background: #F44336;
  border-radius: 8px;
  text-align: center;
  font-weight: bold;
  animation: blink 1s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.dashboard-footer {
  display: flex;
  justify-content: space-between;
  margin-top: 20px;
  padding-top: 12px;
  border-top: 1px solid #333;
  font-size: 0.85rem;
  color: #888;
}

.dashboard.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}
`;

export default ScalarResonanceDashboard;
