// ra_test_backend.js — Express.js API scaffold for Ra Test Dashboard

const express = require('express');
const cors = require('cors');
const app = express();
const PORT = process.env.PORT || 4321;

// Mock data — replace with actual Claude or Clash test logs
let testModules = [
  {
    id: 'prompt17',
    title: 'Biofield Loopback Feedback',
    description: 'Simulates recursive entrainment from biometric input.',
    phase: 'phase1',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt22',
    title: 'Ra.SonicFlux Harmonic Driver',
    description: 'Real-time sonification of field emergence data.',
    phase: 'phase1',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt31',
    title: 'Multi-Core Consent Transformer',
    description: 'Hubbard + Tesla consent transformer with 3:1 step-up.',
    phase: 'phase1',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt32',
    title: 'Self-Regulating Consent Framework',
    description: 'Controls symbolic emergence based on dynamic thresholds.',
    phase: 'phase2',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt33',
    title: 'Biometric Expression Pipeline',
    description: 'Biometric-to-avatar scalar field expression.',
    phase: 'phase2',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt34',
    title: 'Scalar-Interactive Surfaces',
    description: 'Surface shader system with biometric coherence coupling.',
    phase: 'phase2',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt40',
    title: 'Chamber Sync Resonance',
    description: 'Wireless scalar chamber synchronization.',
    phase: 'phase3',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt41',
    title: 'Shell Sync Visualizer',
    description: 'Terminal-based sync network visualization.',
    phase: 'phase3',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt56',
    title: 'Tactile Control Interface',
    description: 'Mind-controlled appendage routing layer.',
    phase: 'phase3',
    status: 'pending',
    tokenUsage: null,
    cost: null
  },
  {
    id: 'prompt60',
    title: 'Ra.Chamber Synthesis',
    description: 'Generates scalar chamber geometry from biometric fields.',
    phase: 'phase4',
    status: 'pending',
    tokenUsage: null,
    cost: null
  }
];

app.use(cors());
app.use(express.json());

// Fetch all test modules
app.get('/api/tests/modules', (req, res) => {
  res.json(testModules);
});

// Run a test module (mock simulation)
app.get('/api/tests/run/:id', (req, res) => {
  const { id } = req.params;
  const index = testModules.findIndex(m => m.id === id);
  if (index === -1) return res.status(404).send('Module not found');

  const tokenUsed = Math.floor(Math.random() * 900) + 100;
  const cost = (tokenUsed * 0.002).toFixed(3);

  testModules[index] = {
    ...testModules[index],
    status: 'complete',
    tokenUsage: tokenUsed,
    cost
  };

  res.json({ status: 'complete', tokenUsage: tokenUsed, cost });
});

// POST version of run (for compatibility)
app.post('/api/tests/run/:id', (req, res) => {
  const { id } = req.params;
  const index = testModules.findIndex(m => m.id === id);
  if (index === -1) return res.status(404).send('Module not found');

  const tokenUsed = Math.floor(Math.random() * 900) + 100;
  const cost = (tokenUsed * 0.002).toFixed(3);

  testModules[index] = {
    ...testModules[index],
    status: 'complete',
    tokenUsage: tokenUsed,
    cost
  };

  res.json({ status: 'complete', tokenUsage: tokenUsed, cost });
});

// Reset all tests
app.post('/api/tests/reset', (req, res) => {
  testModules = testModules.map(m => ({
    ...m,
    status: 'pending',
    tokenUsage: null,
    cost: null
  }));
  res.json({ status: 'reset' });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', modules: testModules.length });
});

app.listen(PORT, () => {
  console.log(`Ra Test Backend running at http://localhost:${PORT}`);
});
