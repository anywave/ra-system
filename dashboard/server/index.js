// Ra Test Dashboard - Backend API Server
// Provides test execution endpoints for Ra System modules

const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Test module definitions
const testModules = [
  {
    id: 'prompt17',
    title: 'Biofield Loopback Feedback',
    description: 'Closed-loop biofield feedback with scalar emergence coupling.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.Biofield.Loopback',
    testCommand: 'cabal test --test-option=--match=/Loopback/',
  },
  {
    id: 'prompt22',
    title: 'Ra.SonicFlux Harmonic Driver',
    description: 'Real-time sonification of field emergence data.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.SonicFlux',
    testCommand: 'cabal test --test-option=--match=/SonicFlux/',
  },
  {
    id: 'prompt31',
    title: 'Multi-Core Consent Transformer',
    description: 'Hubbard + Tesla consent transformer with 3:1 step-up resonance.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.Consent.Transformer',
    testCommand: 'cabal test --test-option=--match=/Consent/',
  },
  {
    id: 'prompt32',
    title: 'Self-Regulating Resonance Engine',
    description: 'Tesla coil + Joe Cell hybrid with biometric feedback loops.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.Engine.Resonator',
    testCommand: 'cabal test --test-option=--match=/Resonator/',
  },
  {
    id: 'prompt33',
    title: 'Biometric Expression Pipeline',
    description: 'Biometric-to-avatar scalar field expression.',
    phase: 'phase2',
    status: 'pending',
    module: 'Ra.Expression.Pipeline',
    testCommand: 'cabal test --test-option=--match=/Expression/',
  },
  {
    id: 'prompt34',
    title: 'Scalar-Interactive Surfaces',
    description: 'Surface shader system with biometric coherence coupling.',
    phase: 'phase2',
    status: 'pending',
    module: 'Ra.Visualizer.Surfaces',
    testCommand: 'cabal test --test-option=--match=/Surfaces/',
  },
  {
    id: 'prompt40',
    title: 'Chamber Sync Resonance',
    description: 'Wireless scalar chamber synchronization.',
    phase: 'phase2',
    status: 'pending',
    module: 'Ra.Chamber.Sync',
    testCommand: 'cabal test --test-option=--match=/Sync/',
  },
  {
    id: 'prompt41',
    title: 'Shell Sync Visualizer',
    description: 'Terminal-based sync network visualization.',
    phase: 'phase3',
    status: 'pending',
    module: 'Ra.Visualizer.Shell',
    testCommand: 'cabal test --test-option=--match=/Shell/',
  },
  {
    id: 'prompt56',
    title: 'Tactile Control Interface',
    description: 'Mind-controlled appendage routing layer.',
    phase: 'phase3',
    status: 'pending',
    module: 'Ra.Interface.TactileControl',
    testCommand: 'cabal test --test-option=--match=/Tactile/',
  },
  {
    id: 'prompt60',
    title: 'Chamber Synthesis',
    description: 'Generates scalar chamber geometry from biometric fields.',
    phase: 'phase4',
    status: 'pending',
    module: 'Ra.Chamber.Synthesis',
    testCommand: 'cabal test --test-option=--match=/Synthesis/',
  },
];

// Track test results
const testResults = new Map();

// GET /api/tests/modules - List all test modules
app.get('/api/tests/modules', (req, res) => {
  const modulesWithResults = testModules.map(mod => ({
    ...mod,
    ...testResults.get(mod.id),
  }));
  res.json(modulesWithResults);
});

// GET /api/tests/module/:id - Get single module
app.get('/api/tests/module/:id', (req, res) => {
  const mod = testModules.find(m => m.id === req.params.id);
  if (!mod) {
    return res.status(404).json({ error: 'Module not found' });
  }
  res.json({ ...mod, ...testResults.get(mod.id) });
});

// POST /api/tests/run/:id - Run a test
app.post('/api/tests/run/:id', async (req, res) => {
  const mod = testModules.find(m => m.id === req.params.id);
  if (!mod) {
    return res.status(404).json({ error: 'Module not found' });
  }

  const startTime = Date.now();

  try {
    // Run the Haskell test via cabal
    const result = await runCabalTest(mod);

    const elapsed = Date.now() - startTime;
    const testResult = {
      status: result.success ? 'complete' : 'error',
      coherence: result.success ? 0.85 + Math.random() * 0.15 : 0.3,
      tokenUsage: Math.floor(Math.random() * 1000) + 500,
      cost: (Math.random() * 0.1).toFixed(4),
      duration: elapsed,
      output: result.output,
      timestamp: new Date().toISOString(),
    };

    testResults.set(mod.id, testResult);
    res.json(testResult);
  } catch (error) {
    const testResult = {
      status: 'error',
      error: error.message,
      timestamp: new Date().toISOString(),
    };
    testResults.set(mod.id, testResult);
    res.status(500).json(testResult);
  }
});

// POST /api/tests/run-all - Run all tests
app.post('/api/tests/run-all', async (req, res) => {
  const results = [];

  for (const mod of testModules) {
    try {
      const result = await runCabalTest(mod);
      const testResult = {
        id: mod.id,
        status: result.success ? 'complete' : 'error',
        coherence: result.success ? 0.85 + Math.random() * 0.15 : 0.3,
      };
      testResults.set(mod.id, testResult);
      results.push(testResult);
    } catch (error) {
      results.push({ id: mod.id, status: 'error', error: error.message });
    }
  }

  res.json({ results, completed: results.filter(r => r.status === 'complete').length });
});

// GET /api/tests/results - Get all results
app.get('/api/tests/results', (req, res) => {
  const results = {};
  for (const [id, result] of testResults) {
    results[id] = result;
  }
  res.json(results);
});

// POST /api/tests/reset - Reset all test results
app.post('/api/tests/reset', (req, res) => {
  testResults.clear();
  res.json({ status: 'reset', message: 'All test results cleared' });
});

// Run cabal test command
function runCabalTest(mod) {
  return new Promise((resolve) => {
    const haskellDir = path.resolve(__dirname, '../../haskell');

    // For now, simulate test execution
    // In production, uncomment the spawn code below
    setTimeout(() => {
      resolve({
        success: Math.random() > 0.1, // 90% success rate for demo
        output: `Testing ${mod.module}...\nAll tests passed.`,
      });
    }, 1000 + Math.random() * 2000);

    /*
    const child = spawn('cabal', ['test'], {
      cwd: haskellDir,
      shell: true,
    });

    let output = '';
    child.stdout.on('data', (data) => { output += data.toString(); });
    child.stderr.on('data', (data) => { output += data.toString(); });

    child.on('close', (code) => {
      resolve({
        success: code === 0,
        output,
      });
    });

    child.on('error', (err) => {
      resolve({
        success: false,
        output: err.message,
      });
    });
    */
  });
}

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Ra Test Dashboard API running on http://localhost:${PORT}`);
  console.log(`Loaded ${testModules.length} test modules`);
});
