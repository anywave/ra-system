# Ra Test Dashboard

Visual interface for testing Ra System prompt compliance and module validation.

## Features

- **Phase-based organization** - Tests grouped into Phase I-IV
- **Real-time execution** - Run individual or batch tests
- **Coherence metrics** - Track module coherence scores
- **Token usage tracking** - Monitor resource consumption
- **Haskell integration** - Executes cabal tests for Ra modules

## Quick Start

```bash
# Install dependencies
npm install

# Run development server (frontend + backend)
npm start

# Or run separately:
npm run dev      # Frontend only (port 3000)
npm run server   # Backend only (port 3001)
```

## Architecture

```
dashboard/
├── src/
│   ├── components/
│   │   ├── ui/           # Reusable UI components
│   │   └── RaTestDashboard.tsx
│   ├── lib/
│   │   └── utils.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── server/
│   └── index.js          # Express API server
├── public/
├── package.json
└── vite.config.ts
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tests/modules` | GET | List all test modules |
| `/api/tests/module/:id` | GET | Get single module details |
| `/api/tests/run/:id` | POST | Run a specific test |
| `/api/tests/run-all` | POST | Run all tests |
| `/api/tests/results` | GET | Get all test results |
| `/api/tests/reset` | POST | Reset all results |
| `/api/health` | GET | Health check |

## Test Phases

### Phase I: Core
- Prompt 17: Biofield Loopback Feedback
- Prompt 22: Ra.SonicFlux Harmonic Driver
- Prompt 31: Multi-Core Consent Transformer
- Prompt 32: Self-Regulating Resonance Engine

### Phase II: Integration
- Prompt 33: Biometric Expression Pipeline
- Prompt 34: Scalar-Interactive Surfaces
- Prompt 40: Chamber Sync Resonance

### Phase III: Interface
- Prompt 41: Shell Sync Visualizer
- Prompt 56: Tactile Control Interface

### Phase IV: Synthesis
- Prompt 60: Chamber Synthesis

## Coherence Scoring

| Score | Level | Color |
|-------|-------|-------|
| ≥ 85% | High | Green |
| ≥ 65% | Moderate | Yellow |
| < 65% | Low | Red |

## Integration with Haskell

The server can execute Haskell tests via cabal:

```javascript
// Enable in server/index.js by uncommenting the spawn code
const child = spawn('cabal', ['test'], { cwd: haskellDir });
```

## Development

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, TailwindCSS
- **Backend**: Express.js, Node.js
- **Testing**: Cabal (Haskell)
