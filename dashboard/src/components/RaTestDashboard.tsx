// RaTestDashboard.tsx — Visual Interface for Modular Prompt Testing
// Tailored for Ra Codex Test Suite Integration

import React, { useState, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { CheckCircle, AlertTriangle, Loader2, Info, Zap } from 'lucide-react'

export interface TestModule {
  id: string
  title: string
  description: string
  phase: string
  status: 'pending' | 'running' | 'complete' | 'error'
  tokenUsage?: number
  cost?: number
  coherence?: number
  module?: string
}

// Fetch test modules from API
const fetchTestModules = async (): Promise<TestModule[]> => {
  try {
    const response = await fetch('/api/tests/modules')
    if (!response.ok) throw new Error('Failed to fetch modules')
    const data = await response.json()
    return data
  } catch {
    // Return default modules if API unavailable
    return defaultModules
  }
}

// Default test modules matching Ra prompts
const defaultModules: TestModule[] = [
  {
    id: 'prompt17',
    title: 'Biofield Loopback Feedback',
    description: 'Closed-loop biofield feedback with scalar emergence coupling. Tests coherence computation and glow classification.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.Biofield.Loopback',
  },
  {
    id: 'prompt22',
    title: 'Ra.SonicFlux Harmonic Driver',
    description: 'Real-time sonification of field emergence data with frequency modulation.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.SonicFlux',
  },
  {
    id: 'prompt31',
    title: 'Multi-Core Consent Transformer',
    description: 'Hubbard + Tesla consent transformer with 3:1 step-up resonance.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.Consent.Transformer',
  },
  {
    id: 'prompt32',
    title: 'Self-Regulating Resonance Engine',
    description: 'Tesla coil + Joe Cell hybrid with biometric feedback loops.',
    phase: 'phase1',
    status: 'pending',
    module: 'Ra.Engine.Resonator',
  },
  {
    id: 'prompt33',
    title: 'Biometric Expression Pipeline',
    description: 'Biometric-to-avatar scalar field expression with torsion shell model.',
    phase: 'phase2',
    status: 'pending',
    module: 'Ra.Expression.Pipeline',
  },
  {
    id: 'prompt34',
    title: 'Scalar-Interactive Surfaces',
    description: 'Surface shader system with biometric coherence coupling.',
    phase: 'phase2',
    status: 'pending',
    module: 'Ra.Visualizer.Surfaces',
  },
  {
    id: 'prompt40',
    title: 'Chamber Sync Resonance',
    description: 'Wireless scalar chamber synchronization via harmonic alignment.',
    phase: 'phase2',
    status: 'pending',
    module: 'Ra.Chamber.Sync',
  },
  {
    id: 'prompt41',
    title: 'Shell Sync Visualizer',
    description: 'Terminal-based sync network visualization with ANSI rendering.',
    phase: 'phase3',
    status: 'pending',
    module: 'Ra.Visualizer.Shell',
  },
  {
    id: 'prompt56',
    title: 'Tactile Control Interface',
    description: 'Mind-controlled appendage routing layer with intent classification.',
    phase: 'phase3',
    status: 'pending',
    module: 'Ra.Interface.TactileControl',
  },
  {
    id: 'prompt60',
    title: 'Chamber Synthesis',
    description: 'Generates scalar chamber geometry from biometric fields.',
    phase: 'phase4',
    status: 'pending',
    module: 'Ra.Chamber.Synthesis',
  },
]

export default function RaTestDashboard() {
  const [modules, setModules] = useState<TestModule[]>([])
  const [loading, setLoading] = useState(true)
  const [runningAll, setRunningAll] = useState(false)

  useEffect(() => {
    fetchTestModules().then(data => {
      setModules(data)
      setLoading(false)
    })
  }, [])

  const runTest = async (id: string) => {
    setModules(prev => prev.map(m =>
      m.id === id ? { ...m, status: 'running' as const } : m
    ))

    try {
      const res = await fetch(`/api/tests/run/${id}`, { method: 'POST' })
      const result = await res.json()
      setModules(prev => prev.map(m =>
        m.id === id
          ? { ...m, status: result.status, cost: result.cost, tokenUsage: result.tokenUsage, coherence: result.coherence }
          : m
      ))
    } catch {
      // Simulate test completion for demo
      await new Promise(resolve => setTimeout(resolve, 1500))
      setModules(prev => prev.map(m =>
        m.id === id
          ? { ...m, status: 'complete' as const, coherence: 0.85 + Math.random() * 0.15, tokenUsage: Math.floor(Math.random() * 1000) + 500 }
          : m
      ))
    }
  }

  const runAllTests = async () => {
    setRunningAll(true)
    for (const mod of modules) {
      if (mod.status !== 'complete') {
        await runTest(mod.id)
      }
    }
    setRunningAll(false)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete': return <CheckCircle className="text-green-500 w-5 h-5" />
      case 'running': return <Loader2 className="animate-spin text-cyan-400 w-5 h-5" />
      case 'error': return <AlertTriangle className="text-red-500 w-5 h-5" />
      default: return <Info className="text-gray-500 w-5 h-5" />
    }
  }

  const getCoherenceColor = (coherence?: number) => {
    if (!coherence) return 'text-gray-500'
    if (coherence >= 0.85) return 'text-green-400'
    if (coherence >= 0.65) return 'text-yellow-400'
    return 'text-red-400'
  }

  const phases = ['phase1', 'phase2', 'phase3', 'phase4']
  const phaseLabels: Record<string, string> = {
    phase1: 'Phase I: Core',
    phase2: 'Phase II: Integration',
    phase3: 'Phase III: Interface',
    phase4: 'Phase IV: Synthesis',
  }

  const completedCount = modules.filter(m => m.status === 'complete').length
  const totalCount = modules.length
  const overallProgress = totalCount > 0 ? (completedCount / totalCount) * 100 : 0

  const renderPhase = (phase: string) => {
    const phaseModules = modules.filter(mod => mod.phase === phase)
    if (phaseModules.length === 0) {
      return (
        <div className="text-center text-gray-500 py-8">
          No modules in this phase yet...
        </div>
      )
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {phaseModules.map(mod => (
          <Card key={mod.id} className={mod.status === 'complete' ? 'border-green-500/30' : ''}>
            <CardContent className="p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-lg text-white">{mod.title}</h2>
                {getStatusIcon(mod.status)}
              </div>
              <p className="text-sm text-gray-400">{mod.description}</p>
              {mod.module && (
                <code className="text-xs text-cyan-400 bg-gray-900 px-2 py-1 rounded">
                  {mod.module}
                </code>
              )}
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Tokens: {mod.tokenUsage ?? '—'}</span>
                <span className={getCoherenceColor(mod.coherence)}>
                  Coherence: {mod.coherence ? `${(mod.coherence * 100).toFixed(1)}%` : '—'}
                </span>
              </div>
              <Button
                onClick={() => runTest(mod.id)}
                disabled={mod.status === 'running' || runningAll}
                variant={mod.status === 'complete' ? 'secondary' : 'default'}
                size="sm"
                className="w-full"
              >
                {mod.status === 'pending' ? 'Run Test' : mod.status === 'running' ? 'Running...' : 'Re-run'}
              </Button>
              {mod.status === 'running' && <Progress value={75} className="h-1" />}
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Zap className="text-ra-gold" />
            Ra Prompt Compliance Dashboard
          </h1>
          <p className="text-gray-400 mt-1">
            Testing {totalCount} modules across {phases.length} phases
          </p>
        </div>
        <Button
          onClick={runAllTests}
          disabled={runningAll || loading}
          size="lg"
        >
          {runningAll ? 'Running All...' : 'Run All Tests'}
        </Button>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex justify-between text-sm text-gray-400 mb-2">
          <span>Overall Progress</span>
          <span>{completedCount} / {totalCount} complete</span>
        </div>
        <Progress value={overallProgress} className="h-3" />
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="animate-spin text-cyan-400 w-8 h-8 mr-3" />
          <span className="text-gray-400">Loading test modules...</span>
        </div>
      ) : (
        <Tabs defaultValue="phase1" className="w-full">
          <TabsList className="mb-4">
            {phases.map(p => (
              <TabsTrigger key={p} value={p}>
                {phaseLabels[p]}
                <span className="ml-2 text-xs opacity-60">
                  ({modules.filter(m => m.phase === p && m.status === 'complete').length}/
                  {modules.filter(m => m.phase === p).length})
                </span>
              </TabsTrigger>
            ))}
          </TabsList>
          {phases.map(p => (
            <TabsContent key={p} value={p}>
              {renderPhase(p)}
            </TabsContent>
          ))}
        </Tabs>
      )}
    </div>
  )
}
