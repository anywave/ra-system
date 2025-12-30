// RaTestDashboard.tsx — Visual Interface for Modular Prompt Testing
// Tailored for Ra Codex Test Suite Integration

import React, { useState, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { CheckCircle, AlertTriangle, Loader2, Info } from 'lucide-react'

interface TestModule {
  id: string
  title: string
  description: string
  phase: string
  status: 'pending' | 'running' | 'complete' | 'error'
  tokenUsage?: number
  cost?: number
}

// Dynamic loader for test module metadata and results
const fetchTestModules = async (): Promise<TestModule[]> => {
  const response = await fetch('/api/tests/modules')
  const data = await response.json()
  return [
    ...data,
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
      id: 'prompt22-pwm',
      title: 'Ra.SonicEmitter → PWM Output',
      description: 'Cascaded scalar amplitude to PWM driver for chamber hardware.',
      phase: 'phase1',
      status: 'pending',
      tokenUsage: null,
      cost: null
    },
    {
      id: 'prompt32',
      title: 'Ra.ConsentFramework — Symbolic Gate Validator',
      description: 'Self-regulating scalar consent logic with override tracking.',
      phase: 'phase2',
      status: 'pending',
      tokenUsage: null,
      cost: null
    }
  ]
}

export default function RaTestDashboard() {
  const [modules, setModules] = useState<TestModule[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchTestModules().then(data => {
      setModules(data)
      setLoading(false)
    })
  }, [])

  const runTest = async (id: string) => {
    setModules(prev => prev.map(m => m.id === id ? { ...m, status: 'running' as const } : m))
    const res = await fetch(`/api/tests/run/${id}`, { method: 'POST' })
    const result = await res.json()
    setModules(prev => prev.map(m => m.id === id ? { ...m, status: result.status, cost: result.cost, tokenUsage: result.tokenUsage } : m))
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete': return <CheckCircle className="text-green-500" />
      case 'running': return <Loader2 className="animate-spin text-blue-500" />
      case 'error': return <AlertTriangle className="text-red-500" />
      default: return <Info className="text-gray-400" />
    }
  }

  const phases = ['phase1', 'phase2', 'phase3', 'phase4']

  const renderPhase = (phase: string) => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {modules.filter(mod => mod.phase === phase).map(mod => (
        <Card key={mod.id} className="shadow-md">
          <CardContent className="p-4 space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-lg">{mod.title}</h2>
              {getStatusIcon(mod.status)}
            </div>
            <p className="text-sm text-muted-foreground">{mod.description}</p>
            <div className="text-xs text-muted-foreground">
              Token Usage: {mod.tokenUsage ?? '—'} | Cost: {mod.cost ?? '—'} RaUnits
            </div>
            <Button onClick={() => runTest(mod.id)} disabled={mod.status === 'running'}>
              {mod.status === 'pending' ? 'Run Test' : mod.status === 'running' ? 'Running...' : 'Re-run'}
            </Button>
            {mod.status === 'running' && <Progress value={75} className="h-1" />}
          </CardContent>
        </Card>
      ))}
    </div>
  )

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">Ra Prompt Compliance Dashboard</h1>
      {loading ? <p className="text-muted-foreground">Loading test modules...</p> : (
        <Tabs defaultValue="phase1" className="w-full">
          <TabsList>
            {phases.map(p => (
              <TabsTrigger key={p} value={p}>{p.toUpperCase()}</TabsTrigger>
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
