// RaTestDashboard.tsx — Updated Phase II with HandshakeGate visual module
// Ra Codex Test Suite Integration with Interactive Handshake Simulation

import React, { useState, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { CheckCircle, AlertTriangle, Loader2, Info } from 'lucide-react'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'

interface TestModule {
  id: string
  title: string
  description: string
  phase: string
  status: 'pending' | 'running' | 'complete' | 'error'
  tokenUsage?: number
  cost?: number
}

interface HandshakeResult {
  handshakeGranted: boolean
  passedBiometric: boolean
  matchedSymbol: boolean
  overrideUsed: boolean
}

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
      status: 'pending'
    },
    {
      id: 'prompt22-pwm',
      title: 'Ra.SonicEmitter → PWM Output',
      description: 'Cascaded scalar amplitude to PWM driver for chamber hardware.',
      phase: 'phase1',
      status: 'pending'
    },
    {
      id: 'prompt32',
      title: 'Ra.ConsentFramework — Symbolic Gate Validator',
      description: 'Self-regulating scalar consent logic with override tracking.',
      phase: 'phase2',
      status: 'pending'
    },
    {
      id: 'prompt32-router',
      title: 'ConsentRouter — Downstream Activation Splitter',
      description: 'Distributes consent state to biometric, gesture, and field pathways.',
      phase: 'phase2',
      status: 'pending'
    },
    {
      id: 'handshakegate',
      title: 'Ra.HandshakeGate — Dual-Factor Consent Link',
      description: 'Validates biometric + symbolic gestures with override support.',
      phase: 'phase2',
      status: 'pending'
    }
  ]
}

export default function RaTestDashboard() {
  const [modules, setModules] = useState<TestModule[]>([])
  const [loading, setLoading] = useState(true)
  const [gesture, setGesture] = useState('3')
  const [biometric, setBiometric] = useState(true)
  const [override, setOverride] = useState(false)
  const [handshakeResult, setHandshakeResult] = useState<HandshakeResult | null>(null)

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

  const simulateHandshake = async () => {
    const res = await fetch('/api/tests/simulate-handshake', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gestureID: parseInt(gesture), biometricMatch: biometric, overrideFlag: override })
    })
    const data = await res.json()
    setHandshakeResult(data)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete': return <CheckCircle className="text-green-500" />
      case 'running': return <Loader2 className="animate-spin text-blue-500" />
      case 'error': return <AlertTriangle className="text-red-500" />
      default: return <Info className="text-gray-400" />
    }
  }

  const renderHandshakeSimulator = () => (
    <Card className="shadow-md border-blue-400">
      <CardContent className="p-4 space-y-3">
        <h2 className="font-semibold text-lg">Handshake Simulation</h2>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium">Gesture ID</label>
            <Select value={gesture} onValueChange={setGesture}>
              <SelectTrigger><SelectValue placeholder="Select gesture ID" /></SelectTrigger>
              <SelectContent>
                {[0,1,2,3,4,5,6,7,8,9].map(id => (
                  <SelectItem key={id} value={id.toString()}>{id}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="block text-sm font-medium">Biometric Coherence</label>
            <Switch checked={biometric} onCheckedChange={setBiometric} />
          </div>
          <div>
            <label className="block text-sm font-medium">Override Flag</label>
            <Switch checked={override} onCheckedChange={setOverride} />
          </div>
        </div>
        <Button onClick={simulateHandshake}>Simulate Handshake</Button>
        {handshakeResult !== null && (
          <div className="text-sm mt-2">
            Granted: <b>{handshakeResult.handshakeGranted ? 'Yes' : 'No'}</b> | Biometric: {String(handshakeResult.passedBiometric)} | Symbol: {String(handshakeResult.matchedSymbol)} | Override: {String(handshakeResult.overrideUsed)}
          </div>
        )}
      </CardContent>
    </Card>
  )

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
            <Button onClick={() => runTest(mod.id)} disabled={mod.status === 'running'}>
              {mod.status === 'pending' ? 'Run Test' : mod.status === 'running' ? 'Running...' : 'Re-run'}
            </Button>
            {mod.status === 'running' && <Progress value={70} className="h-1" />}
          </CardContent>
        </Card>
      ))}
      {phase === 'phase2' && renderHandshakeSimulator()}
    </div>
  )

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">Ra Prompt Compliance Dashboard</h1>
      {loading ? <p className="text-muted-foreground">Loading test modules...</p> : (
        <Tabs defaultValue="phase2" className="w-full">
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
