// RaTestDashboard.tsx — Phase II with Biometric Coherence Matcher (Prompt 33)
// Ra Codex Test Suite with Full Consent Pipeline Visualization

import React, { useState, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { CheckCircle, AlertTriangle, Loader2, Info, Coins, Cpu } from 'lucide-react'
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
      id: 'fieldTransferBus',
      title: 'Ra.FieldTransferBus — Tesla Coherent Transfer',
      description: 'Scalar packet transmission with latency tracking and coherence integrity.',
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
    },
    {
      id: 'fieldSynthesisNode',
      title: 'Ra.FieldSynthesisNode — Chamber State Cascade',
      description: 'Transitions chamber states based on handshakeGrant signal.',
      phase: 'phase2',
      status: 'pending'
    },
    {
      id: 'biometricGenerator',
      title: 'Ra.BiometricGenerator — Runtime Coherence Waveform',
      description: 'Injects dynamic biometric signals for handshake loopback.',
      phase: 'phase2',
      status: 'pending'
    },
    {
      id: 'tokenomicsProfiler',
      title: 'Ra.TokenomicsProfiler — Prompt Cost Analyzer',
      description: 'Logs token + compute spend for Claude operations.',
      phase: 'phase2',
      status: 'pending'
    },
    {
      id: 'biometricMatcher',
      title: 'Ra.BiometricMatcher — Coherence Profile Matcher',
      description: 'Compares biometric waveforms against reference templates for coherence scoring.',
      phase: 'phase2',
      status: 'pending'
    },
    {
      id: 'scalarExpression',
      title: 'Ra.ScalarExpression — Avatar Expression Mapper',
      description: 'Maps biometric coherence + breath phase to aura intensity and limb vector.',
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
  const [pattern, setPattern] = useState('CoherentPulse')
  const [handshakeResult, setHandshakeResult] = useState<HandshakeResult | null>(null)
  const [chamberState, setChamberState] = useState('Idle')
  const [glow, setGlow] = useState(0)
  const [bioSample, setBioSample] = useState(128)
  const [tokenTotal, setTokenTotal] = useState(0)
  const [computeTotal, setComputeTotal] = useState(0)
  const [coherenceScore, setCoherenceScore] = useState<number | null>(null)
  const [coherenceColor, setCoherenceColor] = useState('gray')
  const [expression, setExpression] = useState<{ aura: number; limb: number } | null>(null)
  const [expressionColor, setExpressionColor] = useState('gray')
  const [consentState, setConsentState] = useState<{ granted: boolean; entropy: number; active: number } | null>(null)
  const [transferResult, setTransferResult] = useState<{ signal: number[]; latency: number; ok: boolean } | null>(null)
  const [tokenStats, setTokenStats] = useState<{ tokensUsed: number; computeCost: number } | null>(null)

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

  const updateBiometricSignal = async () => {
    const res = await fetch('/api/tests/biometric-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pattern })
    })
    const out = await res.json()
    setBioSample(out.sample)
  }

  const updateTokenomics = async (opType: string) => {
    const res = await fetch('/api/tests/tokenomics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ opType })
    })
    const t = await res.json()
    setTokenTotal(t.totalTokens)
    setComputeTotal(t.totalCompute)
  }

  const updateCoherenceScore = async (samples: number[], template: string) => {
    const res = await fetch('/api/tests/biometric-match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ waveform: samples, template })
    })
    const score = await res.json()
    setCoherenceScore(score)
    if (score >= 230) setCoherenceColor('green')
    else if (score >= 200) setCoherenceColor('yellow')
    else setCoherenceColor('red')
  }

  const updateAvatarExpression = async (coherence: number, breath: boolean) => {
    const res = await fetch('/api/tests/avatar-expression', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ coherence, breath })
    })
    const { aura, limb } = await res.json()
    setExpression({ aura, limb })
    if (aura >= 200) setExpressionColor('lime')
    else if (aura >= 150) setExpressionColor('gold')
    else setExpressionColor('gray')
  }

  const updateConsent = async (coherence: number, aura: number, votes: boolean[], quorum: number) => {
    const res = await fetch('/api/tests/consent-transform', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ coherence, aura, votes, quorum })
    })
    const { granted, entropy, active } = await res.json()
    setConsentState({ granted, entropy, active })
  }

  const testTransferBus = async () => {
    const res = await fetch('/api/tests/field-transfer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ coherence: 200, signal: [60, 90, 120, 180], send: true })
    })
    const { signal, latency, ok, tokensUsed, computeCost } = await res.json()
    setTransferResult({ signal, latency, ok })
    setTokenStats({ tokensUsed, computeCost })

    const deferReason = applyBackpropGating(tokensUsed, ok)
    if (deferReason) alert(`Activation deferred: ${deferReason}`)
  }

  const tokenColor = (tokens: number) => {
    if (tokens < 100) return 'bg-green-400'
    if (tokens < 200) return 'bg-yellow-400'
    return 'bg-red-500'
  }

  const applyBackpropGating = (tokens: number, ok: boolean): string | null => {
    if (!ok) return 'Signal integrity failure'
    if (tokens > 200) return 'Token strain — defer activation'
    return null
  }

  const exportTokenTelemetry = () => {
    const blob = new Blob([
      JSON.stringify(tokenStats, null, 2)
    ], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'tokenomics_snapshot.json'
    a.click()
  }

  const simulateHandshake = async () => {
    await updateBiometricSignal()
    await updateTokenomics('BioEmit')

    const res = await fetch('/api/tests/simulate-handshake', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gestureID: parseInt(gesture), biometricMatch: biometric, overrideFlag: override })
    })
    const data = await res.json()
    setHandshakeResult(data)
    await updateTokenomics('Handshake')

    if (data.handshakeGranted) {
      const cascade = await fetch('/api/tests/trigger-chamber', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trigger: true })
      })
      const node = await cascade.json()
      setChamberState(node.state)
      setGlow(node.glow)
      await updateTokenomics('ChamberSpin')
    } else {
      setChamberState('Idle')
      setGlow(0)
    }
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

  const renderBiometricVisualizer = () => (
    <Card className="shadow-md border-green-400">
      <CardContent className="p-4 space-y-3">
        <h2 className="font-semibold text-lg">Biometric Pattern Emulator</h2>
        <div className="flex items-center justify-between">
          <span className="text-sm">Pattern:</span>
          <Select value={pattern} onValueChange={setPattern}>
            <SelectTrigger className="w-40"><SelectValue placeholder="Choose Pattern" /></SelectTrigger>
            <SelectContent>
              {['Flatline', 'BreathRise', 'CoherentPulse', 'Arrhythmic'].map(p => (
                <SelectItem key={p} value={p}>{p}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="text-sm">Current Sample: <b>{bioSample}</b></span>
        </div>
        <Progress value={bioSample / 2.55} className="h-2 mt-2" />
      </CardContent>
    </Card>
  )

  const renderChamberVisual = () => (
    <Card className="shadow-md border-purple-400">
      <CardContent className="p-4 space-y-3">
        <h2 className="font-semibold text-lg">Chamber State Monitor</h2>
        <div className="flex items-center justify-between">
          <span className="text-sm">Current State: <b>{chamberState}</b></span>
          <span className="text-sm">Glow Intensity: <b>{glow}</b></span>
        </div>
        <Progress value={glow / 2.55} className="h-2 mt-2" />
      </CardContent>
    </Card>
  )

  const renderCostOverlay = () => (
    <Card className="shadow border-yellow-500">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Tokenomics Profiler</h2>
        <div className="flex items-center justify-between">
          <span className="flex gap-2 items-center text-sm"><Coins size={14}/> Tokens: <b>{tokenTotal}</b></span>
          <span className="flex gap-2 items-center text-sm"><Cpu size={14}/> Compute Units: <b>{computeTotal}</b></span>
        </div>
        <Progress value={(tokenTotal / 1024) * 100} className="h-1" />
      </CardContent>
    </Card>
  )

  const renderCoherenceOverlay = () => (
    <Card className="border-blue-500 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Biometric Coherence Matcher</h2>
        {coherenceScore !== null ? (
          <div className="text-sm">
            Score: <b style={{ color: coherenceColor }}>{coherenceScore}</b>
          </div>
        ) : <div className="text-sm text-muted-foreground">Awaiting input...</div>}
        <div className="flex gap-2">
          <Button size="sm" onClick={() => updateCoherenceScore([128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128], 'TemplateFlat')}>Test Flat</Button>
          <Button size="sm" onClick={() => updateCoherenceScore([128,140,160,180,192,180,160,140,128,116,96,76,64,76,96,116], 'TemplateResonant')}>Test Resonant</Button>
          <Button size="sm" onClick={() => updateCoherenceScore([100,120,140,160,180,160,140,120,100,80,60,40,20,40,60,80], 'TemplateResonant')}>Test Divergence</Button>
        </div>
      </CardContent>
    </Card>
  )

  const renderExpressionOverlay = () => (
    <Card className="border-purple-500 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Avatar Scalar Expression</h2>
        {expression ? (
          <>
            <div className="text-sm">Aura Intensity: <b style={{ color: expressionColor }}>{expression.aura}</b></div>
            <div className="text-sm">Limb Vector: <b>{expression.limb}</b></div>
          </>
        ) : <div className="text-sm text-muted-foreground">Awaiting expression...</div>}
        <div className="flex gap-2">
          <Button size="sm" onClick={() => updateAvatarExpression(255, true)}>Max (Exhale)</Button>
          <Button size="sm" onClick={() => updateAvatarExpression(180, false)}>Mid (Inhale)</Button>
          <Button size="sm" onClick={() => updateAvatarExpression(100, true)}>Low (Exhale)</Button>
        </div>
      </CardContent>
    </Card>
  )

  const renderConsentOverlay = () => (
    <Card className="border-red-500 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Consent Field State</h2>
        {consentState ? (
          <div className="space-y-1">
            <div className="text-sm">Consent: <b style={{ color: consentState.granted ? 'limegreen' : 'gray' }}>{consentState.granted ? 'GRANTED' : 'PENDING'}</b></div>
            <div className="text-sm">Entropy: <b>{consentState.entropy}</b></div>
            <div className="text-sm">Active Votes: <b>{consentState.active}</b></div>
          </div>
        ) : <div className="text-sm text-muted-foreground">Awaiting input...</div>}
        <div className="flex gap-2">
          <Button size="sm" onClick={() => updateConsent(190, 140, [true, true, true], 66)}>All Agree</Button>
          <Button size="sm" onClick={() => updateConsent(170, 100, [true, false, false], 50)}>Low Coherence</Button>
          <Button size="sm" onClick={() => updateConsent(200, 160, [true, true, false], 75)}>Partial</Button>
        </div>
      </CardContent>
    </Card>
  )

  const renderTransferOverlay = () => (
    <Card className="border-blue-400 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Tesla Field Transfer Bus</h2>
        {transferResult ? (
          <div className="space-y-1">
            <div className="text-sm">Signal: <code>[{transferResult.signal.join(', ')}]</code></div>
            <div className="text-sm">Latency: <b>{transferResult.latency} cycles</b></div>
            <div className="text-sm">Integrity: <b style={{ color: transferResult.ok ? 'limegreen' : 'red' }}>{transferResult.ok ? 'PASS' : 'FAIL'}</b></div>
          </div>
        ) : <div className="text-sm text-muted-foreground">No signal transferred yet.</div>}
        <Button size="sm" onClick={testTransferBus} className="mt-2">Simulate Transfer</Button>
      </CardContent>
    </Card>
  )

  const renderTokenOverlay = () => (
    <Card className="border-purple-400 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Tokenomics Heatmap</h2>
        {tokenStats ? (
          <div className="space-y-2">
            <div className="text-sm">Tokens: <b>{tokenStats.tokensUsed}</b></div>
            <div className="text-sm">Compute: <b>{tokenStats.computeCost}</b></div>
            <div className={`h-4 w-full rounded ${tokenColor(tokenStats.tokensUsed)}`} style={{ transition: 'all 0.3s' }} />
            <Button variant="outline" onClick={exportTokenTelemetry} className="w-full mt-2">Export Telemetry</Button>
          </div>
        ) : <div className="text-sm text-muted-foreground">Run a module to view token usage.</div>}
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
      {phase === 'phase1' && (
        <>
          {renderTransferOverlay()}
          {renderTokenOverlay()}
        </>
      )}
      {phase === 'phase2' && (
        <>
          {renderHandshakeSimulator()}
          {renderBiometricVisualizer()}
          {renderChamberVisual()}
          {renderCostOverlay()}
          {renderCoherenceOverlay()}
          {renderExpressionOverlay()}
          {renderConsentOverlay()}
        </>
      )}
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
