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
import { Textarea } from '@/components/ui/textarea'

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
  const [bioResult, setBioResult] = useState<{ motion: boolean; haptic: boolean } | null>(null)
  const [aura, setAura] = useState<number[] | null>(null)
  const [tones, setTones] = useState<number[] | null>(null)
  const [symbolicInputs, setSymbolicInputs] = useState<string>('PhaseShift(0.618) ○ InvertAngle ○ GateThreshold(0.4)')
  const [symbolicResults, setSymbolicResults] = useState<any[] | null>(null)
  const [symbolicConditions, setSymbolicConditions] = useState<string>(
    JSON.stringify([
      { coherence: 100, angle: 80 },
      { coherence: 200, angle: 0 },
      { coherence: 50, angle: 250 },
      { coherence: 0, angle: 127 }
    ], null, 2)
  )
  const [morphInputs, setMorphInputs] = useState<string>(
    JSON.stringify([
      { coherence: 90, instability: 100, form: 'Sphere' },
      { coherence: 120, instability: 50, form: 'Sphere' },
      { coherence: 80, instability: 60, form: 'Cube' },
      { coherence: 150, instability: 80, form: 'Cube' }
    ], null, 2)
  )
  const [morphResults, setMorphResults] = useState<any[] | null>(null)
  const [twistInputs, setTwistInputs] = useState<string>(
    JSON.stringify([
      { modeA: 3, modeB: 2, coherence: 110 },
      { modeA: 3, modeB: 2, coherence: 100 },
      { modeA: 5, modeB: 1, coherence: 108 },
      { modeA: 2, modeB: 3, coherence: 90 }
    ], null, 2)
  )
  const [twistResults, setTwistResults] = useState<any[] | null>(null)

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

  const runBiofeedbackTest = async () => {
    const res = await fetch('/api/tests/biofeedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        breath: [0, 1, 2, 2, 1, 2, 0, 0],
        coherence: [100, 100, 240, 240, 100, 250, 240, 200]
      })
    })
    const { motionIntent, hapticPing } = await res.json()
    setBioResult({ motion: motionIntent, haptic: hapticPing })
  }

  const runAvatarFieldTest = async () => {
    const res = await fetch('/api/tests/avatarfield', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signature: [10, 20, 30, 40],
        chamberState: [0, 1, 5, 3, 5, 2],
        emergenceLevel: [5, 10, 2, 1, 3, 0]
      })
    })
    const { auraPattern } = await res.json()
    setAura(auraPattern)
  }

  const runHarmonicsTest = async () => {
    const res = await fetch('/api/tests/musicharmonics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ coherenceBand: [0, 64, 128, 255] })
    })
    const { overtoneFrequencies } = await res.json()
    setTones(overtoneFrequencies)
  }

  const runSymbolicEval = async () => {
    const res = await fetch('/api/tests/symboliceval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        composition: symbolicInputs,
        inputs: JSON.parse(symbolicConditions)
      })
    })
    const { result } = await res.json()
    setSymbolicResults(result)
  }

  const symbolicCostOverlay = (ops: string): number => {
    if (!ops) return 0
    const opsList = ops.split('○').map(o => o.trim())
    return opsList.reduce((acc, op) => {
      if (op.startsWith('PhaseShift')) return acc + 1.2
      if (op.startsWith('InvertAngle')) return acc + 0.8
      if (op.startsWith('GateThreshold')) return acc + 1.0
      return acc
    }, 0)
  }

  const runMorphFallback = async () => {
    const res = await fetch('/api/tests/morphfallback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputStates: JSON.parse(morphInputs) })
    })
    const { result } = await res.json()
    setMorphResults(result)
  }

  const runTwistEnvelope = async () => {
    const res = await fetch('/api/tests/twistenvelope', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputStates: JSON.parse(twistInputs) })
    })
    const { result } = await res.json()
    setTwistResults(result)
  }

  const morphCostOverlay = (inputStr: string): number => {
    try {
      const inputs = JSON.parse(inputStr)
      return inputs.reduce((acc: number, state: any) => {
        const cost = (state.coherence < 100 && state.instability > 77) ? 1.5 : 0.5
        return acc + cost
      }, 0)
    } catch {
      return 0
    }
  }

  const twistCostOverlay = (inputStr: string): number => {
    try {
      const inputs = JSON.parse(inputStr)
      return inputs.reduce((acc: number, state: any) => {
        // Cost based on coherence threshold: high coherence = more expensive
        const baseCost = state.coherence >= 105 ? 2.0 : 1.2
        return acc + baseCost
      }, 0)
    } catch {
      return 0
    }
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

  const renderBiofeedbackControl = () => (
    <Card className="border-green-400 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Prompt 52: Biofeedback Harness</h2>
        <p className="text-sm text-muted-foreground">
          Exhale to Hold + Coherence &gt; 230 triggers MotionIntent + HapticPing.
        </p>
        <Button onClick={runBiofeedbackTest}>Run Harness Test</Button>
        {bioResult && (
          <div className="text-sm pt-2">
            <div>MotionIntent: <b style={{ color: bioResult.motion ? 'limegreen' : 'red' }}>{bioResult.motion ? 'ACTIVE' : 'INACTIVE'}</b></div>
            <div>HapticPing: <b style={{ color: bioResult.haptic ? 'limegreen' : 'red' }}>{bioResult.haptic ? 'ACTIVE' : 'INACTIVE'}</b></div>
          </div>
        )}
      </CardContent>
    </Card>
  )

  const renderAuraPattern = (pattern: number[]) => (
    <div className="grid grid-cols-4 gap-2 pt-2">
      {pattern.map((val, i) => (
        <div key={i} className="h-6 w-full bg-gradient-to-r from-blue-300 to-purple-500 rounded" style={{ opacity: val / 255 }} />
      ))}
    </div>
  )

  const renderAvatarFieldControl = () => (
    <Card className="border-pink-400 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Prompt 62: Avatar Field Visualizer</h2>
        <p className="text-sm text-muted-foreground">
          Chamber state = 0b101 triggers aura glow with emergence scaling.
        </p>
        <Button onClick={runAvatarFieldTest}>Run Visualizer Test</Button>
        {aura && renderAuraPattern(aura)}
      </CardContent>
    </Card>
  )

  const renderOvertoneBands = (freqs: number[]) => (
    <div className="grid grid-cols-4 gap-2 pt-2">
      {freqs.map((val, i) => (
        <div key={i} className="h-6 bg-gradient-to-r from-yellow-300 to-red-500 rounded flex items-center px-1" style={{ width: `${Math.min(val / 10, 150)}px` }}>
          <span className="text-xs text-white truncate">{val} Hz</span>
        </div>
      ))}
    </div>
  )

  const renderMusicChamberModule = () => (
    <Card className="border-yellow-500 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Prompt 64: Music Chamber Harmonics</h2>
        <p className="text-sm text-muted-foreground">Maps coherence bands to Solfeggio overtone frequencies.</p>
        <Button onClick={runHarmonicsTest}>Run Overtone Mapper</Button>
        {tones && renderOvertoneBands(tones)}
      </CardContent>
    </Card>
  )

  const renderSymbolicOps = () => (
    <Card className="border-purple-600 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Prompt 54: Symbolic Coherence Ops</h2>
        <p className="text-sm text-muted-foreground">Evaluate symbolic DSL on EmergenceConditions.</p>
        <Textarea
          className="text-sm font-mono"
          value={symbolicInputs}
          onChange={e => setSymbolicInputs(e.target.value)}
          rows={2}
        />
        <Textarea
          className="font-mono text-xs"
          value={symbolicConditions}
          onChange={e => setSymbolicConditions(e.target.value)}
          rows={6}
        />
        <div className="flex items-center justify-between">
          <Button onClick={runSymbolicEval}>Evaluate</Button>
          <span className="text-sm text-muted-foreground">
            Token Cost: {symbolicCostOverlay(symbolicInputs).toFixed(2)} units
          </span>
        </div>
        {symbolicResults && (
          <div className="pt-2 space-y-1">
            {symbolicResults.map((step, idx) => (
              <div key={idx} className="text-xs bg-muted p-2 rounded">
                <pre className="whitespace-pre-wrap">{JSON.stringify(step, null, 2)}</pre>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )

  const renderMorphologyFallback = () => (
    <Card className="border-orange-600 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Prompt 44: Chamber Morphology System</h2>
        <p className="text-sm text-muted-foreground">Fallback to Toroid when coherence &lt; 0.39 and instability &gt; 0.30</p>
        <Textarea
          className="font-mono text-xs"
          value={morphInputs}
          onChange={e => setMorphInputs(e.target.value)}
          rows={6}
        />
        <div className="flex items-center justify-between">
          <Button onClick={runMorphFallback}>Run Fallback</Button>
          <span className="text-sm text-muted-foreground">
            Token Cost: {morphCostOverlay(morphInputs).toFixed(2)} units
          </span>
        </div>
        {morphResults && (
          <div className="pt-2 space-y-1">
            {morphResults.map((res, idx) => (
              <div key={idx} className="text-xs bg-muted p-2 rounded">
                <pre className="whitespace-pre-wrap">{JSON.stringify(res, null, 2)}</pre>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )

  const renderTwistEnvelopePanel = () => (
    <Card className="border-yellow-700 shadow">
      <CardContent className="p-4 space-y-2">
        <h2 className="font-semibold text-lg">Prompt 49: Harmonic Inversion Twist</h2>
        <p className="text-sm text-muted-foreground">Y(a,b) inversion mapped to twist vector and duration</p>
        <Textarea
          className="font-mono text-xs"
          value={twistInputs}
          onChange={e => setTwistInputs(e.target.value)}
          rows={6}
        />
        <div className="flex items-center justify-between">
          <Button onClick={runTwistEnvelope}>Run Inversion Test</Button>
          <span className="text-sm text-muted-foreground">
            Token Cost: {twistCostOverlay(twistInputs).toFixed(2)} units
          </span>
        </div>
        {twistResults && (
          <div className="pt-2 space-y-1">
            {twistResults.map((res, idx) => (
              <div key={idx} className="text-xs bg-muted p-2 rounded">
                <pre className="whitespace-pre-wrap">{JSON.stringify(res, null, 2)}</pre>
              </div>
            ))}
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
      {phase === 'phase1' && (
        <>
          {renderTransferOverlay()}
          {renderBiofeedbackControl()}
          {renderAvatarFieldControl()}
          {renderMusicChamberModule()}
          {renderSymbolicOps()}
          {renderMorphologyFallback()}
          {renderTwistEnvelopePanel()}
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
