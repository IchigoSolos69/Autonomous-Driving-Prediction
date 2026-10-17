import { useCallback, useEffect, useRef, useState } from 'react';
import { actorRisk, mostLikelyMode, probability, readable, request } from './data';
import ScenarioCanvas from './ScenarioCanvas';
import type { Health, ModelKind, Prediction, Scenario, ScenarioSummary, StressKind } from './types';

const stressLabels: Record<StressKind, string> = { position_noise: 'Position noise', dropout: 'History dropout', braking: 'Sudden braking', cut_in: 'Lateral cut-in' };
const typeLabels: Record<string, string> = { vehicle: 'Vehicle', pedestrian: 'Pedestrian', cyclist: 'Cyclist', motorcyclist: 'Motorcyclist', bus: 'Bus' };
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'The request could not be completed.';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState('');
  const [sceneError, setSceneError] = useState('');
  const [predictionError, setPredictionError] = useState('');
  const [activeId, setActiveId] = useState('');
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [original, setOriginal] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [selectedActor, setSelectedActor] = useState('');
  const [model, setModel] = useState<ModelKind>('trained');
  const [mode, setMode] = useState(0);
  const [step, setStep] = useState(49);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showReference, setShowReference] = useState(false);
  const [showAllModes, setShowAllModes] = useState(false);
  const [fitKey, setFitKey] = useState(0);
  const [search, setSearch] = useState('');
  const [city, setCity] = useState('');
  const [split, setSplit] = useState('');
  const [visibleCount, setVisibleCount] = useState(100);
  useEffect(() => setVisibleCount(100), [search, city, split]);
  const [stressKind, setStressKind] = useState<StressKind>('braking');
  const [severity, setSeverity] = useState(0.35);
  const [stressActor, setStressActor] = useState('');
  const [drawer, setDrawer] = useState<'scenarios' | 'inspector' | null>(null);
  const predictionRequest = useRef(0);
  const drawerFocus = useRef<HTMLElement | null>(null);

  const loadLibrary = useCallback(async () => {
    setLoading(true); setConnectionError('');
    const [healthResult, libraryResult] = await Promise.allSettled([
      request<Health>('/health'), request<{ scenarios: ScenarioSummary[]; total: number }>('/scenarios'),
    ]);
    if (healthResult.status === 'fulfilled') setHealth(healthResult.value);
    if (libraryResult.status === 'fulfilled') {
      setScenarios(libraryResult.value.scenarios);
      setActiveId((previous) => libraryResult.value.scenarios.some((item) => item.id === previous) ? previous : (libraryResult.value.scenarios.find((item) => item.split === 'evaluation') ?? libraryResult.value.scenarios[0])?.id ?? '');
    }
    const failures = [healthResult, libraryResult].filter((result) => result.status === 'rejected');
    if (failures.length) setConnectionError('Cannot reach the local prediction service. Retry, or close and reopen Interaction Lab. Startup details are in artifacts/desktop/app.log.');
    setLoading(false);
  }, []);
  useEffect(() => { void loadLibrary(); }, [loadLibrary]);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    const controller = new AbortController();
    predictionRequest.current++;
    setSceneLoading(true); setSceneError(''); setPredictionError(''); setPrediction(null); setOriginal(null); setScenario(null); setPlaying(false); setBusy(false);
    request<Scenario>(`/scenarios/${encodeURIComponent(activeId)}`, { signal: controller.signal }).then(async (loaded) => {
      if (cancelled) return;
      setScenario(loaded); setSelectedActor(loaded.focal_track_id); setStep(loaded.observed_steps - 1); setMode(0);
      setStressActor(loaded.actors.find((actor) => !actor.focal && !actor.ego)?.id ?? loaded.focal_track_id);
      setSceneLoading(false);
      const currentRequest = ++predictionRequest.current;
      const initialModel: ModelKind = health?.checkpoint_loaded ? model : 'constant_velocity';
      setBusy(true);
      try {
        const result = await request<Prediction>('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario_id: loaded.id, model: initialModel }), signal: controller.signal });
        if (!cancelled && currentRequest === predictionRequest.current) { setPrediction(result); setMode(mostLikelyMode(result.probabilities)); setModel(initialModel); }
      } catch (error) { if (!cancelled && currentRequest === predictionRequest.current) setPredictionError(errorMessage(error)); }
      finally { if (!cancelled && currentRequest === predictionRequest.current) setBusy(false); }
    }).catch((error) => { if (!cancelled) { setSceneError(errorMessage(error)); setSceneLoading(false); } });
    return () => { cancelled = true; controller.abort(); };
  }, [activeId, health?.checkpoint_id]);

  const maxStep = scenario ? scenario.observed_steps + scenario.future_steps - 1 : 109;
  const visibleScenario = prediction?.scenario ?? scenario;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setStep((current) => Math.min(maxStep, current + 1)), (scenario?.dt ?? 0.1) * 1000 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, maxStep, scenario?.dt]);
  useEffect(() => { if (step >= maxStep) setPlaying(false); }, [step, maxStep]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.key === 'Escape') { setDrawer(null); return; }
      if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(target.tagName)) return;
      if (!scenario) return;
      if (event.code === 'Space') { event.preventDefault(); setStep((value) => value >= maxStep ? 0 : value); setPlaying((value) => !value); }
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); setPlaying(false); setStep((value) => Math.max(0, Math.min(maxStep, value + (event.key === 'ArrowRight' ? 1 : -1)))); }
      if (event.key.toLowerCase() === 'f') setFitKey((value) => value + 1);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [scenario, maxStep]);

  useEffect(() => {
    if (!drawer) return;
    drawerFocus.current = document.activeElement as HTMLElement;
    const panel = document.querySelector<HTMLElement>('.drawer.is-open');
    const elements = () => Array.from(panel?.querySelectorAll<HTMLElement>('button:not(:disabled), input, select, [tabindex="0"]') ?? []).filter((element) => element.offsetParent !== null);
    elements()[0]?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      const items = elements(), first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown', trap);
    return () => { document.removeEventListener('keydown', trap); drawerFocus.current?.focus(); };
  }, [drawer]);

  const runPrediction = async (withStress: boolean) => {
    if (!scenario || busy) return;
    const currentRequest = ++predictionRequest.current;
    setBusy(true); setPredictionError(''); setPlaying(false);
    const predict = (stress?: { kind: StressKind; severity: number; actor_id: string; seed: number }) => request<Prediction>('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario_id: scenario.id, model, ...(stress ? { stress } : {}) }),
    });
    try {
      const base = await predict();
      const result = withStress ? await predict({ kind: stressKind, severity, actor_id: stressActor, seed: 42 }) : base;
      if (currentRequest !== predictionRequest.current) return;
      setOriginal(withStress ? base : null); setPrediction(result); setMode(mostLikelyMode(result.probabilities)); setStep(scenario.observed_steps - 1);
    } catch (error) { if (currentRequest === predictionRequest.current) setPredictionError(errorMessage(error)); }
    finally { if (currentRequest === predictionRequest.current) setBusy(false); }
  };

  const filtered = scenarios.filter((item) => (!city || item.city === city) && (!split || item.split === split) && `${item.id} ${item.city} ${item.focal_type}`.toLowerCase().includes(search.toLowerCase()));
  const actor = visibleScenario?.actors.find((item) => item.id === selectedActor);
  const risk = actorRisk(prediction, selectedActor);
  const rawRisk = actorRisk(prediction, selectedActor, false);
  const baseRisk = actorRisk(original, selectedActor);
  const isBaseline = prediction?.model.kind === 'constant_velocity' || prediction?.model.kind === 'baseline';
  const isCalibrated = !!prediction?.risk.calibrated && !!prediction.calibration_id && !isBaseline && prediction.risk.status === 'calibrated_proxy';
  const insufficientCalibration = !!prediction?.risk.calibrated && prediction.risk.status === 'insufficient_positive_events';
  const forecastTime = scenario ? (step - scenario.observed_steps + 1) * scenario.dt : 0;
  const referenceAvailable = !!visibleScenario?.actors.some((item) => item.reference_future.some(Boolean));
  const historyLabel = scenario && step < scenario.observed_steps ? 'Observed' : 'Forecast';

  return <div className="app-shell">
    <header className="app-header">
      <div className="brand"><svg width="31" height="31" viewBox="0 0 32 32" aria-hidden="true"><rect x="1" y="1" width="30" height="30" rx="6" fill="#1d514c"/><path d="M7 25 13 7h6l6 18M11 19h10" fill="none" stroke="#89d2c1" strokeWidth="2"/><path d="M16 11v4m0 5v4" stroke="#fff" strokeWidth="1.5"/></svg><div><h1>Interaction lab</h1><p>Autonomous driving prediction</p></div></div>
      <div className="header-status"><span className={`status-dot ${health ? 'online' : ''}`} /><span>{health ? 'Local model service connected' : 'Local model service offline'}</span><span className="header-device">{health?.device ?? 'No device'}</span></div>
      <button className="quiet refresh" onClick={() => void loadLibrary()} disabled={loading} title="Refresh data and service status">{loading ? 'Connecting…' : 'Refresh'}</button>
    </header>

    <div className="mobile-nav"><button onClick={() => setDrawer('scenarios')}>Scenarios <span>{scenarios.length}</span></button><button onClick={() => setDrawer('inspector')}>Inspector <span>☰</span></button></div>
    {drawer && <button className="drawer-backdrop" aria-label="Close panel" onClick={() => setDrawer(null)} />}

    <main className="workspace">
      <aside className={`scenario-panel drawer ${drawer === 'scenarios' ? 'is-open' : ''}`} aria-label="Scenario library" role={drawer === 'scenarios' ? 'dialog' : undefined} aria-modal={drawer === 'scenarios' || undefined}>
        <div className="panel-heading"><h2>Scenarios</h2><span className="count">{scenarios.length}</span><button className="mobile-close quiet" aria-label="Close scenario library" onClick={() => setDrawer(null)}>×</button></div>
        <div className="library-filters">
          <label className="search-input"><svg width="15" height="15" viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="5" stroke="currentColor" strokeWidth="1.6"/><path d="m12 12 5 5" stroke="currentColor" strokeWidth="1.6"/></svg><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a scenario…" aria-label="Find a scenario by ID, city or focal type" /></label>
          <div className="filter-row"><select aria-label="Filter by city" value={city} onChange={(event) => setCity(event.target.value)}><option value="">All cities</option>{[...new Set(scenarios.map((item) => item.city))].sort().map((value) => <option key={value}>{value}</option>)}</select><select aria-label="Filter by split" value={split} onChange={(event) => setSplit(event.target.value)}><option value="">All splits</option>{[...new Set(scenarios.map((item) => item.split))].sort().map((value) => <option key={value}>{value}</option>)}</select></div>
        </div>
        <div className="scenario-list" aria-busy={loading}>
          {loading ? <p className="panel-note" role="status">Loading local scenarios…</p> : filtered.slice(0, visibleCount).map((item) => <button className={`scenario-row ${activeId === item.id ? 'active' : ''}`} key={item.id} onClick={() => { setActiveId(item.id); setDrawer(null); }} aria-pressed={activeId === item.id}>
            <div className="row-top"><strong>{readable(item.city)}</strong><span>{item.split}</span></div><div className="scenario-id">{item.id.slice(0, 18)}</div><div className="row-bottom"><span>{typeLabels[item.focal_type] ?? readable(item.focal_type)}</span><span>{item.actor_count} actors</span></div>
          </button>)}
          {!loading && filtered.length > visibleCount && <button className="secondary full-width" onClick={() => setVisibleCount((count) => count + 100)}>Show 100 more ({filtered.length - visibleCount} remaining)</button>}
          {!loading && filtered.length === 0 && <p className="panel-note">{scenarios.length ? 'No scenarios match these filters.' : 'No local scenarios yet.'}</p>}
        </div>
        <div className="library-footer"><span className="dataset-mark">AV2</span><div>Argoverse 2 Motion Forecasting<span>5 s history · 6 s forecast</span></div></div>
      </aside>

      <section className="viewer" aria-label="Scenario viewer">
        <div className="viewer-heading"><div><h2>{scenario ? readable(scenario.city) : 'Scenario workspace'}</h2><p title={scenario?.id}>{scenario?.id ?? 'Select a local scenario to begin'}</p></div><div className="view-badge">Bird’s-eye view</div></div>
        <div className="view-toolbar"><div className="legend"><span><i className="focal-dot" />Focal</span><span><i className="actor-dot" />Other actors</span><span><i className="risk-dot" />Elevated risk</span></div><div className="view-options"><label title={referenceAvailable ? 'Show recorded future trajectories; these are never model inputs.' : 'Reference futures are unavailable for this scene.'}><input type="checkbox" checked={showReference} disabled={!referenceAvailable} onChange={(event) => setShowReference(event.target.checked)} />Reference future</label><label><input type="checkbox" checked={showAllModes} disabled={!prediction} onChange={(event) => setShowAllModes(event.target.checked)} />All actor worlds</label></div></div>
        <div className="canvas-stage" aria-busy={sceneLoading}>
          {visibleScenario && !sceneLoading ? <ScenarioCanvas scenario={visibleScenario} prediction={prediction} original={original} step={step} mode={mode} selected={selectedActor} showReference={showReference} showAllModes={showAllModes} fitKey={fitKey} onSelect={setSelectedActor} /> : <div className="canvas-empty">
            <svg width="74" height="74" viewBox="0 0 74 74" fill="none" aria-hidden="true"><path d="M28 4v66m18-66v66M4 28h66M4 46h66" stroke="#52616c"/><path d="M37 3v12m0 44v12M3 37h12m44 0h12" stroke="#71828e" strokeDasharray="3 3"/><rect x="28" y="30" width="18" height="14" rx="3" stroke="#72b6aa" strokeWidth="2"/></svg>
            <h3>{sceneLoading || loading ? 'Opening scenario…' : connectionError ? 'Connect your local workspace' : sceneError ? 'Scenario could not be opened' : 'Your first scenario starts here'}</h3>
            <p>{sceneError || (connectionError ? 'Close and reopen Interaction Lab to restart its local prediction service.' : sceneLoading || loading ? 'Loading tracks and the local road map.' : 'Download and preprocess an Argoverse 2 subset, then reopen the application to inspect real tracks and forecasts.')}</p>
            {!loading && !sceneLoading && !sceneError && !connectionError && <div className="setup-commands"><span>Prepare a starter subset in a new data folder</span><code>adp download --raw-dir data/starter/raw --train-count 16 --val-count 12<br />adp preprocess --raw-dir data/starter/raw --cache-dir data/starter/cache<br />adp desktop --cache-dir data/starter/cache --baseline</code></div>}
            {!loading && !sceneLoading && <button className="canvas-action" onClick={() => { if (sceneError) { const id = activeId; setActiveId(''); window.setTimeout(() => setActiveId(id), 0); } else void loadLibrary(); }}>Retry connection</button>}
          </div>}
          {prediction && <div className={`prediction-label ${original ? 'stressed' : ''}`}><span className="live-line" />{original ? `${stressLabels[prediction.stress?.kind ?? stressKind]} comparison` : isBaseline ? 'Constant velocity baseline' : 'Trained forecast'}<small>{original ? 'Dashed = original · solid = stressed' : `${prediction.probabilities.length} shared future worlds`}</small></div>}
          {busy && <div className="canvas-status" role="status">Computing {model === 'trained' ? 'trained' : 'baseline'} predictions…</div>}
        </div>
        <div className="timeline">
          <div className="playback-controls"><button className="play-button" aria-label={playing ? 'Pause playback' : 'Play scenario'} disabled={!scenario} onClick={() => { if (step >= maxStep) setStep(0); setPlaying((value) => !value); }}>{playing ? <span className="pause-symbol" /> : <span className="play-symbol" />}</button><button className="quiet step-button" title="Previous frame" aria-label="Previous frame" disabled={!scenario || step === 0} onClick={() => { setPlaying(false); setStep((value) => Math.max(0, value - 1)); }}>‹</button><button className="quiet step-button" title="Next frame" aria-label="Next frame" disabled={!scenario || step >= maxStep} onClick={() => { setPlaying(false); setStep((value) => Math.min(maxStep, value + 1)); }}>›</button><div className="time-readout">{forecastTime >= 0 ? '+' : ''}{forecastTime.toFixed(1)}<span>s</span><small>{historyLabel}</small></div><select aria-label="Playback speed" value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></div>
          <div className="scrubber"><input type="range" aria-label="Scenario frame" aria-valuetext={`${forecastTime.toFixed(1)} seconds, ${historyLabel}`} min={0} max={maxStep} value={step} disabled={!scenario} onChange={(event) => { setPlaying(false); setStep(Number(event.target.value)); }} /><div className="timeline-labels"><span>−4.9 s</span><button disabled={!scenario} onClick={() => { setPlaying(false); setStep((scenario?.observed_steps ?? 50) - 1); }}>Now <i /></button><span>+6.0 s</span></div></div>
          <div className="timeline-foot"><span>{visibleScenario ? `${visibleScenario.selection.retained_dynamic} / ${visibleScenario.selection.total_dynamic} dynamic actors retained` : 'Real scenarios only'}</span><span>Space play / pause <span className="shortcut-separator">·</span> ← → step <span className="shortcut-separator">·</span> F fit</span></div>
        </div>
      </section>

      <aside className={`inspector-panel drawer ${drawer === 'inspector' ? 'is-open' : ''}`} aria-label="Prediction and actor inspector" role={drawer === 'inspector' ? 'dialog' : undefined} aria-modal={drawer === 'inspector' || undefined}>
        <div className="panel-heading"><h2>Inspector</h2><button className="mobile-close quiet" aria-label="Close inspector" onClick={() => setDrawer(null)}>×</button><span className="small-tag">{prediction ? 'Prediction ready' : 'Setup'}</span></div>
        <div className="inspector-scroll">
          {connectionError && <div className="notice error" role="alert">{connectionError}<button className="text-button" onClick={() => void loadLibrary()}>Retry</button></div>}
          <section className="inspector-section"><h3>Forecast model</h3><label className="field-label" htmlFor="model-select">Predictor</label><select id="model-select" value={model} disabled={busy} onChange={(event) => setModel(event.target.value as ModelKind)}><option value="constant_velocity">Constant velocity baseline</option><option value="trained" disabled={!health?.checkpoint_loaded}>Temporal graph model{health?.checkpoint_loaded ? '' : ' — no checkpoint'}</option></select>
            {!health?.checkpoint_loaded && <div className="notice"><strong>No trained checkpoint loaded</strong><p>The baseline extrapolates the last observed velocity. Train a model and reopen the application with its checkpoint to compare learned futures.</p></div>}
            {health?.checkpoint_loaded && <div className="model-detail"><span>Checkpoint</span><code title={health.checkpoint_id ?? ''}>{health.checkpoint_id}</code><span>Calibration</span><span>{health.calibration_loaded ? 'Available' : 'Not loaded'}</span></div>}
            <button className="primary full-width" disabled={!scenario || busy} onClick={() => void runPrediction(false)}>{busy ? 'Computing…' : 'Predict original'}</button>
            {prediction && <p className="model-timing">{prediction.elapsed_ms.toFixed(0)} ms inference <span>{isBaseline ? 'Heuristic baseline' : prediction.model.id}</span></p>}
            {predictionError && <div className="notice error" role="alert">{predictionError}</div>}
          </section>

          <section className="inspector-section"><div className="section-title"><h3>Future worlds</h3><span>{prediction?.probabilities.length ?? 0} modes</span></div><p className="helper">One world shares a probability across all predicted actors.</p><div className="world-list">{prediction ? prediction.probabilities.map((value, index) => <button key={index} className={`world-row ${mode === index ? 'selected' : ''}`} aria-pressed={mode === index} onClick={() => setMode(index)}><span className="world-number">{String(index + 1).padStart(2, '0')}</span><span className="world-bar"><i style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} /></span><strong>{probability(value)}</strong></button>) : <p className="panel-note compact">World probabilities appear after prediction.</p>}</div>{isBaseline && <p className="helper baseline-note">The six baseline worlds are identical; only world 1 carries probability.</p>}</section>

          <section className="inspector-section"><div className="section-title"><h3>Selected actor</h3><span className="actor-role">{actor?.focal ? 'Focal' : actor?.ego ? 'Ego' : actor ? 'Context' : '—'}</span></div><label className="sr-only" htmlFor="actor-select">Select actor</label><select id="actor-select" value={selectedActor} disabled={!visibleScenario} onChange={(event) => setSelectedActor(event.target.value)}>{!visibleScenario && <option value="">No actor selected</option>}{visibleScenario?.actors.map((item) => <option key={item.id} value={item.id}>{typeLabels[item.type] ?? readable(item.type)} {item.focal ? '(focal)' : item.ego ? '(ego)' : item.id.slice(-8)}</option>)}</select>{actor && <p className="actor-id" title={actor.id}>{actor.id}</p>}
            <div className="risk-readout"><div><span>{isCalibrated ? 'Calibrated interaction risk' : 'Raw interaction score'}</span><strong className={(risk ?? 0) >= 0.5 ? 'risk-high' : (risk ?? 0) >= 0.2 ? 'risk-medium' : ''}>{probability(risk)}</strong></div><span className={`risk-status ${isCalibrated ? 'calibrated' : ''}`}>{isCalibrated ? 'Calibrated' : isBaseline ? 'Heuristic · uncalibrated' : 'Uncalibrated'}</span></div>
            {isCalibrated && <div className="metric-line"><span>Before calibration</span><strong>{probability(rawRisk)}</strong></div>}
            {original && <div className="comparison-values"><div><span>Original</span><strong>{probability(baseRisk)}</strong></div><div><span>Stressed</span><strong>{probability(risk)}</strong></div><div><span>Change</span><strong>{risk != null && baseRisk != null ? `${risk - baseRisk >= 0 ? '+' : ''}${((risk - baseRisk) * 100).toFixed(1)} pp` : '—'}</strong></div></div>}
            {insufficientCalibration && <div className="notice"><strong>Insufficient calibration evidence</strong><p>The calibration split has fewer than 50 positive events. Raw scores are shown; fitted adjustments do not establish calibrated risk.</p></div>}
            {prediction?.risk.status === 'untrained_risk_head' && <div className="notice"><strong>No eligible risk supervision</strong><p>This training subset has no fully observed interaction groups. Forecasts are available; the unsupervised risk head is hidden.</p></div>}
            <p className="helper">Another retained dynamic actor within 3 m over 6 s. A research proxy; actor symbols use approximate sizes.</p>
          </section>

          <section className="inspector-section stress-section"><div className="section-title"><h3>Unexpected behavior</h3><span className="small-tag">Stress test</span></div><p className="helper">{stressKind === 'position_noise' || stressKind === 'dropout' ? 'Corrupt observed history and retain recorded future labels, then compare against the original scene.' : 'Change motion from one second before now. Other actors replay recorded paths and do not react to the intervention.'}</p><label className="field-label" htmlFor="stress-kind">Perturbation</label><select id="stress-kind" value={stressKind} onChange={(event) => setStressKind(event.target.value as StressKind)}>{Object.entries(stressLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select><label className="field-label" htmlFor="stress-actor">Affected actor</label><select id="stress-actor" value={stressActor} disabled={!scenario} onChange={(event) => setStressActor(event.target.value)}>{!scenario && <option value="">No actor available</option>}{scenario?.actors.map((item) => <option key={item.id} value={item.id}>{typeLabels[item.type] ?? readable(item.type)} {item.focal ? '(focal)' : item.ego ? '(ego)' : item.id.slice(-8)}</option>)}</select><label className="severity-label" htmlFor="severity">Severity <strong>{Math.round(severity * 100)}%</strong></label><input id="severity" type="range" min={0.05} max={1} step={0.05} value={severity} onChange={(event) => setSeverity(Number(event.target.value))} /><div className="range-labels"><span>Mild</span><span>Strong</span></div><button className="secondary full-width" disabled={!scenario || busy} onClick={() => void runPrediction(true)}>Compare stress test</button><p className="helper seed-note">Repeatable perturbation · seed 42</p></section>
        </div>
        <footer className="inspector-footer">Research workspace <span>Local only</span></footer>
      </aside>
    </main>
  </div>;
}
