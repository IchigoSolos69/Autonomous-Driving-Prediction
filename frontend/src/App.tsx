import { useCallback, useEffect, useRef, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import { actorRisk, mostLikelyMode, probability, readable, request } from './data';
import ScenarioCanvas from './ScenarioCanvas';
import FailureLibrary from './FailureLibrary';
import TrainingScreen from './TrainingScreen';
import type { AnalysisCase, DesktopContext, ExportKind, Health, ImportJob, ModelKind, Prediction, Reports, Scenario, ScenarioSummary, StressKind } from './types';

const stressLabels: Record<StressKind, string> = { position_noise: 'Position noise', dropout: 'History dropout', braking: 'Sudden braking', cut_in: 'Lateral cut-in' };
const typeLabels: Record<string, string> = { vehicle: 'Vehicle', pedestrian: 'Pedestrian', cyclist: 'Cyclist', motorcyclist: 'Motorcyclist', bus: 'Bus' };
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'The request could not be completed.';
const importPending = (job: ImportJob | null) => !!job && ['queued', 'scanning', 'running'].includes(job.status);
const metres = (value: number | undefined) => value == null ? '—' : `${value.toFixed(2)} m`;

export default function App() {
  const [showTraining, setShowTraining] = useState(() => window.location.hash === '#training');
  const [libraryTab, setLibraryTab] = useState<'scenarios' | 'failures'>('scenarios');
  const [recordedCase, setRecordedCase] = useState<AnalysisCase | null>(null);
  const [recordedSeed, setRecordedSeed] = useState<number | null>(42);
  const [recordedLabel, setRecordedLabel] = useState('directional');
  const [comparisonMode, setComparisonMode] = useState(0);
  const [health, setHealth] = useState<Health | null>(null);
  const [desktop, setDesktop] = useState<DesktopContext | null>(null);
  const [desktopError, setDesktopError] = useState('');
  const [settingsMessage, setSettingsMessage] = useState('');
  const [nativeBusy, setNativeBusy] = useState(false);
  const [importJob, setImportJob] = useState<ImportJob | null>(null);
  const [importError, setImportError] = useState('');
  const [cancelRequested, setCancelRequested] = useState(false);
  const [reports, setReports] = useState<Reports | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [exportMessage, setExportMessage] = useState('');
  const [exportError, setExportError] = useState('');
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
  const manualPrediction = useRef<AbortController | null>(null);
  const drawerFocus = useRef<HTMLElement | null>(null);
  const completedImport = useRef('');
  const recorded = libraryTab === 'failures';
  const resetRecordedCase = useCallback(() => {
    predictionRequest.current++;
    manualPrediction.current?.abort();
    setRecordedCase(null); setScenario(null); setPrediction(null); setOriginal(null);
    setPlaying(false); setBusy(false); setSceneLoading(false); setPredictionError(''); setSceneError('');
    setExportMessage(''); setExportError('');
  }, []);
  useEffect(() => {
    if (showTraining) window.history.replaceState(null, '', '#training');
    else if (window.location.hash === '#training') window.history.replaceState(null, '', window.location.pathname + window.location.search);
  }, [showTraining]);
  useEffect(() => () => manualPrediction.current?.abort(), []);
  const openRecordedCase = useCallback((value: AnalysisCase) => {
    const seeds = [...new Set(value.predictions.map((item) => item.analysis_seed).filter((seed): seed is number => seed != null))].sort();
    const label = value.predictions.some((item) => item.analysis_label === 'directional') ? 'directional' : value.predictions[0]?.analysis_label ?? 'reference';
    setRecordedCase(value); setRecordedSeed(seeds[0] ?? null); setRecordedLabel(label); setDrawer(null);
  }, []);
  useEffect(() => {
    if (!recorded || !recordedCase) return;
    const predictions = recordedCase.predictions.filter((item) => item.analysis_seed == null || item.analysis_seed === recordedSeed);
    const primary = predictions.find((item) => item.analysis_label === recordedLabel) ?? predictions[0];
    const comparison = predictions.find((item) => item.analysis_label === (primary?.analysis_label === 'directional' ? 'control' : 'constant_velocity') && item !== primary) ?? predictions.find((item) => item !== primary);
    setScenario(recordedCase.scenario); setPrediction(primary ?? null); setOriginal(comparison ?? null);
    setSelectedActor(recordedCase.scenario.focal_track_id); setStep(recordedCase.scenario.observed_steps - 1);
    setMode(mostLikelyMode(primary?.probabilities ?? [])); setComparisonMode(mostLikelyMode(comparison?.probabilities ?? []));
    setPlaying(false); setShowReference(false); setShowAllModes(false); setExportMessage(''); setExportError('');
  }, [recorded, recordedCase, recordedSeed, recordedLabel]);

  const loadLibrary = useCallback(async (importId?: string) => {
    setLoading(true); setConnectionError('');
    const [healthResult, libraryResult, reportsResult] = await Promise.allSettled([
      request<Health>('/health'), request<{ scenarios: ScenarioSummary[]; total: number }>('/scenarios'),
      request<Reports>('/reports'),
    ]);
    setHealth(healthResult.status === 'fulfilled' ? healthResult.value : null);
    setReports(reportsResult.status === 'fulfilled' ? reportsResult.value : null);
    if (libraryResult.status === 'fulfilled') {
      setScenarios(libraryResult.value.scenarios);
      setActiveId((previous) => libraryResult.value.scenarios.find((item) => importId && item.id.startsWith(`import:${importId}:`))?.id ?? (libraryResult.value.scenarios.some((item) => item.id === previous) ? previous : (libraryResult.value.scenarios.find((item) => item.split === 'evaluation') ?? libraryResult.value.scenarios[0])?.id ?? ''));
    }
    const failures = [healthResult, libraryResult].filter((result) => result.status === 'rejected');
    if (failures.length) setConnectionError('Cannot reach the local prediction service. Retry, or close and reopen Interaction Lab. Startup details are in %LOCALAPPDATA%\\Interaction Lab\\logs\\app.log.');
    setLoading(false);
  }, []);
  useEffect(() => { void loadLibrary(); }, [loadLibrary]);

  useEffect(() => {
    let cancelled = false, connecting = false;
    const connect = async () => {
      const api = window.pywebview?.api;
      if (!api || connecting) return;
      connecting = true;
      try {
        const context = await api.desktop_context();
        if (!cancelled) { setDesktop(context); setDesktopError(''); }
      } catch (error) { if (!cancelled) setDesktopError(errorMessage(error)); }
      finally { connecting = false; }
    };
    window.addEventListener('pywebviewready', connect);
    void connect();
    return () => { cancelled = true; window.removeEventListener('pywebviewready', connect); };
  }, []);

  useEffect(() => {
    if (!importJob || !importPending(importJob) || !desktop) return;
    let cancelled = false;
    let timer: number;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const updated = await request<ImportJob>(`/imports/${encodeURIComponent(importJob.id)}`, { signal: controller.signal }, desktop.token);
        if (!cancelled) { setImportJob(updated); setImportError(''); }
        if (!cancelled && importPending(updated)) timer = window.setTimeout(poll, 800);
      } catch (error) {
        if (!cancelled) { setImportError(`Progress unavailable: ${errorMessage(error)}. Retrying…`); timer = window.setTimeout(poll, 1500); }
      }
    };
    timer = window.setTimeout(poll, 800);
    return () => { cancelled = true; controller.abort(); window.clearTimeout(timer); };
  }, [importJob?.id, importJob?.status, desktop]);

  useEffect(() => {
    if (importJob?.status !== 'completed' || completedImport.current === importJob.id) return;
    completedImport.current = importJob.id;
    setSearch(''); setCity(''); setSplit('');
    void loadLibrary(importJob.id);
  }, [importJob, loadLibrary]);

  const startImport = async () => {
    const api = window.pywebview?.api;
    if (!desktop || !api || importPending(importJob)) return;
    setNativeBusy(true); setImportError(''); setCancelRequested(false);
    try {
      const selected = await api.choose_import_directory();
      if (!selected) return;
      const job = await request<ImportJob>('/imports', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source_token: selected.source_token }) }, desktop.token);
      setImportJob(job);
    } catch (error) { setImportError(errorMessage(error)); }
    finally { setNativeBusy(false); }
  };

  const cancelImport = async () => {
    if (!desktop || !importJob || cancelRequested) return;
    setCancelRequested(true); setImportError('');
    try { setImportJob(await request<ImportJob>(`/imports/${encodeURIComponent(importJob.id)}/cancel`, { method: 'POST' }, desktop.token)); }
    catch (error) { setImportError(errorMessage(error)); setCancelRequested(false); }
  };

  const changeSetting = async (kind: 'data' | 'model' | 'archive') => {
    const api = window.pywebview?.api;
    if (!desktop || !api) return;
    setNativeBusy(true); setDesktopError(''); setSettingsMessage('');
    try {
      const selected = kind === 'data' ? await api.choose_data_directory() : await api.choose_model_bundle(kind === 'archive');
      if (selected) setSettingsMessage(`${kind === 'data' ? 'Data folder' : 'Model bundle'} saved. Close and reopen Interaction Lab to apply it.`);
    } catch (error) { setDesktopError(errorMessage(error)); }
    finally { setNativeBusy(false); }
  };

  const saveExport = async (kind: ExportKind) => {
    if (!desktop) return;
    setExportBusy(true); setExportError(''); setExportMessage('');
    try {
      const result = await request<{ saved: boolean; cancelled: boolean; filename?: string }>('/exports', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind, ...(kind === 'diagnostics' ? {} : { prediction_id: prediction?.prediction_id, ...(original?.prediction_id ? { comparison_prediction_id: original.prediction_id } : {}) }) }) }, desktop.token);
      setExportMessage(result.cancelled ? 'Export cancelled.' : result.saved ? `Saved ${result.filename ?? 'export'}.` : 'No file was saved.');
    } catch (error) { setExportError(errorMessage(error)); }
    finally { setExportBusy(false); }
  };

  useEffect(() => {
    if (showTraining || recorded || !activeId) return;
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
  }, [activeId, health?.checkpoint_id, recorded, showTraining]);

  const maxStep = scenario ? scenario.observed_steps + scenario.future_steps - 1 : 109;
  const visibleScenario = prediction?.scenario ?? scenario;

  useEffect(() => {
    if (showTraining || !playing) return;
    const timer = window.setInterval(() => setStep((current) => Math.min(maxStep, current + 1)), (scenario?.dt ?? 0.1) * 1000 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, maxStep, scenario?.dt, showTraining]);
  useEffect(() => { if (step >= maxStep) setPlaying(false); }, [step, maxStep]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.key === 'Escape') { setDrawer(null); return; }
      if (showTraining) return;
      if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'SUMMARY'].includes(target.tagName)) return;
      if (!scenario) return;
      if (event.code === 'Space') { event.preventDefault(); setStep((value) => value >= maxStep ? 0 : value); setPlaying((value) => !value); }
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); setPlaying(false); setStep((value) => Math.max(0, Math.min(maxStep, value + (event.key === 'ArrowRight' ? 1 : -1)))); }
      if (event.key.toLowerCase() === 'f') setFitKey((value) => value + 1);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [scenario, maxStep, showTraining]);

  useEffect(() => {
    if (!drawer) return;
    drawerFocus.current = document.activeElement as HTMLElement;
    const panel = document.querySelector<HTMLElement>('.drawer.is-open');
    const elements = () => Array.from(panel?.querySelectorAll<HTMLElement>('button:not(:disabled), input, select, summary, [tabindex="0"]') ?? []).filter((element) => element.offsetParent !== null);
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
    if (showTraining || recorded || !scenario || busy) return;
    const currentRequest = ++predictionRequest.current;
    manualPrediction.current?.abort();
    const controller = new AbortController(); manualPrediction.current = controller;
    setBusy(true); setPredictionError(''); setPlaying(false);
    const predict = (stress?: { kind: StressKind; severity: number; actor_id: string; seed: number }) => request<Prediction>('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario_id: scenario.id, model, ...(stress ? { stress } : {}) }), signal: controller.signal,
    });
    try {
      const base = await predict();
      if (controller.signal.aborted || currentRequest !== predictionRequest.current) return;
      const result = withStress ? await predict({ kind: stressKind, severity, actor_id: stressActor, seed: 42 }) : base;
      if (currentRequest !== predictionRequest.current) return;
      setOriginal(withStress ? base : null); setPrediction(result); setMode(mostLikelyMode(result.probabilities)); setStep(scenario.observed_steps - 1);
    } catch (error) { if (currentRequest === predictionRequest.current) setPredictionError(errorMessage(error)); }
    finally { if (currentRequest === predictionRequest.current) setBusy(false); }
  };

  const filtered = scenarios.filter((item) => (!city || item.city === city) && (!split || item.split === split) && `${item.id} ${item.city} ${item.focal_type}`.toLowerCase().includes(search.toLowerCase()));
  const actor = visibleScenario?.actors.find((item) => item.id === selectedActor);
  const risk = actorRisk(prediction, selectedActor, !recorded);
  const rawRisk = actorRisk(prediction, selectedActor, false);
  const baseRisk = actorRisk(original, selectedActor, !recorded);
  const isBaseline = prediction?.model.kind === 'constant_velocity' || prediction?.model.kind === 'baseline';
  const isCalibrated = !recorded && !!prediction?.risk.calibrated && !!prediction.calibration_id && !isBaseline && prediction.risk.status === 'calibrated_proxy';
  const insufficientCalibration = prediction?.risk.status === 'insufficient_positive_events';
  const unknownCalibration = !isBaseline && (prediction?.risk.status === 'population_applicability_unknown' || prediction?.calibration_applicability === 'unknown' || prediction?.risk.applicability === 'unknown');
  const evidence = reports?.risk_calibration_evidence;
  const nativeEnabled = !!desktop && !nativeBusy && !exportBusy;
  const forecastTime = scenario ? (step - scenario.observed_steps + 1) * scenario.dt : 0;
  const referenceAvailable = !!visibleScenario?.actors.some((item) => item.reference_future.some(Boolean));
  const historyLabel = scenario && step < scenario.observed_steps ? 'Observed' : 'Forecast';
  const primaryLabel = recordedCase?.predictions.find((item) => item.prediction_id === prediction?.prediction_id)?.analysis_label ?? 'reference';
  const comparisonLabel = recordedCase?.predictions.find((item) => item.prediction_id === original?.prediction_id)?.analysis_label ?? 'comparison';
  const switchLibrary = (value: 'scenarios' | 'failures') => { if (value !== libraryTab) { resetRecordedCase(); setLibraryTab(value); } };
  const toggleTraining = () => {
    setPlaying(false); predictionRequest.current++; manualPrediction.current?.abort();
    setBusy(false); setSceneLoading(false); setDrawer(null); setShowTraining((value) => !value);
  };
  const tabKey = (event: ReactKeyboardEvent) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const target = event.key === 'Home' ? 'scenarios' : event.key === 'End' ? 'failures' : recorded ? 'scenarios' : 'failures';
    switchLibrary(target); document.getElementById(`${target}-tab`)?.focus();
  };

  return <div className="app-shell">
    <header className={`app-header ${showTraining ? 'training-header' : ''}`}>
      <div className="brand"><svg width="31" height="31" viewBox="0 0 32 32" aria-hidden="true"><rect x="1" y="1" width="30" height="30" rx="6" fill="#1d514c"/><path d="M7 25 13 7h6l6 18M11 19h10" fill="none" stroke="#89d2c1" strokeWidth="2"/><path d="M16 11v4m0 5v4" stroke="#fff" strokeWidth="1.5"/></svg><div><h1>Interaction lab</h1><p>Autonomous driving prediction · Research beta</p></div></div>
      <div className="header-status"><span className={`status-dot ${health ? 'online' : ''}`} /><span>{showTraining ? health ? 'Local study monitor connected' : 'Local study monitor offline' : health ? 'Local model service connected' : 'Local model service offline'}</span><span className="header-device">{showTraining ? 'Read-only' : health?.device ?? 'No device'}</span></div>
      <nav className="view-navigation" aria-label="Workspace navigation"><button className={`training-nav ${!showTraining ? 'selected' : ''}`} aria-pressed={!showTraining} onClick={() => { if (showTraining) toggleTraining(); }}>Scenarios</button><button className={`training-nav ${showTraining ? 'selected' : ''}`} aria-pressed={showTraining} onClick={() => { if (!showTraining) toggleTraining(); }}>Training</button></nav>
      {!showTraining && <button className="quiet refresh" onClick={() => void loadLibrary()} disabled={loading} title="Refresh data and service status">{loading ? 'Connecting…' : 'Refresh'}</button>}
    </header>

    {showTraining && <TrainingScreen />}
    <div className={`mobile-nav ${showTraining ? 'training-hidden' : ''}`} hidden={showTraining}><button onClick={() => setDrawer('scenarios')}>{recorded ? 'Failures' : 'Scenarios'} <span>{recorded ? 'Recorded' : scenarios.length}</span></button><button onClick={() => setDrawer('inspector')}>Inspector <span>☰</span></button></div>
    {drawer && <button className="drawer-backdrop" aria-label="Close panel" onClick={() => setDrawer(null)} />}

    <main className={`workspace ${showTraining ? 'training-hidden' : ''}`} hidden={showTraining}>
      <aside className={`scenario-panel drawer ${drawer === 'scenarios' ? 'is-open' : ''}`} aria-label="Scenario library" role={drawer === 'scenarios' ? 'dialog' : undefined} aria-modal={drawer === 'scenarios' || undefined}>
        <div className="panel-heading library-tabs" role="tablist" aria-label="Workspace library" onKeyDown={tabKey}><button id="scenarios-tab" role="tab" tabIndex={recorded ? -1 : 0} aria-selected={!recorded} aria-controls="scenario-library" onClick={() => switchLibrary('scenarios')}>Scenarios</button><button id="failures-tab" role="tab" tabIndex={recorded ? 0 : -1} aria-selected={recorded} aria-controls="failure-library" onClick={() => switchLibrary('failures')}>Failures</button><button className="mobile-close quiet" aria-label="Close scenario library" onClick={() => setDrawer(null)}>×</button></div>
        {recorded ? <div id="failure-library" role="tabpanel" aria-labelledby="failures-tab" className="failure-tab"><FailureLibrary desktop={!!desktop} nativeBusy={nativeBusy || exportBusy} activeCase={recordedCase?.scenario.id ?? ''} onCase={openRecordedCase} onReset={resetRecordedCase} /></div> : <div id="scenario-library" role="tabpanel" aria-labelledby="scenarios-tab" className="scenario-library-content">
        <div className="import-controls">
          <button className="secondary" disabled={!nativeEnabled || health?.import_available === false || importPending(importJob)} onClick={() => void startImport()}>{nativeBusy ? 'Opening dialog…' : 'Import AV2 folder…'}</button>
          {!desktop && <p className="helper">Open the desktop application to import local AV2 folders.</p>}
          {desktopError && !desktop && <p className="inline-error" role="alert">Native controls unavailable: {desktopError}. Close and reopen the application.</p>}
          {importJob && <div className="import-progress" aria-live="polite"><div><strong>Import {readable(importJob.status)}</strong>{importPending(importJob) && <button className="text-button" disabled={cancelRequested} onClick={() => void cancelImport()}>{cancelRequested ? 'Cancelling…' : 'Cancel import'}</button>}</div><progress aria-label="AV2 import progress" max={Math.max(1, importJob.total)} value={importJob.total ? importJob.processed : undefined} /><p>{importJob.total ? `${importJob.processed} / ${importJob.total} checked · ${importJob.imported} valid` : 'Finding scenario files…'}</p>{importJob.status === 'completed' && <p>Imported scenarios are selected in the library.</p>}{importJob.error && <p className="inline-error">{importJob.error}</p>}{importJob.errors.length > 0 && <details><summary>{importJob.errors.length} scenario errors</summary><ul>{importJob.errors.map((item, index) => <li key={`${item.scenario}-${index}`}><code>{item.scenario}</code>: {item.reason}</li>)}</ul></details>}</div>}
          {importError && <p className="inline-error" role="alert">{importError}</p>}
        </div>
        <div className="library-filters">
          <label className="search-input"><svg width="15" height="15" viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="5" stroke="currentColor" strokeWidth="1.6"/><path d="m12 12 5 5" stroke="currentColor" strokeWidth="1.6"/></svg><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a scenario…" aria-label="Find a scenario by ID, city or focal type" /></label>
          <div className="filter-row"><select aria-label="Filter by city" value={city} onChange={(event) => setCity(event.target.value)}><option value="">All cities</option>{[...new Set(scenarios.map((item) => item.city))].sort().map((value) => <option key={value}>{value}</option>)}</select><select aria-label="Filter by split" value={split} onChange={(event) => setSplit(event.target.value)}><option value="">All splits</option>{[...new Set(scenarios.map((item) => item.split))].sort().map((value) => <option key={value}>{value}</option>)}</select></div>
        </div>
        <div className="scenario-list" aria-busy={loading}>
          {loading ? <p className="panel-note" role="status">Loading local scenarios…</p> : filtered.slice(0, visibleCount).map((item) => <button className={`scenario-row ${activeId === item.id ? 'active' : ''}`} key={item.id} onClick={() => { setActiveId(item.id); setDrawer(null); }} aria-pressed={activeId === item.id}>
            <div className="row-top"><strong>{readable(item.city)}</strong><span>{item.source === 'imported' ? 'Imported' : item.source === 'demo' ? 'Demo' : item.split}</span></div><div className="scenario-id" title={item.id}>{item.id.startsWith('import:') ? item.id.split(':').at(-1)?.slice(0, 18) : item.id.slice(0, 18)}</div><div className="row-bottom"><span>{typeLabels[item.focal_type] ?? readable(item.focal_type)}</span><span>{item.actor_count} actors</span></div>
          </button>)}
          {!loading && filtered.length > visibleCount && <button className="secondary full-width" onClick={() => setVisibleCount((count) => count + 100)}>Show 100 more ({filtered.length - visibleCount} remaining)</button>}
          {!loading && filtered.length === 0 && <p className="panel-note">{scenarios.length ? 'No scenarios match these filters.' : 'No local scenarios yet.'}</p>}
        </div>
        <div className="library-footer"><span className="dataset-mark">AV2</span><div>Argoverse 2 Motion Forecasting<span>5 s history · 6 s forecast</span></div></div>
        </div>}
      </aside>

      <section className="viewer" aria-label="Scenario viewer">
        <div className="viewer-heading"><div><h2>{scenario ? readable(scenario.city) : recorded ? 'Failure review' : 'Scenario workspace'}</h2><p title={scenario?.id}>{scenario?.id ?? (recorded ? 'Choose a recorded case from the failure library' : 'Select a local scenario to begin')}</p></div><div className="view-badge">{recorded ? 'Development · recorded replay' : `${scenario?.source ? `${readable(scenario.source)} · ` : ''}Bird’s-eye view`}</div></div>
        <div className="view-toolbar"><div className="legend"><span><i className="focal-dot" />Focal</span><span><i className="actor-dot" />Other actors</span><span><i className="risk-dot" />Elevated risk</span></div><div className="view-options"><label title={referenceAvailable ? 'Show recorded future trajectories; these are never model inputs.' : 'Reference futures are unavailable for this scene.'}><input type="checkbox" checked={showReference} disabled={!referenceAvailable} onChange={(event) => setShowReference(event.target.checked)} />Reference future</label><label><input type="checkbox" checked={showAllModes} disabled={!prediction} onChange={(event) => setShowAllModes(event.target.checked)} />All actor worlds</label></div></div>
        <div className="canvas-stage" aria-busy={sceneLoading}>
          {visibleScenario && !sceneLoading ? <ScenarioCanvas scenario={visibleScenario} prediction={prediction} original={original} originalMode={recorded ? comparisonMode : undefined} step={step} mode={mode} selected={selectedActor} showReference={showReference} showAllModes={showAllModes} fitKey={fitKey} onSelect={setSelectedActor} /> : <div className="canvas-empty">
            <svg width="74" height="74" viewBox="0 0 74 74" fill="none" aria-hidden="true"><path d="M28 4v66m18-66v66M4 28h66M4 46h66" stroke="#52616c"/><path d="M37 3v12m0 44v12M3 37h12m44 0h12" stroke="#71828e" strokeDasharray="3 3"/><rect x="28" y="30" width="18" height="14" rx="3" stroke="#72b6aa" strokeWidth="2"/></svg>
            {recorded ? <><h3>Inspect a recorded failure</h3><p>Load an analysis folder and select a recorded case. Playback starts paused. Development errors describe the selection set; these previews never run inference.</p></> : <><h3>{sceneLoading || loading ? 'Opening scenario…' : connectionError ? 'Connect your local workspace' : sceneError ? 'Scenario could not be opened' : 'Your first scenario starts here'}</h3>
            <p>{sceneError || (connectionError ? 'Close and reopen Interaction Lab to restart its local prediction service.' : sceneLoading || loading ? 'Loading tracks and the local road map.' : 'Download and preprocess an Argoverse 2 subset, then reopen the application to inspect real tracks and forecasts.')}</p>
            {!loading && !sceneLoading && !sceneError && !connectionError && <button className="canvas-action" disabled={!nativeEnabled || health?.import_available === false} onClick={() => void startImport()}>Import an AV2 scenario folder</button>}
            {!loading && !sceneLoading && <button className="canvas-action" onClick={() => { if (sceneError) { const id = activeId; setActiveId(''); window.setTimeout(() => setActiveId(id), 0); } else void loadLibrary(); }}>Retry connection</button>}</>}
          </div>}
          {prediction && <div className={`prediction-label ${original && !recorded ? 'stressed' : ''}`}><span className="live-line" />{recorded ? `Recorded ${readable(primaryLabel)}${recordedSeed == null ? '' : ` · seed ${recordedSeed}`}` : original ? `${stressLabels[prediction.stress?.kind ?? stressKind]} comparison` : isBaseline ? 'Constant velocity baseline' : 'Trained forecast'}<small>{recorded ? original ? `Solid = ${readable(primaryLabel)} · dashed = ${readable(comparisonLabel)}` : 'Selection-set development replay' : original ? 'Dashed = original · solid = stressed' : `${prediction.probabilities.length} shared future worlds`}</small></div>}
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
          {health?.device_warning && <div className="notice" role="status"><strong>Device fallback</strong><p>{health.device_warning}</p></div>}
          {recorded ? <section className="inspector-section"><h3>Recorded predictions</h3><div className="notice"><strong>Selection-set development evidence</strong><p>Recorded forecasts are experimental. Replay does not run the installed model. Risk remains raw and uncalibrated.</p></div>
            {recordedCase && <><label className="field-label" htmlFor="recorded-seed">Paired training seed</label><select id="recorded-seed" value={recordedSeed ?? ''} disabled={!recordedCase.predictions.some((item) => item.analysis_seed != null)} onChange={(event) => setRecordedSeed(event.target.value ? Number(event.target.value) : null)}>{recordedSeed == null && <option value="">Reference run</option>}{[...new Set(recordedCase.predictions.map((item) => item.analysis_seed).filter((seed): seed is number => seed != null))].sort().map((seed) => <option key={seed} value={seed}>{seed}</option>)}</select><label className="field-label" htmlFor="recorded-model">Solid forecast</label><select id="recorded-model" value={primaryLabel} onChange={(event) => setRecordedLabel(event.target.value)}>{[...new Set(recordedCase.predictions.filter((item) => item.analysis_seed == null || item.analysis_seed === recordedSeed).map((item) => item.analysis_label))].map((label) => <option value={label} key={label}>{readable(label)}</option>)}</select><p className="helper">{original ? `Dashed comparison: ${readable(comparisonLabel)}. World indices are independent between models.` : 'World indices are shared by all actors within this prediction.'}</p><p className="file-path" title={prediction?.model.id}>Recorded model: {prediction?.model.id}</p></>}
          </section> : <section className="inspector-section"><h3>Forecast model</h3><label className="field-label" htmlFor="model-select">Predictor</label><select id="model-select" value={model} disabled={busy} onChange={(event) => setModel(event.target.value as ModelKind)}><option value="constant_velocity">Constant velocity baseline</option><option value="trained" disabled={!health?.checkpoint_loaded}>Temporal graph model{health?.checkpoint_loaded ? '' : ' — no checkpoint'}</option></select>
            {!health?.checkpoint_loaded && <div className="notice"><strong>No trained checkpoint loaded</strong><p>The baseline extrapolates the last observed velocity. Train a model and reopen the application with its checkpoint to compare learned futures.</p></div>}
            {health?.checkpoint_loaded && <div className="model-detail"><span>Checkpoint</span><code title={health.checkpoint_id ?? ''}>{health.checkpoint_id}</code><span>Calibration</span><span>{health.calibration_loaded ? 'Available' : 'Not loaded'}</span></div>}
            <button className="primary full-width" disabled={!scenario || busy} onClick={() => void runPrediction(false)}>{busy ? 'Computing…' : 'Predict original'}</button>
            {prediction && <p className="model-timing">{prediction.elapsed_ms.toFixed(0)} ms inference <span>{isBaseline ? 'Heuristic baseline' : prediction.model.id}</span></p>}
            {predictionError && <div className="notice error" role="alert">{predictionError}</div>}
          </section>}

          <section className="inspector-section"><div className="section-title"><h3>Future worlds</h3><span>{prediction?.probabilities.length ?? 0} modes</span></div><p className="helper">One world shares a probability across all predicted actors.</p><div className="world-list">{prediction ? prediction.probabilities.map((value, index) => <button key={index} aria-label={recorded ? `${readable(primaryLabel)} world ${index + 1}` : undefined} className={`world-row ${mode === index ? 'selected' : ''}`} aria-pressed={mode === index} onClick={() => setMode(index)}><span className="world-number">{String(index + 1).padStart(2, '0')}</span><span className="world-bar"><i style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} /></span><strong>{probability(value)}</strong></button>) : <p className="panel-note compact">World probabilities appear after prediction.</p>}</div>{recorded && original && <><label className="field-label" htmlFor="comparison-world">{readable(comparisonLabel)} world (dashed)</label><select id="comparison-world" aria-label="Comparison world" value={comparisonMode} onChange={(event) => setComparisonMode(Number(event.target.value))}>{original.probabilities.map((value, index) => <option value={index} key={index}>World {index + 1}: {probability(value)}</option>)}</select><p className="helper">Changing this world does not change the solid forecast.</p></>}{isBaseline && <p className="helper baseline-note">The six baseline worlds are identical; only world 1 carries probability.</p>}</section>

          <section className="inspector-section"><div className="section-title"><h3>Selected actor</h3><span className="actor-role">{actor?.focal ? 'Focal' : actor?.ego ? 'Ego' : actor ? 'Context' : '—'}</span></div><label className="sr-only" htmlFor="actor-select">Select actor</label><select id="actor-select" value={selectedActor} disabled={!visibleScenario} onChange={(event) => setSelectedActor(event.target.value)}>{!visibleScenario && <option value="">No actor selected</option>}{visibleScenario?.actors.map((item) => <option key={item.id} value={item.id}>{typeLabels[item.type] ?? readable(item.type)} {item.focal ? '(focal)' : item.ego ? '(ego)' : item.id.slice(-8)}</option>)}</select>{actor && <p className="actor-id" title={actor.id}>{actor.id}</p>}
            <div className="risk-readout"><div><span>{isCalibrated ? 'Calibrated interaction risk' : 'Experimental interaction score'}</span><strong className={(risk ?? 0) >= 0.5 ? 'risk-high' : (risk ?? 0) >= 0.2 ? 'risk-medium' : ''}>{probability(risk)}</strong></div><span className={`risk-status ${isCalibrated ? 'calibrated' : ''}`}>{isCalibrated ? 'Calibrated' : isBaseline ? 'Heuristic · uncalibrated' : 'Raw · uncalibrated'}</span></div>
            {isCalibrated && <div className="metric-line"><span>Before calibration</span><strong>{probability(rawRisk)}</strong></div>}
            {original && <div className="comparison-values"><div><span>{recorded ? readable(comparisonLabel) : 'Original'}</span><strong>{probability(baseRisk)}</strong></div><div><span>{recorded ? readable(primaryLabel) : 'Stressed'}</span><strong>{probability(risk)}</strong></div><div><span>Change</span><strong>{risk != null && baseRisk != null ? `${risk - baseRisk >= 0 ? '+' : ''}${((risk - baseRisk) * 100).toFixed(1)} pp` : '—'}</strong></div></div>}
            {insufficientCalibration && <div className="notice"><strong>Experimental · insufficient calibration evidence</strong><p>Both calibration and evaluation need at least {evidence?.minimum_positives ?? 50} positive events. Raw scores are shown.</p>{evidence && <p>Calibration: {evidence.calibration_positives ?? 'unknown'} / {evidence.calibration_count ?? 'unknown'} eligible actors positive. Evaluation: {evidence.evaluation_positives ?? 'unknown'} / {evidence.evaluation_count ?? 'unknown'} positive.</p>}</div>}
            {prediction?.risk.status === 'evaluation_evidence_unverified' && <div className="notice"><strong>Experimental · evaluation evidence unverified</strong><p>The saved evaluation could not be matched to this model and data. Raw scores are shown.</p></div>}
            {unknownCalibration && <div className="notice"><strong>Experimental · calibration applicability unknown</strong><p>{recorded ? 'These development predictions use raw world probabilities and risk scores. No calibrated-risk claim applies to this analysis.' : 'This imported or stressed scene is outside the original calibration population. World probabilities and risk scores are uncalibrated.'}</p></div>}
            {prediction?.risk.status === 'untrained_risk_head' && <div className="notice"><strong>No eligible risk supervision</strong><p>This training subset has no fully observed interaction groups. Forecasts are available; the unsupervised risk head is hidden.</p></div>}
            <p className="helper">Another retained dynamic actor within 3 m over 6 s. A research proxy; actor symbols use approximate sizes.</p>
          </section>

          {!recorded && <section className="inspector-section stress-section"><div className="section-title"><h3>Unexpected behavior</h3><span className="small-tag">Stress test</span></div><p className="helper">{stressKind === 'position_noise' || stressKind === 'dropout' ? 'Corrupt observed history and retain recorded future labels, then compare against the original scene.' : 'Change motion from one second before now. Other actors replay recorded paths and do not react to the intervention.'}</p><label className="field-label" htmlFor="stress-kind">Perturbation</label><select id="stress-kind" value={stressKind} onChange={(event) => setStressKind(event.target.value as StressKind)}>{Object.entries(stressLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select><label className="field-label" htmlFor="stress-actor">Affected actor</label><select id="stress-actor" value={stressActor} disabled={!scenario} onChange={(event) => setStressActor(event.target.value)}>{!scenario && <option value="">No actor available</option>}{scenario?.actors.map((item) => <option key={item.id} value={item.id}>{typeLabels[item.type] ?? readable(item.type)} {item.focal ? '(focal)' : item.ego ? '(ego)' : item.id.slice(-8)}</option>)}</select><label className="severity-label" htmlFor="severity">Severity <strong>{Math.round(severity * 100)}%</strong></label><input id="severity" type="range" min={0.05} max={1} step={0.05} value={severity} onChange={(event) => setSeverity(Number(event.target.value))} /><div className="range-labels"><span>Mild</span><span>Strong</span></div><button className="secondary full-width" disabled={!scenario || busy} onClick={() => void runPrediction(true)}>Compare stress test</button><p className="helper seed-note">Repeatable perturbation · seed 42</p></section>}
          <section className="inspector-section">
            <details className="utility-details"><summary>Installed model evaluation</summary>{reports?.evaluation ? <><p className="helper">Original benchmark · {reports.evaluation.scenario_count.toLocaleString()} scenarios. Imported scenes are not part of this report.</p><dl className="report-metrics"><dt>minADE</dt><dd>{metres(reports.evaluation.minADE)}</dd><dt>minFDE</dt><dd>{metres(reports.evaluation.minFDE)}</dd><dt>Miss rate at 2 m</dt><dd>{probability(reports.evaluation.miss_rate)}</dd><dt>Footprint overlap proxy</dt><dd>{probability(reports.evaluation.footprint_collision_proxy)}</dd></dl><p className="helper">Risk evidence: {readable(reports.risk_calibration_status)}. Proximity and footprint overlap are research proxies.</p>{reports.provenance?.training_manifest_sha256 && <p className="file-path" title={reports.provenance.training_manifest_sha256}>Manifest {reports.provenance.training_manifest_sha256}</p>}</> : <p className="helper">This model has no saved evaluation report. Predictions are experimental.</p>}</details>
          </section>
          <section className="inspector-section">
            <details className="utility-details"><summary>Export & diagnostics</summary><p className="helper">Save the current prediction and, when shown, its comparison. Reference errors use available steps and are separate from benchmark scores.</p><div className="utility-buttons"><button disabled={!desktop || exportBusy || !prediction?.prediction_id} onClick={() => void saveExport('prediction_json')}>Prediction JSON…</button><button disabled={!desktop || exportBusy || !prediction?.prediction_id} onClick={() => void saveExport('report_csv')}>Actor report CSV…</button><button disabled={!desktop || exportBusy || !prediction?.prediction_id} onClick={() => void saveExport('report_png')}>Scenario report PNG…</button><button disabled={!desktop || exportBusy} onClick={() => void saveExport('diagnostics')}>Diagnostics ZIP…</button></div>{!desktop && <p className="helper">Native save dialogs are available in the desktop application.</p>}{exportBusy && <p className="helper" role="status">Choose a destination in the save dialog…</p>}{exportMessage && <p className="helper" role="status">{exportMessage}</p>}{exportError && <p className="inline-error" role="alert">{exportError} {recorded ? 'Reopen this recorded case if its export expired, then retry.' : 'Run the prediction again if it expired, then retry.'}</p>}</details>
          </section>
          <section className="inspector-section">
            <details className="utility-details"><summary>Application settings</summary><p className="field-label">Current data folder</p><p className="file-path">{desktop?.data_root ?? health?.data_root ?? 'Managed by the local service'}</p><div className="utility-buttons"><button disabled={!nativeEnabled || importPending(importJob)} onClick={() => void changeSetting('data')}>Choose data folder…</button><button disabled={!nativeEnabled || importPending(importJob)} onClick={() => void changeSetting('model')}>Load model folder…</button><button disabled={!nativeEnabled || importPending(importJob)} onClick={() => void changeSetting('archive')}>Load model ZIP…</button></div><p className="helper">Folder and model changes apply after restarting the application. Training checkpoints must first be exported as an inference bundle.</p>{settingsMessage && <p className="notice" role="status">{settingsMessage}</p>}{desktopError && <p className="inline-error" role="alert">{desktopError}</p>}</details>
          </section>
        </div>
        <footer className="inspector-footer">Research workspace <span>Local only</span></footer>
      </aside>
    </main>
  </div>;
}
