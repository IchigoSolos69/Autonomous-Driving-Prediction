import { useEffect, useRef, useState } from 'react';
import { readable, request } from './data';
import type { AnalysisCase, AnalysisDetail, AnalysisMetrics, AnalysisSummary } from './types';

const filters = [
  ['city', 'City'], ['actor_type', 'Actor type'], ['role', 'Actor role'],
  ['speed_bin', 'Observed speed'], ['turn_bin', 'Observed turning'],
  ['density_bin', 'Nearby actor density'], ['history_bin', 'History coverage'],
] as const;
type FilterKey = typeof filters[number][0];
const message = (error: unknown) => error instanceof Error ? error.message : 'The analysis could not be opened.';
const number = (value: number | null | undefined) => value == null ? '—' : value.toFixed(3);
const metricNames: Record<string, string> = { minADE: 'minADE (m)', minFDE: 'minFDE (m)', focal_minFDE: 'Focal minFDE (m)', jointFDE: 'Joint minFDE (m)', miss_rate: 'Miss rate at 2 m', focal_miss_rate: 'Focal miss rate', retained_target_coverage: 'Retained-target coverage', incomplete_future_exclusions: 'Incomplete-future exclusions' };

export function Metrics({ metrics }: { metrics: AnalysisMetrics }) {
  return <div className="failure-metrics">{Object.entries(metrics).map(([label, values]) => <div key={label}>
    <strong>{readable(label)}</strong><dl className="report-metrics">{Object.entries(values).filter(([, value]) => value == null || typeof value === 'number').map(([name, value]) => <div key={name}><dt>{metricNames[name] ?? readable(name)}</dt><dd>{name.includes('rate') || name.includes('coverage') ? value == null ? '—' : `${(value * 100).toFixed(1)}%` : number(value)}</dd></div>)}</dl>
  </div>)}</div>;
}

interface Props { desktop: boolean; nativeBusy: boolean; activeCase: string; onCase: (value: AnalysisCase) => void; onReset: () => void }

export default function FailureLibrary({ desktop, nativeBusy, activeCase, onCase, onReset }: Props) {
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [analysisId, setAnalysisId] = useState('');
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [options, setOptions] = useState<Partial<Record<FilterKey, string[]>>>({});
  const [selected, setSelected] = useState<Partial<Record<FilterKey, string>>>({});
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [opening, setOpening] = useState('');
  const [importing, setImporting] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const caseRequest = useRef<AbortController | null>(null);
  const filterQuery = new URLSearchParams(Object.entries(selected).filter(([, value]) => value)).toString();

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true); setError('');
    request<{ analyses: AnalysisSummary[] }>('/analyses', { signal: controller.signal }).then((result) => {
      if (cancelled) return;
      setAnalyses(result.analyses);
      setAnalysisId((current) => result.analyses.some((item) => item.id === current) ? current : result.analyses[0]?.id ?? '');
    }).catch((error) => { if (!cancelled) setError(message(error)); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; controller.abort(); };
  }, [revision]);

  useEffect(() => {
    if (!analysisId) return;
    const controller = new AbortController();
    let cancelled = false;
    caseRequest.current?.abort(); setOpening(''); onReset();
    setDetailLoading(true); setError(''); setDetail(null);
    request<AnalysisDetail>(`/analyses/${encodeURIComponent(analysisId)}${filterQuery ? `?${filterQuery}` : ''}`, { signal: controller.signal }).then((result) => {
      if (cancelled) return;
      setDetail(result);
      if (!filterQuery) setOptions(Object.fromEntries(filters.map(([key]) => [key, [...new Set(result.slices.filter((slice) => slice.dimension === key).map((slice) => slice.value))].sort()])));
    }).catch((error) => { if (!cancelled) setError(message(error)); }).finally(() => { if (!cancelled) setDetailLoading(false); });
    return () => { cancelled = true; controller.abort(); };
  }, [analysisId, filterQuery, revision, onReset]);
  useEffect(() => () => caseRequest.current?.abort(), []);

  const openCase = async (scenarioId: string) => {
    caseRequest.current?.abort();
    const controller = new AbortController(); caseRequest.current = controller;
    setOpening(scenarioId); setError('');
    try {
      const result = await request<AnalysisCase>(`/analyses/${encodeURIComponent(analysisId)}/cases/${encodeURIComponent(scenarioId)}`, { signal: controller.signal });
      if (!controller.signal.aborted) onCase(result);
    } catch (error) { if (!controller.signal.aborted) setError(message(error)); }
    finally { if (!controller.signal.aborted) setOpening(''); }
  };

  const importAnalysis = async () => {
    const api = window.pywebview?.api;
    if (!desktop || !api?.choose_analysis_directory) return;
    setImporting(true); setError(''); setNotice('');
    try {
      const result = await api.choose_analysis_directory();
      if (!result) { setNotice('Analysis selection cancelled.'); return; }
      setSelected({}); setOptions({}); setAnalysisId(result.id); setRevision((value) => value + 1);
      setNotice('Analysis imported. Source files are preserved.');
    } catch (error) { setError(`${message(error)} Select a valid exported analysis folder and try again.`); }
    finally { setImporting(false); }
  };

  return <div className="failure-library" aria-label="Failure analysis library">
    <div className="import-controls">
      <button disabled={!desktop || nativeBusy || importing || !window.pywebview?.api?.choose_analysis_directory} onClick={() => void importAnalysis()}>{importing ? 'Opening analysis…' : 'Load analysis folder…'}</button>
      {!desktop && <p className="helper">Use the desktop application to import a recorded analysis folder.</p>}
      <p className="helper">Recorded predictions · selection set · development evidence. These previews do not run inference.</p>
      {notice && <p className="helper" role="status">{notice}</p>}
      {error && <div className="inline-error" role="alert">{error}<button className="text-button" onClick={() => setRevision((value) => value + 1)}>Retry analysis</button></div>}
    </div>
    {loading ? <p className="panel-note" role="status">Loading analyses…</p> : !analyses.length ? <p className="panel-note">No saved analyses. Generate a failure report, then load its analysis folder.</p> : <>
      <div className="failure-filter-block">
        <label className="field-label" htmlFor="analysis-select">Analysis</label><select id="analysis-select" value={analysisId} onChange={(event) => { setSelected({}); setOptions({}); setAnalysisId(event.target.value); }}>{analyses.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select>
        <details className="failure-filters"><summary>Filter actors and cases</summary>{filters.map(([key, label]) => <label key={key} className="failure-filter-label">{label}<select aria-label={`Failure ${label.toLowerCase()}`} value={selected[key] ?? ''} onChange={(event) => setSelected((current) => ({ ...current, [key]: event.target.value }))}><option value="">All</option>{options[key]?.map((value) => <option key={value} value={value}>{readable(value)}</option>)}</select></label>)}{filterQuery && <button className="text-button" onClick={() => setSelected({})}>Clear failure filters</button>}</details>
      </div>
      {detailLoading ? <p className="panel-note" role="status">Computing slice summary…</p> : detail && <div className="failure-results">
        <div className="failure-counts"><strong>{detail.counts.actors.toLocaleString()} actors</strong><span>{detail.counts.scenarios.toLocaleString()} unique scenarios</span>{detail.counts.small_sample && <span className="small-sample">Small sample: fewer than 50 scenarios</span>}<span>{detail.cases.length} recorded previews; statistics cover all matching selection scenarios.</span></div>
        <details className="failure-stats"><summary>Displacement and coverage</summary><Metrics metrics={detail.metrics} /><p className="helper">Primary displacement errors require complete scored futures. Exclusions are reported separately.</p><p className="helper">{detail.joint_metric_scope ?? 'Joint errors use whole scenes containing matching actors; worlds remain shared.'} Retained-target coverage also uses these whole scenes.</p></details>
        <details className="failure-stats"><summary>Slice evidence</summary>{detail.slices.map((slice) => <details key={`${slice.dimension}:${slice.value}`} className="failure-slice"><summary>{readable(slice.dimension)}: {readable(slice.value)}<span>{slice.actor_count} actors / {slice.scenario_count} scenarios{slice.small_sample ? ' · Small sample' : ''}</span></summary><Metrics metrics={slice.metrics} /></details>)}</details>
        <details className="failure-stats"><summary>Frozen slice definitions</summary><p className="helper">Speed: &lt;1, 1–5, 5–15, ≥15 m/s. Turning: signed heading change beyond ±15° in the final observed second; incomplete history is unknown. Density: other retained dynamic actors within 20 m, grouped 0, 1–3, 4+. History: below or at least 80% observed.</p></details>
        <div className="failure-case-heading">Recorded cases <span>{detail.cases.length} / 64 maximum</span></div>
        {detail.cases.length ? detail.cases.map((item) => <button key={item.scenario_id} className={`scenario-row ${activeCase === item.scenario_id ? 'active' : ''}`} aria-pressed={activeCase === item.scenario_id} onClick={() => void openCase(item.scenario_id)}>
          <div className="row-top"><strong>{readable(item.city)}</strong><span>{readable(item.focal_type)}</span></div><div className="scenario-id" title={item.scenario_id}>{item.scenario_id.slice(0, 20)}</div><div className="failure-case-values"><span>Focal minFDE <strong>{number(item.focal_minFDE)} m</strong></span><span>Joint minFDE <strong>{number(item.jointFDE)} m</strong></span>{item.delta_jointFDE != null && <span>Directional − control <strong>{item.delta_jointFDE >= 0 ? '+' : ''}{number(item.delta_jointFDE)} m</strong></span>}</div>{opening === item.scenario_id && <p role="status">Opening recorded case…</p>}
        </button>) : <p className="panel-note">No recorded previews match these actor filters. Full slice counts above may include unrecorded scenarios.</p>}
        <details className="failure-stats"><summary>Analysis provenance</summary><pre className="failure-provenance">{JSON.stringify(detail.provenance, null, 2)}</pre></details>
      </div>}
    </>}
  </div>;
}
