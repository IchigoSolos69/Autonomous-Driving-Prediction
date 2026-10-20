// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { AnalysisCase, AnalysisDetail, AnalysisSummary, DesktopApi, RecordedPrediction, Scenario } from './types';

vi.mock('./ScenarioCanvas', () => ({
  default: ({ prediction, original, mode, originalMode }: { prediction: RecordedPrediction; original: RecordedPrediction; mode: number; originalMode: number }) => <div data-testid="canvas" data-primary={prediction?.prediction_id} data-comparison={original?.prediction_id} data-mode={mode} data-comparison-mode={originalMode} />,
}));
const response = (value: unknown) => ({ ok: true, json: async () => value }) as Response;
const scenario: Scenario = {
  id: 'recorded', city: 'pittsburgh', focal_track_id: 'car', ego_track_id: null, dt: 0.1, observed_steps: 2, future_steps: 1,
  coordinate_frame: { origin: [0, 0], rotation: 0, units: 'm', name: 'focal' },
  actors: [{ id: 'car', type: 'vehicle', focal: true, ego: false, history: [[0, 0], [1, 0]], headings: [0, 0], reference_future: [[2, 0]] }],
  map: { lanes: [], crosswalks: [], drivable_areas: [] }, selection: { retained_dynamic: 1, total_dynamic: 1 },
};
const summary: AnalysisSummary = { id: 'opaque-analysis', title: 'Direction experiment', split: 'validation', development_only: true, counts: { actors: 82, scenarios: 49, cases: 1, small_sample: true } };
const detail: AnalysisDetail = { ...summary, metrics: { 'control-42': { jointFDE: 4, focal_minFDE: 3, minADE: 1, miss_rate: 0.2, retained_target_coverage: 0.8, incomplete_future_exclusions: 2 } }, model_labels: ['control-42', 'directional-42'], provenance: { protocol_sha256: 'frozen' },
  slices: [{ dimension: 'speed_bin', value: '1-5', actor_count: 82, scenario_count: 49, small_sample: true, metrics: {} }, { dimension: 'city', value: 'pittsburgh', actor_count: 82, scenario_count: 49, small_sample: true, metrics: {} }],
  cases: [{ scenario_id: 'recorded', city: 'pittsburgh', focal_type: 'vehicle', actor_count: 1, focal_minFDE: 3, jointFDE: 4, delta_jointFDE: 1, actors: [] }],
};
const prediction = (label: string, seed: number): RecordedPrediction => ({
  scenario_id: 'recorded', prediction_id: `${label}-${seed}`, actor_ids: ['car'], trajectories: [[[[2, 0]]], [[[3, 0]]]], scales: [[[[1, 1]]], [[[1, 1]]]], probabilities: label === 'directional' ? [0.2, 0.8] : [0.8, 0.2],
  risk: { raw: [0.1], calibrated: null, status: 'population_applicability_unknown' }, model: { id: `${label}-model-${seed}`, kind: 'trained' }, calibration_id: null,
  coordinate_frame: scenario.coordinate_frame, scenario, stress: null, elapsed_ms: 1, analysis_label: label, analysis_seed: seed,
});
const recordedCase: AnalysisCase = { analysis_id: summary.id, scenario, predictions: [42, 43, 44].flatMap((seed) => [prediction('control', seed), prediction('directional', seed)]) };
const native = (choose: () => Promise<AnalysisSummary | null> = async () => summary): DesktopApi => ({
  desktop_context: vi.fn(async () => ({ desktop: true as const, token: 'session-token', data_root: 'C:\\研究', settings: {} })),
  choose_analysis_directory: vi.fn(choose), choose_import_directory: vi.fn(async () => null), choose_data_directory: vi.fn(async () => null), choose_model_bundle: vi.fn(async () => null),
});

describe('recorded failure review', () => {
  let container: HTMLDivElement, root: Root;
  let fetchMock: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    container = document.createElement('div'); document.body.append(container); root = createRoot(container);
    window.pywebview = { api: native() };
    fetchMock = vi.fn(async (url: string) => {
      if (url === '/api/health') return response({ device: 'cpu', checkpoint_loaded: false, checkpoint_id: null, calibration_loaded: false });
      if (url === '/api/scenarios') return response({ scenarios: [], total: 0 });
      if (url === '/api/reports') return response({ evaluation: null, experimental: true, risk_calibration_status: 'not_fitted' });
      if (url === '/api/analyses') return response({ analyses: [summary] });
      if (url === `/api/analyses/${summary.id}/cases/recorded`) return response(recordedCase);
      if (url.startsWith(`/api/analyses/${summary.id}`)) return response(detail);
      if (url === '/api/exports') return response({ saved: true, cancelled: false, filename: 'recorded.json' });
      throw new Error(`Unexpected request ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(async () => { await act(async () => root.unmount()); container.remove(); delete window.pywebview; vi.unstubAllGlobals(); });
  const button = (text: string) => {
    const result = [...container.querySelectorAll('button')].find((item) => item.textContent === text);
    if (!result) throw new Error(`Missing ${text}`);
    return result;
  };
  const change = async (select: HTMLSelectElement, value: string) => {
    await act(async () => { select.value = value; select.dispatchEvent(new Event('change', { bubbles: true })); });
  };
  const openLibrary = async () => { await act(async () => root.render(<App />)); await act(async () => button('Failures').click()); };
  const openCase = async () => { await openLibrary(); await act(async () => container.querySelector<HTMLButtonElement>('.failure-results .scenario-row')!.click()); };

  it('replays saved predictions without inference, starts paused, and keeps model worlds independent', async () => {
    await openCase();
    const canvas = () => container.querySelector<HTMLElement>('[data-testid="canvas"]')!;
    expect(canvas().dataset.primary).toBe('directional-42'); expect(canvas().dataset.comparison).toBe('control-42');
    expect(canvas().dataset.mode).toBe('1'); expect(canvas().dataset.comparisonMode).toBe('0');
    expect(container.querySelector('[aria-label="Play scenario"]')).not.toBeNull();
    expect(container.textContent).toContain('Selection-set development evidence');
    expect(container.textContent).not.toContain('Predict original'); expect(container.textContent).not.toContain('Compare stress test');
    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="Directional world 1"]')!.click());
    expect(canvas().dataset.mode).toBe('0'); expect(canvas().dataset.comparisonMode).toBe('0');
    await change(container.querySelector('#comparison-world')!, '1');
    expect(canvas().dataset.mode).toBe('0'); expect(canvas().dataset.comparisonMode).toBe('1');
    await change(container.querySelector('#recorded-seed')!, '43');
    expect(canvas().dataset.primary).toBe('directional-43'); expect(canvas().dataset.comparison).toBe('control-43');
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/predict')).toBe(false);
  });

  it('exports the displayed saved pair using opaque prediction IDs and the native session token', async () => {
    await openCase(); await change(container.querySelector('#recorded-seed')!, '44');
    await act(async () => button('Prediction JSON…').click());
    const exportCall = fetchMock.mock.calls.find(([url]) => url === '/api/exports')!;
    const init = exportCall[1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ kind: 'prediction_json', prediction_id: 'directional-44', comparison_prediction_id: 'control-44' });
    expect(new Headers(init.headers).get('X-Interaction-Token')).toBe('session-token');
    expect(container.textContent).toContain('Saved recorded.json.');
  });

  it('filters full slice statistics server-side and labels small scenario samples', async () => {
    await openLibrary();
    expect(container.textContent).toContain('82 actors'); expect(container.textContent).toContain('49 unique scenarios');
    expect(container.textContent).toContain('Small sample: fewer than 50 scenarios');
    expect(container.textContent).toContain('Incomplete-future exclusions');
    expect(container.textContent).toContain('Joint errors use whole scenes containing matching actors');
    expect(container.textContent).toContain('Retained-target coverage also uses these whole scenes.');
    await change(container.querySelector('[aria-label="Failure observed speed"]')!, '1-5');
    expect(fetchMock.mock.calls.some(([url]) => url === `/api/analyses/${summary.id}?speed_bin=1-5`)).toBe(true);
    await change(container.querySelector('[aria-label="Failure city"]')!, 'pittsburgh');
    expect(fetchMock.mock.calls.some(([url]) => url === `/api/analyses/${summary.id}?speed_bin=1-5&city=pittsburgh`)).toBe(true);
    expect(container.querySelector('[aria-label="Failure observed speed"]')?.getAttribute('value')).not.toBe('');
  });

  it('ignores a late case response after filters change and aborts the saved-case request', async () => {
    let complete!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => { complete = resolve; });
    const initial = fetchMock.getMockImplementation()! as (url: string, init?: RequestInit) => Promise<Response>;
    fetchMock.mockImplementation((url: string, init?: RequestInit) => url.includes('/cases/') ? pending : initial(url, init));
    await openLibrary();
    await act(async () => container.querySelector<HTMLButtonElement>('.failure-results .scenario-row')!.click());
    const caseCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/cases/'))!;
    await change(container.querySelector('[aria-label="Failure observed speed"]')!, '1-5');
    expect((caseCall[1] as RequestInit).signal?.aborted).toBe(true);
    await act(async () => complete(response(recordedCase)));
    expect(container.querySelector('[data-testid="canvas"]')).toBeNull();
  });

  it('preserves current analysis when native folder selection is cancelled', async () => {
    const api = native(async () => null); window.pywebview = { api };
    await openCase(); await act(async () => button('Load analysis folder…').click());
    expect(api.choose_analysis_directory).toHaveBeenCalledOnce();
    expect(container.textContent).toContain('Analysis selection cancelled.');
    expect(container.querySelector('[data-testid="canvas"]')?.getAttribute('data-primary')).toBe('directional-42');
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('C:'))).toBe(false);
  });

  it('shows an actionable invalid-folder error without discarding the active replay', async () => {
    window.pywebview = { api: native(async () => { throw new Error('Analysis hash mismatch'); }) };
    await openCase(); await act(async () => button('Load analysis folder…').click());
    expect(container.textContent).toContain('Analysis hash mismatch Select a valid exported analysis folder and try again.');
    expect(container.querySelector('[data-testid="canvas"]')?.getAttribute('data-primary')).toBe('directional-42');
  });
});
