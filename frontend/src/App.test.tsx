// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { DesktopApi, Health, ImportJob, Prediction, Scenario, ScenarioSummary } from './types';

vi.mock('./ScenarioCanvas', () => ({
  default: ({ scenario, prediction }: { scenario: Scenario; prediction: Prediction | null }) =>
    <div data-testid="canvas" data-scenario={scenario.id} data-prediction={prediction?.scenario_id} />,
}));

const health: Health = { device: 'cpu', checkpoint_loaded: true, checkpoint_id: 'checkpoint', calibration_loaded: false };
const response = (value: unknown) => ({ ok: true, json: async () => value }) as Response;
const summary = (id: string): ScenarioSummary => ({ id, city: 'pittsburgh', actor_count: 1, focal_type: 'vehicle', split: 'evaluation' });
const scene = (id: string): Scenario => ({
  id, city: 'pittsburgh', focal_track_id: 'car', ego_track_id: null, dt: 0.1, observed_steps: 2, future_steps: 1,
  coordinate_frame: { origin: [0, 0], rotation: 0, units: 'm', name: 'focal' },
  actors: [{ id: 'car', type: 'vehicle', focal: true, ego: false, history: [[0, 0], [1, 0]], headings: [0, 0], reference_future: [[2, 0]] }],
  map: { lanes: [], crosswalks: [], drivable_areas: [] }, selection: { retained_dynamic: 1, total_dynamic: 1 },
});
const forecast = (id: string): Prediction => ({
  scenario_id: id, prediction_id: `prediction-${id}`, actor_ids: ['car'], trajectories: [[[[2, 0]]]], scales: [[[[1, 1]]]], probabilities: [1],
  risk: { raw: [0.1], calibrated: null, status: 'uncalibrated' }, model: { id: 'checkpoint', kind: 'trained' },
  calibration_id: null, coordinate_frame: scene(id).coordinate_frame, scenario: scene(id), stress: null, elapsed_ms: 1,
});
const nativeApi = (): DesktopApi => ({
  desktop_context: vi.fn(async () => ({ desktop: true as const, token: 'native-session-secret', data_root: 'C:\\Interaction data', settings: {} })),
  choose_import_directory: vi.fn(async () => ({ source_token: 'native-source-selection-token', display_name: 'AV2 scenarios' })),
  choose_data_directory: vi.fn(async () => ({ data_root: 'D:\\Scenarios', restart_required: true as const })),
  choose_model_bundle: vi.fn(async () => ({ bundle_id: 'model-bundle', restart_required: true as const })),
});
const job = (status: ImportJob['status']): ImportJob => ({ id: 'job1', status, processed: status === 'completed' ? 1 : 0, total: 1, imported: status === 'completed' ? 1 : 0, errors: [] });
const publicResponse = (url: string, init?: RequestInit, ids = ['alpha']) => {
  if (url === '/api/health') return response({ ...health, desktop: true, import_available: true });
  if (url === '/api/scenarios') return response({ scenarios: ids.map((id) => ({ ...summary(id), source: id.startsWith('import:') ? 'imported' : 'benchmark' })), total: ids.length });
  if (url === '/api/reports') return response({ evaluation: null, model: null, provenance: null, risk_calibration_status: 'not_fitted', experimental: true });
  if (url.startsWith('/api/scenarios/')) return response(scene(decodeURIComponent(url.split('/').at(-1)!)));
  if (url === '/api/predict') return response(forecast((JSON.parse(init?.body as string) as { scenario_id: string }).scenario_id));
  throw new Error(`Unexpected request: ${url}`);
};

describe('scenario application', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    delete window.pywebview;
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  const button = (container: HTMLElement, text: string) => {
    const found = Array.from(container.querySelectorAll('button')).find((item) => item.textContent === text);
    if (!found) throw new Error(`Missing button: ${text}`);
    return found;
  };

  it('shows offline after a previously successful health check fails on refresh', async () => {
    let connected = true;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/health') {
        if (!connected) throw new Error('Connection refused');
        return response(health);
      }
      if (url === '/api/scenarios') return response({ scenarios: [], total: 0 });
      throw new Error(`Unexpected request: ${url}`);
    }));

    await act(async () => root.render(<App />));
    expect(container.querySelector('.header-status')?.textContent).toContain('Local model service connected');
    expect(container.querySelector('.status-dot.online')).not.toBeNull();

    connected = false;
    await act(async () => button(container, 'Refresh').click());
    expect(container.querySelector('.header-status')?.textContent).toContain('Local model service offline');
    expect(container.querySelector('.header-device')?.textContent).toBe('No device');
    expect(container.querySelector('.status-dot.online')).toBeNull();
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('Cannot reach the local prediction service');
  });

  it('keeps the new scene when an older manual prediction completes late', async () => {
    let alphaPredictions = 0;
    let completeOldPrediction!: (value: Response) => void;
    const pendingPrediction = new Promise<Response>((resolve) => { completeOldPrediction = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/health') return response(health);
      if (url === '/api/scenarios') return response({ scenarios: ['alpha', 'beta'].map(summary), total: 2 });
      if (url.startsWith('/api/scenarios/')) return response(scene(url.split('/').at(-1)!));
      if (url === '/api/predict') {
        const { scenario_id: id } = JSON.parse(init?.body as string) as { scenario_id: string };
        if (id === 'alpha' && ++alphaPredictions === 2) return pendingPrediction;
        return response(forecast(id));
      }
      throw new Error(`Unexpected request: ${url}`);
    }));

    await act(async () => root.render(<App />));
    expect(container.querySelector('[data-testid="canvas"]')?.getAttribute('data-prediction')).toBe('alpha');
    await act(async () => button(container, 'Predict original').click());
    await act(async () => {
      const beta = Array.from(container.querySelectorAll<HTMLButtonElement>('.scenario-row')).find((item) => item.textContent?.includes('beta'));
      expect(beta).toBeDefined();
      beta!.click();
    });
    expect(container.querySelector('[data-testid="canvas"]')?.getAttribute('data-prediction')).toBe('beta');

    await act(async () => completeOldPrediction(response(forecast('alpha'))));
    const canvas = container.querySelector('[data-testid="canvas"]');
    expect(canvas?.getAttribute('data-scenario')).toBe('beta');
    expect(canvas?.getAttribute('data-prediction')).toBe('beta');
    expect(button(container, 'Predict original').disabled).toBe(false);
  });

  it.each([true, false])('enables native actions when the bridge is ready (already present: %s)', async (alreadyReady) => {
    const api = nativeApi();
    if (alreadyReady) window.pywebview = { api };
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => publicResponse(url, init, [])));
    await act(async () => root.render(<App />));
    if (!alreadyReady) {
      expect(button(container, 'Import AV2 folder…').disabled).toBe(true);
      await act(async () => { window.pywebview = { api }; window.dispatchEvent(new Event('pywebviewready')); });
    }
    expect(api.desktop_context).toHaveBeenCalledOnce();
    expect(button(container, 'Import AV2 folder…').disabled).toBe(false);
    expect(container.querySelector('.file-path')?.textContent).toBe('C:\\Interaction data');
  });

  it('imports through native selection tokens, polls once at a time, and selects the new scene', async () => {
    vi.useFakeTimers();
    const api = nativeApi();
    window.pywebview = { api };
    let completed = false;
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.startsWith('/api/imports')) {
        expect(new Headers(init?.headers).get('X-Interaction-Token')).toBe('native-session-secret');
        if (url === '/api/imports') {
          expect(JSON.parse(init?.body as string)).toEqual({ source_token: 'native-source-selection-token' });
          return response(job('queued'));
        }
        completed = true;
        return response(job('completed'));
      }
      return publicResponse(url, init, completed ? ['alpha', 'import:job1:beta'] : ['alpha']);
    });
    vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<App />));
    await act(async () => button(container, 'Import AV2 folder…').click());
    expect(api.choose_import_directory).toHaveBeenCalledOnce();
    expect(container.querySelector('.import-progress')?.textContent).toContain('Import Queued');

    await act(async () => { await vi.advanceTimersByTimeAsync(800); });
    expect(container.querySelector('.import-progress')?.textContent).toContain('Import Completed');
    expect(container.querySelector('[data-testid="canvas"]')?.getAttribute('data-scenario')).toBe('import:job1:beta');
    expect(container.querySelector('.scenario-row.active')?.textContent).toContain('Imported');
    const pollCount = fetchMock.mock.calls.filter(([url]) => url === '/api/imports/job1').length;
    await act(async () => { await vi.advanceTimersByTimeAsync(2400); });
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/imports/job1')).toHaveLength(pollCount);
  });

  it('cancels imports with the native token and stops polling after cancellation', async () => {
    vi.useFakeTimers();
    window.pywebview = { api: nativeApi() };
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/imports') return response(job('running'));
      if (url === '/api/imports/job1/cancel') {
        expect(init?.method).toBe('POST');
        expect(new Headers(init?.headers).get('X-Interaction-Token')).toBe('native-session-secret');
        return response(job('cancelled'));
      }
      return publicResponse(url, init);
    });
    vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<App />));
    await act(async () => button(container, 'Import AV2 folder…').click());
    await act(async () => button(container, 'Cancel import').click());
    expect(container.querySelector('.import-progress')?.textContent).toContain('Import Cancelled');
    await act(async () => { await vi.advanceTimersByTimeAsync(1600); });
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/imports/job1')).toBe(false);
  });

  it('stops progress polling when the application unmounts', async () => {
    vi.useFakeTimers();
    window.pywebview = { api: nativeApi() };
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => url === '/api/imports' ? response(job('running')) : publicResponse(url, init));
    vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<App />));
    await act(async () => button(container, 'Import AV2 folder…').click());
    await act(async () => root.unmount());
    await act(async () => { await vi.advanceTimersByTimeAsync(2400); });
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/imports/job1')).toBe(false);
  });

  it('shows export failures without claiming a file was saved', async () => {
    window.pywebview = { api: nativeApi() };
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/exports') {
        expect(new Headers(init?.headers).get('X-Interaction-Token')).toBe('native-session-secret');
        expect(JSON.parse(init?.body as string)).toEqual({ kind: 'prediction_json', prediction_id: 'prediction-alpha' });
        return { ok: false, status: 422, statusText: 'Unprocessable Entity', json: async () => ({ detail: 'Export failed: disk full' }) } as Response;
      }
      return publicResponse(url, init);
    }));
    await act(async () => root.render(<App />));
    const exports = Array.from(container.querySelectorAll('details')).find((item) => item.querySelector('summary')?.textContent === 'Export & diagnostics');
    exports!.open = true;
    await act(async () => button(container, 'Prediction JSON…').click());
    expect(exports?.textContent).toContain('Export failed: disk full');
    expect(exports?.textContent).not.toContain('Saved export');
    expect(button(container, 'Prediction JSON…').disabled).toBe(false);
  });

  it.each([false, true])('reports the native save result (cancelled: %s)', async (cancelled) => {
    window.pywebview = { api: nativeApi() };
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => url === '/api/exports'
      ? response({ saved: !cancelled, cancelled, filename: cancelled ? undefined : 'scene.json' }) : publicResponse(url, init)));
    await act(async () => root.render(<App />));
    await act(async () => button(container, 'Prediction JSON…').click());
    expect(container.textContent).toContain(cancelled ? 'Export cancelled.' : 'Saved scene.json.');
    if (cancelled) expect(container.textContent).not.toContain('Saved scene.json.');
  });

  it('persists native model and data selections while explaining that a restart applies them', async () => {
    const api = nativeApi();
    window.pywebview = { api };
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => publicResponse(url, init)));
    await act(async () => root.render(<App />));
    await act(async () => button(container, 'Choose data folder…').click());
    expect(api.choose_data_directory).toHaveBeenCalledOnce();
    expect(container.textContent).toContain('Data folder saved. Close and reopen Interaction Lab to apply it.');
    expect(container.querySelector('.file-path')?.textContent).toBe('C:\\Interaction data');
    await act(async () => button(container, 'Load model ZIP…').click());
    expect(api.choose_model_bundle).toHaveBeenCalledWith(true);
    expect(container.textContent).toContain('Model bundle saved. Close and reopen Interaction Lab to apply it.');
  });

  it('keeps insufficient-evidence warnings visible when calibrated scores are withheld', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (url === '/api/predict') return response({ ...forecast('alpha'), risk: { raw: [0.1], calibrated: null, status: 'insufficient_positive_events' } });
      if (url === '/api/reports') return response({ evaluation: null, risk_calibration_status: 'insufficient_positive_events', experimental: true, risk_calibration_evidence: { calibration_positives: 2, calibration_count: 21, evaluation_positives: 14, evaluation_count: 95, minimum_positives: 50 } });
      return publicResponse(url, init);
    }));
    await act(async () => root.render(<App />));
    expect(container.textContent).toContain('Experimental · insufficient calibration evidence');
    expect(container.textContent).toContain('Calibration: 2 / 21 eligible actors positive. Evaluation: 14 / 95 positive.');
    expect(container.querySelector('.risk-readout')?.textContent).toContain('10.0%');
  });
});
