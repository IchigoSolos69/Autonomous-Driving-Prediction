// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { Health, Prediction, Scenario, ScenarioSummary } from './types';

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
  scenario_id: id, actor_ids: ['car'], trajectories: [[[[2, 0]]]], scales: [[[[1, 1]]]], probabilities: [1],
  risk: { raw: [0.1], calibrated: null, status: 'uncalibrated' }, model: { id: 'checkpoint', kind: 'trained' },
  calibration_id: null, coordinate_frame: scene(id).coordinate_frame, scenario: scene(id), stress: null, elapsed_ms: 1,
});

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
});
