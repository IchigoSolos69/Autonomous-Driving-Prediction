// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import TrainingScreen from './TrainingScreen';
import App from './App';
import type { ComputeHostStatus, TrainingRun, TrainingSnapshot } from './types';

vi.mock('./ScenarioCanvas', () => ({ default: () => <div data-testid="scenario-canvas" /> }));
const response = (value: unknown) => ({ ok: true, json: async () => value }) as Response;
const names = ['full-42', 'no_graph-42', 'no_map-42', 'world_ce-42', 'full-43', 'full-44'];
const runs: TrainingRun[] = names.map((name, index) => ({ id: name, name, configuration: 'Six shared worlds', seed: index < 4 ? 42 : index === 4 ? 43 : 44,
  state: index === 0 ? 'stopped' : 'pending', planned_epochs: 15, completed_epochs: index === 0 ? 1 : 0,
  epoch_number: index === 0 ? 2 : null, committed_examples: index === 0 ? 19104 : null, training_examples: 40000,
  committed_batches: index === 0 ? 4776 : null, batches_per_epoch: 10000, committed_fraction: index === 0 ? .4776 : null,
  stage: index === 0 ? 'train' : 'pending', global_step: index === 0 ? 1847 : null,
  examples_per_second: index === 0 ? 43.6 : null, peak_vram_mib: null,
  checkpoint: { updated_at: index === 0 ? '2026-10-19T04:30:00Z' : null, age_seconds: index === 0 ? 318 : null, stale: false },
  history: index === 0 ? [{ epoch: 1, train_loss: 3.4, validation_loss: 3.8, joint_minFDE: 7.2, focal_minFDE: 5.1 }] : [], issue: null,
}));
const snapshot: TrainingSnapshot = { version: 1, generated_at: '2026-10-19T04:35:18Z', poll_seconds: 5, studies: [
  { id: 'original', title: 'Original frozen study', status: 'stopped', phase: 'training', message: 'Storage paused: preserve 8 GiB free.', issues: [], current_run_id: 'full-42',
    progress: { planned_epochs: 90, completed_epochs: 1, committed_epoch_fraction: .4776, overall_fraction: 1 / 90, completed_runs: 0, total_runs: 6 },
    budget: { limit_hours: 40, charged_hours: 1.1, reserved_hours: .25, remaining_hours: 38.65, note: 'Conservative wall-time accounting; reservations are not measured GPU execution.' },
    storage: { free_gib: 3.37, minimum_free_gib: 8, below_reserve: true }, runs },
  { id: 'direction', title: 'Map-direction experiment', status: 'waiting', phase: 'waiting', message: 'Waiting for the original frozen study to complete.', issues: [], current_run_id: null,
    progress: { planned_epochs: null, completed_epochs: 0, committed_epoch_fraction: 0, overall_fraction: null, completed_runs: 0, total_runs: 6 },
    budget: { limit_hours: 12, charged_hours: null, reserved_hours: null, remaining_hours: null, note: 'Pilot and all six runs share the 12-hour ceiling.' },
    storage: { free_gib: 3.37, minimum_free_gib: 8, below_reserve: true }, runs: [] },
] };
const computeHosts: ComputeHostStatus[] = [
  { id: 'local', label: 'Local RTX 5050', status: 'running', phase: 'frozen_study', gpu: 'NVIDIA RTX 5050 Laptop GPU', updated_at: '2026-10-19T04:35:00Z', freshness: { age_seconds: 18, stale: false },
    training: { completed_epochs: 2, planned_epochs: 15, committed_examples: 24000, total_examples: 40000, examples_per_second: 3.9 },
    transfer: { status: 'local', verified_at: null, local_path: 'D:\\InteractionLab\\study-v1' }, cleanup: { status: 'not_required', message: 'Original research data is preserved.' } },
  { id: 'private', label: 'Private 940MX', status: 'running', phase: 'qualification', gpu: 'NVIDIA GeForce 940MX', message: 'Qualification pilot only; not part of the frozen study.',
    freshness: { age_seconds: 10800, stale: false }, training: null, transfer: null, cleanup: { status: 'pending', message: 'Wait for verified local transfer.' } },
  { id: 'colab', label: 'Colab T4', status: 'completed', phase: 'transferred', gpu: 'Tesla T4', freshness: { age_seconds: 300, stale: false },
    training: { completed_epochs: 3, planned_epochs: 3, committed_examples: null, total_examples: null, examples_per_second: 28.4 },
    transfer: { status: 'verified', verified_at: '2026-10-19T04:30:00Z', local_path: 'D:\\InteractionLab\\imports\\colab-checkpoint.pt' }, cleanup: { status: 'completed', message: 'Only generated runtime files were removed after transfer verification.' } },
  { id: 'kaggle', label: 'Kaggle T4 × 2', status: 'unknown', phase: 'qualification', gpu: 'Tesla T4 × 2', message: 'A fresh remote check is required.', last_observed_status: 'running',
    freshness: { age_seconds: 18000, stale: true }, transfer: { status: 'not_verified', verified_at: null, local_path: null }, cleanup: { status: 'blocked', message: 'Retain runtime files until transfer verification.' } },
];

describe('read-only training monitor', () => {
  let container: HTMLDivElement, root: Root, hidden: boolean, unmounted: boolean;
  beforeEach(() => {
    vi.useFakeTimers(); vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    hidden = false; unmounted = false; vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden);
    container = document.createElement('div'); document.body.append(container); root = createRoot(container);
    window.history.replaceState(null, '', '/');
  });
  afterEach(async () => { if (!unmounted) await act(async () => root.unmount()); container.remove(); window.history.replaceState(null, '', '/'); vi.restoreAllMocks(); vi.useRealTimers(); vi.unstubAllGlobals(); });
  const button = (text: string) => {
    const result = [...container.querySelectorAll('button')].find((item) => item.textContent === text);
    if (!result) throw new Error(`Missing ${text}`);
    return result;
  };
  const change = async (select: HTMLSelectElement, value: string) => { await act(async () => { select.value = value; select.dispatchEvent(new Event('change', { bubbles: true })); }); };

  it('shows checkpoint-based six-run progress, storage pause, honest measurements and one-axis learning curves', async () => {
    const fetchMock = vi.fn(async () => response(snapshot)); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<TrainingScreen />));
    expect(container.textContent).toContain('1 saved epochs / 90');
    expect((container.querySelector('[aria-label="Overall saved training progress"]') as HTMLProgressElement).value).toBeCloseTo(1 / 90);
    expect(container.textContent).toContain('4,776 / 10,000'); expect(container.textContent).toContain('19,104 / 40,000');
    expect(container.textContent).toContain('5 min ago'); expect(container.textContent).toContain('43.6 scenarios/s');
    expect(container.textContent).toContain('Not reported yet'); expect(container.textContent).toContain('Storage is below the 8 GiB reserve');
    expect(container.querySelectorAll('.training-run-section .training-run-table tbody tr')).toHaveLength(6);
    expect(container.querySelector('svg[aria-label]')?.getAttribute('aria-label')).toContain('Selection joint minFDE (m)');
    await change(container.querySelector('[aria-label="Learning curve metric"]')!, 'train_loss');
    expect(container.querySelector('svg[aria-label]')?.getAttribute('aria-label')).toContain('Training loss');
    expect(container.querySelector('.training-curve-unit')?.textContent).toBe('Training loss · lower is better');
    expect(fetchMock.mock.calls).toHaveLength(1);
  });

  it('selects the waiting follow-up without suggesting it is training or showing an invented schedule', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response(snapshot)));
    await act(async () => root.render(<TrainingScreen />));
    await change(container.querySelector('[aria-label="Training study"]')!, 'direction');
    expect(container.textContent).toContain('Waiting for the original frozen study to complete.');
    expect(container.textContent).toContain('No follow-up training is running.');
    expect(container.textContent).toContain('schedule not frozen');
    expect(container.textContent).toContain('12 h');
  });

  it('separates finished training progress from ongoing selection scoring', async () => {
    const scoring = { ...snapshot, studies: [{ ...snapshot.studies[0], status: 'running', phase: 'selection_scoring', message: null,
      current_run_id: null, progress: { ...snapshot.studies[0].progress, completed_runs: 6, completed_epochs: 90, overall_fraction: 1 } }] };
    vi.stubGlobal('fetch', vi.fn(async () => response(scoring)));
    await act(async () => root.render(<TrainingScreen />));
    expect(container.textContent).toContain('All six training runs have finished. Selection Scoring continues separately');
    expect(container.textContent).toContain('6 / 6 runs complete');
    expect(container.textContent).not.toContain('Run details appear after the study creates its first checkpoint');
  });

  it('polls five seconds after completion, never overlaps requests, and aborts on unmount', async () => {
    let complete!: (value: Response) => void;
    const pending = new Promise<Response>((resolve) => { complete = resolve; });
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => pending); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<TrainingScreen />));
    await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
    expect(fetchMock).toHaveBeenCalledOnce(); expect(button('Refresh status').disabled).toBe(true);
    await act(async () => complete(response(snapshot)));
    await act(async () => { await vi.advanceTimersByTimeAsync(4999); }); expect(fetchMock).toHaveBeenCalledOnce();
    await act(async () => { await vi.advanceTimersByTimeAsync(1); }); expect(fetchMock).toHaveBeenCalledTimes(2);
    const init = fetchMock.mock.calls[1][1] as RequestInit;
    await act(async () => root.unmount()); unmounted = true;
    expect(init.signal?.aborted).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); }); expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('retains last good values after an error and recovers with explicit retry', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(response(snapshot)).mockRejectedValueOnce(new Error('Connection refused')).mockResolvedValue(response(snapshot)); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<TrainingScreen />));
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('Status is stale');
    expect(container.textContent).toContain('Connection refused'); expect(container.textContent).toContain('19,104 / 40,000');
    await act(async () => button('Retry status').click());
    expect(container.querySelector('[role="alert"]')).toBeNull(); expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it('pauses while hidden, aborts an active read, and ignores its late response', async () => {
    let complete!: (value: Response) => void;
    const fetchMock = vi.fn().mockResolvedValueOnce(response(snapshot)).mockImplementationOnce(() => new Promise<Response>((resolve) => { complete = resolve; })).mockResolvedValue(response(snapshot));
    vi.stubGlobal('fetch', fetchMock); await act(async () => root.render(<TrainingScreen />));
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    const init = fetchMock.mock.calls[1][1] as RequestInit;
    await act(async () => { hidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    expect(init.signal?.aborted).toBe(true); expect(container.textContent).toContain('Refresh paused while hidden');
    await act(async () => complete(response({ ...snapshot, studies: [] })));
    await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
    expect(fetchMock).toHaveBeenCalledTimes(2); expect(container.textContent).toContain('19,104 / 40,000');
    await act(async () => { hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it('explains an unconfigured service without fake progress', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({ version: 1, generated_at: null, poll_seconds: 5, studies: [], message: 'Launch with configured research study folders.' })));
    await act(async () => root.render(<TrainingScreen />));
    expect(container.textContent).toContain('No study folders configured');
    expect(container.textContent).toContain('Launch with configured research study folders.'); expect(container.querySelector('progress')).toBeNull();
    expect(container.querySelectorAll('[data-host-id]')).toHaveLength(4);
    expect(container.querySelector('[data-host-id="private"]')?.textContent).toContain('No status yet');
    expect(container.querySelector('[data-host-id="private"]')?.textContent).toContain('No cleanup report');
    expect(container.querySelector('[data-host-id="private"]')?.textContent).not.toContain('Running');
  });

  it('shows all four actual compute reports, verified transfers and cleanup separately from frozen study totals', async () => {
    const fetchMock = vi.fn(async () => response({ ...snapshot, compute_hosts: computeHosts })); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<TrainingScreen />));
    expect(container.querySelectorAll('[data-host-id]')).toHaveLength(4);
    expect(container.querySelector('[data-host-id="local"]')?.textContent).toContain('24,000 / 40,000');
    expect(container.querySelector('[data-host-id="private"]')?.textContent).toContain('Qualification pilot only');
    expect(container.querySelector('[data-host-id="colab"]')?.textContent).toContain('Verified');
    expect(container.querySelector('[data-host-id="colab"] time')?.getAttribute('datetime')).toBe('2026-10-19T04:30:00Z');
    expect(container.querySelector('[data-host-id="colab"] code')?.textContent).toBe('D:\\InteractionLab\\imports\\colab-checkpoint.pt');
    expect(container.querySelector('[data-host-id="colab"]')?.textContent).toContain('Only generated runtime files were removed after transfer verification.');
    expect(container.querySelector('[data-host-id="kaggle"]')?.textContent).toContain('Retain runtime files until transfer verification.');
    expect(container.textContent).toContain('1 saved epochs / 90');
    expect(container.querySelectorAll('.training-compute-section button')).toHaveLength(0);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('uses provider freshness without inventing a shorter remote timeout and makes old running reports explicit', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({ ...snapshot, compute_hosts: computeHosts })));
    await act(async () => root.render(<TrainingScreen />));
    const privateRow = container.querySelector('[data-host-id="private"]')!;
    const kaggleRow = container.querySelector('[data-host-id="kaggle"]')!;
    expect(privateRow.textContent).toContain('Last checked 3.0 h ago');
    expect(privateRow.classList.contains('training-compute-stale')).toBe(false);
    expect(kaggleRow.textContent).toContain('Stale report · last checked 5.0 h ago');
    expect(kaggleRow.textContent).toContain('Last observed: Running');
    expect(kaggleRow.classList.contains('training-compute-stale')).toBe(true);
  });

  it('labels provider-stale evidence explicitly when no observation timestamp exists', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => response({ ...snapshot, compute_hosts: [{ id: 'private', status: 'unknown', freshness: { age_seconds: null, stale: true } }] })));
    await act(async () => root.render(<TrainingScreen />));
    expect(container.querySelector('[data-host-id="private"]')?.textContent).toContain('Stale report · No update time reported');
  });

  it('renders remote reports without a configured study and updates them on the existing read-only poll', async () => {
    const old = { ...snapshot, studies: [], compute_hosts: computeHosts.map((host) => host.id === 'colab' ? { ...host, status: 'running', transfer: null, cleanup: null } : host) };
    const fetchMock = vi.fn().mockResolvedValueOnce(response(old)).mockResolvedValue(response({ ...snapshot, studies: [], compute_hosts: computeHosts })); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<TrainingScreen />));
    expect(container.querySelector('[data-host-id="colab"]')?.textContent).toContain('Running');
    expect(container.querySelector('[data-host-id="colab"]')?.textContent).toContain('No transfer report');
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(container.querySelector('[data-host-id="colab"]')?.textContent).toContain('Completed');
    expect(container.querySelector('[data-host-id="colab"]')?.textContent).toContain('Verified');
    expect(container.querySelector('progress')).toBeNull();
    expect(fetchMock.mock.calls.every(([url, init]) => url === '/api/training' && !init.method)).toBe(true);
  });

  it('separates a recent cloud connection check from the older training observation', async () => {
    const host = { ...computeHosts.find((item) => item.id === 'colab')!, status: 'completed',
      freshness: { age_seconds: 7200, stale: true },
      connection: { status: 'unconfigured', checked_at: '2026-10-20T11:55:00Z', message: 'Connect the approved results folder to enable unattended checks.' } };
    const local = { ...computeHosts.find((item) => item.id === 'local')!, connection: { status: 'unconfigured' } };
    vi.stubGlobal('fetch', vi.fn(async () => response({ ...snapshot, compute_hosts: [host, local] })));
    await act(async () => root.render(<TrainingScreen />));
    const row = container.querySelector('[data-host-id="colab"]');
    expect(row?.textContent).toContain('Stale report · last checked 2.0 h ago');
    expect(row?.textContent).toContain('Monitor connection: Unconfigured');
    expect(row?.textContent).toContain('Connect the approved results folder');
    expect(row?.querySelector('time[datetime="2026-10-20T11:55:00Z"]')).not.toBeNull();
    expect(container.querySelector('[data-host-id="local"]')?.textContent).not.toContain('Monitor connection');
  });

  it('shows an actual SSH check separately from private training progress', async () => {
    const host = { ...computeHosts.find((item) => item.id === 'private')!, connection: {
      status: 'unavailable', checked_at: '2026-10-20T11:55:00Z', message: 'SSH check failed; saved progress retained.' } };
    vi.stubGlobal('fetch', vi.fn(async () => response({ ...snapshot, compute_hosts: [host] })));
    await act(async () => root.render(<TrainingScreen />));
    expect(container.querySelector('[data-host-id="private"]')?.textContent).toContain('Monitor connection: Unavailable');
    expect(container.querySelector('[data-host-id="private"]')?.textContent).toContain('saved progress retained');
  });

  const appResponse = (url: string) => {
    if (url === '/api/health') return response({ device: 'cpu', checkpoint_loaded: false, checkpoint_id: null, calibration_loaded: false });
    if (url === '/api/scenarios') return response({ scenarios: [{ id: 'alpha', city: 'PIT', actor_count: 0, focal_type: 'vehicle', split: 'validation' }] });
    if (url === '/api/reports') return response({ evaluation: null, experimental: true, risk_calibration_status: 'not_fitted' });
    if (url === '/api/training') return response(snapshot);
    if (url === '/api/analyses') return response({ analyses: [] });
    if (url === '/api/scenarios/alpha') return response({ id: 'alpha', city: 'PIT', focal_track_id: 'car', observed_steps: 50, future_steps: 60, dt: .1, actors: [], selection: { retained_dynamic: 0, total_dynamic: 0 }, map: { lanes: [], crosswalks: [], drivable_areas: [] } });
    if (url === '/api/predict') return response({ scenario_id: 'alpha', actor_ids: [], trajectories: [], probabilities: [1], risk: { raw: [], calibrated: null, status: 'uncalibrated' }, model: { id: 'baseline', kind: 'baseline' }, calibration_id: null, scenario: { id: 'alpha', city: 'PIT', actors: [], selection: { retained_dynamic: 0, total_dynamic: 0 }, observed_steps: 50, future_steps: 60, dt: .1 }, elapsed_ms: 1, stress: null });
    throw new Error(`Unexpected ${url}`);
  };

  it('opens directly at #training without loading scenarios for inference and returns to the workspace', async () => {
    window.history.replaceState(null, '', '/#training');
    const fetchMock = vi.fn(async (url: string) => appResponse(url)); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<App />));
    expect(container.querySelector('[aria-label="Training monitor"]')).not.toBeNull(); expect(container.querySelector('.workspace')?.hasAttribute('hidden')).toBe(true);
    expect(container.querySelector('.header-status')?.textContent).toContain('Local study monitor connected');
    expect(container.querySelector('.header-device')?.textContent).toBe('Read-only');
    expect(button('Training').getAttribute('aria-pressed')).toBe('true');
    expect(button('Scenarios').getAttribute('aria-pressed')).toBe('false');
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/predict' || url === '/api/scenarios/alpha')).toBe(false);
    await act(async () => { window.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', key: ' ' })); });
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/predict')).toBe(false);
    await act(async () => button('Scenarios').click());
    expect(window.location.hash).toBe(''); expect(container.querySelector('.workspace')?.hasAttribute('hidden')).toBe(false);
    expect(button('Scenarios').getAttribute('aria-pressed')).toBe('true');
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/predict')).toBe(true);
  });

  it('aborts a pending manual comparison and prevents its second prediction after opening Training', async () => {
    let complete!: (value: Response) => void; let predictions = 0;
    const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
      if (url === '/api/predict' && ++predictions === 2) return new Promise<Response>((resolve) => { complete = resolve; });
      return appResponse(url);
    }); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<App />));
    await act(async () => button('Compare stress test').click());
    const predictionCall = fetchMock.mock.calls.filter(([url]) => url === '/api/predict')[1];
    await act(async () => button('Training').click());
    expect((predictionCall[1] as RequestInit).signal?.aborted).toBe(true);
    await act(async () => complete(appResponse('/api/predict')));
    expect(predictions).toBe(2); expect(window.location.hash).toBe('#training');
    expect(container.querySelector('[aria-label="Training monitor"]')).not.toBeNull();
  });

  it('returns to the previously selected Failures tab and stops the training poll', async () => {
    const fetchMock = vi.fn(async (url: string) => appResponse(url)); vi.stubGlobal('fetch', fetchMock);
    await act(async () => root.render(<App />));
    await act(async () => button('Failures').click());
    await act(async () => button('Training').click());
    expect(container.querySelector('.workspace')?.hasAttribute('hidden')).toBe(true);
    await act(async () => button('Scenarios').click());
    expect(container.querySelector('#failures-tab')?.getAttribute('aria-selected')).toBe('true');
    expect(container.querySelector('[aria-label="Failure analysis library"]')).not.toBeNull();
    expect(container.querySelector('[aria-label="Training monitor"]')).toBeNull();
    const statusCalls = fetchMock.mock.calls.filter(([url]) => url === '/api/training').length;
    await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/training')).toHaveLength(statusCalls);
  });
});
