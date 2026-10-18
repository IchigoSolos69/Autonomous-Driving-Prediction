import type { Actor, Point, Prediction, Scenario } from './types';

export async function request<T>(path: string, init?: RequestInit, desktopToken?: string): Promise<T> {
  const headers = new Headers(init?.headers);
  if (desktopToken) headers.set('X-Interaction-Token', desktopToken);
  const response = await fetch(`/api${path}`, { ...init, headers });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch { /* Preserve the HTTP status if the server returned no JSON. */ }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function actorPosition(actor: Actor, step: number, scenario: Scenario, prediction: Prediction | null, mode: number): Point | null {
  if (step < scenario.observed_steps) return actor.history[step] ?? null;
  const index = prediction?.actor_ids.indexOf(actor.id) ?? -1;
  if (index >= 0) return prediction?.trajectories[mode]?.[index]?.[step - scenario.observed_steps] ?? null;
  // Actors without predictions disappear after the observation window.
  return null;
}

export function lastPosition(actor: Actor): Point | null {
  for (let index = actor.history.length - 1; index >= 0; index--) {
    if (actor.history[index]) return actor.history[index];
  }
  return null;
}

export function probability(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
}

export function mostLikelyMode(probabilities: number[]): number {
  return probabilities.reduce((best, value, index) => value > probabilities[best] ? index : best, 0);
}

export function actorRisk(prediction: Prediction | null, actorId: string, calibrated = true): number | null {
  if (!prediction) return null;
  if (prediction.risk.status === 'untrained_risk_head') return null;
  const actor = prediction.scenario?.actors.find((item) => item.id === actorId);
  if (actor && !['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus'].includes(actor.type)) return null;
  const index = prediction.actor_ids.indexOf(actorId);
  if (index < 0) return null;
  return (calibrated && prediction.risk.status === 'calibrated_proxy' && prediction.risk.calibrated ? prediction.risk.calibrated : prediction.risk.raw)[index] ?? null;
}

export function readable(value: string): string { return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()); }
