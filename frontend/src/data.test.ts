import { describe, expect, it } from 'vitest';
import { actorPosition, actorRisk, mostLikelyMode, probability } from './data';
import type { Actor, Prediction, Scenario } from './types';

const actor = { id: 'car', history: [[1, 2], null], reference_future: [[99, 99]] } as Actor;
const scenario = { observed_steps: 2 } as Scenario;
const prediction = { actor_ids: ['car'], trajectories: [[[[3, 4]]], [[[5, 6]]]], risk: { raw: [0.4], calibrated: [0.25], status: 'calibrated_proxy' } } as Prediction;

describe('forecast inspection', () => {
  it('uses the selected shared world after the observation window', () => {
    expect(actorPosition(actor, 0, scenario, prediction, 0)).toEqual([1, 2]);
    expect(actorPosition(actor, 1, scenario, prediction, 0)).toBeNull();
    expect(actorPosition(actor, 2, scenario, prediction, 1)).toEqual([5, 6]);
  });
  it('does not substitute recorded futures or stale positions when predictions are absent', () => {
    expect(actorPosition(actor, 2, scenario, null, 0)).toBeNull();
    expect(actorPosition({ ...actor, id: 'unpredicted' }, 2, scenario, prediction, 0)).toBeNull();
  });
  it('looks up risk by actor ID and keeps missing risk distinct from zero', () => {
    expect(actorRisk(prediction, 'car')).toBe(0.25);
    expect(actorRisk(prediction, 'car', false)).toBe(0.4);
    expect(actorRisk(prediction, 'missing')).toBeNull();
    expect(probability(null)).toBe('—');
    expect(probability(0)).toBe('0.0%');
  });
  it('selects the highest-probability world without reordering actor trajectories', () => {
    expect(mostLikelyMode([0.1, 0.15, 0.75])).toBe(2);
    expect(mostLikelyMode([1, 0, 0, 0, 0, 0])).toBe(0);
  });
  it('keeps raw scores when calibration evidence is insufficient', () => {
    expect(actorRisk({ ...prediction, risk: { ...prediction.risk, status: 'insufficient_positive_events' } }, 'car')).toBe(0.4);
  });
  it('hides risk from heads without supervision and from static context', () => {
    expect(actorRisk({ ...prediction, risk: { ...prediction.risk, status: 'untrained_risk_head' } }, 'car')).toBeNull();
    const staticScene = { actors: [{ id: 'car', type: 'static' }] } as Scenario;
    expect(actorRisk({ ...prediction, scenario: staticScene }, 'car')).toBeNull();
  });
});
