export type Point = [number, number];
export type ModelKind = 'trained' | 'constant_velocity';
export type StressKind = 'position_noise' | 'dropout' | 'braking' | 'cut_in';
export interface Health {
  device: string;
  checkpoint_loaded: boolean;
  checkpoint_id: string | null;
  calibration_loaded: boolean;
  risk_definition?: string;
}
export interface ScenarioSummary { id: string; city: string; actor_count: number; focal_type: string; split: string }
export interface Actor {
  id: string;
  type: string;
  focal: boolean;
  ego: boolean;
  history: (Point | null)[];
  headings: (number | null)[];
  reference_future: (Point | null)[];
}
export interface CoordinateFrame { origin: Point; rotation: number; units: string; name: string }
export interface Scenario {
  id: string;
  city: string;
  focal_track_id: string;
  ego_track_id: string | null;
  dt: number;
  observed_steps: number;
  future_steps: number;
  coordinate_frame: CoordinateFrame;
  actors: Actor[];
  map: {
    lanes: { id: string; points: Point[]; left_boundary: Point[]; right_boundary: Point[] }[];
    crosswalks: Point[][];
    drivable_areas: Point[][];
  };
  selection: { retained_dynamic: number; total_dynamic: number };
}
export interface Stress { kind: StressKind; severity: number; actor_id?: string; seed: number }
export interface Prediction {
  scenario_id: string;
  actor_ids: string[];
  trajectories: Point[][][];
  scales: Point[][][];
  probabilities: number[];
  risk: { raw: number[]; calibrated: number[] | null; status: string };
  model: { id: string; kind: string };
  calibration_id: string | null;
  coordinate_frame: CoordinateFrame;
  scenario: Scenario;
  stress: Stress | null;
  elapsed_ms: number;
}
