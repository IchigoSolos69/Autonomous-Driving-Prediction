export type Point = [number, number];
export type ModelKind = 'trained' | 'constant_velocity';
export type StressKind = 'position_noise' | 'dropout' | 'braking' | 'cut_in';
export type ScenarioSource = 'demo' | 'benchmark' | 'imported';
export interface DesktopContext { desktop: true; token: string; data_root: string; settings: Record<string, unknown> }
export interface DesktopApi {
  desktop_context(): Promise<DesktopContext>;
  choose_import_directory(): Promise<{ source_token: string; display_name: string } | null>;
  choose_data_directory(): Promise<{ data_root: string; restart_required: true } | null>;
  choose_model_bundle(archive?: boolean): Promise<{ bundle_id: string; restart_required: true } | null>;
}
declare global { interface Window { pywebview?: { api?: DesktopApi } } }
export interface ImportJob {
  id: string;
  status: 'queued' | 'scanning' | 'running' | 'completed' | 'cancelled' | 'failed';
  processed: number; total: number; imported: number;
  errors: { scenario: string; reason: string }[];
  error?: string;
  display_name?: string;
}
export type ExportKind = 'prediction_json' | 'report_csv' | 'report_png' | 'diagnostics';
export interface Reports {
  evaluation: { scenario_count: number; minADE?: number; minFDE?: number; miss_rate?: number; footprint_collision_proxy?: number; risk?: { eligible_count?: number; positive_count?: number } } | null;
  model: unknown;
  provenance: { training_manifest_sha256?: string; source_checkpoint_sha256?: string } | null;
  risk_calibration_status: string;
  risk_calibration_evidence?: { calibration_positives: number | null; calibration_count: number | null; evaluation_positives: number | null; evaluation_count: number | null; minimum_positives: number };
  experimental: boolean;
}
export interface Health {
  device: string;
  checkpoint_loaded: boolean;
  checkpoint_id: string | null;
  calibration_loaded: boolean;
  risk_definition?: string;
  desktop?: boolean;
  import_available?: boolean;
  device_warning?: string | null;
  data_root?: string | null;
  risk_calibration_status?: string;
}
export interface ScenarioSummary { id: string; city: string; actor_count: number; focal_type: string; split: string; source?: ScenarioSource }
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
  source?: ScenarioSource;
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
  prediction_id?: string;
  source?: ScenarioSource;
  calibration_applicability?: string;
  actor_ids: string[];
  trajectories: Point[][][];
  scales: Point[][][];
  probabilities: number[];
  risk: { raw: number[]; calibrated: number[] | null; status: string; applicability?: string };
  model: { id: string; kind: string };
  calibration_id: string | null;
  coordinate_frame: CoordinateFrame;
  scenario: Scenario;
  stress: Stress | null;
  elapsed_ms: number;
}
