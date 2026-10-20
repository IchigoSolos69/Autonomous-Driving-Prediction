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
  choose_analysis_directory?(): Promise<AnalysisSummary | null>;
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
export interface AnalysisCounts { actors: number; scenarios: number; cases: number; small_sample: boolean }
export type AnalysisMetrics = Record<string, Record<string, number | null>>;
export interface AnalysisSummary { id: string; title: string; split: string; development_only: boolean; counts: AnalysisCounts }
export interface AnalysisActor { actor_id: string; actor_type: string; role: string; speed_bin: string; turn_bin: string; density_bin: string; history_bin: string }
export interface AnalysisCaseSummary { scenario_id: string; city: string; focal_type: string; actor_count: number; focal_minFDE: number | null; jointFDE: number | null; delta_jointFDE?: number | null; actors: AnalysisActor[] }
export interface AnalysisDetail extends AnalysisSummary {
  metrics: AnalysisMetrics;
  slices: { dimension: string; value: string; actor_count: number; scenario_count: number; small_sample: boolean; metrics: AnalysisMetrics }[];
  cases: AnalysisCaseSummary[];
  model_labels: string[];
  provenance: Record<string, unknown>;
  joint_metric_scope?: string;
}
export interface RecordedPrediction extends Prediction { analysis_label: string; analysis_seed: number | null }
export interface AnalysisCase { analysis_id: string; scenario: Scenario; predictions: RecordedPrediction[] }
export interface TrainingHistory {
  epoch: number; train_loss: number | null; validation_loss: number | null;
  joint_minFDE: number | null; focal_minFDE: number | null;
  train_objective?: Record<string, number | null> | null;
  validation_objective?: Record<string, number | null> | null;
}
export interface TrainingRun {
  id: string; name: string; configuration: string | null; seed: number;
  state: 'pending' | 'running' | 'completed' | 'stopped' | 'unavailable';
  planned_epochs: number | null; completed_epochs: number; epoch_number: number | null;
  committed_examples: number | null; training_examples: number | null; committed_fraction: number | null;
  committed_batches: number | null; batches_per_epoch: number | null;
  stage: 'train' | 'validation_pending' | 'epoch_boundary' | 'pending' | 'completed';
  global_step: number | null; examples_per_second: number | null; peak_vram_mib: number | null;
  checkpoint: { updated_at: string | null; age_seconds: number | null; stale: boolean };
  history: TrainingHistory[]; issue: string | null;
}
export interface TrainingStudy {
  id: 'original' | 'direction'; title: string;
  status: 'running' | 'waiting' | 'stopped' | 'completed' | 'not_started' | 'unavailable';
  phase: string; message: string | null; issues: string[]; current_run_id: string | null;
  progress: { planned_epochs: number | null; completed_epochs: number; committed_epoch_fraction: number; overall_fraction: number | null; completed_runs: number; total_runs: number };
  budget: { limit_hours: number; charged_hours: number | null; reserved_hours: number | null; remaining_hours: number | null; note: string };
  storage: { free_gib: number | null; minimum_free_gib: number; below_reserve: boolean | null };
  runs: TrainingRun[];
}
export interface ComputeHostStatus {
  id: string; label?: string | null; status?: string | null; phase?: string | null;
  message?: string | null; gpu?: string | null; updated_at?: string | null;
  freshness?: { age_seconds: number | null; stale: boolean };
  stale?: boolean; last_observed_status?: string | null;
  connection?: { status: string; phase?: string | null; message?: string | null; scope?: string | null; checked_at?: string | null; saved_run_status?: string | null } | null;
  training?: { completed_epochs: number | null; planned_epochs: number | null; committed_examples: number | null; total_examples: number | null; examples_per_second: number | null } | null;
  transfer?: { status: string | null; verified_at: string | null; local_path: string | null } | null;
  cleanup?: { status: string | null; message: string | null } | null;
}
export interface TrainingSnapshot { version: 1; generated_at: string | null; poll_seconds: number; studies: TrainingStudy[]; message?: string; compute_hosts?: ComputeHostStatus[] }
