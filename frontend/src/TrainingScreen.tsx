import { useEffect, useRef, useState } from 'react';
import { readable, request } from './data';
import type { ComputeHostStatus, TrainingHistory, TrainingRun, TrainingSnapshot } from './types';

const format = (value: number | null | undefined, digits = 1) => value == null ? '—' : value.toFixed(digits);
const fraction = (value: number | null) => value == null ? 'Not measured' : `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`;
const age = (seconds: number | null) => seconds == null ? 'No checkpoint' : seconds < 60 ? `${Math.floor(seconds)} s ago` : seconds < 3600 ? `${Math.floor(seconds / 60)} min ago` : `${(seconds / 3600).toFixed(1)} h ago`;
const stage = (run: TrainingRun) => run.stage === 'validation_pending' ? 'Validation pending' : run.stage === 'train' ? 'Training' : readable(run.stage);
const curveMetrics = { train_loss: 'Training loss', validation_loss: 'Selection loss', joint_minFDE: 'Selection joint minFDE (m)' } as const;
type CurveMetric = keyof typeof curveMetrics;
const curveColors = ['#21665b', '#2f75a3', '#916123', '#a3473c', '#62688c', '#47705b'];
const computeHostLabels = [['local', 'Local RTX'], ['private', 'Private 940MX'], ['colab', 'Colab T4'], ['kaggle', 'Kaggle T4 × 2']] as const;

function ComputeHosts({ hosts }: { hosts?: ComputeHostStatus[] }) {
  return <section className="training-section training-compute-section" aria-labelledby="compute-hosts-title"><div className="training-section-heading"><div><h3 id="compute-hosts-title">Compute hosts</h3><p className="helper">Last reported status from each runtime. These reports are separate from the frozen study totals and are not direct GPU telemetry.</p></div><span>Local, private and free cloud compute</span></div><div className="training-table-scroll"><table className="training-run-table training-compute-table"><thead><tr><th scope="col">Host / reported GPU</th><th scope="col">Last reported state</th><th scope="col">Training</th><th scope="col">Local transfer</th><th scope="col">Cleanup</th></tr></thead><tbody>{computeHostLabels.map(([id, fallbackLabel]) => {
    const host = hosts?.find((item) => item.id === id);
    const timestamp = host?.updated_at ? Date.parse(host.updated_at) : NaN;
    const seconds = host?.freshness?.age_seconds ?? (Number.isFinite(timestamp) ? Math.max(0, (Date.now() - timestamp) / 1000) : null);
    const stale = host?.freshness?.stale ?? host?.stale ?? false;
    const verified = host?.transfer?.verified_at ? Date.parse(host.transfer.verified_at) : NaN;
    const training = host?.training;
    const connection = host?.connection;
    return <tr key={id} data-host-id={id} className={stale ? 'training-compute-stale' : ''}>
      <th scope="row"><strong>{host?.label || fallbackLabel}</strong><span>{host?.gpu || 'GPU not reported'}</span></th>
      <td><strong>{host?.status ? readable(host.status) : 'No status yet'}</strong><small>{host?.phase ? `Phase: ${readable(host.phase)}` : host ? 'Phase not reported' : 'Not configured'}</small><small className={stale ? 'training-stale' : ''}>{seconds == null ? `${stale ? 'Stale report · ' : ''}No update time reported` : `${stale ? 'Stale report · last checked ' : 'Last checked '}${age(seconds)}`}</small>{stale && host?.last_observed_status && host.last_observed_status !== host.status && <small>Last observed: {readable(host.last_observed_status)}</small>}{host?.message && <p>{host.message}</p>}{connection && <><small>Monitor connection: {readable(connection.status)}{connection.checked_at && <> · checked <time dateTime={connection.checked_at}>{new Date(connection.checked_at).toLocaleTimeString()}</time></>}</small>{connection.message && <p>{connection.message}</p>}</>}</td>
      <td>{training ? <><span>{training.completed_epochs == null ? 'Epochs not reported' : `${training.completed_epochs} / ${training.planned_epochs ?? '—'} epochs complete`}</span><small>Committed examples: {training.committed_examples?.toLocaleString() ?? '—'} / {training.total_examples?.toLocaleString() ?? '—'}</small><small>{training.examples_per_second == null ? 'Throughput not measured' : `${format(training.examples_per_second)} scenarios/s measured`}</small></> : <span>No training measurements</span>}</td>
      <td><span>{host?.transfer?.status ? readable(host.transfer.status) : 'No transfer report'}</span>{Number.isFinite(verified) && <small>Verified <time dateTime={host!.transfer!.verified_at!}>{new Date(verified).toLocaleString()}</time></small>}{host?.transfer?.local_path && <code className="training-compute-path" title={host.transfer.local_path}>{host.transfer.local_path}</code>}</td>
      <td><span>{host?.cleanup?.status ? readable(host.cleanup.status) : 'No cleanup report'}</span>{host?.cleanup?.message && <p>{host.cleanup.message}</p>}</td>
    </tr>;
  })}</tbody></table></div><p className="helper">The local program checks configured sources every five minutes. A connection check does not refresh the age of a training observation. Cloud observations require a connected provider. Transfer verification and cleanup require saved evidence. This screen does not delete files.</p></section>;
}

function LearningCurve({ runs, metric }: { runs: TrainingRun[]; metric: CurveMetric }) {
  const series = runs.map((run, index) => ({ run, color: curveColors[index % curveColors.length], dashed: index > 2, points: run.history.filter((item) => item[metric] != null && Number.isFinite(item[metric])) }));
  const values = series.flatMap((item) => item.points.map((point) => point[metric] as number));
  if (!values.length) return <div className="training-curve-empty">No completed-epoch {curveMetrics[metric].toLowerCase()} yet. Curves appear after validation and an epoch checkpoint.</div>;
  const maxEpoch = Math.max(1, ...series.flatMap((item) => item.points.map((point) => point.epoch)));
  const low = Math.min(...values), high = Math.max(...values), padding = Math.max((high - low) * 0.1, Math.abs(high) * 0.02, 0.01);
  const minY = low - padding, maxY = high + padding;
  const x = (epoch: number) => 56 + (epoch / maxEpoch) * 604;
  const y = (value: number) => 190 - (value - minY) / (maxY - minY) * 166;
  return <>
    <svg className="training-curve" viewBox="0 0 688 228" role="img" aria-label={`${curveMetrics[metric]} by completed epoch. ${series.filter((item) => item.points.length).length} runs.`}>
      <title>{curveMetrics[metric]} by completed epoch; lower is better</title>
      {[0, 0.5, 1].map((ratio) => <g key={ratio}><line x1="56" x2="660" y1={y(minY + ratio * (maxY - minY))} y2={y(minY + ratio * (maxY - minY))} stroke="#dde2e4" /><text x="48" y={y(minY + ratio * (maxY - minY)) + 4} textAnchor="end">{format(minY + ratio * (maxY - minY), 2)}</text></g>)}
      <line x1="56" x2="660" y1="190" y2="190" stroke="#8a999e" />
      {[0, maxEpoch].map((epoch) => <text key={epoch} x={x(epoch)} y="207" textAnchor="middle">{epoch}</text>)}
      <text x="358" y="225" textAnchor="middle">Completed epoch</text>
      {series.map(({ run, color, dashed, points }) => <g key={run.id}><polyline points={points.map((point) => `${x(point.epoch)},${y(point[metric] as number)}`).join(' ')} fill="none" stroke={color} strokeWidth="2" strokeDasharray={dashed ? '6 4' : undefined} />{points.map((point) => <circle key={point.epoch} cx={x(point.epoch)} cy={y(point[metric] as number)} r="3" fill={color}><title>{run.name}, epoch {point.epoch}: {format(point[metric], 4)}</title></circle>)}</g>)}
    </svg>
    <div className="training-curve-legend">{series.filter((item) => item.points.length).map(({ run, color, dashed }) => <span key={run.id}><i style={{ borderTopColor: color, borderTopStyle: dashed ? 'dashed' : 'solid' }} />{run.name}</span>)}</div>
  </>;
}

function Progress({ label, value }: { label: string; value: number | null }) {
  return value == null ? <div className="training-progress-track" role="img" aria-label={`${label}: not measured`} /> : <progress aria-label={label} max={1} value={Math.max(0, Math.min(1, value))} />;
}

export default function TrainingScreen() {
  const [snapshot, setSnapshot] = useState<TrainingSnapshot | null>(null);
  const [selectedStudy, setSelectedStudy] = useState('original');
  const [metric, setMetric] = useState<CurveMetric>('joint_minFDE');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [visible, setVisible] = useState(!document.hidden);
  const [receivedAt, setReceivedAt] = useState<string | null>(null);
  const refresh = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false, inFlight = false;
    let timer: number | undefined;
    let controller: AbortController | null = null;
    const poll = async () => {
      if (cancelled || document.hidden || inFlight) return;
      window.clearTimeout(timer); inFlight = true; controller = new AbortController(); setPending(true);
      try {
        const result = await request<TrainingSnapshot>('/training', { signal: controller.signal });
        if (!cancelled && !controller.signal.aborted) { setSnapshot(result); setError(''); setReceivedAt(new Date().toLocaleTimeString()); }
      } catch (error) {
        if (!cancelled && !controller.signal.aborted) setError(error instanceof Error ? error.message : 'Training status is unavailable.');
      } finally {
        inFlight = false;
        if (!cancelled) { setPending(false); if (!document.hidden) timer = window.setTimeout(poll, 5000); }
      }
    };
    const visibility = () => {
      setVisible(!document.hidden); window.clearTimeout(timer);
      if (document.hidden) controller?.abort(); else void poll();
    };
    refresh.current = () => { window.clearTimeout(timer); void poll(); };
    document.addEventListener('visibilitychange', visibility);
    void poll();
    return () => { cancelled = true; controller?.abort(); window.clearTimeout(timer); refresh.current = null; document.removeEventListener('visibilitychange', visibility); };
  }, []);

  const study = snapshot?.studies.find((item) => item.id === selectedStudy) ?? snapshot?.studies[0];
  const current = study?.runs.find((run) => run.id === study.current_run_id);
  const historyKeys: (keyof TrainingHistory)[] = ['train_loss', 'validation_loss', 'joint_minFDE'];

  return <main className="training-screen" aria-label="Training monitor">
    <div className="training-heading"><div><h2>Training</h2><p>Live study status from saved checkpoints. This screen is read-only.</p></div><div className="training-poll-status" aria-live="polite"><span>{!visible ? 'Refresh paused while hidden' : pending ? 'Reading study status…' : receivedAt ? `Last read ${receivedAt}` : 'Waiting for study status'}</span><button disabled={pending || !visible} onClick={() => refresh.current?.()}>{error ? 'Retry status' : 'Refresh status'}</button></div></div>
    {error && <div className="notice error training-status-notice" role="alert"><strong>{snapshot ? 'Status is stale' : 'Cannot read training status'}</strong><p>{error} {snapshot ? 'Last received values remain below. Retry, or reopen the application if the local service is offline.' : 'Retry, or reopen Interaction Lab to restart its local service.'}</p></div>}
    {!snapshot ? <div className="training-empty"><h3>{pending ? 'Reading local study files…' : 'No training status available'}</h3><p>Configure the original and map-direction study folders when launching the research application. This monitor does not run forecasts or change training.</p></div> : !study ? <div className="training-empty"><h3>No study folders configured</h3><p>{snapshot.message ?? 'Launch the research application with the original and map-direction study folders, then refresh status.'}</p></div> : <>
      <div className="training-study-bar"><label>Study<select aria-label="Training study" value={study.id} onChange={(event) => setSelectedStudy(event.target.value)}>{snapshot.studies.map((item) => <option key={item.id} value={item.id}>{item.title} · {item.budget.limit_hours} GPU-hours</option>)}</select></label><span className={`training-state state-${study.status}`}>{readable(study.status)}</span><span className="training-phase">Phase: {readable(study.phase)}</span></div>
      {(study.message || study.issues.length > 0) && <div className={`notice training-status-notice ${study.status === 'stopped' || study.status === 'unavailable' ? 'error' : ''}`} role={study.status === 'stopped' ? 'status' : undefined}>{study.message && <p>{study.message}</p>}{study.issues.length > 0 && <ul>{study.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul>}</div>}
      <section className="training-progress-section" aria-labelledby="training-progress-title"><div className="training-section-heading"><h3 id="training-progress-title">Six-run training progress</h3><strong>{fraction(study.progress.overall_fraction)}</strong></div><Progress label="Overall saved training progress" value={study.progress.overall_fraction} /><div className="training-progress-caption"><span>{study.progress.completed_runs} / {study.progress.total_runs} runs complete · {study.progress.completed_epochs} saved epochs{study.progress.planned_epochs == null ? ' · schedule not frozen' : ` / ${study.progress.planned_epochs}`}</span><span>Checkpoint-based; updated about every 5 minutes and at epoch boundaries</span></div><p className="helper">Overall progress counts completed epochs only; the partial epoch is shown below. Calibration, evaluation and analysis are separate. No completion-time estimate is inferred.</p></section>
      <ComputeHosts hosts={snapshot.compute_hosts} />
      <div className="training-detail-grid"><section className="training-section" aria-labelledby="active-run-title"><div className="training-section-heading"><h3 id="active-run-title">{study.status === 'running' ? 'Active run' : 'Latest saved run'}</h3><span>{current ? `${readable(current.state)} · ${stage(current)}` : 'No active training run'}</span></div>{current ? <><div className="training-run-title"><code>{current.name}</code><span>Seed {current.seed}</span></div><p className="helper">{current.configuration ?? 'Configuration recorded in the study protocol'}</p><Progress label="Active run saved epoch progress" value={current.committed_fraction} /><dl className="training-values"><div><dt>Epoch</dt><dd>{current.epoch_number ?? '—'} / {current.planned_epochs ?? 'Not frozen'}</dd></div><div><dt>Committed batches this epoch</dt><dd>{current.committed_batches?.toLocaleString() ?? '—'} / {current.batches_per_epoch?.toLocaleString() ?? '—'}</dd></div><div><dt>Saved examples this epoch</dt><dd>{current.committed_examples?.toLocaleString() ?? '—'} / {current.training_examples?.toLocaleString() ?? '—'}</dd></div><div><dt>Last checkpoint</dt><dd className={current.checkpoint.stale ? 'training-stale' : ''}>{age(current.checkpoint.age_seconds)}{current.checkpoint.stale ? ' · stale checkpoint' : ''}</dd></div><div><dt>Measured throughput</dt><dd>{current.examples_per_second == null ? 'Not measured' : `${format(current.examples_per_second)} scenarios/s`}</dd></div><div><dt>Peak allocated VRAM</dt><dd>{current.peak_vram_mib == null ? 'Not reported yet' : `${format(current.peak_vram_mib, 0)} MiB`}</dd></div><div><dt>Saved optimizer steps</dt><dd>{current.global_step?.toLocaleString() ?? '—'}</dd></div></dl>{current.issue && <p className="inline-error">{current.issue}</p>}</> : <p className="helper">{study.status === 'waiting' ? 'The follow-up experiment waits for the original frozen study to finish. No follow-up training is running.' : study.status === 'completed' ? 'All recorded training runs have finished.' : study.progress.completed_runs === study.progress.total_runs ? `All six training runs have finished. ${readable(study.phase)} continues separately; training progress does not measure this phase.` : 'Run details appear after the study creates its first checkpoint. Check the study status and storage reserve before resuming.'}</p>}</section>
        <section className="training-section" aria-labelledby="training-resources-title"><h3 id="training-resources-title">Budget and storage</h3><dl className="training-values"><div><dt>GPU-hour ceiling</dt><dd>{study.budget.limit_hours.toFixed(0)} h</dd></div><div><dt>Charged</dt><dd>{format(study.budget.charged_hours, 2)} h</dd></div><div><dt>Reserved</dt><dd>{format(study.budget.reserved_hours, 2)} h</dd></div><div><dt>Remaining</dt><dd>{format(study.budget.remaining_hours, 2)} h</dd></div><div><dt>Drive free</dt><dd className={study.storage.below_reserve ? 'training-stale' : ''}>{format(study.storage.free_gib, 2)} GiB</dd></div><div><dt>Required free reserve</dt><dd>{study.storage.minimum_free_gib} GiB</dd></div></dl><p className="helper">{study.budget.note}</p>{study.storage.below_reserve && <p className="inline-error">Storage is below the {study.storage.minimum_free_gib} GiB reserve. Training must pause; free space before resuming the study from its CLI.</p>}</section></div>
      <section className="training-section training-run-section" aria-labelledby="training-runs-title"><div className="training-section-heading"><h3 id="training-runs-title">Study runs</h3><span>Saved progress across all six runs</span></div><div className="training-table-scroll"><table className="training-run-table"><thead><tr><th scope="col">Run</th><th scope="col">Seed</th><th scope="col">Status</th><th scope="col">Epochs complete</th><th scope="col">Saved epoch</th><th scope="col">Checkpoint age</th></tr></thead><tbody>{study.runs.map((run) => <tr key={run.id} className={current?.id === run.id ? 'active' : ''}><th scope="row"><code>{run.name}</code>{run.configuration && <span>{run.configuration}</span>}</th><td>{run.seed}</td><td>{readable(run.state)}<small>{stage(run)}</small>{run.issue && <small className="inline-error">{run.issue}</small>}</td><td>{run.completed_epochs} / {run.planned_epochs ?? '—'}</td><td>{fraction(run.committed_fraction)}</td><td>{age(run.checkpoint.age_seconds)}{run.checkpoint.stale && <small>Stale checkpoint</small>}</td></tr>)}</tbody></table></div></section>
      <section className="training-section training-curves-section" aria-labelledby="learning-curves-title"><div className="training-section-heading"><div><h3 id="learning-curves-title">Learning curves</h3><p className="helper">Completed epochs only. {study.id === 'direction' ? 'Compare final fixed-epoch checkpoints; no best-epoch selection.' : 'Checkpoint selection uses selection-set joint minFDE.'}</p></div><label>Metric<select aria-label="Learning curve metric" value={metric} onChange={(event) => setMetric(event.target.value as CurveMetric)}>{historyKeys.map((key) => <option key={key} value={key}>{curveMetrics[key as CurveMetric]}</option>)}</select></label></div><p className="training-curve-unit">{curveMetrics[metric]} · lower is better</p><LearningCurve runs={study.runs} metric={metric} /></section>
    </>}
    {!study && <ComputeHosts hosts={snapshot?.compute_hosts} />}
    <footer className="training-footer">Read-only monitor · no start, stop or resume commands are issued · selection metrics are development evidence</footer>
  </main>;
}
