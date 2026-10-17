import { useCallback, useEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { actorPosition, actorRisk, lastPosition } from './data';
import type { Point, Prediction, Scenario } from './types';

interface Props {
  scenario: Scenario;
  prediction: Prediction | null;
  original: Prediction | null;
  step: number;
  mode: number;
  selected: string;
  showReference: boolean;
  showAllModes: boolean;
  fitKey: number;
  onSelect: (id: string) => void;
}
interface Camera { x: number; y: number; zoom: number }
const colors = { focal: '#5dd2bd', actor: '#87b6e8', selected: '#ffffff', high: '#f0786e', medium: '#f3bb67' };

export default function ScenarioCanvas(props: Props) {
  const { scenario, prediction, original, step, mode, selected, showReference, showAllModes, fitKey, onSelect } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; camera: Camera; moved: boolean } | null>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const [camera, setCamera] = useState<Camera>({ x: 0, y: 0, zoom: 3 });

  useEffect(() => {
    if (!wrapperRef.current) return;
    const observer = new ResizeObserver(([entry]) => setSize({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(wrapperRef.current);
    return () => observer.disconnect();
  }, []);

  const fit = useCallback(() => {
    const points: Point[] = scenario.actors.flatMap((actor) => actor.history.filter((point): point is Point => point !== null));
    if (prediction) {
      const focal = prediction.actor_ids.indexOf(scenario.focal_track_id);
      if (focal >= 0) points.push(...(prediction.trajectories[mode]?.[focal] ?? []));
    }
    if (!points.length) return;
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    setCamera({ x: (minX + maxX) / 2, y: (minY + maxY) / 2, zoom: Math.min(size.width / Math.max(40, maxX - minX + 25), size.height / Math.max(40, maxY - minY + 25)) });
  }, [scenario, prediction, mode, size]);

  // A scene or explicit fit command resets the view; prediction updates preserve inspection.
  useEffect(() => { fit(); }, [scenario.id, fitKey, size.width, size.height]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * ratio);
    canvas.height = Math.round(size.height * ratio);
    ctx.scale(ratio, ratio);
    ctx.fillStyle = '#1c242b';
    ctx.fillRect(0, 0, size.width, size.height);
    const project = ([x, y]: Point): Point => [size.width / 2 + (x - camera.x) * camera.zoom, size.height / 2 - (y - camera.y) * camera.zoom];
    const line = (points: (Point | null)[], color: string, width = 1, dash: number[] = [], alpha = 1) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.globalAlpha = alpha; ctx.beginPath();
      let started = false;
      for (const point of points) {
        if (!point) { started = false; continue; }
        const [x, y] = project(point);
        if (!started) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        started = true;
      }
      ctx.stroke(); ctx.restore();
    };
    const polygon = (points: Point[], fill: string, stroke?: string) => {
      if (!points.length) return;
      ctx.beginPath();
      points.forEach((point, index) => { const [x, y] = project(point); if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
    };

    const gridStep = camera.zoom < 1 ? 50 : 10;
    ctx.strokeStyle = '#26313a'; ctx.lineWidth = 0.5;
    const left = camera.x - size.width / camera.zoom / 2;
    const right = camera.x + size.width / camera.zoom / 2;
    const bottom = camera.y - size.height / camera.zoom / 2;
    const top = camera.y + size.height / camera.zoom / 2;
    for (let x = Math.floor(left / gridStep) * gridStep; x < right; x += gridStep) line([[x, bottom], [x, top]], '#29333c', 0.5);
    for (let y = Math.floor(bottom / gridStep) * gridStep; y < top; y += gridStep) line([[left, y], [right, y]], '#29333c', 0.5);

    scenario.map.drivable_areas.forEach((area) => polygon(area, '#283139'));
    scenario.map.lanes.forEach((lane) => {
      if (lane.left_boundary.length && lane.right_boundary.length) polygon([...lane.left_boundary, ...lane.right_boundary.slice().reverse()], '#2c363f');
      line(lane.left_boundary, '#66737d', 0.8, [], 0.55);
      line(lane.right_boundary, '#66737d', 0.8, [], 0.55);
      line(lane.points, '#8a969f', 0.6, [4, 7], 0.27);
    });
    scenario.map.crosswalks.forEach((crosswalk) => polygon(crosswalk, '#61758124', '#6f808b'));

    scenario.actors.forEach((actor) => {
      const isSelected = actor.id === selected;
      const color = actor.focal ? colors.focal : colors.actor;
      const until = Math.min(scenario.observed_steps, step + 1);
      line(actor.history.slice(0, until), color, isSelected ? 2.3 : 1.3, [], isSelected || actor.focal ? 0.85 : 0.4);
      if (showReference) line([lastPosition(actor), ...actor.reference_future], '#d5e0e5', isSelected ? 1.8 : 1, [3, 5], isSelected ? 0.8 : 0.22);
    });

    if (original) original.actor_ids.forEach((id, index) => {
      const actor = original.scenario.actors.find((candidate) => candidate.id === id);
      line([actor ? lastPosition(actor) : null, ...(original.trajectories[mode]?.[index] ?? [])], '#bdc8d1', id === selected ? 2 : 1, [5, 6], id === selected ? 0.85 : 0.27);
    });

    if (prediction) prediction.actor_ids.forEach((id, index) => {
      const actor = scenario.actors.find((candidate) => candidate.id === id);
      const isSelected = id === selected;
      const risk = actorRisk(prediction, id) ?? 0;
      const color = risk >= 0.5 ? colors.high : risk >= 0.2 ? colors.medium : actor?.focal ? colors.focal : colors.actor;
      const modes = showAllModes && isSelected ? prediction.trajectories.map((_, index) => index) : [mode];
      modes.forEach((world) => line([actor ? lastPosition(actor) : null, ...(prediction.trajectories[world]?.[index] ?? [])], color, world === mode ? isSelected ? 2.7 : 1.3 : 1, [], world === mode ? isSelected ? 1 : 0.6 : 0.2));
      if (isSelected) {
        const path = prediction.trajectories[mode]?.[index];
        if (path?.length) {
          const [x, y] = project(path[path.length - 1]);
          ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
        }
      }
    });

    scenario.actors.forEach((actor) => {
      const point = actorPosition(actor, step, scenario, prediction, mode);
      if (!point) return;
      const [x, y] = project(point);
      const isSelected = actor.id === selected;
      const risk = actorRisk(prediction, actor.id) ?? 0;
      const color = risk >= 0.5 ? colors.high : risk >= 0.2 ? colors.medium : actor.focal ? colors.focal : colors.actor;
      const previous = step > 0 ? actorPosition(actor, step - 1, scenario, prediction, mode) : null;
      const heading = step >= scenario.observed_steps && previous && Math.hypot(point[0] - previous[0], point[1] - previous[1]) > 0.01
        ? Math.atan2(point[1] - previous[1], point[0] - previous[0])
        : actor.headings[Math.min(step, actor.headings.length - 1)] ?? 0;
      ctx.save(); ctx.translate(x, y); ctx.rotate(-heading);
      ctx.fillStyle = color;
      if (actor.type === 'pedestrian' || actor.type === 'cyclist') {
        ctx.beginPath(); ctx.arc(0, 0, Math.max(3.5, Math.min(6, camera.zoom * 0.5)), 0, Math.PI * 2); ctx.fill();
      } else {
        const width = Math.max(9, Math.min(24, camera.zoom * (actor.type === 'bus' ? 7 : 4)));
        const height = Math.max(5, Math.min(12, camera.zoom * 1.8));
        ctx.fillRect(-width / 2, -height / 2, width, height);
        ctx.strokeStyle = '#1c242b'; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(width / 4, -height / 2); ctx.lineTo(width / 4, height / 2); ctx.stroke();
      }
      ctx.restore();
      if (isSelected) { ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke(); }
      if (isSelected || actor.focal || actor.ego) {
        ctx.fillStyle = '#dfe9ed'; ctx.font = '11px Segoe UI, sans-serif';
        ctx.fillText(actor.focal ? 'Focal' : actor.ego ? 'Ego' : actor.id.slice(-6), x + 17, y - 9);
      }
    });

    const scaleMeters = camera.zoom >= 3 ? 10 : 25;
    const scaleWidth = scaleMeters * camera.zoom;
    ctx.strokeStyle = '#acbac6'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(25, size.height - 28); ctx.lineTo(25 + scaleWidth, size.height - 28); ctx.stroke();
    ctx.fillStyle = '#acbac6'; ctx.font = '11px Segoe UI, sans-serif'; ctx.fillText(`${scaleMeters} m`, 25, size.height - 38);
  }, [scenario, prediction, original, step, mode, selected, showReference, showAllModes, camera, size]);

  const handleDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { x: event.clientX, y: event.clientY, camera, moved: false };
  };
  const handleMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!drag.current) return;
    const dx = event.clientX - drag.current.x, dy = event.clientY - drag.current.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.current.moved = true;
    setCamera({ ...drag.current.camera, x: drag.current.camera.x - dx / camera.zoom, y: drag.current.camera.y + dy / camera.zoom });
  };
  const handleUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (drag.current && !drag.current.moved) {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      let nearest = 20, target: string | null = null;
      scenario.actors.forEach((actor) => {
        const point = actorPosition(actor, step, scenario, prediction, mode);
        if (!point) return;
        const distance = Math.hypot(size.width / 2 + (point[0] - camera.x) * camera.zoom - x, size.height / 2 - (point[1] - camera.y) * camera.zoom - y);
        if (distance < nearest) { nearest = distance; target = actor.id; }
      });
      if (target) onSelect(target);
    }
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  return <div className="canvas-wrap" ref={wrapperRef}>
    <canvas ref={canvasRef} tabIndex={0} aria-label="Scenario map. Drag to pan, scroll to zoom. Use plus and minus to zoom, F to fit. Select actors from the inspector using the keyboard."
      onPointerDown={handleDown} onPointerMove={handleMove} onPointerUp={handleUp} onPointerCancel={() => { drag.current = null; }}
      onWheel={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const px = event.clientX - rect.left - size.width / 2, py = event.clientY - rect.top - size.height / 2;
        setCamera((old) => { const zoom = Math.min(30, Math.max(0.15, old.zoom * Math.exp(-event.deltaY * 0.0015))); return { x: old.x + px / old.zoom - px / zoom, y: old.y - py / old.zoom + py / zoom, zoom }; });
      }}
      onKeyDown={(event) => {
        if (event.key === '+' || event.key === '=') { event.preventDefault(); setCamera((old) => ({ ...old, zoom: Math.min(30, old.zoom * 1.2) })); }
        if (event.key === '-') { event.preventDefault(); setCamera((old) => ({ ...old, zoom: Math.max(0.15, old.zoom / 1.2) })); }
        if (event.key.toLowerCase() === 'f') { event.preventDefault(); fit(); }
      }} />
    <div className="map-orientation" aria-hidden="true"><span>↑</span>Local y</div>
    <div className="map-tools">
      <button title="Zoom in" aria-label="Zoom in" onClick={() => setCamera((old) => ({ ...old, zoom: Math.min(30, old.zoom * 1.25) }))}>+</button>
      <button title="Zoom out" aria-label="Zoom out" onClick={() => setCamera((old) => ({ ...old, zoom: Math.max(0.15, old.zoom / 1.25) }))}>−</button>
      <button title="Fit scenario (F)" aria-label="Fit scenario" onClick={fit}>⌖</button>
    </div>
    <div className="map-coordinate">Focal local frame <span>x → · y ↑</span></div>
  </div>;
}
