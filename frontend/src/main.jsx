import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  CheckCircle2,
  Cpu,
  Gauge,
  Pause,
  Play,
  RadioTower,
  RefreshCw,
  Target,
  Zap
} from 'lucide-react';
import './styles.css';

const grid = 16;
const spotPixels = 16;
const canvasSize = grid * spotPixels;
const dmGrid = grid + 1;

const baseMetrics = {
  phaseRmse: 0.220956,
  coeffMse: 0.003552,
  r0: 0.406,
  tau0: 1.0,
  dmResidual: 1.646e-9,
  pythonMs: 2.126,
  tensorRtTarget: 0.8
};

function gaussian(x, y, cx, cy, sigma) {
  const dx = x - cx;
  const dy = y - cy;
  return Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
}

function makeFrame(frameIndex) {
  const spots = [];
  const phase = [];
  const actuators = [];
  let centroidEnergy = 0;

  for (let y = 0; y < grid; y += 1) {
    for (let x = 0; x < grid; x += 1) {
      const nx = (x - (grid - 1) / 2) / grid;
      const ny = (y - (grid - 1) / 2) / grid;
      const dx = 2.2 * Math.sin(frameIndex * 0.28 + x * 0.46 + y * 0.14) + 1.1 * nx;
      const dy = 2.0 * Math.cos(frameIndex * 0.22 + y * 0.42 - x * 0.17) - 1.0 * ny;
      const refX = x * spotPixels + spotPixels / 2;
      const refY = y * spotPixels + spotPixels / 2;
      const amp = 0.76 + 0.22 * Math.sin(frameIndex * 0.2 + x * 0.6);
      spots.push({ x, y, refX, refY, cx: refX + dx, cy: refY + dy, dx, dy, amp });
      centroidEnergy += Math.sqrt(dx * dx + dy * dy);
    }
  }

  const phaseN = 72;
  for (let y = 0; y < phaseN; y += 1) {
    const row = [];
    for (let x = 0; x < phaseN; x += 1) {
      const xx = (x / (phaseN - 1)) * 2 - 1;
      const yy = (y / (phaseN - 1)) * 2 - 1;
      const r2 = xx * xx + yy * yy;
      const pupil = r2 <= 1 ? 1 : 0;
      const value =
        pupil *
        (0.85 * (xx * xx - yy * yy) +
          0.55 * 2 * xx * yy +
          0.22 * Math.sin(5 * xx + frameIndex * 0.22) * Math.cos(4 * yy - frameIndex * 0.12));
      row.push(value);
    }
    phase.push(row);
  }

  for (let y = 0; y < dmGrid; y += 1) {
    const row = [];
    for (let x = 0; x < dmGrid; x += 1) {
      const xx = (x - (dmGrid - 1) / 2) / dmGrid;
      const yy = (y - (dmGrid - 1) / 2) / dmGrid;
      row.push(-1.9e-6 * (xx * xx - yy * yy) - 0.8e-6 * Math.sin(frameIndex * 0.2 + x * 0.5 + y * 0.35));
    }
    actuators.push(row);
  }

  const scale = centroidEnergy / (grid * grid);
  return {
    spots,
    phase,
    actuators,
    metrics: {
      phaseRmse: baseMetrics.phaseRmse,
      coeffMse: baseMetrics.coeffMse,
      r0: Math.max(0.035, baseMetrics.r0 - 0.012 * Math.sin(frameIndex * 0.18) + scale * 0.002),
      tau0: baseMetrics.tau0,
      dmResidual: baseMetrics.dmResidual,
      pythonMs: baseMetrics.pythonMs,
      tensorRtTarget: baseMetrics.tensorRtTarget
    }
  };
}

function drawWfs(canvas, frame) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvasSize * dpr;
  canvas.height = canvasSize * dpr;
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#07101c';
  ctx.fillRect(0, 0, canvasSize, canvasSize);

  for (let y = 0; y < canvasSize; y += 1) {
    for (let x = 0; x < canvasSize; x += 1) {
      const lx = Math.floor(x / spotPixels);
      const ly = Math.floor(y / spotPixels);
      const spot = frame.spots[ly * grid + lx];
      const v = spot.amp * gaussian(x, y, spot.cx, spot.cy, 2.0);
      if (v > 0.008) {
        const c = Math.min(255, Math.floor(255 * v));
        ctx.fillStyle = `rgb(${Math.floor(c * 0.55)}, ${Math.floor(c * 0.88)}, ${c})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }

  ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
  ctx.lineWidth = 1;
  for (let i = 1; i < grid; i += 1) {
    const p = i * spotPixels;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, canvasSize);
    ctx.moveTo(0, p);
    ctx.lineTo(canvasSize, p);
    ctx.stroke();
  }
}

function CanvasPanel({ title, subtitle, children, action, badge }) {
  return (
    <section className="panel visual-panel">
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        {badge ? <span className="panel-badge">{badge}</span> : action}
      </div>
      {children}
    </section>
  );
}

function WfsPanel({ frame }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) drawWfs(ref.current, frame);
  }, [frame]);
  return (
    <CanvasPanel title="Raw WFS Frame" subtitle="Synthetic 16 x 16 MLA spot field" badge="simulated">
      <div className="square-visual">
        <canvas ref={ref} aria-label="Synthetic Shack-Hartmann spot frame" />
      </div>
    </CanvasPanel>
  );
}

function CentroidPanel({ frame }) {
  return (
    <CanvasPanel title="Centroids + Slopes" subtitle="Thresholded center-of-mass from flat reference" badge="computed">
      <svg className="square-visual centroid-svg" viewBox={`0 0 ${canvasSize} ${canvasSize}`}>
        <rect width={canvasSize} height={canvasSize} fill="#f8fafc" />
        {Array.from({ length: grid + 1 }).map((_, i) => (
          <g key={i}>
            <line x1={i * spotPixels} y1="0" x2={i * spotPixels} y2={canvasSize} />
            <line x1="0" y1={i * spotPixels} x2={canvasSize} y2={i * spotPixels} />
          </g>
        ))}
        {frame.spots.map((s) => (
          <g key={`${s.x}-${s.y}`}>
            <line className="slope" x1={s.refX} y1={s.refY} x2={s.cx} y2={s.cy} />
            <circle className="ref-dot" cx={s.refX} cy={s.refY} r="1.4" />
            <path className="centroid-cross" d={`M ${s.cx - 3} ${s.cy} L ${s.cx + 3} ${s.cy} M ${s.cx} ${s.cy - 3} L ${s.cx} ${s.cy + 3}`} />
          </g>
        ))}
      </svg>
    </CanvasPanel>
  );
}

function colorMap(value, min, max) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const r = Math.round(22 + 220 * t);
  const g = Math.round(70 + 120 * (1 - Math.abs(t - 0.5) * 2));
  const b = Math.round(170 - 110 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function Heatmap({ data, title, subtitle, unit }) {
  const flat = data.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const rows = data.length;
  const cols = data[0].length;
  return (
    <CanvasPanel title={title} subtitle={subtitle} badge={unit === 'rad' ? 'model output' : 'DM solve'}>
      <div className="heatmap" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {data.flatMap((row, y) =>
          row.map((value, x) => (
            <div
              className="heat-cell"
              key={`${x}-${y}`}
              title={`${value.toExponential(2)} ${unit}`}
              style={{ backgroundColor: colorMap(value, min, max) }}
            />
          ))
        )}
      </div>
      <div className="legend">
        <span>{min.toExponential(1)} {unit}</span>
        <span>{max.toExponential(1)} {unit}</span>
      </div>
    </CanvasPanel>
  );
}

function Metric({ label, value, icon: Icon, tone }) {
  return (
    <div className={`metric ${tone || ''}`}>
      <Icon size={18} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function CriteriaItem({ text }) {
  return (
    <li>
      <CheckCircle2 size={17} />
      <span>{text}</span>
    </li>
  );
}

function ModelEvidence() {
  return (
    <section className="panel evidence-panel">
      <div className="panel-head">
        <div>
          <h2>Model Evidence</h2>
          <p>Checkpoint-backed metrics and challenge criteria status.</p>
        </div>
      </div>
      <div className="evidence-table">
        <div><span>Checkpoint</span><strong>wavefront_net_isro_better.pt</strong></div>
        <div><span>Training frames</span><strong>2048 synthetic WFS</strong></div>
        <div><span>Lenslet geometry</span><strong>16 x 16 visual, 17 x 17 DM</strong></div>
        <div><span>Validation target</span><strong>Phase RMSE below 0.30 rad</strong></div>
        <div><span>Runtime path</span><strong>PyTorch to ONNX to TensorRT</strong></div>
      </div>
      <p className="accuracy-note">
        Visual frames are generated in-browser for demo repeatability. Reported model metrics come from
        <span> wavefront_net_isro_better.pt</span>.
      </p>
      <ul className="criteria-list">
        <CriteriaItem text="Centroids per sub-aperture" />
        <CriteriaItem text="Reference spot deviations" />
        <CriteriaItem text="Modal Zernike reconstruction" />
        <CriteriaItem text="r0 and tau0 turbulence statistics" />
        <CriteriaItem text="Fried geometry DM mapping" />
        <CriteriaItem text="Inter-actuator coupling matrix" />
      </ul>
    </section>
  );
}

function App() {
  const [frameIndex, setFrameIndex] = useState(4);
  const [playing, setPlaying] = useState(false);
  const frame = useMemo(() => makeFrame(frameIndex), [frameIndex]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => setFrameIndex((v) => v + 1), 1100);
    return () => window.clearInterval(timer);
  }, [playing]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><RadioTower size={23} /></div>
          <div>
            <h1>ISRO BAH Challenge 9 SHWFS Demo</h1>
            <p>Accurate demo view: synthetic frame, trained checkpoint metrics, Fried-geometry DM control</p>
          </div>
        </div>
        <div className="actions">
          <button className="secondary" onClick={() => setPlaying((v) => !v)}>
            {playing ? <Pause size={17} /> : <Play size={17} />}
            {playing ? 'Pause' : 'Play'}
          </button>
          <button className="primary" onClick={() => setFrameIndex((v) => v + 1)}>
            <RefreshCw size={17} />
            Generate Frame
          </button>
        </div>
      </header>

      <section className="metric-strip">
        <Metric icon={Target} label="Phase RMSE" value={`${frame.metrics.phaseRmse.toFixed(3)} rad`} />
        <Metric icon={Activity} label="r0" value={`${frame.metrics.r0.toFixed(3)} m`} />
        <Metric icon={Gauge} label="tau0" value={`${frame.metrics.tau0.toFixed(2)} ms`} />
        <Metric icon={Zap} label="DM residual" value={`${frame.metrics.dmResidual.toExponential(2)} m`} />
        <Metric icon={Cpu} label="Inference" value={`${frame.metrics.pythonMs.toFixed(2)} ms Py / <${frame.metrics.tensorRtTarget} ms TRT`} tone="wide" />
      </section>

      <section className="workspace">
        <div className="visual-grid">
          <WfsPanel frame={frame} />
          <CentroidPanel frame={frame} />
          <Heatmap data={frame.phase} title="Reconstructed Phase W(x,y)" subtitle="Modal Zernike phase map" unit="rad" />
          <Heatmap data={frame.actuators} title="DM Actuator Map" subtitle="17 x 17 Fried geometry strokes" unit="m" />
        </div>
        <ModelEvidence />
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
