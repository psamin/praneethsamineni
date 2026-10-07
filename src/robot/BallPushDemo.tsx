import { useEffect, useMemo, useRef, useState } from "react";
import { Sim, clampToArena, config as C, sampleScene, type Scene, type Vec } from "./simulation";
import { TinyPolicy, type PolicyFile } from "./tinyPolicy";
import weights from "./policyWeights.json";
import verified from "./scenes.json";

const W = C.WORLD_MAX_X - C.WORLD_MIN_X;
const H = C.WORLD_MAX_Y - C.WORLD_MIN_Y;
const ACTION_DT = C.PHYSICS_DT * C.SUBSTEPS_PER_ACTION; // one policy decision
const SUCCESS_PAUSE_MS = 900;
const TIMEOUT_PAUSE_MS = 400;
const TRAIL = 22;
// End an episode early (counted as unsolved) if the ball stops making progress.
const STALL_STEPS = Math.round(6 / ACTION_DT);
const STALL_PROGRESS = 3;

type Snap = { rx: number; ry: number; bx: number; by: number; vx: number; vy: number };
const snap = (s: Sim): Snap => ({ rx: s.rx, ry: s.ry, bx: s.bx, by: s.by, vx: s.rvx, vy: s.rvy });
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** A starting scene the shipped policy is known to solve (training/verified_scenes.py), with its seed. */
function nextScene(): { scene: Scene; seed: number } {
  const list = verified.scenes as number[][];
  if (!list.length) return { scene: sampleScene(), seed: (Math.random() * 2 ** 32) >>> 0 };
  const [rx, ry, bx, by, gx, gy, seed] = list[Math.floor(Math.random() * list.length)];
  return { scene: { robot: [rx, ry], ball: [bx, by], goal: [gx, gy] }, seed };
}

function readTheme(el: HTMLElement) {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    fg: v("--fg", "#f2f2f2"),
    muted: v("--muted", "#7d7d7d"),
    line: v("--line", "#1b1b1b"),
    accent: v("--accent", "#00e5ff"),
    bg: v("--bg", "#000"),
  };
}

export default function BallPushDemo() {
  const policy = useMemo(() => new TinyPolicy(weights as PolicyFile), []);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [paused, setPaused] = useState(
    () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [hz, setHz] = useState(0);

  // mutable sim state lives in a ref; React only renders the chrome
  const st = useRef({
    sim: null as unknown as Sim,
    prev: null as Snap | null,
    cur: null as Snap | null,
    acc: 0,
    last: 0,
    trail: [] as [number, number][],
    pauseUntil: 0,
    celebrateAt: 0,
    resetPending: false,
    dragBall: false,
    visible: true,
    paused,
    best: Infinity,
    bestStep: 0,
    spin: 0,
    hzCount: 0,
    hzStart: 0,
    lastBall: null as [number, number] | null,
    pointer: null as Vec | null,
  });
  st.current.paused = paused;

  useEffect(() => {
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;
    const theme = readTheme(wrap);
    const s = st.current;

    let scale = 1;
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssW = wrap.clientWidth;
      scale = cssW / W;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssW * (H / W)}px`;
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssW * (H / W) * dpr);
      ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const io = new IntersectionObserver(([e]) => (s.visible = e.isIntersecting));
    io.observe(wrap);

    const newEpisode = () => {
      const { scene, seed } = nextScene();
      policy.reset(seed);
      s.sim = new Sim(scene);
      s.cur = s.prev = snap(s.sim);
      s.trail = [];
      s.acc = 0;
      s.resetPending = false;
      s.best = Infinity;
    };

    // chunked policies (diffusion, ACT) keep a plan; drop it whenever the user changes the scene
    const replan = () => policy.reset((Math.random() * 2 ** 32) >>> 0);
    newEpisode();

    const step = () => {
      s.prev = s.cur;
      if (!s.dragBall) {
        const [ax, ay] = policy.act(s.sim.state());
        s.sim.applyAction(ax, ay);
      }
      s.cur = snap(s.sim);
      s.hzCount += 1;
      s.trail.push([s.sim.rx, s.sim.ry]);
      if (s.trail.length > TRAIL) s.trail.shift();
      const d = s.sim.goalDistance();
      if (d < s.best - STALL_PROGRESS || s.dragBall) { s.best = d; s.bestStep = s.sim.steps; }
      const stalled = s.sim.steps - s.bestStep > STALL_STEPS;
      if ((s.sim.done() || stalled) && !s.dragBall) {
        const ok = s.sim.success();
        s.resetPending = true;
        s.pauseUntil = performance.now() + (ok ? SUCCESS_PAUSE_MS : TIMEOUT_PAUSE_MS);
        if (ok) s.celebrateAt = performance.now();
      }
    };

    let raf = 0;
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dt = Math.min((now - (s.last || now)) / 1000, 0.1); // clamp after tab switches
      s.last = now;
      if (!s.visible) { s.hzStart = 0; return; }
      if (!s.paused) {
        if (s.resetPending) {
          if (now >= s.pauseUntil) newEpisode();
        } else {
          s.acc += dt;
          while (s.acc >= ACTION_DT) {
            s.acc -= ACTION_DT;
            step();
            if (s.resetPending) break;
          }
        }
      }
      // live decision rate, refreshed twice a second
      if (now - s.hzStart >= 500) {
        setHz(s.hzStart ? (s.hzCount * 1000) / (now - s.hzStart) : 0);
        s.hzStart = now;
        s.hzCount = 0;
      }
      const alpha = s.resetPending || s.paused ? 1 : s.acc / ACTION_DT;
      draw(ctx, s, alpha, now, theme);
    };
    raf = requestAnimationFrame(frame);

    // ---- interaction: drag the ball, click anywhere else to move the goal
    const toWorld = (e: PointerEvent): Vec => {
      const r = canvas.getBoundingClientRect();
      return [(e.clientX - r.left) / scale, (e.clientY - r.top) / scale];
    };
    let downAt: Vec | null = null;
    const onDown = (e: PointerEvent) => {
      const p = toWorld(e);
      downAt = p;
      if (Math.hypot(p[0] - s.sim.bx, p[1] - s.sim.by) < C.BALL_RADIUS * 2.2) {
        s.dragBall = true;
        canvas.setPointerCapture(e.pointerId);
      }
    };
    const near = (p: Vec, x: number, y: number, r: number) => Math.hypot(p[0] - x, p[1] - y) < r;
    const onLeave = () => (s.pointer = null);
    const onMove = (e: PointerEvent) => {
      s.pointer = toWorld(e); // the robot's eyes follow the cursor
      if (!s.dragBall) {
        const p = toWorld(e);
        canvas.style.cursor = near(p, s.sim.bx, s.sim.by, C.BALL_RADIUS * 2.2) ? "grab" : "crosshair";
        return;
      }
      const [x, y] = clampToArena(toWorld(e), C.BALL_RADIUS);
      // don't let the user drop the ball inside the robot
      const dx = x - s.sim.rx, dy = y - s.sim.ry, d = Math.hypot(dx, dy);
      const min = C.ROBOT_RADIUS + C.BALL_RADIUS;
      const [bx, by] = d < min && d > 1e-6 ? [s.sim.rx + (dx / d) * min, s.sim.ry + (dy / d) * min] : [x, y];
      s.sim.bx = bx; s.sim.by = by; s.sim.bvx = 0; s.sim.bvy = 0;
      s.cur = snap(s.sim); s.prev = s.cur;
    };
    const onUp = (e: PointerEvent) => {
      const p = toWorld(e);
      if (s.dragBall) {
        s.dragBall = false;
        s.sim.steps = 0; s.sim.hold = 0; s.best = Infinity;
        replan();
        s.resetPending = false;
      } else if (downAt && near(p, downAt[0], downAt[1], 6)) {
        const [gx, gy] = clampToArena(p, C.GOAL_SPAWN_MARGIN);
        s.sim.gx = gx; s.sim.gy = gy;
        s.sim.steps = 0; s.sim.hold = 0; s.best = Infinity;
        replan();
        s.resetPending = false;
      }
      downAt = null;
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointerleave", onLeave);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, [policy]);

  const m = weights.meta;
  return (
    <div className="robot-demo">
      <div ref={wrapRef} className="robot-arena">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="A small robot head pushing a soccer ball toward a goal circle, controlled by a tiny neural network running in your browser"
        />
      </div>
      <div className="robot-bar">
        <ul className="robot-stats">
          <li><b>{policy.params.toLocaleString()}</b> parameters</li>
          <li><b>{hz.toFixed(0)} Hz</b> live</li>
        </ul>
        <div className="robot-controls">
          <button type="button" onClick={() => setPaused((p) => !p)}>{paused ? "▶ resume" : "❚❚ pause"}</button>
          <button
            type="button"
            onClick={() => {
              const s = st.current;
              const { scene, seed } = nextScene();
              policy.reset(seed);
              s.sim = new Sim(scene);
              s.cur = s.prev = snap(s.sim);
              s.trail = []; s.acc = 0; s.resetPending = false; s.best = Infinity;
            }}
          >
            ↻ new scene
          </button>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ drawing

type Theme = ReturnType<typeof readTheme>;

type DrawState = {
  prev: Snap | null; cur: Snap | null; sim: Sim; trail: [number, number][];
  celebrateAt: number; resetPending: boolean; spin: number; lastBall: [number, number] | null;
  pointer: Vec | null;
};

function draw(ctx: CanvasRenderingContext2D, s: DrawState, alpha: number, now: number, t: Theme) {
  const a = s.prev!, b = s.cur!;
  const rx = lerp(a.rx, b.rx, alpha), ry = lerp(a.ry, b.ry, alpha);
  const bx = lerp(a.bx, b.bx, alpha), by = lerp(a.by, b.by, alpha);
  const vx = lerp(a.vx, b.vx, alpha), vy = lerp(a.vy, b.vy, alpha);
  const { gx, gy } = s.sim;

  ctx.clearRect(0, 0, W, H);

  // dot grid
  ctx.fillStyle = t.line;
  for (let x = 20; x < W; x += 40) for (let y = 20; y < H; y += 40) ctx.fillRect(x - 1, y - 1, 2, 2);

  // goal
  const since = now - s.celebrateAt;
  const celebrating = s.resetPending && since < SUCCESS_PAUSE_MS;
  const pulse = 1 + 0.04 * Math.sin(now / 420);
  const pop = celebrating ? 1 + 0.18 * Math.sin(Math.min(since / 260, 1) * Math.PI) : 1;
  const gr = C.GOAL_RADIUS * pulse * pop;
  ctx.save();
  ctx.globalAlpha = celebrating ? 0.22 : 0.08;
  ctx.fillStyle = t.accent;
  ctx.beginPath(); ctx.arc(gx, gy, gr, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = t.accent;
  ctx.globalAlpha = celebrating ? 1 : 0.55;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(celebrating ? [] : [4, 5]);
  ctx.lineDashOffset = -now / 60;
  if (celebrating) { ctx.shadowColor = t.accent; ctx.shadowBlur = 16; }
  ctx.beginPath(); ctx.arc(gx, gy, gr, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();

  // trail
  for (let i = 1; i < s.trail.length; i++) {
    const k = i / s.trail.length;
    ctx.globalAlpha = k * k * 0.35;
    ctx.strokeStyle = t.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(s.trail[i - 1][0], s.trail[i - 1][1]);
    ctx.lineTo(s.trail[i][0], s.trail[i][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // ball: spins with the distance it has rolled
  if (s.lastBall) s.spin += Math.hypot(bx - s.lastBall[0], by - s.lastBall[1]) / C.BALL_RADIUS * Math.sign(bx - s.lastBall[0] || 1);
  s.lastBall = [bx, by];
  drawSoccerBall(ctx, bx, by, C.BALL_RADIUS, s.spin);

  // checkmark on success
  if (celebrating) {
    const k = Math.min(since / 220, 1);
    ctx.save();
    ctx.strokeStyle = t.fg;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.globalAlpha = k;
    ctx.beginPath();
    ctx.moveTo(gx - 8, gy - C.GOAL_RADIUS - 14);
    ctx.lineTo(gx - 2, gy - C.GOAL_RADIUS - 8);
    ctx.lineTo(gx + 9, gy - C.GOAL_RADIUS - 20);
    ctx.stroke();
    ctx.restore();
  }

  drawRobot(ctx, rx, ry, vx, vy, now, t, s.pointer);
}

function drawRobot(ctx: CanvasRenderingContext2D, x: number, y: number, vx: number, vy: number, now: number, t: Theme, look: Vec | null) {
  const R = C.ROBOT_RADIUS;
  const sp = Math.hypot(vx, vy);
  const k = Math.min(sp / C.MAX_ROBOT_SPEED, 1);
  const ux = sp > 1e-3 ? vx / sp : 0, uy = sp > 1e-3 ? vy / sp : 0;
  const tilt = ux * k * 0.1; // a few degrees, never more

  ctx.save();
  ctx.translate(x, y);

  // shadow
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.beginPath(); ctx.ellipse(2, 5, R * 0.95, R * 0.8, 0, 0, Math.PI * 2); ctx.fill();

  ctx.rotate(tilt);

  // antenna (leans against motion)
  const lean = -ux * k * 5;
  ctx.strokeStyle = t.muted;
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(0, -R * 0.78); ctx.quadraticCurveTo(lean * 0.3, -R * 1.05, lean, -R * 1.25); ctx.stroke();
  ctx.fillStyle = t.accent;
  ctx.beginPath(); ctx.arc(lean, -R * 1.25, 2.4, 0, Math.PI * 2); ctx.fill();

  // head: rounded square, the collision disc fits inside it
  const s = R * 0.86;
  roundRect(ctx, -s, -s * 0.82, s * 2, s * 1.64, 7);
  ctx.fillStyle = "#121212";
  ctx.fill();
  ctx.strokeStyle = t.fg;
  ctx.globalAlpha = 0.85;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.globalAlpha = 1;

  // visor
  roundRect(ctx, -s * 0.72, -s * 0.42, s * 1.44, s * 0.84, 5);
  ctx.fillStyle = "#050505";
  ctx.fill();

  // eyes look where it's going; blink every few seconds
  const blink = (now % 4200) < 120 ? 0.15 : 1;
  // eyes look at the cursor if it's over the arena, otherwise where it's going
  const lx = look ? look[0] - x : 0, ly = look ? look[1] - y : 0, ld = Math.hypot(lx, ly);
  const ex = look && ld > 1 ? (lx / ld) * 3 : ux * 3 * k;
  const ey = look && ld > 1 ? (ly / ld) * 2 : uy * 2 * k;
  ctx.fillStyle = t.accent;
  ctx.shadowColor = t.accent;
  ctx.shadowBlur = 6;
  for (const sx of [-1, 1]) {
    ctx.beginPath();
    ctx.ellipse(sx * s * 0.33 + ex, ey, 2.6, 3.2 * blink, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** White ball, black pentagon panels, seams. `spin` rotates the pattern. */
function drawSoccerBall(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, spin: number) {
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.beginPath(); ctx.ellipse(x + 2, y + 4, r, r * 0.8, 0, 0, Math.PI * 2); ctx.fill();

  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r);
  g.addColorStop(0, "#ffffff");
  g.addColorStop(0.7, "#e8e8e8");
  g.addColorStop(1, "#9a9a9a");
  ctx.fillStyle = g;
  ctx.fill();
  ctx.clip();

  const pent = (cx: number, cy: number, pr: number, rot: number) => {
    ctx.beginPath();
    for (let k = 0; k < 5; k++) {
      const a = rot + (k * 2 * Math.PI) / 5 - Math.PI / 2;
      const px = cx + pr * Math.cos(a), py = cy + pr * Math.sin(a);
      if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
  };
  ctx.fillStyle = "#151515";
  ctx.strokeStyle = "#151515";
  ctx.lineWidth = 0.8;
  pent(x, y, r * 0.38, spin);
  for (let k = 0; k < 5; k++) {
    const a = spin + (k * 2 * Math.PI) / 5 - Math.PI / 2;
    // seam from the centre panel's corner out to an edge panel
    ctx.beginPath();
    ctx.moveTo(x + r * 0.38 * Math.cos(a), y + r * 0.38 * Math.sin(a));
    ctx.lineTo(x + r * 0.78 * Math.cos(a), y + r * 0.78 * Math.sin(a));
    ctx.stroke();
    const b = a + Math.PI / 5;
    pent(x + r * 1.05 * Math.cos(b), y + r * 1.05 * Math.sin(b), r * 0.3, b + Math.PI / 2);
  }
  ctx.restore();
}
