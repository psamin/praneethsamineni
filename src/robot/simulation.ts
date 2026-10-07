// Ball-pushing simulator — a line-for-line mirror of training/environment.py.
// Both read src/robot/simConfig.json. If you change one, change the other and
// rerun `python training/parity.py` (it checks the two agree to ~1e-9).

import C from "./simConfig.json";

const X0 = C.WORLD_MIN_X, X1 = C.WORLD_MAX_X;
const Y0 = C.WORLD_MIN_Y, Y1 = C.WORLD_MAX_Y;
const RR = C.ROBOT_RADIUS, RB = C.BALL_RADIUS;
const CONTACT_DISTANCE = RR + RB;
const DT = C.PHYSICS_DT;
const VMAX = C.MAX_ROBOT_SPEED;

export const config = C;

export type Vec = [number, number];
export type Scene = { robot: Vec; ball: Vec; goal: Vec };

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

export class Sim {
  rx: number; ry: number;
  bx: number; by: number;
  gx: number; gy: number;
  rvx = 0; rvy = 0;
  bvx = 0; bvy = 0;
  steps = 0;
  hold = 0;

  constructor({ robot, ball, goal }: Scene) {
    [this.rx, this.ry] = robot;
    [this.bx, this.by] = ball;
    [this.gx, this.gy] = goal;
  }

  /** ax, ay: desired velocity as a fraction of MAX_ROBOT_SPEED. */
  applyAction(ax: number, ay: number) {
    const m = Math.sqrt(ax * ax + ay * ay);
    if (m > 1.0) { ax /= m; ay /= m; }
    const tx = ax * VMAX, ty = ay * VMAX;
    let dx = tx - this.rvx, dy = ty - this.rvy;
    const d = Math.sqrt(dx * dx + dy * dy);
    const lim = C.MAX_ACTION_DELTA;
    if (d > lim) { dx *= lim / d; dy *= lim / d; }
    this.rvx += dx;
    this.rvy += dy;
    for (let i = 0; i < C.SUBSTEPS_PER_ACTION; i++) this.substep();
    this.steps += 1;
    if (this.goalDistance() <= C.GOAL_SUCCESS_RADIUS) this.hold += 1;
    else this.hold = 0;
  }

  private substep() {
    this.rx = clamp(this.rx + this.rvx * DT, X0 + RR, X1 - RR);
    this.ry = clamp(this.ry + this.rvy * DT, Y0 + RR, Y1 - RR);

    this.bx += this.bvx * DT;
    this.by += this.bvy * DT;
    this.bvx *= C.BALL_DAMPING_PER_SUBSTEP;
    this.bvy *= C.BALL_DAMPING_PER_SUBSTEP;
    const sp = Math.sqrt(this.bvx * this.bvx + this.bvy * this.bvy);
    const dec = C.BALL_FRICTION_DECEL * DT;
    if (sp <= dec) { this.bvx = 0; this.bvy = 0; }
    else { this.bvx -= (this.bvx / sp) * dec; this.bvy -= (this.bvy / sp) * dec; }
    this.ballWalls();

    let [nx, ny, dist] = this.contact();
    if (dist < CONTACT_DISTANCE) {
      let pen = CONTACT_DISTANCE - dist;
      this.bx += nx * pen;
      this.by += ny * pen;
      const rvx = this.bvx - this.rvx, rvy = this.bvy - this.rvy;
      const vn = rvx * nx + rvy * ny;
      if (vn < 0.0) {
        const j = -(1.0 + C.ROBOT_BALL_RESTITUTION) * vn;
        this.bvx += j * nx;
        this.bvy += j * ny;
        const tx = -ny, ty = nx;
        const vt = rvx * tx + rvy * ty;
        const f = C.CONTACT_FRICTION;
        this.bvx -= f * vt * tx;
        this.bvy -= f * vt * ty;
      }
      this.ballWalls();
      [nx, ny, dist] = this.contact();
      if (dist < CONTACT_DISTANCE) {
        pen = CONTACT_DISTANCE - dist;
        this.rx = clamp(this.rx - nx * pen, X0 + RR, X1 - RR);
        this.ry = clamp(this.ry - ny * pen, Y0 + RR, Y1 - RR);
      }
    }
  }

  private contact(): [number, number, number] {
    const dx = this.bx - this.rx, dy = this.by - this.ry;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 1e-9) return [1.0, 0.0, 0.0];
    return [dx / dist, dy / dist, dist];
  }

  private ballWalls() {
    const e = C.WALL_RESTITUTION;
    if (this.bx < X0 + RB) { this.bx = X0 + RB; if (this.bvx < 0.0) this.bvx = -e * this.bvx; }
    else if (this.bx > X1 - RB) { this.bx = X1 - RB; if (this.bvx > 0.0) this.bvx = -e * this.bvx; }
    if (this.by < Y0 + RB) { this.by = Y0 + RB; if (this.bvy < 0.0) this.bvy = -e * this.bvy; }
    else if (this.by > Y1 - RB) { this.by = Y1 - RB; if (this.bvy > 0.0) this.bvy = -e * this.bvy; }
  }

  goalDistance() {
    return Math.hypot(this.gx - this.bx, this.gy - this.by);
  }
  success() {
    return this.hold >= C.SUCCESS_HOLD_STEPS;
  }
  done() {
    return this.success() || this.steps >= C.MAX_EPISODE_STEPS;
  }
  state(): [number, number, number, number, number, number] {
    return [this.rx, this.ry, this.bx, this.by, this.gx, this.gy];
  }
}

const dist = (a: Vec, b: Vec) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Rejection-sample a valid scene. Same rules as sample_scene() in Python. */
export function sampleScene(rand: () => number = Math.random): Scene {
  const m = C.SPAWN_MARGIN, gm = C.GOAL_SPAWN_MARGIN;
  const u = (lo: number, hi: number) => lo + (hi - lo) * rand();
  for (;;) {
    const robot: Vec = [u(X0 + m, X1 - m), u(Y0 + m, Y1 - m)];
    const ball: Vec = [u(X0 + m, X1 - m), u(Y0 + m, Y1 - m)];
    const goal: Vec = [u(X0 + gm, X1 - gm), u(Y0 + gm, Y1 - gm)];
    if (dist(robot, ball) < RR + RB + C.SPAWN_CLEARANCE) continue;
    if (dist(ball, goal) < C.MIN_BALL_GOAL_DIST) continue;
    if (dist(robot, goal) < C.MIN_ROBOT_GOAL_DIST) continue;
    return { robot, ball, goal };
  }
}

/** Clamp a user-placed point so a disc of radius r fits (with margin). */
export function clampToArena([x, y]: Vec, r: number): Vec {
  return [clamp(x, X0 + r, X1 - r), clamp(y, Y0 + r, Y1 - r)];
}
