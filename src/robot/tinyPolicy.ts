// Tiny policies, hand-written forward passes, no ML runtime.
// Mirrors training/policies.py (Runner, mlp_forward, diffusion_sample, act_forward).
// Weights come from training/export.py as base64 little-endian float32.

import C from "./simConfig.json";

export type FeatureKind = "abs" | "rel" | "rel_ball";
type B64Layer = { in: number; out: number; w: string; b: string };

export type PolicyFile =
  | { type?: "mlp"; features: FeatureKind; layers: B64Layer[]; meta: Record<string, unknown> }
  | {
      type: "diffusion";
      features: FeatureKind;
      layers: B64Layer[];
      H: number; E: number; temb: number;
      alpha_bar: number[]; ddim_steps: number[];
      meta: Record<string, unknown>;
    }
  | {
      type: "act";
      features: FeatureKind;
      d: number; heads: number; n_layers: number; H: number; ensemble_m: number;
      tensors: Record<string, { shape: number[]; data: string }>;
      meta: Record<string, unknown>;
    };

type Layer = { nIn: number; nOut: number; w: Float64Array; b: Float64Array };

const X0 = C.WORLD_MIN_X, X1 = C.WORLD_MAX_X, Y0 = C.WORLD_MIN_Y, Y1 = C.WORLD_MAX_Y;
const HALF_W = (X1 - X0) / 2, HALF_H = (Y1 - Y0) / 2;
const CX = (X0 + X1) / 2, CY = (Y0 + Y1) / 2;
const SCALE = X1 - X0;

function decode(b64: string): Float64Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return Float64Array.from(new Float32Array(bytes.buffer));
}

/** Same features as environment.features() in Python. */
export function features(s: readonly number[], kind: FeatureKind, out: Float64Array) {
  const [rx, ry, bx, by, gx, gy] = s;
  if (kind === "abs") {
    out[0] = (rx - CX) / HALF_W; out[1] = (ry - CY) / HALF_H;
    out[2] = (bx - CX) / HALF_W; out[3] = (by - CY) / HALF_H;
    out[4] = (gx - CX) / HALF_W; out[5] = (gy - CY) / HALF_H;
    return;
  }
  out[0] = (bx - rx) / SCALE; out[1] = (by - ry) / SCALE;
  out[2] = (gx - bx) / SCALE; out[3] = (gy - by) / SCALE;
  out[4] = (gx - rx) / SCALE; out[5] = (gy - ry) / SCALE;
  if (kind === "rel_ball") {
    out[6] = (bx - CX) / HALF_W; out[7] = (by - CY) / HALF_H;
  }
}

// ------------------------------------------------------------------ RNG
/** mulberry32 + Box-Muller. Bit-identical to policies.Rng in Python. */
export class Rng {
  private a: number;
  constructor(seed: number) { this.a = seed >>> 0; }
  uniform() {
    this.a = (this.a + 0x6d2b79f5) >>> 0;
    const a = this.a;
    let t = Math.imul(a ^ (a >>> 15), 1 | a) >>> 0;
    t = (((t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0) ^ t) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  normal() {
    const u1 = Math.max(this.uniform(), 1e-12);
    const u2 = this.uniform();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// ------------------------------------------------------------------ MLP
function mlp(layers: Layer[], x: Float64Array, bufs: Float64Array[]): Float64Array {
  let h = x;
  for (let li = 0; li < layers.length; li++) {
    const { nIn, nOut, w, b } = layers[li];
    const y = bufs[li];
    const relu = li < layers.length - 1;
    for (let o = 0; o < nOut; o++) {
      let acc = b[o];
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) acc += w[row + i] * h[i];
      y[o] = relu && acc < 0 ? 0 : acc;
    }
    h = y;
  }
  return h;
}

const toLayers = (ls: B64Layer[]): Layer[] => ls.map((l) => ({ nIn: l.in, nOut: l.out, w: decode(l.w), b: decode(l.b) }));

function stepEmbedding(k: number, n: number, out: Float64Array, offset: number) {
  const half = n / 2;
  for (let i = 0; i < half; i++) {
    const f = Math.exp((-Math.log(1000) * i) / half);
    out[offset + i] = Math.sin(k * f);
    out[offset + half + i] = Math.cos(k * f);
  }
}

// ------------------------------------------------------------------ ACT
type Mat = { rows: number; cols: number; v: Float64Array };

function linear(x: Float64Array, nIn: number, W: Mat, b: Float64Array, out: Float64Array) {
  // out = W @ x + b, W is (out, in) row-major
  for (let o = 0; o < W.rows; o++) {
    let acc = b[o];
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) acc += W.v[row + i] * x[i];
    out[o] = acc;
  }
}

function layerNorm(x: Float64Array, g: Float64Array, b: Float64Array, out: Float64Array) {
  const n = x.length;
  let mu = 0;
  for (let i = 0; i < n; i++) mu += x[i];
  mu /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mu) * (x[i] - mu);
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mu) * inv * g[i] + b[i];
}

class TinyAct {
  private t: Record<string, Mat>;
  constructor(private f: Extract<PolicyFile, { type: "act" }>) {
    this.t = {};
    for (const [k, { shape, data }] of Object.entries(f.tensors)) {
      this.t[k] = { rows: shape[0], cols: shape[1] ?? 1, v: decode(data) };
    }
  }
  get params() {
    return Object.values(this.t).reduce((n, m) => n + m.v.length, 0);
  }
  /** Decoder with z = 0. Returns H rows of [vx, vy]. */
  forward(obs: Float64Array): Float64Array[] {
    const { d, heads, n_layers: L, H } = this.f;
    const T = 2 + H, hs = d / heads, t = this.t;
    const tok = Array.from({ length: T }, () => new Float64Array(d));
    const tmp = new Float64Array(d);
    for (let j = 0; j < d; j++) tok[0][j] = t["z_proj.bias"].v[j];
    linear(obs, obs.length, t["obs_proj.weight"], t["obs_proj.bias"].v, tok[1]);
    for (let r = 0; r < T; r++) for (let j = 0; j < d; j++) tok[r][j] += t["pos"].v[r * d + j];

    const qkv = Array.from({ length: T }, () => new Float64Array(3 * d));
    const att = Array.from({ length: T }, () => new Float64Array(d));
    const ff = new Float64Array(2 * d);
    const scores = new Float64Array(T);
    for (let li = 0; li < L; li++) {
      const p = `blocks.${li}.`;
      for (let r = 0; r < T; r++) {
        layerNorm(tok[r], t[p + "ln1.weight"].v, t[p + "ln1.bias"].v, tmp);
        linear(tmp, d, t[p + "qkv.weight"], t[p + "qkv.bias"].v, qkv[r]);
      }
      for (let h = 0; h < heads; h++) {
        for (let r = 0; r < T; r++) {
          let mx = -Infinity;
          for (let c = 0; c < T; c++) {
            let s = 0;
            for (let j = 0; j < hs; j++) s += qkv[r][h * hs + j] * qkv[c][d + h * hs + j];
            scores[c] = s / Math.sqrt(hs);
            if (scores[c] > mx) mx = scores[c];
          }
          let sum = 0;
          for (let c = 0; c < T; c++) { scores[c] = Math.exp(scores[c] - mx); sum += scores[c]; }
          for (let j = 0; j < hs; j++) {
            let acc = 0;
            for (let c = 0; c < T; c++) acc += (scores[c] / sum) * qkv[c][2 * d + h * hs + j];
            att[r][h * hs + j] = acc;
          }
        }
      }
      for (let r = 0; r < T; r++) {
        linear(att[r], d, t[p + "proj.weight"], t[p + "proj.bias"].v, tmp);
        for (let j = 0; j < d; j++) tok[r][j] += tmp[j];
        layerNorm(tok[r], t[p + "ln2.weight"].v, t[p + "ln2.bias"].v, tmp);
        linear(tmp, d, t[p + "ff1.weight"], t[p + "ff1.bias"].v, ff);
        for (let j = 0; j < 2 * d; j++) if (ff[j] < 0) ff[j] = 0;
        linear(ff, 2 * d, t[p + "ff2.weight"], t[p + "ff2.bias"].v, tmp);
        for (let j = 0; j < d; j++) tok[r][j] += tmp[j];
      }
    }
    const out: Float64Array[] = [];
    for (let r = 2; r < T; r++) {
      layerNorm(tok[r], t["ln_f.weight"].v, t["ln_f.bias"].v, tmp);
      const a = new Float64Array(2);
      linear(tmp, d, t["head.weight"], t["head.bias"].v, a);
      out.push(a);
    }
    return out;
  }
}

// ------------------------------------------------------------------ runner
/** Stateful closed-loop controller around any policy family. Mirrors policies.Runner. */
export class TinyPolicy {
  readonly type: "mlp" | "diffusion" | "act";
  readonly params: number;
  private obs: Float64Array;
  private layers: Layer[] = [];
  private bufs: Float64Array[] = [];
  private act_?: TinyAct;
  private rng = new Rng(0);
  private queue: [number, number][] = [];
  private history: { chunk: Float64Array[]; t0: number }[] = [];
  private t = 0;

  constructor(private file: PolicyFile, seed = 0) {
    this.type = file.type ?? "mlp";
    this.obs = new Float64Array(file.features === "rel_ball" ? 8 : 6);
    if (file.type === "act") {
      this.act_ = new TinyAct(file);
      this.params = this.act_.params;
    } else {
      this.layers = toLayers(file.layers);
      this.bufs = this.layers.map((l) => new Float64Array(l.nOut));
      this.params = this.layers.reduce((n, l) => n + l.w.length + l.b.length, 0);
    }
    this.reset(seed);
  }

  reset(seed = 0) {
    this.rng = new Rng(seed);
    this.queue = [];
    this.history = [];
    this.t = 0;
  }

  /** One diffusion chunk (DDIM, eta = 0), exposed for the parity test. */
  sampleChunk(state: readonly number[]): [number, number][] {
    const f = this.file as Extract<PolicyFile, { type: "diffusion" }>;
    const n = 2 * f.H, nObs = this.layers[0].nIn - n - f.temb;
    const obs = new Float64Array(nObs);
    features(state, f.features, obs);
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = this.rng.normal();
    const inp = new Float64Array(this.layers[0].nIn);
    const S = f.ddim_steps.length;
    for (let i = 0; i < S; i++) {
      const k = f.ddim_steps[i];
      const ab = f.alpha_bar[k];
      const abPrev = i + 1 < S ? f.alpha_bar[f.ddim_steps[i + 1]] : 1;
      inp.set(x, 0);
      inp.set(obs, n);
      stepEmbedding(k, f.temb, inp, n + nObs);
      const eps = mlp(this.layers, inp, this.bufs);
      for (let j = 0; j < n; j++) {
        let x0 = (x[j] - Math.sqrt(1 - ab) * eps[j]) / Math.sqrt(ab);
        x0 = x0 < -1 ? -1 : x0 > 1 ? 1 : x0;
        const e = (x[j] - Math.sqrt(ab) * x0) / Math.sqrt(1 - ab);
        x[j] = Math.sqrt(abPrev) * x0 + Math.sqrt(1 - abPrev) * e;
      }
    }
    const out: [number, number][] = [];
    for (let i = 0; i < f.H; i++) out.push([x[2 * i], x[2 * i + 1]]);
    return out;
  }

  /** ACT chunk for a state (z = 0), exposed for the parity test. */
  actChunk(state: readonly number[]): Float64Array[] {
    features(state, this.file.features, this.obs);
    return this.act_!.forward(this.obs);
  }

  /** state -> desired velocity as a fraction of MAX_ROBOT_SPEED. */
  act(state: readonly number[]): [number, number] {
    if (this.type === "diffusion") {
      const f = this.file as Extract<PolicyFile, { type: "diffusion" }>;
      if (!this.queue.length) this.queue = this.sampleChunk(state).slice(0, f.E);
      return this.queue.shift()!;
    }
    if (this.type === "act") {
      const f = this.file as Extract<PolicyFile, { type: "act" }>;
      this.history.push({ chunk: this.actChunk(state), t0: this.t });
      this.history = this.history.filter((h) => this.t - h.t0 < f.H);
      let nx = 0, ny = 0, den = 0;
      this.history.forEach((h, i) => {
        const w = Math.exp(-f.ensemble_m * i); // i = 0 is the oldest
        const a = h.chunk[this.t - h.t0];
        nx += w * a[0]; ny += w * a[1]; den += w;
      });
      this.t += 1;
      return [nx / den, ny / den];
    }
    features(state, this.file.features, this.obs);
    const out = mlp(this.layers, this.obs, this.bufs);
    return [out[0], out[1]];
  }
}
