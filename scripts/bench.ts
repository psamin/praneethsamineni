// Policy inference + simulation step latency in V8 (Node). Run: npx tsx scripts/bench.ts
import weights from "../src/robot/policyWeights.json";
import { Sim, sampleScene } from "../src/robot/simulation";
import { TinyPolicy, type PolicyFile } from "../src/robot/tinyPolicy";

const policy = new TinyPolicy(weights as PolicyFile);
const s = [100, 100, 300, 200, 500, 300];
for (let i = 0; i < 50_000; i++) policy.act(s);
const runs: number[] = [];
for (let r = 0; r < 7; r++) {
  const n = 200_000;
  const t0 = performance.now();
  for (let i = 0; i < n; i++) { s[0] = 100 + (i & 63); policy.act(s); }
  runs.push(((performance.now() - t0) * 1000) / n);
}
runs.sort((a, b) => a - b);

const sim = new Sim(sampleScene());
const n = 100_000;
const t0 = performance.now();
for (let i = 0; i < n; i++) { sim.applyAction(0.3, 0.1); if (sim.done()) sim.steps = 0; }
const simUs = ((performance.now() - t0) * 1000) / n;

console.log(JSON.stringify({
  model: weights.meta.model,
  params: policy.params,
  inference_us_median: +runs[3].toFixed(4),
  inference_us_min: +runs[0].toFixed(4),
  inference_us_max: +runs[6].toFixed(4),
  sim_step_us: +simUs.toFixed(3),
  runtime: `node ${process.version} (V8)`,
}, null, 2));
