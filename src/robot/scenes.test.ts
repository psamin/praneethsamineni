// Every scene the demo can spawn must be solved by the shipped policy, in the
// browser's own simulator and policy code, under the demo's give-up rule.
import { expect, test } from "vitest";
import verified from "./scenes.json";
import weights from "./policyWeights.json";
import { Sim, config as C } from "./simulation";
import { TinyPolicy, type PolicyFile } from "./tinyPolicy";

const STALL_STEPS = Math.round(6 / (C.PHYSICS_DT * C.SUBSTEPS_PER_ACTION)); // as in BallPushDemo.tsx
const STALL_PROGRESS = 3;

test("scene list was made for the shipped policy", () => {
  expect(verified.model).toBe(weights.meta.model);
  expect(verified.scenes.length).toBeGreaterThan(100);
});

test("the policy solves every spawnable scene", () => {
  const failures: number[] = [];
  for (const [rx, ry, bx, by, gx, gy, seed] of verified.scenes as number[][]) {
    const sim = new Sim({ robot: [rx, ry], ball: [bx, by], goal: [gx, gy] });
    const policy = new TinyPolicy(weights as PolicyFile, seed);
    let best = Infinity, bestStep = 0, stalled = false;
    while (!sim.done()) {
      sim.applyAction(...policy.act(sim.state()));
      const d = sim.goalDistance();
      if (d < best - STALL_PROGRESS) { best = d; bestStep = sim.steps; }
      if (sim.steps - bestStep > STALL_STEPS) { stalled = true; break; }
    }
    if (stalled || !sim.success()) failures.push(seed);
  }
  expect(failures).toEqual([]);
});
