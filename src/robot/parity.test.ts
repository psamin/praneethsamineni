// Replays training/parity.py's recording. Simulator and policy are checked
// separately so a 1-ulp libm difference can't compound across an episode.
import { expect, test } from "vitest";
import fixture from "./parity.fixture.json";
import weights from "./policyWeights.json";
import { Sim } from "./simulation";
import { TinyPolicy, type PolicyFile } from "./tinyPolicy";

type Ep = (typeof fixture.episodes)[number];
const eps = fixture.episodes.map((e, i) => [i, e] as [number, Ep]);

test("fixture matches the exported policy", () => {
  expect(fixture.model).toBe(weights.meta.model);
});

test.each(eps)("episode %i: simulator reproduces Python given the same actions", (_, ep) => {
  const [rx, ry, bx, by, gx, gy] = ep.start;
  const sim = new Sim({ robot: [rx, ry], ball: [bx, by], goal: [gx, gy] });
  let maxErr = 0;
  ep.actions.forEach(([ax, ay], t) => {
    sim.applyAction(ax, ay);
    const got = sim.state(), want = ep.states[t + 1];
    for (let k = 0; k < 6; k++) maxErr = Math.max(maxErr, Math.abs(got[k] - want[k]));
  });
  expect(sim.success()).toBe(ep.success);
  expect(maxErr).toBeLessThan(1e-9);
});

test.each(eps)("episode %i: policy reproduces Python given the same states", (_, ep) => {
  const policy = new TinyPolicy(weights as PolicyFile, ep.seed);
  let maxErr = 0;
  ep.actions.forEach(([ax, ay], t) => {
    const [px, py] = policy.act(ep.states[t]);
    maxErr = Math.max(maxErr, Math.abs(px - ax), Math.abs(py - ay));
  });
  expect(maxErr).toBeLessThan(1e-9);
});
