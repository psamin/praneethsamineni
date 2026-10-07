"""Pick the starting scenes the website shows: only ones the shipped policy solves.

Replays the exported policy (exact float32 weights, same Runner as the browser)
on random scenes with integer coordinates, applying the website's own rules:
success = ball held in the goal; give up when the ball makes no progress for
6 s. Keeps the solved ones, with the seed the browser must use (only matters
for the diffusion policy's noise). The TS simulator matches Python to 1e-9, so
a verified scene plays out the same way in the browser.

The reported success rate (results/evaluation.json) is still measured on
unfiltered random scenes; this list only decides what the demo opens with.

Run on the cluster after export.py:  python verified_scenes.py [count]
Writes src/robot/scenes.json.
"""

import json
import random
import sys
from pathlib import Path

from environment import C, Sim, sample_scene
from parity import POLICY, weights
from policies import Runner

ROOT = Path(__file__).resolve().parent.parent
STALL_STEPS = round(6 / (C["PHYSICS_DT"] * C["SUBSTEPS_PER_ACTION"]))  # BallPushDemo.tsx STALL_STEPS
STALL_PROGRESS = 3                                                     # BallPushDemo.tsx STALL_PROGRESS
SEED0 = 700_000  # never used by training or evaluation


def solves(scene, seed, kind, W):
    sim = Sim(*scene)
    runner = Runner(kind, W, POLICY["features"], seed)
    best, best_step = float("inf"), 0
    while not sim.done():
        sim.apply_action(*runner.act(sim.state()))
        d = sim.goal_distance()
        if d < best - STALL_PROGRESS:
            best, best_step = d, sim.steps
        if sim.steps - best_step > STALL_STEPS:
            return False
    return sim.success()


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    kind = POLICY.get("type", "mlp")
    W = weights()
    kept, tried = [], 0
    while len(kept) < want:
        seed = SEED0 + tried
        tried += 1
        r, b, g = sample_scene(random.Random(seed))
        scene = tuple(tuple(float(round(v)) for v in p) for p in (r, b, g))  # integers: compact and exact in JSON
        if solves(scene, seed, kind, W):
            kept.append([int(v) for p in scene for v in p] + [seed])
    out = ROOT / "src" / "robot" / "scenes.json"
    out.write_text(json.dumps({"model": POLICY["meta"]["model"], "tried": tried, "scenes": kept}, separators=(",", ":")))
    print(f"kept {len(kept)} of {tried} scenes ({len(kept) / tried:.1%}) -> {out.name}")


if __name__ == "__main__":
    main()
