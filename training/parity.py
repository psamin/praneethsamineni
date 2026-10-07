"""Write the fixture that src/robot/parity.test.ts replays in TypeScript.

Runs the exported policy closed-loop in the Python simulator and records, per
step, the state the policy saw and the action it returned. The TS test checks
two things separately (so 1-ulp libm differences can't compound):
  * simulator parity: TS Sim fed the recorded actions reproduces the states;
  * policy parity: the TS runner fed the recorded states (same seed)
    reproduces the actions -- covering diffusion noise, chunk queues and
    ACT temporal ensembling.
Run (after export.py): python parity.py && cd .. && npm test
"""

import base64
import json
import random
from pathlib import Path

import numpy as np

from environment import Sim, sample_scene
from policies import Runner

ROOT = Path(__file__).resolve().parent.parent
POLICY = json.loads((ROOT / "src" / "robot" / "policyWeights.json").read_text())


def f32(s, shape=None):
    a = np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64)
    return a.reshape(shape) if shape else a


def weights():
    kind = POLICY.get("type", "mlp")
    if kind in ("mlp", "diffusion"):
        return [(f32(l["w"], (l["out"], l["in"])), f32(l["b"])) for l in POLICY["layers"]]
    return {k: f32(t["data"], t["shape"]) for k, t in POLICY["tensors"].items()}


def main():
    kind = POLICY.get("type", "mlp")
    W = weights()
    episodes = []
    for i in range(4):
        seed = 900_000 + i
        sim = Sim(*sample_scene(random.Random(seed)))
        start = sim.state()
        runner = Runner(kind, W, POLICY["features"], seed)
        states, actions = [], []
        while not sim.done():
            s = sim.state()
            a = runner.act(s)
            states.append(list(s))
            actions.append(list(a))
            sim.apply_action(*a)
        states.append(list(sim.state()))
        episodes.append({"seed": seed, "start": list(start), "states": states, "actions": actions,
                         "success": sim.success()})
    out = ROOT / "src" / "robot" / "parity.fixture.json"
    out.write_text(json.dumps({"model": POLICY["meta"]["model"], "episodes": episodes}))
    print(f"wrote {out.name}: {[len(e['actions']) for e in episodes]} steps, success {[e['success'] for e in episodes]}")


if __name__ == "__main__":
    main()
