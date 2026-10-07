"""Closed-loop evaluation on unseen random episodes.

Policies are evaluated with float32 weights (exactly what the website loads),
forward pass in float64 like JavaScript. Seeds 300000+ are never used in data.
Run: python training/evaluate.py [n_episodes]  ->  results/evaluation.json
     python training/evaluate.py 2000 "abs_*"          (only models matching a pattern)
     python training/evaluate.py pymunk <model> [n]   (cross-check in Pymunk)
"""

import json
import math
import random
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

from environment import FEATURE_DIMS, RB, X0, X1, Y0, Y1, Sim, sample_scene
from expert import expert_action
from policies import Runner, build, numpy_weights

HERE = Path(__file__).resolve().parent
MODELS, RESULTS = HERE / "models", HERE.parent / "results"
EVAL_SEED = 300_000
NEAR_WALL = 70  # ball or goal spawned within this distance of a wall


def load_policy(path):
    """-> (family, feature kind, float32-rounded weights). Older MLP checkpoints have no "type"."""
    ck = torch.load(path, weights_only=False)
    family = ck.get("type", "mlp")
    model = build(family, FEATURE_DIMS[ck["kind"]], ck.get("size") or ck.get("hidden"))
    model.load_state_dict(ck["state_dict"])
    return family, ck["kind"], numpy_weights(family, model)


def wall_dist(x, y):
    return min(x - X0, X1 - x, y - Y0, Y1 - y)


def run_episode(args):
    seed, policy, engine = args
    sc = sample_scene(random.Random(seed))
    if engine == "pymunk":
        from pymunk_env import PymunkSim
        sim = PymunkSim(*sc)
    else:
        sim = Sim(*sc)
    if policy == "expert":
        act = expert_action
    else:
        family, kind, weights = policy
        act = Runner(family, weights, kind, seed).act
    touched_wall = False
    while not sim.done():
        sim.apply_action(*act(sim.state()))
        _, _, bx, by, _, _ = sim.state()
        touched_wall |= wall_dist(bx, by) <= RB + 0.5
    _, _, bx, by, gx, gy = sim.state()
    near = min(wall_dist(*sc[1]), wall_dist(*sc[2])) < NEAR_WALL
    return dict(success=sim.success(), steps=sim.steps, final_dist=sim.goal_distance(),
                near_wall=near, touched_wall=touched_wall)


def summarize(rows):
    ok = [r for r in rows if r["success"]]
    near = [r for r in rows if r["near_wall"]]
    steps = [r["steps"] for r in ok] or [0]
    return {
        "episodes": len(rows),
        "success_rate": len(ok) / len(rows),
        "mean_steps_to_success": st.mean(steps),
        "median_steps_to_success": st.median(steps),
        "mean_final_distance": st.mean(r["final_dist"] for r in rows),
        "near_wall_episodes": len(near),
        "near_wall_failure_rate": (sum(not r["success"] for r in near) / len(near)) if near else 0.0,
        "ball_touched_wall_rate": sum(r["touched_wall"] for r in rows) / len(rows),
    }


def evaluate(policy, n, engine="mirror", pool=None):
    jobs = [(EVAL_SEED + i, policy, engine) for i in range(n)]
    return summarize(pool.map(run_episode, jobs, chunksize=8))


def crosscheck(name, n):
    """Evaluate one policy (and the expert) in the independent Pymunk simulator."""
    path = RESULTS / "evaluation.json"
    out = json.loads(path.read_text())
    with Pool() as pool:
        pcs = out.get("pymunk_crosscheck", {})
        if "model" in pcs:  # older single-model layout
            pcs = {pcs["model"]: pcs}
        pcs[name] = {
            "expert": evaluate("expert", n, "pymunk", pool),
            "policy": evaluate(load_policy(MODELS / f"{name}.pt"), n, "pymunk", pool),
        }
        out["pymunk_crosscheck"] = pcs
    path.write_text(json.dumps(out, indent=2))
    pc = pcs[name]
    print(f"pymunk: expert {pc['expert']['success_rate']:.4f}  {name} {pc['policy']['success_rate']:.4f}  (n={n})")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pymunk":
        return crosscheck(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 500)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*"  # e.g. "abs_*" to evaluate only the MLPs
    RESULTS.mkdir(exist_ok=True)
    train_log = {}
    for name in ("train_log.json", "train_log_chunked.json"):
        if (MODELS / name).exists():
            train_log.update(json.loads((MODELS / name).read_text()))
    out = {"n_episodes": n, "eval_seed_start": EVAL_SEED, "models": {}}
    with Pool() as pool:
        t = time.time()
        out["expert"] = evaluate("expert", n, pool=pool)
        print(f"{'expert':12s} success={out['expert']['success_rate']:.4f}  ({time.time() - t:.0f}s)", flush=True)
        for path in sorted(MODELS.glob(f"{pattern}.pt")):
            t = time.time()
            pol = load_policy(path)
            r = evaluate(pol, n, pool=pool)
            r.update(train_log.get(path.stem, {}))
            out["models"][path.stem] = r
            print(f"{path.stem:12s} params={r.get('params', 0):5d} success={r['success_rate']:.4f} "
                  f"median_steps={r['median_steps_to_success']:.0f} near_wall_fail={r['near_wall_failure_rate']:.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)
    (RESULTS / "evaluation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
