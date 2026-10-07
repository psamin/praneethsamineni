"""Train tiny MLP policies by behaviour cloning (MSE to expert actions).

Grid: feature set x hidden width, architecture  in -> h -> h -> 2  (ReLU).

Round 0 is plain BC on the DART dataset from generate_data.py. Rounds 1..N
are DAgger (Ross et al., 2011): roll out the *learned* policy, ask the
scripted expert what it would have done in each visited state, add those
(state, expert_action) pairs, retrain. Still supervised imitation of the
expert -- no reward, no RL. Models after round 0 are kept as "<name>_bc".

Run: python training/train.py [features] [widths] [dagger_rounds]
     ->  training/models/<features>_<h>.pt (+ <features>_<h>_bc.pt)
"""

import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment import FEATURE_DIMS, Sim, features, sample_scene
from expert import expert_action

HERE = Path(__file__).resolve().parent
DATA, MODELS = HERE / "data", HERE / "models"
FEATURE_SETS = ["abs", "rel", "rel_ball"]
WIDTHS = [8, 16, 32, 64]
EPOCHS, BATCH, LR = 30, 1024, 3e-3
DAGGER_ROUNDS, DAGGER_EPISODES, DAGGER_SEED = 6, 1000, 400_000


def make_mlp(n_in, h):
    return nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))


def load(split):
    d = np.load(DATA / f"{split}.npz")
    return d["states"].astype(np.float64), d["actions"]


def featurize(S, kind):
    return torch.from_numpy(np.array([features(s, kind) for s in S], dtype=np.float32))


def fit(X, Y, n_in, h, seed=0):
    torch.manual_seed(seed)
    model = make_mlp(n_in, h)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    steps = EPOCHS * (len(X) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(len(X) // BATCH):
            idx = perm[i * BATCH:(i + 1) * BATCH]
            loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


def numpy_layers(model):
    return [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64))
            for m in model if isinstance(m, nn.Linear)]


def rollout(args):
    """Run the learned policy (with light DART noise); label visited states with the expert."""
    seed, kind, layers = args
    rng = random.Random(seed)
    sim = Sim(*sample_scene(rng))
    sigma = rng.choice([0.0, 0.15, 0.3])
    nx = ny = 0.0
    S, A = [], []
    while not sim.done():
        s = sim.state()
        h = np.asarray(features(s, kind))
        for i, (W, b) in enumerate(layers):
            h = W @ h + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0.0)
        S.append(s)
        A.append(expert_action(s))
        nx += -0.15 * nx + sigma * math.sqrt(0.3) * rng.gauss(0, 1)
        ny += -0.15 * ny + sigma * math.sqrt(0.3) * rng.gauss(0, 1)
        sim.apply_action(h[0] + nx, h[1] + ny)
    return S, A, sim.success()


def val_mse(model, kind):
    S, A = load("val")
    with torch.no_grad():
        return nn.functional.mse_loss(model(featurize(S, kind)), torch.from_numpy(A)).item()


def main():
    torch.set_num_threads(4)
    MODELS.mkdir(exist_ok=True)
    kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else FEATURE_SETS
    widths = [int(w) for w in sys.argv[2].split(",")] if len(sys.argv) > 2 else WIDTHS
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else DAGGER_ROUNDS
    log_path = MODELS / "train_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    S0, A0 = load("train")
    with Pool() as pool:
        for kind in kinds:
            X0 = featurize(S0, kind)
            for h in widths:
                t = time.time()
                name = f"{kind}_{h}"
                X, Y = X0, torch.from_numpy(A0)
                model = fit(X, Y, FEATURE_DIMS[kind], h)
                save(model, kind, h, f"{name}_bc", log, len(X), 0)
                for r in range(1, rounds + 1):
                    layers = numpy_layers(model)
                    seeds = range(DAGGER_SEED + 10_000 * r, DAGGER_SEED + 10_000 * r + DAGGER_EPISODES)
                    res = pool.map(rollout, [(sd, kind, layers) for sd in seeds], chunksize=8)
                    succ = sum(x[2] for x in res) / len(res)
                    S = [s for x in res for s in x[0]]
                    A = [a for x in res for a in x[1]]
                    X = torch.cat([X, featurize(S, kind)])
                    Y = torch.cat([Y, torch.tensor(A, dtype=torch.float32)])
                    print(f"  {name} dagger round {r}: policy rollout success {succ:.3f}, dataset {len(X):,}", flush=True)
                    model = fit(X, Y, FEATURE_DIMS[kind], h)
                save(model, kind, h, name, log, len(X), rounds)
                print(f"{name:12s} done ({time.time() - t:.0f}s)", flush=True)
                log_path.write_text(json.dumps(log, indent=2))


def save(model, kind, h, name, log, n_samples, rounds):
    torch.save({"kind": kind, "hidden": h, "state_dict": model.state_dict()}, MODELS / f"{name}.pt")
    n = sum(p.numel() for p in model.parameters())
    log[name] = {"params": n, "val_mse": val_mse(model, kind), "train_samples": n_samples, "dagger_rounds": rounds}
    print(f"{name:12s} params={n:5d}  val_mse={log[name]['val_mse']:.5f}", flush=True)


if __name__ == "__main__":
    main()
