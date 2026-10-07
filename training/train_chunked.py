"""Train the chunked policy families (Diffusion Policy, ACT) by imitation.

Label for a state = the expert's next H actions if it took over from there
(the simulator is copied and the expert rolled forward H steps). Round 0 uses
DART-noised expert rollouts; rounds 1..N are DAgger (learned policy drives,
expert relabels). Same recipe as train.py, only the label is a chunk.

Run on the cluster (see train.sbatch):
    python train_chunked.py diffusion 64,128 [rounds]
    python train_chunked.py act 32,64 [rounds]
Writes models/<family>_<size>.pt and models/<family>_<size>_bc.pt.
"""

import copy
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from environment import FEATURE_DIMS, Sim, features, sample_scene
from expert import expert_action
from policies import H, Runner, build, deploy_params, numpy_weights

HERE = Path(__file__).resolve().parent
DATA, MODELS = HERE / "data", HERE / "models"
FEATURES = "abs"
BASE_EPISODES, BASE_SEED = 3000, 0
DAGGER_ROUNDS, DAGGER_EPISODES, DAGGER_SEED = 4, 1000, 500_000
EPOCHS = {"diffusion": 500, "act": 150}
BATCH, LR = 2048, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def expert_chunk(sim):
    """What the expert would do over the next H steps from this exact state."""
    s = copy.copy(sim)
    out = []
    for _ in range(H):
        a = expert_action(s.state())
        out.append(a)
        s.apply_action(*a)
    return out


def _ou(rng, sigma, n):
    return n - 0.15 * n + sigma * math.sqrt(0.3) * rng.gauss(0, 1)


def expert_episode(seed):
    rng = random.Random(seed)
    sim = Sim(*sample_scene(rng))
    sigma = rng.choice([0.0, 0.15, 0.3, 0.5])
    nx = ny = 0.0
    S, A = [], []
    while not sim.done():
        S.append(sim.state())
        A.append(expert_chunk(sim))
        ax, ay = A[-1][0]
        nx, ny = _ou(rng, sigma, nx), _ou(rng, sigma, ny)
        sim.apply_action(ax + nx, ay + ny)
    return S, A, sim.success()


def policy_episode(args):
    seed, kind, weights = args
    rng = random.Random(seed)
    sim = Sim(*sample_scene(rng))
    runner = Runner(kind, weights, FEATURES, seed)
    sigma = rng.choice([0.0, 0.15, 0.3])
    nx = ny = 0.0
    S, A = [], []
    while not sim.done():
        S.append(sim.state())
        A.append(expert_chunk(sim))
        ax, ay = runner.act(sim.state())
        nx, ny = _ou(rng, sigma, nx), _ou(rng, sigma, ny)
        sim.apply_action(ax + nx, ay + ny)
    return S, A, sim.success()


def to_tensors(S, A):
    X = torch.tensor([features(s, FEATURES) for s in S], dtype=torch.float32)
    Y = torch.tensor(A, dtype=torch.float32)  # (N, H, 2)
    return X, Y


def fit(kind, size, X, Y, seed=0):
    torch.manual_seed(seed)
    model = build(kind, FEATURE_DIMS[FEATURES], size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS[kind] * (len(X) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    Xd, Yd = X.to(DEVICE), Y.to(DEVICE)
    model.train()
    for _ in range(EPOCHS[kind]):
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(len(X) // BATCH):
            idx = perm[i * BATCH:(i + 1) * BATCH]
            loss = model.loss(Xd[idx], Yd[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model.cpu()


def save(model, kind, size, name, log, n, rounds):
    torch.save({"type": kind, "kind": FEATURES, "size": size, "state_dict": model.state_dict()}, MODELS / f"{name}.pt")
    log[name] = {"params": deploy_params(kind, model), "train_samples": n, "dagger_rounds": rounds}
    print(f"{name:16s} params={log[name]['params']:6d} samples={n:,}", flush=True)


def main():
    kind = sys.argv[1]
    sizes = [int(x) for x in sys.argv[2].split(",")]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else DAGGER_ROUNDS
    MODELS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    log_path = MODELS / "train_log_chunked.json"  # separate from train.py's log: the two jobs run concurrently
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    with Pool() as pool:
        cache = DATA / "chunks_train.npz"
        if cache.exists():
            d = np.load(cache)
            X0, Y0 = torch.from_numpy(d["X"]), torch.from_numpy(d["Y"])
        else:
            t = time.time()
            res = pool.map(expert_episode, range(BASE_SEED, BASE_SEED + BASE_EPISODES), chunksize=16)
            X0, Y0 = to_tensors([s for r in res for s in r[0]], [a for r in res for a in r[1]])
            np.savez_compressed(cache, X=X0.numpy(), Y=Y0.numpy())
            print(f"chunk data: {len(X0):,} samples, noisy-expert success {sum(r[2] for r in res) / len(res):.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)

        for size in sizes:
            t = time.time()
            name = f"{kind}_{size}"
            X, Y = X0, Y0
            model = fit(kind, size, X, Y)
            save(model, kind, size, f"{name}_bc", log, len(X), 0)
            for r in range(1, rounds + 1):
                w = numpy_weights(kind, model)
                seeds = range(DAGGER_SEED + 10_000 * r, DAGGER_SEED + 10_000 * r + DAGGER_EPISODES)
                res = pool.map(policy_episode, [(sd, kind, w) for sd in seeds], chunksize=8)
                Xr, Yr = to_tensors([s for x in res for s in x[0]], [a for x in res for a in x[1]])
                X, Y = torch.cat([X, Xr]), torch.cat([Y, Yr])
                print(f"  {name} dagger round {r}: rollout success {sum(x[2] for x in res) / len(res):.3f}, "
                      f"dataset {len(X):,}", flush=True)
                model = fit(kind, size, X, Y)
            save(model, kind, size, name, log, len(X), rounds)
            print(f"{name} done ({time.time() - t:.0f}s)", flush=True)
            log_path.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
