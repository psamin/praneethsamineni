"""Generate behaviour-cloning data from the scripted expert.

DART-style noise injection (Laskey et al., 2017): the robot executes
expert_action + correlated noise, but every sample is LABELLED with the clean
expert action at the visited state. The dataset therefore contains states
slightly off the expert's path together with the correction back. This is
still plain supervised imitation; there is no reward and no RL.

Splits use disjoint seeds: train / val / test.
Output: training/data/{train,val,test}.npz with states (N,6), actions (N,2).
"""

import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from environment import Sim, sample_scene
from expert import expert_action

OUT = Path(__file__).resolve().parent / "data"
SPLITS = {"train": (0, 3000), "val": (100_000, 300), "test": (200_000, 300)}
OU_THETA = 0.15  # per-step mean reversion of the noise process


def episode(seed):
    rng = random.Random(seed)
    sim = Sim(*sample_scene(rng))
    sigma = rng.choice([0.0, 0.15, 0.3, 0.5])  # some clean, some noisy episodes
    nx = ny = 0.0
    states, actions = [], []
    while not sim.done():
        s = sim.state()
        ax, ay = expert_action(s)
        states.append(s)
        actions.append((ax, ay))
        nx += -OU_THETA * nx + sigma * math.sqrt(2 * OU_THETA) * rng.gauss(0, 1)
        ny += -OU_THETA * ny + sigma * math.sqrt(2 * OU_THETA) * rng.gauss(0, 1)
        sim.apply_action(ax + nx, ay + ny)
    return states, actions, sim.success()


def main():
    OUT.mkdir(exist_ok=True)
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    with Pool() as pool:
        for name, (base, n) in SPLITS.items():
            t = time.time()
            n = max(1, int(n * scale))
            res = pool.map(episode, range(base, base + n), chunksize=16)
            S = np.array([s for r in res for s in r[0]], dtype=np.float32)
            A = np.array([a for r in res for a in r[1]], dtype=np.float32)
            succ = sum(r[2] for r in res) / n
            np.savez_compressed(OUT / f"{name}.npz", states=S, actions=A)
            print(f"{name:5s}: {n} episodes, {len(S):,} samples, noisy-expert success {succ:.3f}  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
