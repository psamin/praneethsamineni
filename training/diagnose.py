"""Categorize closed-loop failures of a trained policy.
Run: python training/diagnose.py models/<name>.pt [episodes]"""
import math
import random
import sys
from collections import Counter

import numpy as np

from environment import CONTACT_DISTANCE, RB, Sim, sample_scene
from evaluate import EVAL_SEED, load_policy, wall_dist
from policies import Runner
from expert import expert_action

family, kind, W = load_policy(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
cats = Counter()
examples = {}
for i in range(n):
    s = Sim(*sample_scene(random.Random(EVAL_SEED + i)))
    runner = Runner(family, W, kind, EVAL_SEED + i)
    spd, rb, err, gd = [], [], [], []
    while not s.done():
        st = s.state()
        a = runner.act(st)
        e = expert_action(st)
        err.append(math.hypot(a[0] - e[0], a[1] - e[1]))
        s.apply_action(*a)
        spd.append(math.hypot(s.rvx, s.rvy)); rb.append(math.hypot(s.bx - s.rx, s.by - s.ry)); gd.append(s.goal_distance())
    if s.success():
        continue
    tail = slice(-150, None)
    progress = gd[-150] - gd[-1]
    k = ("ball_on_wall " if wall_dist(s.bx, s.by) < RB + 2 else "") + \
        ("still" if np.mean(spd[tail]) < 15 else "moving") + \
        ("/contact" if np.mean(rb[tail]) < CONTACT_DISTANCE + 3 else "/apart") + \
        ("/near_goal" if s.goal_distance() < 40 else "/far") + \
        ("/progressing" if progress > 20 else "/stuck")
    cats[k] += 1
    examples.setdefault(k, []).append((i, round(s.goal_distance()), round(float(np.mean(err[tail])), 2)))
print(f"failures {sum(cats.values())}/{n}")
for k, v in cats.most_common():
    print(f"{v:4d}  {k:45s} e.g. {examples[k][:3]}")
