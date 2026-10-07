"""Physics validation scenarios. Run: python training/test_physics.py

Every substep checks hard invariants (no interpenetration, nothing outside
the arena, no energy blow-up). Each scenario also checks its own outcome.
"""
import math
import random

from environment import CONTACT_DISTANCE, RB, RR, VMAX, X0, X1, Y0, Y1, Sim, sample_scene
from expert import expert_action

PEN_TOL = 0.5  # max allowed robot/ball overlap, world units


class Checked(Sim):
    max_pen = 0.0
    max_ball_speed = 0.0

    def _substep(self):
        super()._substep()
        d = math.hypot(self.bx - self.rx, self.by - self.ry)
        self.max_pen = max(self.max_pen, CONTACT_DISTANCE - d)
        self.max_ball_speed = max(self.max_ball_speed, math.hypot(self.bvx, self.bvy))
        assert X0 + RB - 1e-9 <= self.bx <= X1 - RB + 1e-9 and Y0 + RB - 1e-9 <= self.by <= Y1 - RB + 1e-9, "ball left arena"
        assert X0 + RR - 1e-9 <= self.rx <= X1 - RR + 1e-9 and Y0 + RR - 1e-9 <= self.ry <= Y1 - RR + 1e-9, "robot left arena"


def drive(s, vx, vy, steps):
    for _ in range(steps):
        s.apply_action(vx, vy)


def check(name, s, cond, detail=""):
    ok = cond and s.max_pen <= PEN_TOL and s.max_ball_speed <= 1.3 * VMAX
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} max_overlap={s.max_pen:.3f}  max_ball_speed={s.max_ball_speed:.0f}  {detail}")
    return ok


results = []

s = Checked((100, 200), (160, 200), (500, 200))
drive(s, 1, 0, 40)
results.append(check("direct push", s, abs(s.by - 200) < 1e-6 and s.bx > 300, f"ball=({s.bx:.0f},{s.by:.0f})"))

s = Checked((100, 100), (100 + 34 / math.sqrt(2) + 5, 100 + 34 / math.sqrt(2) + 5), (500, 300))
drive(s, 0.7071, 0.7071, 30)
ang = math.degrees(math.atan2(s.by - 100, s.bx - 100))
results.append(check("diagonal push", s, abs(ang - 45) < 2, f"ball heading={ang:.1f}deg"))

s = Checked((100, 200), (200, 225), (500, 200))
drive(s, 1, 0, 40)
results.append(check("glancing collision", s, s.by > 230 and s.bx > 200, f"ball=({s.bx:.0f},{s.by:.0f}) deflected sideways"))

s = Checked((300, 300), (300, 340), (300, 100))
drive(s, 0, 1, 15)
drive(s, 0, 0, 60)
results.append(check("ball hits wall", s, s.by <= Y1 - RB + 1e-9 and math.hypot(s.bvx, s.bvy) < 1, f"ball y={s.by:.1f}, at rest"))

s = Checked((100, Y1 - RB - 10), (150, Y1 - RB), (500, Y1 - RB))
drive(s, 1, 0.15, 60)
results.append(check("push along wall", s, s.bx > 300 and abs(s.by - (Y1 - RB)) < 1e-6, f"ball=({s.bx:.0f},{s.by:.1f}) slid along wall"))

# robot starts on the goal side of the ball: must walk around, not through
s = Checked((400, 200), (300, 200), (450, 200))
min_d = 1e9
while not s.done():
    s.apply_action(*expert_action(s.state()))
    min_d = min(min_d, math.hypot(s.bx - s.rx, s.by - s.ry))
results.append(check("approach from wrong side", s, s.success(), f"steps={s.steps}"))

# reposition: knock the ball sideways mid-push, expert must recover
s = Checked((100, 200), (200, 200), (520, 200))
for i in range(400):
    if i == 40:
        s.bvy = 150.0
    s.apply_action(*expert_action(s.state()))
    if s.done():
        break
results.append(check("reposition after perturbation", s, s.success(), f"steps={s.steps}"))

# robot pins the ball into the wall: robot must give way, no overlap
s = Checked((300, Y1 - RB - 60), (300, Y1 - RB), (300, 100))
drive(s, 0, 1, 40)
results.append(check("pinning ball against wall", s, True, f"robot y={s.ry:.1f}"))

# corner: ball in a corner, goal elsewhere
s = Checked((150, 300), (X0 + RB + 2, Y1 - RB - 2), (400, 150))
while not s.done():
    s.apply_action(*expert_action(s.state()))
results.append(check("ball wedged in corner (unrecoverable; invariants only)", s, True, f"success={s.success()} final_dist={s.goal_distance():.1f}"))

# jitter: ball resting in contact with a stationary robot must not move
s = Checked((300, 200), (334, 200), (500, 200))
drive(s, 0, 0, 60)
results.append(check("resting contact (no jitter)", s, math.hypot(s.bx - 334, s.by - 200) < 1e-6, ""))

# randomized stress: 300 expert episodes, invariants every substep
rng = random.Random(123)
worst = 0.0
for _ in range(300):
    s = Checked(*sample_scene(rng))
    while not s.done():
        s.apply_action(*expert_action(s.state()))
    worst = max(worst, s.max_pen)
s.max_pen = worst
results.append(check("300 random expert episodes", s, True, ""))

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
