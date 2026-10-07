"""Ball-pushing simulator. Authoritative dynamics for training AND the website.

This file is mirrored line-for-line by src/robot/simulation.ts. Both read the
same constants from src/robot/simConfig.json. If you change the physics here,
change it there too and rerun `python training/parity.py`.

Model (top-down tabletop, no gravity):
  * robot: kinematic disc, velocity controlled, acceleration limited.
  * ball: dynamic disc with linear damping + constant rolling friction.
  * contact: robot is infinitely massive; impulse with restitution and a
    little tangential friction; positional correction removes overlap.
  * walls: four static boundaries; the ball reflects, everything is clamped.
Fixed timestep, several substeps per policy action (no tunnelling: the
fastest object moves < 1 unit per substep, discs are >= 14 units).
"""

import json
import math
import random
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "src" / "robot" / "simConfig.json"
C = json.loads(CONFIG_PATH.read_text())

X0, X1 = C["WORLD_MIN_X"], C["WORLD_MAX_X"]
Y0, Y1 = C["WORLD_MIN_Y"], C["WORLD_MAX_Y"]
RR, RB = C["ROBOT_RADIUS"], C["BALL_RADIUS"]
CONTACT_DISTANCE = RR + RB
DT = C["PHYSICS_DT"]
VMAX = C["MAX_ROBOT_SPEED"]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Sim:
    def __init__(self, robot, ball, goal):
        self.rx, self.ry = robot
        self.bx, self.by = ball
        self.gx, self.gy = goal
        self.rvx = self.rvy = 0.0  # actual robot velocity (after accel limit)
        self.bvx = self.bvy = 0.0
        self.steps = 0
        self.hold = 0  # consecutive policy steps with ball inside success radius

    # ------------------------------------------------------------- physics
    def apply_action(self, ax, ay):
        """ax, ay: desired velocity as a fraction of MAX_ROBOT_SPEED."""
        m = math.sqrt(ax * ax + ay * ay)
        if m > 1.0:
            ax /= m
            ay /= m
        tx, ty = ax * VMAX, ay * VMAX
        dx, dy = tx - self.rvx, ty - self.rvy
        d = math.sqrt(dx * dx + dy * dy)
        lim = C["MAX_ACTION_DELTA"]
        if d > lim:
            dx *= lim / d
            dy *= lim / d
        self.rvx += dx
        self.rvy += dy
        for _ in range(C["SUBSTEPS_PER_ACTION"]):
            self._substep()
        self.steps += 1
        if self.goal_distance() <= C["GOAL_SUCCESS_RADIUS"]:
            self.hold += 1
        else:
            self.hold = 0

    def _substep(self):
        # robot: integrate, keep inside the arena
        self.rx = clamp(self.rx + self.rvx * DT, X0 + RR, X1 - RR)
        self.ry = clamp(self.ry + self.rvy * DT, Y0 + RR, Y1 - RR)

        # ball: integrate, damping, rolling friction
        self.bx += self.bvx * DT
        self.by += self.bvy * DT
        self.bvx *= C["BALL_DAMPING_PER_SUBSTEP"]
        self.bvy *= C["BALL_DAMPING_PER_SUBSTEP"]
        sp = math.sqrt(self.bvx * self.bvx + self.bvy * self.bvy)
        dec = C["BALL_FRICTION_DECEL"] * DT
        if sp <= dec:
            self.bvx = self.bvy = 0.0
        else:
            self.bvx -= self.bvx / sp * dec
            self.bvy -= self.bvy / sp * dec
        self._ball_walls()

        # robot-ball contact
        nx, ny, dist = self._contact()
        if dist < CONTACT_DISTANCE:
            pen = CONTACT_DISTANCE - dist
            self.bx += nx * pen
            self.by += ny * pen
            rvx, rvy = self.bvx - self.rvx, self.bvy - self.rvy
            vn = rvx * nx + rvy * ny
            if vn < 0.0:
                j = -(1.0 + C["ROBOT_BALL_RESTITUTION"]) * vn
                self.bvx += j * nx
                self.bvy += j * ny
                # tangential friction: bleed relative sliding velocity
                tx, ty = -ny, nx
                vt = rvx * tx + rvy * ty
                f = C["CONTACT_FRICTION"]
                self.bvx -= f * vt * tx
                self.bvy -= f * vt * ty
            self._ball_walls()
            # ball pinned against a wall: the robot gives way instead
            nx, ny, dist = self._contact()
            if dist < CONTACT_DISTANCE:
                pen = CONTACT_DISTANCE - dist
                self.rx = clamp(self.rx - nx * pen, X0 + RR, X1 - RR)
                self.ry = clamp(self.ry - ny * pen, Y0 + RR, Y1 - RR)

    def _contact(self):
        dx, dy = self.bx - self.rx, self.by - self.ry
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-9:
            return 1.0, 0.0, 0.0
        return dx / dist, dy / dist, dist

    def _ball_walls(self):
        e = C["WALL_RESTITUTION"]
        if self.bx < X0 + RB:
            self.bx = X0 + RB
            if self.bvx < 0.0:
                self.bvx = -e * self.bvx
        elif self.bx > X1 - RB:
            self.bx = X1 - RB
            if self.bvx > 0.0:
                self.bvx = -e * self.bvx
        if self.by < Y0 + RB:
            self.by = Y0 + RB
            if self.bvy < 0.0:
                self.bvy = -e * self.bvy
        elif self.by > Y1 - RB:
            self.by = Y1 - RB
            if self.bvy > 0.0:
                self.bvy = -e * self.bvy

    # ------------------------------------------------------------- queries
    def goal_distance(self):
        return math.hypot(self.gx - self.bx, self.gy - self.by)

    def success(self):
        return self.hold >= C["SUCCESS_HOLD_STEPS"]

    def done(self):
        return self.success() or self.steps >= C["MAX_EPISODE_STEPS"]

    def state(self):
        return (self.rx, self.ry, self.bx, self.by, self.gx, self.gy)


# ----------------------------------------------------------------- spawning
def sample_scene(rng: random.Random):
    """Rejection-sample a valid (robot, ball, goal). Mirrored in simulation.ts."""
    m, gm = C["SPAWN_MARGIN"], C["GOAL_SPAWN_MARGIN"]
    while True:
        r = (rng.uniform(X0 + m, X1 - m), rng.uniform(Y0 + m, Y1 - m))
        b = (rng.uniform(X0 + m, X1 - m), rng.uniform(Y0 + m, Y1 - m))
        g = (rng.uniform(X0 + gm, X1 - gm), rng.uniform(Y0 + gm, Y1 - gm))
        if math.dist(r, b) < RR + RB + C["SPAWN_CLEARANCE"]:
            continue
        if math.dist(b, g) < C["MIN_BALL_GOAL_DIST"]:
            continue
        if math.dist(r, g) < C["MIN_ROBOT_GOAL_DIST"]:
            continue
        return r, b, g


# ----------------------------------------------------------------- features
HALF_W = (X1 - X0) / 2
HALF_H = (Y1 - Y0) / 2
CX, CY = (X0 + X1) / 2, (Y0 + Y1) / 2
SCALE = X1 - X0  # relative offsets / arena width -> roughly [-1, 1]


def features(s, kind):
    rx, ry, bx, by, gx, gy = s
    if kind == "abs":
        return [(rx - CX) / HALF_W, (ry - CY) / HALF_H, (bx - CX) / HALF_W,
                (by - CY) / HALF_H, (gx - CX) / HALF_W, (gy - CY) / HALF_H]
    rel = [(bx - rx) / SCALE, (by - ry) / SCALE, (gx - bx) / SCALE,
           (gy - by) / SCALE, (gx - rx) / SCALE, (gy - ry) / SCALE]
    if kind == "rel":
        return rel
    if kind == "rel_ball":  # relative + where the ball is (wall awareness)
        return rel + [(bx - CX) / HALF_W, (by - CY) / HALF_H]
    raise ValueError(kind)


FEATURE_DIMS = {"abs": 6, "rel": 6, "rel_ball": 8}
