"""Independent cross-check: the same task in Pymunk / Chipmunk2D.

Not used for training or on the website. evaluate.py runs the final policy
here to show it is not exploiting quirks of the mirror simulator.
Setup follows Push-T: zero gravity, kinematic velocity-controlled agent,
static segment walls. Table friction on the ball is a PivotJoint to a static
body with bounded force (the same trick Push-T-style envs use for a top-down
table), plus global damping.
"""

import math

import pymunk

from environment import C, RB, RR, VMAX, X0, X1, Y0, Y1

BALL_MASS = 1.0


class PymunkSim:
    def __init__(self, robot, ball, goal):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        # per-second velocity retention matching the mirror sim's per-substep damping
        self.space.damping = C["BALL_DAMPING_PER_SUBSTEP"] ** (1 / C["PHYSICS_DT"])
        static = self.space.static_body
        for a, b in [((X0, Y0), (X1, Y0)), ((X1, Y0), (X1, Y1)), ((X1, Y1), (X0, Y1)), ((X0, Y1), (X0, Y0))]:
            seg = pymunk.Segment(static, a, b, 0)
            seg.elasticity = C["WALL_RESTITUTION"]
            seg.friction = 0.2
            self.space.add(seg)

        self.robot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.robot.position = robot
        rs = pymunk.Circle(self.robot, RR)
        rs.elasticity, rs.friction = C["ROBOT_BALL_RESTITUTION"], C["CONTACT_FRICTION"]

        self.ball = pymunk.Body(BALL_MASS, pymunk.moment_for_circle(BALL_MASS, 0, RB))
        self.ball.position = ball
        bs = pymunk.Circle(self.ball, RB)
        bs.elasticity, bs.friction = 1.0, 1.0
        friction = pymunk.PivotJoint(static, self.ball, (0, 0), (0, 0))
        friction.max_bias = 0
        friction.max_force = BALL_MASS * C["BALL_FRICTION_DECEL"]
        self.space.add(self.robot, rs, self.ball, bs, friction)

        self.gx, self.gy = goal
        self.vx = self.vy = 0.0
        self.steps = self.hold = 0

    def apply_action(self, ax, ay):
        m = math.hypot(ax, ay)
        if m > 1:
            ax, ay = ax / m, ay / m
        dx, dy = ax * VMAX - self.vx, ay * VMAX - self.vy
        d = math.hypot(dx, dy)
        if d > C["MAX_ACTION_DELTA"]:
            dx, dy = dx * C["MAX_ACTION_DELTA"] / d, dy * C["MAX_ACTION_DELTA"] / d
        self.vx += dx
        self.vy += dy
        for _ in range(C["SUBSTEPS_PER_ACTION"]):
            # keep the kinematic robot inside the arena (it ignores walls)
            x, y = self.robot.position
            vx = 0.0 if (x <= X0 + RR and self.vx < 0) or (x >= X1 - RR and self.vx > 0) else self.vx
            vy = 0.0 if (y <= Y0 + RR and self.vy < 0) or (y >= Y1 - RR and self.vy > 0) else self.vy
            self.robot.velocity = (vx, vy)
            self.space.step(C["PHYSICS_DT"])
            x, y = self.robot.position
            self.robot.position = (min(max(x, X0 + RR), X1 - RR), min(max(y, Y0 + RR), Y1 - RR))
        self.steps += 1
        self.hold = self.hold + 1 if self.goal_distance() <= C["GOAL_SUCCESS_RADIUS"] else 0

    def goal_distance(self):
        bx, by = self.ball.position
        return math.hypot(self.gx - bx, self.gy - by)

    def success(self):
        return self.hold >= C["SUCCESS_HOLD_STEPS"]

    def done(self):
        return self.success() or self.steps >= C["MAX_EPISODE_STEPS"]

    def state(self):
        (rx, ry), (bx, by) = self.robot.position, self.ball.position
        return (rx, ry, bx, by, self.gx, self.gy)
