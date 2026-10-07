"""Scripted geometric expert. Memoryless: action = f(robot, ball, goal).

Being a pure function of the state (no internal mode flag) is deliberate: a
feed-forward MLP with the same inputs can represent it, so behaviour cloning
has a well-posed target. Push/approach modes are blended smoothly instead of
switched, which also makes the target easier to fit.

  1. push_dir = normalize(goal - ball), bent away from walls if the robot
     could not physically stand behind the ball (near-wall handling).
  2. theta = angle of the robot around the ball, measured from "directly
     behind" (-push_dir). theta == 0 means perfectly lined up.
  3. orbit term: move to a point on a circle of radius ORBIT_RADIUS around the
     ball, rotated toward theta = 0 by at most ORBIT_STEP. Chords of that
     circle stay outside the ball, so the robot walks around it, not into it.
  4. push term: pure pursuit toward a point LOOKAHEAD past the ball along
     push_dir, slowing as the ball nears the goal (so it stops in it).
  5. weight w = smoothstep on |theta|: push when aligned, orbit otherwise.
"""

import math

from environment import CONTACT_DISTANCE, RR, VMAX, X0, X1, Y0, Y1

APPROACH_DISTANCE = CONTACT_DISTANCE + 10   # where the robot waits behind the ball
ORBIT_RADIUS = CONTACT_DISTANCE + 14
ORBIT_STEP = math.radians(50)
ALIGNMENT_THRESHOLD = math.radians(4)       # fully in push mode below this
PUSHING_ALIGNMENT_THRESHOLD = math.radians(12)  # fully in orbit mode above this
NEAR_GOAL_DISTANCE = 50.0  # alignment relaxes inside this ball->goal distance
NEAR_GOAL_RELAX = math.radians(20)
LOOKAHEAD = 40.0        # pursuit point this far past the ball along push_dir
APPROACH_GAIN = 8.0     # 1/s, position gain toward orbit targets
PUSH_GAIN = 1.8         # 1/s, push speed per unit of ball->goal distance
MIN_PUSH_SPEED = 45.0
TAPER_DISTANCE = 25.0    # push speed fades to 0 inside this ball->goal distance
CLOSE_GAIN = 3.0        # 1/s, extra speed while closing the gap to the ball
WALL_PAD = 2.0          # approach point must be this far inside robot bounds


def _robot_reachable(x, y, pad=WALL_PAD):
    return X0 + RR + pad <= x <= X1 - RR - pad and Y0 + RR + pad <= y <= Y1 - RR - pad


def _violation(x, y):
    vx = max(X0 + RR - x, 0.0, x - (X1 - RR))
    vy = max(Y0 + RR - y, 0.0, y - (Y1 - RR))
    return vx + vy


def push_direction(bx, by, gx, gy):
    """Goal direction, rotated to the nearest direction whose approach point
    the robot can actually reach (it may not fit between ball and wall)."""
    base = math.atan2(gy - by, gx - bx)
    best, best_v = base, float("inf")
    for k in range(0, 19):  # 0, +-5deg, ..., +-90deg
        for sgn in ((1,) if k == 0 else (1, -1)):
            a = base + sgn * math.radians(5 * k)
            ax = bx - math.cos(a) * APPROACH_DISTANCE
            ay = by - math.sin(a) * APPROACH_DISTANCE
            if _robot_reachable(ax, ay):
                return math.cos(a), math.sin(a)
            v = _violation(ax, ay)
            if v < best_v:
                best, best_v = a, v
    return math.cos(best), math.sin(best)


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def expert_action(s):
    """Returns desired velocity as a fraction of MAX_ROBOT_SPEED."""
    rx, ry, bx, by, gx, gy = s
    dx, dy = push_direction(bx, by, gx, gy)
    px, py = -dy, dx  # perpendicular

    ox, oy = rx - bx, ry - by
    dist = math.hypot(ox, oy)
    # theta: signed angle from "behind" (-d) to the robot's offset direction
    bxd, byd = -dx, -dy
    theta = math.atan2(bxd * oy - byd * ox, bxd * ox + byd * oy)

    # ---- orbit term
    step = max(-ORBIT_STEP, min(ORBIT_STEP, -theta))
    t = theta + step
    cx = bxd * math.cos(t) - byd * math.sin(t)
    cy = bxd * math.sin(t) + byd * math.cos(t)
    rad = APPROACH_DISTANCE if abs(theta) < PUSHING_ALIGNMENT_THRESHOLD else ORBIT_RADIUS
    tx = min(max(bx + cx * rad, X0 + RR), X1 - RR)
    ty = min(max(by + cy * rad, Y0 + RR), Y1 - RR)
    vox, voy = APPROACH_GAIN * (tx - rx), APPROACH_GAIN * (ty - ry)

    # ---- push term: pure pursuit toward a point beyond the ball
    goal_d = math.hypot(gx - bx, gy - by)
    # continuous taper to zero at the goal centre (a cliff here is hard to clone)
    v_push = PUSH_GAIN * goal_d + MIN_PUSH_SPEED * min(1.0, goal_d / TAPER_DISTANCE)
    gap = max(dist - CONTACT_DISTANCE, 0.0)
    v_fwd = min(VMAX, v_push + CLOSE_GAIN * gap)
    aim_x, aim_y = bx + dx * LOOKAHEAD - rx, by + dy * LOOKAHEAD - ry
    an = math.hypot(aim_x, aim_y)
    vpx, vpy = aim_x / an * v_fwd, aim_y / an * v_fwd

    # ---- blend
    # near the goal, push_dir swings wildly with tiny ball drift, but an
    # off-axis push still lands the ball: relax alignment as the ball arrives
    relax = NEAR_GOAL_RELAX * max(0.0, 1.0 - goal_d / NEAR_GOAL_DISTANCE)
    w = 1.0 - smoothstep(ALIGNMENT_THRESHOLD + relax, PUSHING_ALIGNMENT_THRESHOLD + 2 * relax, abs(theta))
    vx = w * vpx + (1 - w) * vox
    vy = w * vpy + (1 - w) * voy

    sp = math.hypot(vx, vy)
    if sp > VMAX:
        vx, vy = vx / sp * VMAX, vy / sp * VMAX
    return vx / VMAX, vy / VMAX
