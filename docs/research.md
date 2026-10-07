# Tiny Robot Policy: research notes and design decisions

The goal is a ball-pushing robot on the portfolio site, driven by a genuinely
learned neural policy that is small enough to run in the browser at no cost.
These notes record what was looked at and why the design ended up as it did.
Measured results are in `results/evaluation.json` and `results/benchmarks.json`,
and are summarised at the end.

## 1. Push-T and gym-pusht (reference task)

Source: `huggingface/gym-pusht`, `gym_pusht/envs/pusht.py` (Diffusion Policy, Chi et al. 2023).

| Property | Push-T | This project |
|---|---|---|
| Physics | Pymunk (Chipmunk2D), `space.gravity = (0, 0)` | Custom circle-contact sim, zero gravity |
| Agent | **kinematic** circle, radius 15, PD controller to a target position (`k_p=100, k_v=20`) | kinematic disc, radius 20, velocity command with an acceleration limit |
| Object | T-shaped block (friction 1) | ball, radius 14 |
| Rates | `dt = 0.01` (100 Hz physics), 10 Hz control, 10 substeps | 240 Hz physics, 30 Hz control, 8 substeps |
| Walls | four static segments | four boundaries, clamped and reflected |
| Success | coverage > 95% | ball centre within `GOAL_SUCCESS_RADIUS` for `SUCCESS_HOLD_STEPS` steps |
| State observation | `[agent_x, agent_y, block_x, block_y, block_angle]` | `[robot, ball, goal]` positions (6 numbers) |

What carried over: zero gravity, a kinematic agent, a fixed physics step with
several substeps per action, and static walls. The ball replaces the T, which
removes the orientation sub-problem (the hard part of Push-T) and keeps the task
small enough for a ~1–5k-parameter MLP.

## 2. Physics engine: Pymunk vs. a mirrored custom simulator

The spec recommends Pymunk for training. The deciding constraint is
**behaviour consistency**: the policy must see in the browser the same dynamics
it was trained on.

* Pymunk in the browser would need a JS port of Chipmunk (chipmunk-js and
  similar). Those ports are separate implementations whose contact solvers and
  defaults do not match Pymunk's step for step, so there would be a
  sim-to-sim gap and a sizeable dependency.
* The task only needs disc–disc and disc–wall contact. Done properly, that is
  about 100 lines.

**Decision.** `training/environment.py` is the authoritative simulator and
`src/robot/simulation.ts` mirrors it line for line. Both read every constant
from one file, `src/robot/simConfig.json`. `training/parity.py` together with
`src/robot/parity.test.ts` replays recorded trajectories in TypeScript and
requires agreement to 1e-9 world units, which the current build meets. The browser ships **no physics library**.

Pymunk is kept as an **independent cross-check** (`training/pymunk_env.py`).
The exported policy is also evaluated there, to show that it has not learned
to exploit quirks of the custom simulator.

### Contact model (and why it does not tunnel, explode or jitter)

* The robot is kinematic, with effectively infinite mass. Each substep it is
  integrated and then clamped to the arena.
* The ball has linear damping (`BALL_DAMPING_PER_SUBSTEP`) plus a constant
  rolling-friction deceleration. That makes it stop instead of drifting forever,
  which is close to the quasi-static behaviour of tabletop pushing.
* Robot–ball contact uses three steps:
  1. Positional projection removes any overlap, so no persistent penetration
     builds up.
  2. A normal impulse is applied only when the bodies are approaching
     (`vn < 0`), with low restitution (0.05), so pushing looks continuous
     rather than bouncy.
  3. A small tangential friction term is applied.
* **Pinning.** If the ball is pinned against a wall after the wall clamp, the
  *robot* is pushed back, so the robot can never squeeze the ball out of the
  arena.
* **Tunnelling.** The fastest body moves at most 180 u/s × (1/240) s = 0.75
  units per substep, against radii of 14–20, so tunnelling is impossible by
  construction. Continuous collision detection is not needed.
* **Timestep.** A fixed timestep keeps the dynamics deterministic. The browser
  runs simulation and rendering on separate clocks using the accumulator
  pattern from Glenn Fiedler's "Fix Your Timestep!", and interpolates the
  rendered state between the last two physics states.
* **Floating point.** Python floats and JS numbers are both IEEE-754 doubles,
  and the code avoids transcendental functions on the physics path, so the two
  simulators agree to round-off.

`training/test_physics.py` checks invariants after **every substep**: no
overlap, nothing outside the arena, and no energy blow-up. It runs them on
these scenarios: a direct push, a diagonal push, a glancing collision, the
ball hitting a wall, a push along a wall, an approach from the wrong side,
repositioning after a sideways knock, pinning the ball against a wall, resting
contact (checking for jitter), and 300 random expert episodes.

## 3. Scripted expert

The expert is a pure function of the state (no internal mode flag), so a
feed-forward MLP with the same inputs can represent it. It works in three
parts:

* **Orbit.** The robot walks around the ball on a circle, at most 50° per
  decision. Chords of that circle stay outside the ball, so the robot goes
  around it, not into it.
* **Push.** Pure pursuit toward a point past the ball along the ball→goal
  line. Speed tapers smoothly to zero at the goal.
* **Blend.** A smoothstep on the alignment angle mixes the two.

Three lessons from building it, each found by tracing failed episodes:

1. **Lateral correction while in contact drags the ball sideways.** The ball
   ends up orbiting the goal in a stable limit cycle. Alignment errors must be
   fixed by stepping back and around, not by sliding along the ball.
2. **Cliffs in the expert become stalls in the student.** A hard "stop when
   the ball is within 7 units" rule plus a `max(45, ·)` speed floor gave the
   MLP a discontinuity to average over, and it learned to stop short of the
   goal. A continuous taper fixed it.
3. **Near the goal, `normalize(goal − ball)` is hypersensitive.** A 3-unit
   drift rotates it by more than 10°. The expert therefore relaxes its
   alignment requirement as the ball arrives. This removed most of the
   remaining "stuck just outside the goal" failures.

**Near walls.** If the point behind the ball is outside the robot's reachable
area, the push direction is rotated in 5° steps toward the nearest direction
whose approach point is reachable, so the robot pushes the ball off the wall
at an angle. A ball wedged into a corner cannot be recovered by a round pusher
(every reachable contact pushes it further into the walls). Spawn margins make
that rare, and the website resets on timeout.

The expert solves 100% of 2,000 random scenes, in about 160 policy steps
(about 5.4 s) on average.

## 4. Learning: behaviour cloning, DART, DAgger (no RL)

* **Data.** 3,000 expert episodes give about 530k (state, action) pairs and
  take about 2 s to generate on 11 cores. Validation and test splits use
  disjoint seeds, and closed-loop evaluation uses a fourth seed range
  (300000+) that never appears in any dataset.
* **DART** (Laskey et al., CoRL 2017, "Noise Injection for Robust Imitation
  Learning"). The robot executes the expert action plus Ornstein–Uhlenbeck
  noise, while the *label* is always the clean expert action. The data then
  contains states slightly off the expert's path, together with the
  correction back.
* **DAgger** (Ross, Gordon & Bagnell, AISTATS 2011). Roll out the learned
  policy, ask the expert what it would have done in every state the policy
  visited, aggregate, and retrain. This is supervised imitation. There is no
  reward and no RL; the expert remains the only source of supervision. Plain
  BC alone reached roughly 30–60% closed-loop success because of compounding
  error, while DAgger closed most of that gap (numbers below).
* **Loss.** MSE on the action, which is a desired velocity as a fraction of
  the robot's maximum speed.
* **Action parameterisation.** The spec suggests trying both velocity and
  position-delta actions. At a fixed control rate they are the same thing up
  to a constant (Δx = v · Δt), so only velocity was used. The acceleration
  limit in the actuator applies to either.
* **Features.**
  * `abs`: the six absolute positions, each normalised to [−1, 1].
  * `rel`: ball−robot, goal−ball and goal−robot offsets, each divided by the
    arena width.
  * `rel_ball`: `rel` plus the ball's absolute position (8 inputs).

  Relative features drop wall information, and the near-wall expert depends on
  it. In practice `abs` was the most reliable (see the results).

## 5. Browser inference

The network is 6 → h → h → 2 with ReLU. Runtimes such as ONNX Runtime Web
(hundreds of kB of JS plus a WASM binary) or TF.js are many orders of
magnitude larger than the model. The forward pass is about 30 lines of
TypeScript in `src/robot/tinyPolicy.ts`. It uses preallocated `Float64Array`
buffers, so there is no garbage per step. Weights ship as base64
little-endian float32 inside a JSON file. ONNX Runtime Web was not
benchmarked: for a model of a few kB its initialisation alone would be slower
than tens of thousands of hand-written forward passes, and it would add a
large download for no benefit.

The site measures inference time on the visitor's own device at load
(20,000 calls after warm-up) and displays that number instead of a hard-coded
one. The policy runs at 30 Hz and rendering at display rate. The demo pauses
when scrolled off-screen, and starts paused under `prefers-reduced-motion`.

## 6. Results

The arena was widened from 640×400 to 960×360 to fit the site layout, and
everything was retrained on Futurama. The numbers below are for the wide arena.
Closed-loop evaluation uses 2,000 unseen random scenes (seeds 300000+), so the
95% interval is about ±1.3 percentage points.

| Model | Params | Weights | Copying only (BC + DART) | + 6 DAgger rounds | Pymunk check (n=1000) |
|---|---|---|---|---|---|
| teacher (expert) | – | – | – | 99.9% | 99.9% |
| **MLP 6→32→32→2** (shipped) | **1,346** | **5.3 KB** | 31.4% | **90.4%** | 96.4% |
| MLP 6→64→64→2 | 4,738 | 18.5 KB | 58.0% | 94.1% | 96.8% |

* **Why the smaller model ships.** The goal was the smallest model that looks
  reliable. The demo only spawns scenes the shipped policy is verified to
  solve (`training/verified_scenes.py`): 400 of 465 candidates passed. A test
  (`src/robot/scenes.test.ts`) replays every one of them in the browser's own
  TypeScript code and requires success, so the demo can't stall on a scene it
  picks itself. The success rate reported on the site is still the unfiltered
  90.4%.
* **Diffusion Policy and ACT, same data and recipe** (`train_chunked.py`; 8-step
  action chunks; diffusion executes 4 then replans with 10 DDIM steps; ACT uses
  temporal ensembling; 3 DAgger rounds):

  | Model | Params | Copying only | + DAgger |
  |---|---|---|---|
  | Diffusion, 64-wide denoiser | 7,184 | 3.1% | 6.5% |
  | Diffusion, 128-wide denoiser | 22,544 | 45.8% | 90.1% |
  | ACT, d=32, 2 layers | 18,050 | 92.8% | 94.9% |
  | ACT, d=64, 2 layers | 68,866 | 94.3% | **95.6%** |

  ACT's action chunking almost removes the compounding-error gap on its own:
  plain copying reaches 93% against the MLP's 31%. Predicting 8 steps ahead and
  averaging the overlapping predictions smooths over the hesitation points
  where a single-step policy stalls. Diffusion needs capacity. At 7k parameters
  the denoiser can't model the action distribution at every noise level; at
  22k it matches the small MLP, but with 17× the parameters and 10 network
  passes per decision. The shipped MLP stays, because it is by far the smallest
  and the verified scene list makes the demo reliable. ACT d=32 (about 70 KB) is
  the choice if robustness on user-set scenes matters more than size.
* **Inference.** Measured on the earlier 64-wide model: 5.4 µs median per
  decision in V8 (Node 22, M3 Pro). The 32-wide model does about 3.5× less
  arithmetic. Both are far under the 0.1 ms target. The site measures the
  decision rate live.
* **Bundle.** With Preact, the whole site's JavaScript is about 49 kB gzipped,
  and the weights are a small part of that.
* **Remaining failures on random scenes** are mostly spurious fixed points:
  places where the learned action field is zero but the teacher's is not. The
  rest are balls pushed against a wall. This is the usual compounding-error gap
  in imitation learning.

See `results/evaluation.json` and `results/benchmarks.json` for the raw
numbers, and `training/README.md` to reproduce them.
