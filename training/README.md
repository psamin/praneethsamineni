# Training the tiny robot policy

This directory holds everything used to train, evaluate and export the
ball-pushing policy that runs on the website. The design reasoning is in
[../docs/research.md](../docs/research.md).

## Setup

```bash
cd training
uv sync
```

## Pipeline

**Run training on the cluster, not on a laptop.** From `training/` on Futurama:

```bash
bash jobs/submit_all.sh                      # MLP + Diffusion/ACT in parallel, then evaluation
MODEL=abs_32 sbatch jobs/finalize.sbatch     # evaluate, export, parity fixture, verified scenes
```

Afterwards, copy `../src/robot/{policyWeights,policyMeta,parity.fixture,scenes}.json`
and `../results/` back, then run `npm test` locally. It checks Python/TypeScript
parity and that every verified scene is solved in the browser code.

The individual steps, in order (times from an 11-core M3 Pro):

```bash
uv run python test_physics.py      # contact/wall invariants, every substep          (~3 s)
uv run python generate_data.py     # DART expert demos -> data/{train,val,test}.npz   (~3 s)
uv run python train.py             # BC + DAgger grid  -> models/*.pt                 (~minutes)
uv run python evaluate.py 2000     # closed loop, unseen seeds -> ../results/evaluation.json
uv run python export.py            # smallest reliable model -> ../src/robot/policyWeights.json
uv run python evaluate.py pymunk <model> 500   # optional cross-check in Pymunk
uv run python parity.py            # fixture for the Python<->TypeScript parity test
cd .. && npm test                  # TS simulator + policy must reproduce Python
```

`train.py` takes optional arguments: `uv run python train.py abs 32,64 6`
trains feature set `abs`, widths 32 and 64, with 6 DAgger rounds.

## Files

| File | Purpose |
|---|---|
| `environment.py` | Authoritative simulator, spawning and features. Mirrored by `src/robot/simulation.ts`. |
| `expert.py` | Scripted geometric expert. |
| `generate_data.py` | DART-noised expert rollouts. |
| `train.py` | Behaviour cloning, then DAgger rounds. Also keeps a `<name>_bc` model for comparison. |
| `evaluate.py` | Closed-loop success, steps, near-wall failure rate. |
| `diagnose.py` | Groups failures by mode: stalled, stuck on a wall, and so on. |
| `export.py` | Writes the web weights file and `results/benchmarks.json`. |
| `pymunk_env.py` | The same task in Pymunk, used only as an independent check. |
| `parity.py` | Writes `src/robot/parity.fixture.json` for `npm test`. |
| `policies.py` | MLP, Diffusion Policy and ACT models, plus the inference code the browser mirrors. |
| `train_chunked.py` | Trains the Diffusion Policy and ACT families. |
| `verified_scenes.py` | Picks the starting scenes the demo may spawn: only ones the shipped policy solves. |
| `jobs/` | Slurm scripts for Futurama. |

All physics constants live in `../src/robot/simConfig.json`, which both
simulators read. If you change the physics in Python, make the same change in
`simulation.ts` and rerun `parity.py` and `npm test`.
