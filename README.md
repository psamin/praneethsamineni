# praneethsamineni.com

My personal site. It's built with TypeScript and Vite, with React-style
components rendered by Preact (smaller), and hosted on GitHub Pages.

The home page runs a **tiny learned robot policy** live in the browser: a
6 → 32 → 32 → 2 MLP (1,346 parameters, 5.3 KB) that dribbles a ball to a goal.
It was trained by imitating a scripted expert (behaviour cloning with DART and
DAgger, no RL). There's no ML runtime and no server; the forward pass is
hand-written TypeScript.

```bash
npm install
npm run dev      # local dev server
npm test         # TS simulator + policy must reproduce the Python trajectories
npm run build    # -> dist/
```

| Path | What |
|---|---|
| `src/` | The site. `src/data/projects.ts` holds the project list. |
| `src/robot/` | Browser simulator, policy, demo component and exported weights. |
| `training/` | Python simulator, expert, data generation, training, evaluation, export. See [training/README.md](training/README.md). |
| `results/` | Evaluation and benchmark numbers shown on the site. |
| `docs/research.md` | Design notes and results for the robot policy. |
| `docs/hosting.md` | How the domain and GitHub Pages deployment are set up. |
