# DMAI Assignment 3 — wind farm layout optimization

We optimize the placement of five wind turbines using the provided XGBoost
surrogate, across four scenarios (A expensive, B constrained, C multi-objective,
D bonus). This repo holds the shared **framework** that everyone builds on, so
the algorithm comparison stays consistent across scenarios.

## Setup

```bash
pip install -r requirements.txt
```

`Ensemble.pkl` must sit next to the code (it does in this repo). The first run
may print an XGBoost version warning; that is expected and harmless.

## Modules

| file | what it does |
|------|--------------|
| `problem.py` | the given `objective1` / `objective2` / `constraint1`, unchanged, made importable |
| `harness.py` | `EvaluationHarness`: counts evaluations, times the run, does the Scenario A budget accounting, tracks best-so-far |
| `experiment.py` | `run_experiment(...)`: runs an algorithm ≥5 times and bundles the results |
| `stats_tests.py` | Wilcoxon / Mann-Whitney comparisons between algorithms |
| `plotting.py` | convergence (mean ± std), Pareto fronts, wind farm layout pictures |
| `example_usage.py` | reference example tying it all together (run `python example_usage.py`) |

## Writing an algorithm

An algorithm is just a function that takes a harness and returns the best layout
it found. It must go through the harness for *every* objective/constraint call so
the evaluations are counted:

```python
import numpy as np

def my_algorithm(harness, seed=None):
    rng = np.random.default_rng(seed)
    for _ in range(harness.budget_remaining() or 500):
        x = harness.sample(rng)        # uniform layout in [0, 1]^10
        harness.f1(x)                  # objective1, counted + logged
    return harness.best_x
```

Useful harness calls:

* `harness.f1(x)` — objective1 (negative yearly energy); updates best-so-far.
* `harness.evaluate_constrained(x)` → `(value, feasible)` — Scenario B; tracks the
  best *feasible* layout.
* `harness.evaluate_multi(x)` → `(f1, f2)` — Scenario C; logs the pair for the
  Pareto front.
* `harness.sample(rng)`, `harness.clip(x)`, `harness.budget_remaining()`.

Then:

```python
from experiment import run_experiment
import plotting, stats_tests

res = run_experiment(my_algorithm, scenario="A", n_runs=5)
plotting.plot_convergence([res], savepath="convergence.png")
print(res.table_row())                 # mean energy, evals, time
```

## Scenario configuration

| scenario | objective(s) | constraint | budget |
|----------|--------------|------------|--------|
| A | objective1 | – | 30 s/eval, 5 h → ~600 evals |
| B | objective1 | spacing (constraint1) | – |
| C | objective1 + objective2 | – | – |
| D (bonus) | objective1 + objective2 | spacing | 30 s/eval, 5 h |

The harness picks the right settings from the `scenario` string.

## Conventions

* We **minimize** `objective1` (it is minus the yearly energy in GWh), so the best
  value is the most negative. Plots show energy = `-objective1` in GWh for
  readability.
* Turbine `k` sits at `(x[2k], x[2k+1])` in `[0, 1]`, scaled by 1666.65 m. The
  spacing rule needs the centres at least two rotor diameters (252 m) apart.
* Runs are seeded `0, 1, 2, ...` and shared across algorithms, so comparisons are
  paired (Wilcoxon signed-rank).
