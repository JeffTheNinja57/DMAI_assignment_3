"""Reference example showing how to use the framework.

This is the template the scenario owners (tasks 3-5) follow: write an algorithm
that takes a harness, run it through ``run_experiment``, then feed the results to
the plotting and statistics helpers. The optimizer library itself (random search,
Bayesian optimisation, CMA-ES, NSGA-II) is task 2's job; the tiny random searches
below are only here to exercise the plumbing.

Run it with:  python example_usage.py
"""

import os

import numpy as np

from experiment import run_experiment
import stats_tests
import plotting


# --- a couple of throwaway demo algorithms ---------------------------------
def random_search(harness, seed=None, budget=200):
    """Plain random search on objective1 (Scenarios A/D shape)."""
    rng = np.random.default_rng(seed)
    cap = harness.budget_remaining()
    if cap is not None:
        budget = min(budget, cap)
    for _ in range(budget):
        harness.f1(harness.sample(rng))
    return harness.best_x


def random_search_constrained(harness, seed=None, budget=200):
    """Random search that only keeps feasible layouts (Scenario B shape)."""
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        harness.evaluate_constrained(harness.sample(rng))
    return harness.best_feasible_x


def random_search_multi(harness, seed=None, budget=300):
    """Random search logging both objectives (Scenario C shape)."""
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        harness.evaluate_multi(harness.sample(rng))
    return None


def main(outdir="demo_outputs"):
    os.makedirs(outdir, exist_ok=True)

    # Scenario A: compare two evaluation budgets, with a statistical test.
    rs_small = run_experiment(random_search, scenario="A", n_runs=5,
                              budget=50, name="RS (50 evals)")
    rs_large = run_experiment(random_search, scenario="A", n_runs=5,
                              budget=200, name="RS (200 evals)")
    print("Scenario A summary rows:")
    for res in (rs_small, rs_large):
        print(" ", res.table_row())
    print(stats_tests.format_table(
        stats_tests.pairwise_table([rs_small, rs_large])))

    plotting.plot_convergence([rs_small, rs_large],
                              savepath=os.path.join(outdir, "convergence_A.png"))

    # Scenario B: feasible random search, check every run ends feasible.
    rs_b = run_experiment(random_search_constrained, scenario="B", n_runs=5,
                          budget=200, name="feasible RS")
    print("\nScenario B all runs feasible:", rs_b.all_feasible())
    plotting.plot_convergence([rs_b], metric="feasible",
                              savepath=os.path.join(outdir, "convergence_B.png"))

    # Scenario C: a Pareto front from random sampling.
    rs_c = run_experiment(random_search_multi, scenario="C", n_runs=5,
                          budget=300, name="RS Pareto")
    plotting.plot_pareto([rs_c], savepath=os.path.join(outdir, "pareto_C.png"))

    # Best layout picture of the best Scenario A run.
    best_run = min(rs_large.runs, key=lambda r: r.best_value)
    plotting.plot_layout(best_run.best_x, energy=-best_run.best_value,
                         savepath=os.path.join(outdir, "best_layout_A.png"))
    print(f"\nFigures written to {outdir}/")


if __name__ == "__main__":
    main()
