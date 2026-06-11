"""Runner that executes an algorithm several times and collects the results.

``run_experiment`` is the single entry point everyone uses to produce the
numbers that go into the report: it runs an algorithm ``n_runs`` times (5 by
default, as the assignment asks), each run on a fresh harness with its own seed,
and bundles everything into an ``ExperimentResult`` that the plotting and
statistics helpers understand.

Seeds default to 0, 1, 2, ... and are reused across algorithms. Running every
algorithm on the same set of seeds means the runs are paired, which lets us use
the (more powerful) Wilcoxon signed-rank test later on.
"""

import numpy as np

from harness import EvaluationHarness


class RunResult:
    """Everything we keep from a single run of a single algorithm."""

    def __init__(self, harness, seed, returned_x=None):
        self.seed = seed
        self.returned_x = returned_x

        self.best_x = harness.best_x
        self.best_value = harness.best_value
        self.best_feasible_x = harness.best_feasible_x
        self.best_feasible_value = harness.best_feasible_value

        self.history = list(harness.history)
        self.feasible_history = list(harness.feasible_history)
        self.mo_log = list(harness.mo_log)

        self.n_obj1 = harness.n_obj1
        self.n_obj2 = harness.n_obj2
        self.n_constraint = harness.n_constraint

        self.wall_seconds = harness.wall_seconds
        self.computation_time_seconds = harness.computation_time_seconds()
        self.within_budget = harness.within_budget()


class ExperimentResult:
    """The collection of runs for one algorithm in one scenario."""

    def __init__(self, algorithm_name, scenario, runs):
        self.algorithm_name = algorithm_name
        self.scenario = scenario
        self.runs = runs

    # -- per-run arrays, convenient for stats / tables -----------------------
    def best_values(self):
        return np.array([r.best_value for r in self.runs])

    def best_feasible_values(self):
        return np.array([r.best_feasible_value for r in self.runs])

    def computation_times(self):
        return np.array([r.computation_time_seconds for r in self.runs])

    def eval_counts(self):
        return np.array([r.n_obj1 for r in self.runs])

    def constraint_counts(self):
        return np.array([r.n_constraint for r in self.runs])

    def all_feasible(self):
        """True if every run returned a feasible best solution (Scenario B)."""
        return all(np.isfinite(r.best_feasible_value) for r in self.runs)

    def table_row(self, energy=True):
        """One summary row (mean +/- std) for a results table.

        With ``energy=True`` the objective is reported as energy in GWh (i.e.
        ``-objective1``) so it reads naturally for a decision-maker.
        """
        vals = self.best_values()
        if energy:
            vals = -vals
        return {
            "algorithm": self.algorithm_name,
            "objective_mean": float(np.mean(vals)),
            "objective_std": float(np.std(vals)),
            "evals_mean": float(np.mean(self.eval_counts())),
            "time_mean_s": float(np.mean(self.computation_times())),
        }


def run_experiment(algorithm, scenario="A", n_runs=5, seeds=None, name=None,
                   harness_kwargs=None, **algo_kwargs):
    """Run ``algorithm`` ``n_runs`` times and return an ``ExperimentResult``.

    Parameters
    ----------
    algorithm : callable
        Takes ``(harness, seed=...)`` and optionally returns the best layout it
        found. It must use the harness for every objective/constraint call.
    scenario : str
        One of "A", "B", "C", "D"; selects the harness configuration.
    n_runs : int
        How many independent repetitions (>= 5 for the assignment).
    seeds : sequence of int, optional
        Seeds to use; defaults to ``range(n_runs)`` so runs are paired across
        algorithms.
    harness_kwargs : dict, optional
        Extra keyword arguments forwarded to ``EvaluationHarness`` (e.g.
        ``{"repeats": 3}`` to average the noisy objective).
    **algo_kwargs :
        Extra keyword arguments forwarded to the algorithm.
    """
    if seeds is None:
        seeds = list(range(n_runs))
    if name is None:
        name = getattr(algorithm, "__name__", "algorithm")
    harness_kwargs = dict(harness_kwargs or {})

    runs = []
    for seed in seeds:
        harness = EvaluationHarness(scenario=scenario, **harness_kwargs)
        harness.tic()
        returned_x = algorithm(harness, seed=seed, **algo_kwargs)
        harness.toc()
        runs.append(RunResult(harness, seed, returned_x))

    return ExperimentResult(name, scenario, runs)
