"""Evaluation harness shared by all four scenarios.

The idea is that nobody calls ``objective1`` / ``objective2`` / ``constraint1``
directly. Instead every algorithm gets an ``EvaluationHarness`` and asks it for
function values. That way the counting of evaluations, the wall-clock timing and
the Scenario A budget accounting are done in exactly one place and are identical
no matter which optimizer or scenario is being run.

Contract for an algorithm (this is what task 2's optimizers follow):

    def my_algorithm(harness, seed=None):
        rng = np.random.default_rng(seed)
        x = harness.sample(rng)
        value = harness.f1(x)          # single objective
        ...
        return harness.best_x          # the best layout it found

The harness records a best-so-far history on every objective1 call, which is
what the convergence plots are built from.
"""

import time

import numpy as np

import problem


# Scenario A: pretend every objective1 evaluation costs 30 seconds, with a total
# budget of 5 hours for evaluations *and* algorithm time together.
DEFAULT_SECONDS_PER_EVAL = 30.0
BUDGET_SECONDS = 5 * 60 * 60          # 5 hours -> 18000 s
# 18000 / 30 = 600 evaluations if the algorithm itself took no time at all.
SCENARIO_A_EVAL_BUDGET = int(BUDGET_SECONDS // DEFAULT_SECONDS_PER_EVAL)

# How many seconds of (pretend) evaluation time each scenario charges per call.
# Only Scenario A and the bonus Scenario D have an expensive objective.
_SECONDS_PER_EVAL_BY_SCENARIO = {"A": DEFAULT_SECONDS_PER_EVAL,
                                 "B": 0.0,
                                 "C": 0.0,
                                 "D": DEFAULT_SECONDS_PER_EVAL}


class EvaluationHarness:
    """Counts evaluations, times the run and tracks the best solution so far."""

    def __init__(self, scenario="A", seconds_per_eval=None, repeats=1,
                 record_history=True):
        self.scenario = scenario
        if seconds_per_eval is None:
            seconds_per_eval = _SECONDS_PER_EVAL_BY_SCENARIO.get(scenario, 0.0)
        self.seconds_per_eval = float(seconds_per_eval)
        # How many noisy samples to average per objective1 call. >1 trades budget
        # for a less noisy estimate (each repeat still counts as one evaluation).
        self.repeats = int(repeats)
        self.record_history = record_history

        # search space, copied from the problem definition for convenience
        self.dim = problem.DIM
        self.lower = np.full(self.dim, problem.LOWER_BOUND)
        self.upper = np.full(self.dim, problem.UPPER_BOUND)

        # evaluation counters
        self.n_obj1 = 0
        self.n_obj2 = 0
        self.n_constraint = 0

        # best-so-far bookkeeping (we minimise, so smaller is better)
        self.best_value = np.inf
        self.best_x = None
        self.best_feasible_value = np.inf
        self.best_feasible_x = None

        # logs used by the plotting / analysis helpers
        self.history = []            # (n_obj1, best_value) after each obj1 call
        self.feasible_history = []   # (n_obj1, best_feasible_value)
        self.mo_log = []             # (x, f1, f2) for the multi-objective scenario

        # timing: the real wall-clock spent inside the algorithm
        self.wall_seconds = 0.0
        self._tic = None

    # -- search-space helpers -------------------------------------------------
    def sample(self, rng=None):
        """A uniform random layout in the [0, 1]^10 box."""
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.lower, self.upper)

    def clip(self, x):
        """Clip a layout back into the box (handy after a mutation step)."""
        return np.clip(x, self.lower, self.upper)

    # -- objective / constraint wrappers -------------------------------------
    def f1(self, x):
        """Objective 1 (negative yearly energy). Counts one evaluation per
        repeat, optionally averaging the noise, and updates the best so far."""
        if self.repeats == 1:
            value = float(problem.objective1(x))
        else:
            value = float(np.mean([problem.objective1(x)
                                   for _ in range(self.repeats)]))
        self.n_obj1 += self.repeats

        if value < self.best_value:
            self.best_value = value
            self.best_x = np.array(x, dtype=float)
        if self.record_history:
            self.history.append((self.n_obj1, self.best_value))
        return value

    def f2(self, x):
        """Objective 2 (biodiversity metric, smaller = wider bird corridor)."""
        self.n_obj2 += 1
        return float(problem.objective2(x))

    def c1(self, x):
        """The spacing constraint. 1 if satisfied, 0 if two turbines are too
        close together."""
        self.n_constraint += 1
        return int(problem.constraint1(x))

    def evaluate_constrained(self, x):
        """Scenario B helper: evaluate the objective and the constraint together
        and keep track of the best *feasible* layout seen so far. Returns
        ``(value, feasible)``."""
        value = self.f1(x)
        feasible = self.c1(x) == 1
        if feasible and value < self.best_feasible_value:
            self.best_feasible_value = value
            self.best_feasible_x = np.array(x, dtype=float)
        if self.record_history:
            self.feasible_history.append((self.n_obj1, self.best_feasible_value))
        return value, feasible

    def evaluate_multi(self, x):
        """Scenario C helper: evaluate both objectives and log the pair so a
        Pareto front can be extracted afterwards. Returns ``(f1, f2)``."""
        f1 = self.f1(x)
        f2 = self.f2(x)
        self.mo_log.append((np.array(x, dtype=float), f1, f2))
        return f1, f2

    # -- timing ---------------------------------------------------------------
    def tic(self):
        self._tic = time.perf_counter()

    def toc(self):
        if self._tic is not None:
            self.wall_seconds += time.perf_counter() - self._tic
            self._tic = None
        return self.wall_seconds

    # -- budget / cost accounting --------------------------------------------
    def computation_time_seconds(self):
        """Total reported computation time. For the expensive scenarios this is
        the real algorithm time plus the pretended 30 s per objective1 call (the
        post-processing trick described in the assignment)."""
        return self.wall_seconds + self.seconds_per_eval * self.n_obj1

    def eval_budget(self):
        """How many objective1 evaluations fit in the budget, ignoring algorithm
        time. ``None`` when the scenario has no budget."""
        if self.seconds_per_eval <= 0:
            return None
        return int(BUDGET_SECONDS // self.seconds_per_eval)

    def budget_remaining(self):
        """Objective1 evaluations still allowed, or ``None`` if uncapped."""
        budget = self.eval_budget()
        return None if budget is None else budget - self.n_obj1

    def within_budget(self):
        """Whether the run respected the 5 hour budget (only meaningful for the
        expensive scenarios)."""
        if self.seconds_per_eval <= 0:
            return True
        return self.computation_time_seconds() <= BUDGET_SECONDS

    def summary(self):
        """A small dict with the headline numbers for one run."""
        return {
            "scenario": self.scenario,
            "n_obj1": self.n_obj1,
            "n_obj2": self.n_obj2,
            "n_constraint": self.n_constraint,
            "best_value": self.best_value,
            "best_feasible_value": self.best_feasible_value,
            "wall_seconds": self.wall_seconds,
            "computation_time_seconds": self.computation_time_seconds(),
            "within_budget": self.within_budget(),
        }
