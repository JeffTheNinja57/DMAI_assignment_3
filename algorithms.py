"""Optimizer library for the wind farm layout assignment.

Four algorithms that all follow the harness contract:

    def algorithm(harness, seed=None, **kwargs) -> best_x

Every objective/constraint call must go through the harness so evaluations
are counted, timed, and budget-tracked identically for all scenarios.

Algorithms
----------
random_search     — baseline, uniform sampling
bayes_opt         — Optuna TPE (Tree-structured Parzen Estimator), the
                    course's surrogate-based BO method with minimal adaptation
cma_es            — CMA-ES (Hansen 2016), a state-of-the-art evolutionary
                    strategy that adapts the covariance of its sampling
                    distribution; goes beyond the course material
nsga2             — NSGA-II (Deb et al. 2002), the reference multi-objective
                    evolutionary algorithm via pymoo; goes beyond the course

References
----------
Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
    multiobjective genetic algorithm: NSGA-II. IEEE TEC, 6(2), 182-197.
"""

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# 1. Random Search
# ---------------------------------------------------------------------------

def random_search(harness, seed=None, budget=None):
    """Uniform random search — the baseline.

    Uses the evaluation budget remaining in the harness when no explicit
    ``budget`` is given (correct for Scenario A), or falls back to 200 evals.
    """
    rng = np.random.default_rng(seed)
    cap = harness.budget_remaining()
    if budget is None:
        budget = cap if cap is not None else 200
    else:
        if cap is not None:
            budget = min(budget, cap)

    for _ in range(budget):
        harness.f1(harness.sample(rng))
    return harness.best_x


def random_search_constrained(harness, seed=None, budget=500):
    """Random search for Scenario B: only keeps feasible layouts."""
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        harness.evaluate_constrained(harness.sample(rng))
    return harness.best_feasible_x


def random_search_multi(harness, seed=None, budget=500):
    """Random search for Scenario C: logs both objectives for Pareto front."""
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        harness.evaluate_multi(harness.sample(rng))
    return None


# ---------------------------------------------------------------------------
# 2. Bayesian Optimisation — Optuna TPE
# ---------------------------------------------------------------------------

def bayes_opt(harness, seed=None, budget=None, n_startup=20):
    """Bayesian optimisation using Optuna's TPE sampler.

    TPE (Tree-structured Parzen Estimator) fits two density models — one over
    good solutions, one over bad ones — and samples from the ratio.  This is
    exactly the surrogate-based BO method covered in the course, implemented
    here via the Optuna library with minimal adaptation: we just wrap the
    harness call as an Optuna objective function.

    Adaptation for this problem
    ---------------------------
    The surrogate is noisy and the landscape is multimodal (five turbines with
    no canonical ordering), so we set ``n_startup_trials`` to spend the first
    ``n_startup`` evaluations on random exploration before switching to TPE.
    The study is configured to minimise (matching the harness convention).

    References
    ----------
    Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter
        Optimization Framework. KDD 2019.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cap = harness.budget_remaining()
    if budget is None:
        budget = cap if cap is not None else 200
    else:
        if cap is not None:
            budget = min(budget, cap)

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(n_startup, budget),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def optuna_objective(trial):
        x = np.array([
            trial.suggest_float(f"x{i}", harness.lower[i], harness.upper[i])
            for i in range(harness.dim)
        ])
        return harness.f1(x)

    study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
    return harness.best_x


def bayes_opt_constrained(harness, seed=None, budget=500, n_startup=30):
    """TPE for Scenario B: infeasible solutions are penalised heavily."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=min(n_startup, budget)
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    BIG = 1e6

    def optuna_objective(trial):
        x = np.array([
            trial.suggest_float(f"x{i}", harness.lower[i], harness.upper[i])
            for i in range(harness.dim)
        ])
        value, feasible = harness.evaluate_constrained(x)
        return value if feasible else BIG + value

    study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
    return harness.best_feasible_x


def bayes_opt_multi(harness, seed=None, budget=500, n_startup=30):
    """TPE for Scenario C: weighted-sum scalarisation over a grid of weights.

    This is the course-level multi-objective method: convert to single-
    objective by scalarising f1 + λ·f2 for a range of λ values so we
    approximate the whole Pareto front, not just one extreme.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.default_rng(seed)
    n_weights = 10
    lambdas = np.linspace(0.0, 1.0, n_weights)
    per_weight = max(1, budget // n_weights)

    for lam in lambdas:
        sampler = optuna.samplers.TPESampler(
            seed=int(rng.integers(1 << 31)),
            n_startup_trials=min(n_startup, per_weight),
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def make_obj(lam=lam):
            def obj(trial):
                x = np.array([
                    trial.suggest_float(f"x{i}", harness.lower[i], harness.upper[i])
                    for i in range(harness.dim)
                ])
                f1, f2 = harness.evaluate_multi(x)
                return (1.0 - lam) * f1 + lam * f2
            return obj

        study.optimize(make_obj(lam), n_trials=per_weight,
                       show_progress_bar=False)
    return None


# ---------------------------------------------------------------------------
# 3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
# ---------------------------------------------------------------------------

def cma_es(harness, seed=None, budget=None, sigma0=0.3, popsize=None):
    """CMA-ES for Scenario A/B (single objective).

    CMA-ES adapts the full covariance matrix of a Gaussian search
    distribution. It is state-of-the-art for continuous, single-objective,
    moderate-dimension problems (dim ≤ a few hundred), making it the natural
    choice beyond-the-course for Scenarios A and B.

    Adaptations for this problem
    ----------------------------
    * Initial sigma set to 0.3 (30 % of the [0,1] box) — wide enough to
      explore a multimodal landscape without immediately converging.
    * Box bounds enforced via the cma library's built-in ``bounds`` argument.
    * Budget cap from the harness is respected: iteration stops as soon as
      ``harness.budget_remaining() == 0``.
    * Restarts: if CMA-ES converges early (``StopIteration``), we restart
      from a fresh random centre so the budget is not wasted.

    References
    ----------
    Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.
    """
    import cma as cmalib

    cap = harness.budget_remaining()
    if budget is None:
        budget = cap if cap is not None else 200
    else:
        if cap is not None:
            budget = min(budget, cap)

    rng = np.random.default_rng(seed)

    opts = {
        "bounds": [harness.lower.tolist(), harness.upper.tolist()],
        "maxfevals": budget,
        "verbose": -9,
        "seed": int(rng.integers(1 << 31)),
    }
    if popsize is not None:
        opts["popsize"] = popsize

    evals_used = 0
    while evals_used < budget:
        x0 = harness.sample(rng)
        opts["maxfevals"] = budget - evals_used
        opts["seed"] = int(rng.integers(1 << 31))
        es = cmalib.CMAEvolutionStrategy(x0, sigma0, opts)
        try:
            while not es.stop():
                xs = es.ask()
                # Check budget per individual so we never overshoot by a whole
                # generation. Pad exhausted slots with a large dummy so
                # es.tell() still receives a full population.
                fitnesses = []
                for x in xs:
                    if evals_used >= budget:
                        fitnesses.append(1e6)
                    else:
                        fitnesses.append(harness.f1(x))
                        evals_used += 1
                es.tell(xs, fitnesses)
                if evals_used >= budget:
                    break
        except cmalib.evolution_strategy.CMAEvolutionStrategyResult:
            pass
        except Exception:
            pass
        if evals_used >= budget:
            break
    return harness.best_x


def cma_es_constrained(harness, seed=None, budget=500, sigma0=0.3):
    """CMA-ES for Scenario B: penalty method for the spacing constraint."""
    import cma as cmalib

    rng = np.random.default_rng(seed)
    BIG = 100.0

    opts = {
        "bounds": [harness.lower.tolist(), harness.upper.tolist()],
        "maxfevals": budget,
        "verbose": -9,
        "seed": int(rng.integers(1 << 31)),
    }

    evals_used = 0
    while evals_used < budget:
        x0 = harness.sample(rng)
        remaining = budget - evals_used
        opts["maxfevals"] = remaining
        opts["seed"] = int(rng.integers(1 << 31))
        es = cmalib.CMAEvolutionStrategy(x0, sigma0, opts)
        try:
            while not es.stop():
                xs = es.ask()
                fitnesses = []
                for x in xs:
                    val, feasible = harness.evaluate_constrained(x)
                    fitnesses.append(val if feasible else BIG + abs(val))
                evals_used += len(xs)
                es.tell(xs, fitnesses)
        except Exception:
            pass
        if evals_used >= budget:
            break
    return harness.best_feasible_x


# ---------------------------------------------------------------------------
# 4. NSGA-II (via pymoo) — multi-objective
# ---------------------------------------------------------------------------

def nsga2(harness, seed=None, budget=500, pop_size=50):
    """NSGA-II for Scenario C (multi-objective optimisation).

    NSGA-II (Deb et al. 2002) is the reference algorithm for multi-objective
    evolutionary optimisation. It uses non-dominated sorting and crowding
    distance to maintain a diverse set of Pareto-optimal solutions.

    Adaptation
    ----------
    * We wrap the two harness objectives in a pymoo Problem so the MO logging
      (needed for the Pareto plot) happens inside the harness as required.
    * Population size and number of generations are derived from the budget:
        n_gen = budget // pop_size
    * pymoo's RandomSampling + SimulatedBinaryCrossover + PolynomialMutation
      are kept at their defaults — these are well-proven for continuous
      problems in [0, 1]^d.

    References
    ----------
    Deb, K. et al. (2002). A fast and elitist multiobjective genetic
        algorithm: NSGA-II. IEEE TEC, 6(2), 182-197.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    n_gen = max(1, budget // pop_size)

    class WindFarmProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=harness.dim,
                n_obj=2,
                n_ieq_constr=0,
                xl=harness.lower,
                xu=harness.upper,
            )

        def _evaluate(self, X, out, *_, **__):
            f1s, f2s = [], []
            for x in X:
                f1, f2 = harness.evaluate_multi(x)
                f1s.append(f1)
                f2s.append(f2)
            out["F"] = np.column_stack([f1s, f2s])

    problem = WindFarmProblem()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / harness.dim, eta=20),
        eliminate_duplicates=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        minimize(
            problem,
            algorithm,
            ("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
    return None


def nsga2_constrained(harness, seed=None, budget=500, pop_size=50):
    """NSGA-II with constraint handling for Scenario D (constrained MO)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    n_gen = max(1, budget // pop_size)

    class WindFarmProblemConstrained(Problem):
        def __init__(self):
            super().__init__(
                n_var=harness.dim,
                n_obj=2,
                n_ieq_constr=1,   # constraint1 >= 1 → g = 1 - c1 <= 0
                xl=harness.lower,
                xu=harness.upper,
            )

        def _evaluate(self, X, out, *_, **__):
            f1s, f2s, gs = [], [], []
            for x in X:
                f1, f2 = harness.evaluate_multi(x)
                c = harness.c1(x)
                f1s.append(f1)
                f2s.append(f2)
                gs.append(1 - c)   # g <= 0 means feasible
            out["F"] = np.column_stack([f1s, f2s])
            out["G"] = np.array(gs).reshape(-1, 1)

    problem = WindFarmProblemConstrained()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / harness.dim, eta=20),
        eliminate_duplicates=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        minimize(
            problem,
            algorithm,
            ("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
    return None
