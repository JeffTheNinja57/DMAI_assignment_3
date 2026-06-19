"""Scenario B 

Scenario B adds a safety constraint (problem.constraint1): every pair of the five
turbines must be at least 252 m (two rotor diameters) apart. We compare three
algorithms, all taken from the shared optimizer library, each of which handles
the constraint with a different classic strategy:

    random_search_constrained : REJECTION: only feasible layouts can become the
                                "best so far", infeasible draws are evaluated but
                                can never win.
    bayes_opt_constrained     : PENALTY: Optuna TPE; an infeasible layout is
                                scored as (1e6 + value), a fine far larger than the
                                real energy scale, so the model learns to avoid it.
    cma_es_constrained        : PENALTY: CMA-ES, an infeasible layout is scored
                                as (100 + |value|), which pulls the sampling
                                distribution back towards the feasible region.

Two rules keep the comparison valid:
  * every algorithm is given the SAME evaluation budget, so no method is
    advantaged by simply being allowed more tries
  * every algorithm returns the best *feasible* layout it saw, so the final
    answer of every run is guaranteed to satisfy the constraint, which is
    exactly what Scenario B requires.

Running this file reproduces the three Scenario B deliverables:
  - a printed summary of best feasible energy + a feasibility check,
  - a paired statistical test (which algorithm is genuinely better),
  - scenario_b_outputs/convergence_B.png  (best feasible energy vs evaluations),
  - scenario_b_outputs/best_layout_B.png  (the best feasible wind-farm layout).
"""

import os

import numpy as np

from experiment import run_experiment
import algorithms as alg
import plotting
import stats_tests
import problem


# Configuration
BUDGET = 500            # objective1 evaluations per run, identical for all algorithms
N_RUNS = 12             # well above the required minimum of 5. More runs are needed
                        # so the statistical test has power: with only 5 paired runs
                        # a two-sided Wilcoxon test cannot drop below p = 0.0625, so
                        # it can never reach significance no matter how large the
                        # real effect is. Scenario B has no compute budget, so extra
                        # runs are essentially free.
DENOISE_REPEATS = 50    # re-evaluations of the best layout for an honest energy value
GLOBAL_SEED = 0         # best-effort reproducibility (the surrogate is still noisy)
OUTDIR = "outputs_B"

# (algorithm function, label used in tables and plots)
ALGORITHMS = [
    (alg.random_search_constrained, "Random search"),
    (alg.bayes_opt_constrained,     "Bayesian opt (TPE)"),
    (alg.cma_es_constrained,         "CMA-ES"),
]


# Experiment
def run_all():
    """Run every algorithm N_RUNS times at the same budget on paired seeds.

    run_experiment uses seeds 0..N_RUNS-1 and reuses the same seeds across the
    three algorithms, so run k of every algorithm shares a seed. That pairing is
    what allows the paired Wilcoxon signed-rank test used below.
    """
    results = []
    for fn, name in ALGORITHMS:
        results.append(
            run_experiment(fn, scenario="B", n_runs=N_RUNS, budget=BUDGET, name=name)
        )
    return results


def feasible_energy(result):
    """Best feasible energy per run, in GWh (energy = -objective1).

    We deliberately use the best *feasible* value rather than the best overall
    value: the penalty methods also evaluate infeasible layouts (which can look
    excellent on energy alone), but only feasible layouts are valid answers here.
    """
    return -result.best_feasible_values()


def find_champion(results):
    """Return (algorithm_name, seed, layout) of the best feasible layout overall."""
    best = None
    for res in results:
        for run in res.runs:
            if run.best_feasible_x is not None and np.isfinite(run.best_feasible_value):
                if best is None or run.best_feasible_value < best[2]:
                    best = (res.algorithm_name, run.seed,
                            run.best_feasible_value, run.best_feasible_x)
    name, seed, _value, x = best
    return name, seed, x


def denoise_energy(x, repeats=DENOISE_REPEATS):
    """Honest expected energy of a fixed layout (GWh): (mean, std) over repeats.

    The optimiser's recorded best is the minimum over many noisy evaluations, so
    it is optimistically biased. Re-evaluating the fixed layout many times and
    averaging removes that bias and also reveals how confident the surrogate is
    (a small spread means its sub-models agree on this layout).
    """
    samples = -np.array([float(problem.objective1(x)) for _ in range(repeats)])
    return samples.mean(), samples.std()



def main():
    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(GLOBAL_SEED)

    results = run_all()

    # summary: best feasible energy and feasibility 
    print(f"Scenario B - {N_RUNS} runs each, budget = {BUDGET} evaluations\n")
    header = (f"{'algorithm':<20}{'energy mean':>12}{'energy std':>12}"
              f"{'obj1 evals':>12}{'all feasible':>14}")
    print(header)
    print("-" * len(header))
    for res in results:
        e = feasible_energy(res)
        print(f"{res.algorithm_name:<20}{e.mean():>12.2f}{e.std():>12.2f}"
              f"{int(res.eval_counts().mean()):>12}{str(res.all_feasible()):>14}")

    # statistical tests: paired Wilcoxon on best feasible energy 
    print("\nPairwise Wilcoxon signed-rank test (best feasible value):")
    print(stats_tests.format_table(
        stats_tests.pairwise_table(results, metric="feasible")))

    # feasibility check: every run of every algorithm must end feasible 
    all_ok = all(res.all_feasible() for res in results)
    print(f"\nFeasibility check - every run ended with a legal layout: {all_ok}")

    # best layout + honest (denoised) energy 
    name, seed, x = find_champion(results)
    mean_e, std_e = denoise_energy(x)
    print(f"\nBest feasible layout: found by {name} (seed {seed})")
    print(f"  energy (denoised over {DENOISE_REPEATS} evals): "
          f"{mean_e:.2f} +/- {std_e:.2f} GWh")
    print(f"  min turbine spacing: {plotting.min_turbine_distance(x):.0f} m "
          f"(rule: >= {2 * problem.ROTOR_DIAMETER} m)")

    # figures 
    plotting.plot_convergence(results, metric="feasible", energy=True,
                              savepath=os.path.join(OUTDIR, "convergence_B.png"))
    plotting.plot_layout(x, energy=mean_e,
                         savepath=os.path.join(OUTDIR, "best_layout_B.png"))
    print(f"\nFigures written to {OUTDIR}/ (convergence_B.png, best_layout_B.png)")

    return results


if __name__ == "__main__":
    main()