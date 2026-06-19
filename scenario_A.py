"""Scenario A: Expensive Optimization

Scenario A pretends that evaluating objective1 takes 30 seconds of computation 
time. We have a strict total budget of 5 hours (= 18,000 seconds), so at most
600 evaluations are allowed per run (fewer if the algorithm itself takes 
noticeable wall time).

We compare:
    Random Search (RS) : Evaluation-count baseline; needs no model
    Bayesian Opt (TPE) : Course-taught surrogate method; adapts spending to 
                         promising regions
    CMA-ES             : Beyond the course; adapts the covariance of a Gaussian
                         sampler; state-of-the-art for expensive continuous 
                         problems (Hansen 2016)

Each algorithm is run 5 times with seeds 0-4 (paired across algorithms for the
Wilcoxon test).
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Framework imports
from experiment import run_experiment
import plotting
import stats_tests

# Algorithm library
from algorithms import random_search, bayes_opt, cma_es

N_RUNS = 5
SCENARIO = 'A'
# Scenario A budget: theoretical max is 600 evals (18000s / 30s per eval).
# We use 590 to leave a safe margin for real algorithm overhead (wall time
# + eval_time must both fit inside 5 hours).
BUDGET_A = 590


def main():
    os.makedirs('outputs_A', exist_ok=True)
    
    print('Framework and algorithms loaded OK')
    print(f'Running {N_RUNS} runs x 3 algorithms x up to {BUDGET_A} evals each...')
    print('(This takes ~30 s of real time since the surrogate is actually fast)\n')

    # Run Experiments
    res_rs = run_experiment(
        random_search, scenario=SCENARIO, n_runs=N_RUNS,
        budget=BUDGET_A, name='Random Search'
    )
    print('Random Search done')

    res_bo = run_experiment(
        bayes_opt, scenario=SCENARIO, n_runs=N_RUNS,
        budget=BUDGET_A, n_startup=30, name='Bayesian Opt (TPE)'
    )
    print('Bayesian Opt done')

    res_cma = run_experiment(
        cma_es, scenario=SCENARIO, n_runs=N_RUNS,
        budget=BUDGET_A, sigma0=0.3, name='CMA-ES'
    )
    print('CMA-ES done')

    results_A = [res_rs, res_bo, res_cma]

    # Results Table - Mean energy, eval count, and computation time
    rows = [r.table_row(energy=True) for r in results_A]
    df = pd.DataFrame(rows)
    df.columns = ['Algorithm', 'Energy mean (GWh)', 'Energy std', 'Evals mean', 'Time mean (s)']

    # Add pretend 30-s-per-eval accounting explicitly
    for i, res in enumerate(results_A):
        times = res.computation_times()
        df.loc[i, 'Time mean (s)'] = times.mean()
        df.loc[i, 'Time std (s)'] = times.std()
        df.loc[i, 'Within budget'] = all(r.within_budget for r in res.runs)

    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # Convergence Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plotting.plot_convergence(results_A, ax=ax, energy=True)
    ax.set_title('Scenario A: Convergence (mean ± std, 5 runs each)')
    fig.tight_layout()
    fig.savefig(os.path.join('outputs_A', 'convergence_A.png'), dpi=150)
    print('Saved outputs_A/convergence_A.png')

    # Statistical Tests - Wilcoxon Signed-Rank on best solutions
    rows_stat = stats_tests.pairwise_table(results_A, metric='best', paired=True)
    print(stats_tests.format_table(rows_stat))

    # Detailed per-run best energies
    print('\nPer-run best energies (GWh):')
    for res in results_A:
        vals = -res.best_values()
        print(f'  {res.algorithm_name:25s}: {[round(v,2) for v in vals]}')
        print(f'  {"":25s}  mean={vals.mean():.2f}  std={vals.std():.2f}')

    # Best layout visualization - the single best run across all algorithms
    # Find the single best run across all three algorithms
    best_run = None
    best_alg_name = ''
    for res in results_A:
        for run in res.runs:
            if best_run is None or run.best_value < best_run.best_value:
                best_run = run
                best_alg_name = res.algorithm_name

    energy_gwh = -best_run.best_value
    print(f'\nBest layout found by: {best_alg_name}')
    print(f'Energy: {energy_gwh:.2f} GWh')
    print(f'Layout x vector: {best_run.best_x.round(4)}')

    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    plotting.plot_layout(best_run.best_x, ax=ax2, energy=energy_gwh,
                         title=f'Best layout — {best_alg_name} ({energy_gwh:.2f} GWh)')
    fig2.tight_layout()
    fig2.savefig(os.path.join('outputs_A', 'best_layout_A.png'), dpi=150)
    print('Saved outputs_A/best_layout_A.png')

    # Computation time analysis
    # In scenario A the '30s per eval' post-processing domintes.
    # The key metric is therefore how much energy can each algorithm extract within 
    # 600 evals, not raw wall time. The table above already shows this; this plots it
    fig3, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Left: pretend computation time per algorithm
    names = [r.algorithm_name for r in results_A]
    times_mean = [r.computation_times().mean() / 3600 for r in results_A]  # hours
    times_std  = [r.computation_times().std()  / 3600 for r in results_A]
    axes[0].bar(names, times_mean, yerr=times_std, capsize=5, color=['C0','C1','C2'])
    axes[0].axhline(5.0, color='red', linestyle='--', label='5-hour budget')
    axes[0].set_ylabel('Pretend computation time (hours)')
    axes[0].set_title('Scenario A: Time per run (incl. 30 s/eval)')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=15)

    # Right: number of evaluations per algorithm
    evals_mean = [r.eval_counts().mean() for r in results_A]
    evals_std  = [r.eval_counts().std()  for r in results_A]
    axes[1].bar(names, evals_mean, yerr=evals_std, capsize=5, color=['C0','C1','C2'])
    axes[1].axhline(600, color='red', linestyle='--', label='600-eval cap')
    axes[1].set_ylabel('Number of objective1 evaluations')
    axes[1].set_title('Scenario A: Evaluations per run')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=15)

    fig3.tight_layout()
    fig3.savefig(os.path.join('outputs_A', 'timing_A.png'), dpi=150)
    print('Saved outputs_A/timing_A.png')


if __name__ == "__main__":
    main()
