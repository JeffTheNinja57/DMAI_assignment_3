import os 
import csv
import numpy as np
import matplotlib.pyplot as plt

import problem
import algorithms
import plotting
from experiment import run_experiment

OUTDIR = 'outputs_C'
N_RUNS = 5
BUDGET = 600
ROBUST_REPEATS = 15
ENERGY_FLOOR_FRAC = 0.5   # drop front layouts below this fraction of the peak energy 

# the 3 different algorithms being compared
# random search (baseline), nsga-ii (taught in lecture 4), and sms-emoa
# (beyond-course)
ALGORITHMS = [(algorithms.random_search_multi, 'Random search'),
              (algorithms.nsga2, 'NSGA-II'),
              (algorithms.smsemoa, 'SMS-EMOA'),]


def pooled_solutions(result):
    '''stack every (x, f1, f2) logged across a results run
    
        args:
            result: an ExperimentResult holding several runs, each with mo_log
        
        returns:
            a tuple (xs, F) where xs in an (n, dim) array of layouts and F is an (n, 2)
            array of (objective1, objective2 ) pairs
        '''
    xs, F = [], []
    for run in result.runs:
        for x, f1, f2 in run.mo_log:
            xs.append(x)
            F.append((f1, f2))

    return np.array(xs), np.array(F, dtype= float)


def robust_front(xs, repeats = ROBUST_REPEATS):
    '''as both objectives are stochastic, a point can look non- dom by luck 
        average a few samples per layout 
        
        args: 
            xs: (n, dim) array of candidate layouts to re-score
            repeats: number of samples averaged per objective layout
            
        returns:
            a tuple (front_xs, front_F) of the non-dominated layouts and their 
            averaged (objective1, objective2) values, sorted by biodiversity'''
    
    F = np.array([
        [np.mean([problem.objective1(x) for _ in range(repeats)]),
         np.mean([problem.objective2(x) for _ in range(repeats)])]
        for x in xs
    ])
    mask = plotting.pareto_front(F)
    order = np.argsort(F[mask, 1])          # sort by biodiversity for a tidy curve
    return xs[mask][order], F[mask][order]

def tradeoff_summary(F, pcts=(0, 5, 10, 15, 20, 25)):
    """how much wider the corridor gets per % of peak energy given up.

    Args:
        F: (n, 2) array of (objective1, objective2) values on the front.
        pcts: percentages of the peak energy we are willing to sacrifice.

    Returns:
        A tuple (e_best, b_best, rows): the peak energy in gwh, its biodiversity,
        and a list of dicts with keys energy_pct, energy_gwh, biodiversity and
        corridor_gain.
    """
    energy = -F[:, 0]
    biodiv = F[:, 1]
    i_best = int(np.argmax(energy))
    e_best, b_best = energy[i_best], biodiv[i_best]

    rows = []
    for pct in pcts:
        # layouts whose energy is within this % of the peak
        reachable = energy >= e_best * (1 - pct / 100)
        # widest corridor (smallest biodiversity) reachable within that budget
        j = int(np.argmin(np.where(reachable, biodiv, np.inf)))
        rows.append({
            "energy_pct": pct,
            "energy_gwh": energy[j],
            "biodiversity": biodiv[j],
            "corridor_gain": b_best - biodiv[j],   # drop in objective2 vs the peak point
        })
    return e_best, b_best, rows


def main(outdir = OUTDIR):
    '''run the algorithms, save the figs, print analysis
    
    Args: 
        outdir: directory the figures and csv are saved to 
        
    Returns:
        A list of ExperimentResult, one per algorithm, for reuse in the notebook'''
    
    np.random.seed(0)
    os.makedirs(outdir, exist_ok=True)
    results = []

    # run every algorithm n_runes times on the same paired seeds
    for fn, label in ALGORITHMS:
        res = run_experiment(fn, scenario="C", n_runs=N_RUNS, budget=BUDGET,
                            name=label)
        results.append(res)

    # print the results
    print(f"{'algorithm':<18}{'obj1 evals':<12}{'obj2 evals':<12}{'time (s)':<10}")
    print("-" * 52)
    for res in results:
        n1 = int(np.mean(res.eval_counts())) #mean objective 1 calls per run 
        n2 = int(np.mean([r.n_obj2 for r in res.runs])) #mean objective 2 calls per run
        t = float(np.mean(res.computation_times())) #mean time
        print(f"{res.algorithm_name:<18}{n1:<12}{n2:<12}{t:<10.2f}")

    plotting.plot_pareto(results,
                        savepath=os.path.join(outdir, "pareto_C_raw.png"))
    
    # pool every layout any algorithm visited
    all_xs = np.vstack([pooled_solutions(r)[0] for r in results])
    # rescore them with averaging and keep non dominated set
    front_xs, front_F = robust_front(all_xs)

    # drop non-viable low-energy layouts. non-dominated only by a sliver
    # of biodiversity but no decision-maker would build a near-zero-energy farm, but
    # they stretch the plot axes and distort the knee, so keep only layouts within
    # ENERGY_FLOOR_FRAC of the peak energy
    peak = (-front_F[:, 0]).max()
    keep = (-front_F[:, 0]) >= ENERGY_FLOOR_FRAC * peak
    front_xs, front_F = front_xs[keep], front_F[keep]

    # plot with same convention as plot_pareto
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(front_F[:, 1], -front_F[:, 0], marker="o", color="C3")
    ax.set_xlabel("Biodiversity metric objective2 (lower = wider corridor)")
    ax.set_ylabel("Energy (GWh)")
    ax.set_title("Combined Pareto front (denoised, all algorithms)")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(outdir, "pareto_C_combined.png"),
                dpi=150, bbox_inches="tight")
    
    # same front to csv 
    with open(os.path.join(outdir, 'pareto_C_front.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["energy_gwh", "biodiversity_objective2"]
                   + [f"x{i}" for i in range(problem.DIM)])   # header + the 10 coords
        for x, (f1, f2) in zip(front_xs, front_F):
            w.writerow([-f1, f2, *x])

    # best layout pictures at the two ends of the trade-off
    energy = -front_F[:, 0] #gwh per front point
    i_energy = int(np.argmax(energy)) # one extreme
    
    # the other point of interest is the knee as best compromise
    norm = (front_F - front_F.min(0)) / np.ptp(front_F, 0)
    i_knee = int(np.argmin(np.linalg.norm(norm, axis=1)))

    # draw both as wind-farm layouts to see the difference
    plotting.plot_layout(front_xs[i_energy], energy=energy[i_energy],
                        savepath=os.path.join(outdir, "best_layout_C_energy.png"),
                        title="Max-energy layout")
    plotting.plot_layout(front_xs[i_knee], energy=energy[i_knee],
                        savepath=os.path.join(outdir, "best_layout_C_knee.png"),
                        title="Knee layout (best trade-off)")

    # trade-off table: how much the corridor widens per % of peak energy given up.
    # written to csv 
    e_best, b_best, rows = tradeoff_summary(front_F)
    with open(os.path.join(outdir, "tradeoff_C.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["energy_pct_given_up", "energy_kept_gwh",
                    "biodiversity_objective2", "corridor_gain"])
        for r in rows:
            w.writerow([r["energy_pct"], f"{r['energy_gwh']:.2f}",
                        f"{r['biodiversity']:.3f}", f"{r['corridor_gain']:.3f}"])

    print(f"\nfigures, pareto_C_front.csv and tradeoff_C.csv written to {outdir}/")
    return results


if __name__ == "__main__":
    main()