"""Reusable plots for the report.

Three things are needed across the scenarios:

* convergence plots (best objective so far vs number of evaluations), averaged
  over the runs with a mean +/- std band, separated by algorithm;
* Pareto fronts for the multi-objective scenario;
* wind farm layout pictures of the best found solution.

All objective values can be shown as energy in GWh (``-objective1``) so the axes
read naturally for a decision-maker, which is what the rubric asks for.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import problem


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------
def _align_histories(histories, max_evals=None):
    """Put a list of (evals, best) step-histories onto one evaluation grid.

    Runs can stop at different evaluation counts. We step-interpolate each run
    onto a shared grid 1..max_evals; once a run has finished its best value is
    simply carried forward, which is the right thing for a best-so-far curve.
    """
    histories = [h for h in histories if h]
    if not histories:
        return np.array([]), np.empty((0, 0))
    if max_evals is None:
        max_evals = max(h[-1][0] for h in histories)

    grid = np.arange(1, max_evals + 1)
    matrix = np.full((len(histories), len(grid)), np.nan)
    for i, h in enumerate(histories):
        evals = np.array([e for e, _ in h])
        best = np.array([b for _, b in h])
        idx = np.searchsorted(evals, grid, side="right") - 1
        valid = idx >= 0
        matrix[i, valid] = best[idx[valid]]
    return grid, matrix


def plot_convergence(results, metric="best", energy=True, ax=None,
                     max_evals=None, savepath=None):
    """Convergence plot with one mean line and std band per algorithm.

    ``results`` is a list of ExperimentResult. ``metric="feasible"`` uses the
    best-feasible-so-far history (Scenario B).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    for result in results:
        histories = [r.feasible_history if metric == "feasible" else r.history
                     for r in result.runs]
        grid, matrix = _align_histories(histories, max_evals=max_evals)
        if grid.size == 0:
            continue
        if energy:                       # show energy in GWh instead of -energy
            matrix = -matrix
        # Before the first feasible point the best-feasible value is +/-inf;
        # treat those as "nothing to show yet" so they do not poison the band.
        matrix = np.where(np.isfinite(matrix), matrix, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-nan columns
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
        shown = np.isfinite(mean)
        line, = ax.plot(grid[shown], mean[shown], label=result.algorithm_name)
        ax.fill_between(grid[shown], (mean - std)[shown], (mean + std)[shown],
                        alpha=0.2, color=line.get_color())

    ax.set_xlabel("Number of objective1 evaluations")
    ax.set_ylabel("Best energy so far (GWh)" if energy
                  else "Best objective1 so far")
    ax.set_title("Convergence (mean +/- std over runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if savepath:
        ax.figure.savefig(savepath, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Pareto fronts
# ---------------------------------------------------------------------------
def pareto_front(F):
    """Return the boolean mask of non-dominated rows of F (minimising every
    column)."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if it is <= in all objectives and < in at least one
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def _collect_objectives(result):
    """Stack all (objective1, objective2) pairs logged across a result's runs."""
    pairs = [(f1, f2) for r in result.runs for (_, f1, f2) in r.mo_log]
    return np.array(pairs, dtype=float) if pairs else np.empty((0, 2))


def plot_pareto(results, ax=None, energy=True, savepath=None):
    """Plot the (non-dominated) Pareto front of each algorithm.

    x-axis: the biodiversity metric objective2 (smaller = wider bird corridor).
    y-axis: energy in GWh (``-objective1``), so up-and-to-the-left is best.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    for result in results:
        F = _collect_objectives(result)
        if len(F) == 0:
            continue
        mask = pareto_front(F)
        front = F[mask]
        front = front[np.argsort(front[:, 1])]      # sort by biodiversity
        biodiversity = front[:, 1]
        energy_axis = -front[:, 0] if energy else front[:, 0]
        ax.plot(biodiversity, energy_axis, marker="o", linestyle="-",
                label=result.algorithm_name, alpha=0.8)

    ax.set_xlabel("Biodiversity metric objective2 (lower = wider corridor)")
    ax.set_ylabel("Energy (GWh)" if energy else "objective1")
    ax.set_title("Pareto fronts: energy vs biodiversity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if savepath:
        ax.figure.savefig(savepath, dpi=150, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# wind farm layout
# ---------------------------------------------------------------------------
def layout_coords(x):
    """Turn a decision vector into (x, y) turbine coordinates in meters.

    Following the assignment text, [x1, x2] is the first turbine and [x9, x10]
    the last, i.e. the vector reshapes to (5, 2) just like in constraint1.
    """
    coords = np.asarray(x, dtype=float).reshape(problem.N_TURBINES, 2)
    return coords * problem.FARM_LENGTH


def min_turbine_distance(x):
    """Smallest distance between any two turbines, in meters."""
    coords = layout_coords(x)
    dmin = np.inf
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dmin = min(dmin, np.linalg.norm(coords[i] - coords[j]))
    return dmin


def plot_layout(x, ax=None, energy=None, savepath=None, title=None):
    """Draw a single wind farm layout.

    Each turbine is drawn with a circle of one rotor diameter radius; the spacing
    rule (centres at least two rotor diameters apart) is violated exactly when two
    such circles overlap, so the picture makes feasibility easy to read.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))

    coords = layout_coords(x)
    side = problem.FARM_LENGTH
    radius = problem.ROTOR_DIAMETER

    # farm boundary
    ax.add_patch(plt.Rectangle((0, 0), side, side, fill=False,
                               edgecolor="gray", linestyle="--"))
    for k, (cx, cy) in enumerate(coords):
        ax.add_patch(Circle((cx, cy), radius, alpha=0.25, color="C0"))
        ax.plot(cx, cy, "o", color="C0")
        ax.annotate(f"T{k + 1}", (cx, cy), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)

    dmin = min_turbine_distance(x)
    feasible = dmin >= 2 * radius
    ax.set_xlim(-radius, side + radius)
    ax.set_ylim(-radius, side + radius)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    if title is None:
        bits = []
        if energy is not None:
            bits.append(f"{energy:.2f} GWh")
        bits.append(f"min spacing {dmin:.0f} m")
        bits.append("feasible" if feasible else "INFEASIBLE")
        title = "Wind farm layout (" + ", ".join(bits) + ")"
    ax.set_title(title)
    if savepath:
        ax.figure.savefig(savepath, dpi=150, bbox_inches="tight")
    return ax
