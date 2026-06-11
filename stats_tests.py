"""Statistical tests for comparing two algorithms over their repeated runs.

The assignment asks for a test that says whether one algorithm is better than
another. Because we run every algorithm on the same seeds, the runs are paired,
so the Wilcoxon signed-rank test is the natural choice. If for some reason the
runs are not paired (different number of runs, different seeds) we fall back to
the Mann-Whitney U test. Both are non-parametric, which is sensible here: with
only ~5 runs we cannot assume the best values are normally distributed.
"""

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

ALPHA = 0.05


def _values(result, metric="best"):
    """Pull the per-run best values out of an ExperimentResult."""
    if metric == "feasible":
        return result.best_feasible_values()
    return result.best_values()


def compare(result_a, result_b, metric="best", paired=True,
            alternative="two-sided"):
    """Compare two algorithms. Lower objective values are better (minimisation).

    Returns a dict with the test used, its statistic and p-value, the median of
    each algorithm and which one looks better.
    """
    a = np.asarray(_values(result_a, metric), dtype=float)
    b = np.asarray(_values(result_b, metric), dtype=float)

    # A non-finite best value means a run never produced a usable solution (for
    # example no feasible layout in Scenario B). Feeding that into the tests
    # would give a meaningless p-value, so fail loudly instead of hiding it.
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError(
            f"non-finite {metric} value(s) in "
            f"{result_a.algorithm_name!r} or {result_b.algorithm_name!r}; "
            "a run likely found no usable (feasible) solution")

    is_paired = paired and len(a) == len(b)
    if is_paired:
        diff = a - b
        if np.allclose(diff, 0.0):
            # Wilcoxon is undefined when every difference is zero.
            test, stat, p = "wilcoxon", 0.0, 1.0
        else:
            test = "wilcoxon"
            stat, p = wilcoxon(a, b, alternative=alternative)
    else:
        test = "mannwhitneyu"
        stat, p = mannwhitneyu(a, b, alternative=alternative)

    median_a, median_b = float(np.median(a)), float(np.median(b))
    # Direction of effect. For paired runs use the median of the per-seed
    # differences, which is what the signed-rank test actually looks at; for
    # unpaired runs fall back to comparing the two medians.
    center = float(np.median(a - b)) if is_paired else median_a - median_b
    if center < 0:
        better = result_a.algorithm_name
    elif center > 0:
        better = result_b.algorithm_name
    else:
        better = "tie"

    return {
        "algorithm_a": result_a.algorithm_name,
        "algorithm_b": result_b.algorithm_name,
        "test": test,
        "statistic": float(stat),
        "p_value": float(p),
        "median_a": median_a,
        "median_b": median_b,
        "better": better,
        "significant": bool(p < ALPHA),
    }


def pairwise_table(results, metric="best", paired=True):
    """Run :func:`compare` on every pair in a list of ExperimentResults."""
    rows = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            rows.append(compare(results[i], results[j], metric=metric,
                                paired=paired))
    return rows


def format_table(rows):
    """Render the comparison rows as a small plain-text table for the report."""
    header = f"{'A vs B':<32}{'test':<14}{'p-value':<10}{'better':<14}{'sig.'}"
    lines = [header, "-" * len(header)]
    for r in rows:
        pair = f"{r['algorithm_a']} vs {r['algorithm_b']}"
        sig = "yes" if r["significant"] else "no"
        lines.append(f"{pair:<32}{r['test']:<14}{r['p_value']:<10.4f}"
                     f"{r['better']:<14}{sig}")
    return "\n".join(lines)
