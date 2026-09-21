import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def pair_key_to_methods(pair_key: str):
    # expects: "MethodA__vs__MethodB"
    a, b = pair_key.split("__vs__")
    return a, b

def build_winprob_matrix(res: dict):
    """
    Returns:
      methods (list[str])
      P (ndarray): P[i,j] ~ probability method i beats method j
      N (ndarray): number of component-comparisons aggregated for (i,j)
    Tie handling: tie contributes 0.5 win to each side.
    """
    methods = list(res["methods"])
    idx = {m: i for i, m in enumerate(methods)}
    n = len(methods)

    # accumulate "points" and counts
    pts = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=float)

    for pair_key, pdata in res["pairs"].items():
        A, B = pair_key_to_methods(pair_key)
        i, j = idx[A], idx[B]
        wins = pdata["wins"]  # {"A": x, "B": y, "tie": z}

        wA = float(wins.get("A", 0))
        wB = float(wins.get("B", 0))
        t  = float(wins.get("tie", 0))
        total = wA + wB + t
        if total <= 0:
            continue

        # A vs B direction
        pts[i, j] += wA + 0.5 * t
        cnt[i, j] += total

        # B vs A direction
        pts[j, i] += wB + 0.5 * t
        cnt[j, i] += total

    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(cnt > 0, pts / cnt, np.nan)

    # set diagonal to 0.5 for display
    for k in range(n):
        P[k, k] = 0.5
        cnt[k, k] = 0.0

    return methods, P, cnt

def mean_score_from_P(P: np.ndarray):
    """
    Score for method i: average P[i,j] over j != i.
    """
    n = P.shape[0]
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        row = np.delete(P[i, :], i)
        scores[i] = np.nanmean(row)
    return scores

def main(paths):
    runs = [load_results(p) for p in paths]

    # Per-run matrices + scores
    methods0 = None
    scores_runs = []
    P_runs = []

    for r in runs:
        methods, P, _ = build_winprob_matrix(r)
        if methods0 is None:
            methods0 = methods
        else:
            if methods != methods0:
                raise ValueError("Methods list differs across runs; ensure same --methods order each time.")
        P_runs.append(P)
        scores_runs.append(mean_score_from_P(P))

    scores_runs = np.vstack(scores_runs)  # (R, M)
    score_mean = scores_runs.mean(axis=0)
    score_std = scores_runs.std(axis=0, ddof=1) if scores_runs.shape[0] > 1 else np.zeros_like(score_mean)

    P_mean = np.nanmean(np.stack(P_runs, axis=0), axis=0)

    # ---- Plot 1: heatmap of average win probability
    plt.figure()
    im = plt.imshow(P_mean, interpolation="nearest", aspect="auto")
    plt.colorbar(im, label=r"Average win prob. $\hat{p}(i \succ j)$")

    plt.xticks(range(len(methods0)), methods0, rotation=60, ha="right")
    plt.yticks(range(len(methods0)), methods0)
    plt.title(r"Pairwise Win Probability Matrix (mean of 3 runs)")
    plt.tight_layout()
    plt.savefig("heatmap.pdf", bbox_inches="tight")

    # ---- Plot 2: method scores (mean ± std over runs)
    plt.figure()
    x = np.arange(len(methods0))
    plt.bar(x, score_mean)
    plt.errorbar(x, score_mean, yerr=score_std, fmt="none", capsize=3)
    plt.xticks(x, methods0, rotation=60, ha="right")
    plt.ylabel(r"Score $\hat{s}_i = \frac{1}{M-1}\sum_{j\neq i}\hat{p}(i \succ j)$")
    plt.title(r"Overall Method Scores (mean $\pm$ std across 3 runs)")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig("scores.pdf", bbox_inches="tight")

    # plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_pairwise.py results_run1.json results_run2.json results_run3.json")
        raise SystemExit(2)
    main(sys.argv[1:])
