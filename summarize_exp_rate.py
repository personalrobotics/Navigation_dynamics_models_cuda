#!/usr/bin/env python3
"""Expansion-weighted rate from run_bag_playback.py pickles.

The playback script prints the mean of each problem's time/expansion.
That mean is dominated by searches that stop after one expansion.
The number used in the comparisons is total expansions / total solve time.

Solved means some best_cost < 1e4 at an index strictly below the budget
(the same rule as playback's "solved within N expansions").

Playback lives in hound_nav/hound_nav/run_bag_playback.py. This file only
reads the pickles it writes. Example:

  python3 summarize_exp_rate.py results_10k_hyst2000_nopreempt --exp 10000
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

GROUPS = (
    ("IGHA*", "IGHAStar_test_*_bag.pkl"),
    ("BiIGHA*", "BiIGHAStar_LCR_1_*_bag.pkl"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", type=Path)
    p.add_argument("--exp", type=int, default=0, help="budget; 0 = max n_expansions")
    args = p.parse_args()
    budget = args.exp
    if budget <= 0:
        seen = []
        for _, pat in GROUPS:
            for f in args.results_dir.glob(pat):
                seen.append(int(pickle.load(f.open("rb"))["n_expansions"]))
        if not seen:
            raise SystemExit(f"no playback pickles in {args.results_dir}")
        budget = max(seen)
    for label, pat in GROUPS:
        files = sorted(args.results_dir.glob(pat))
        if not files:
            continue
        nexp = wall = succ_n = succ_d = 0.0
        solved = 0
        for f in files:
            r = pickle.load(f.open("rb"))
            ne = int(r["n_expansions"])
            nexp += ne
            wall += float(r["solve_time_s"])
            costs = np.asarray(r["best_cost"], dtype=float)
            idx = np.where(costs < 1e4)[0]
            if len(idx) and int(idx[0]) < budget:
                solved += 1
            a = float(r.get("avg_successor_us", float("nan")))
            if a == a and ne:
                succ_n += a * ne
                succ_d += ne
        print(
            f"{label}: {solved}/{len(files)} solved  "
            f"{nexp / wall:.0f} exp/s  {1e6 * wall / nexp:.1f} us/exp  "
            f"mean_exp={nexp / len(files):.0f}  "
            f"avg_succ={succ_n / succ_d:.1f} us  budget={budget}"
        )


if __name__ == "__main__":
    main()
