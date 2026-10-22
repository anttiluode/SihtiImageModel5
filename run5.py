"""Run the preregistered test (docs/PREREG.md) and write results/<name>/metrics.json + figures.

    python run5.py                       # the preregistered run on the held-out photographs
    python run5.py --folders ~/THINGS/unseen --name folders --n 24
                                         # pans over your own photographs (not a gate run)
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import models
import sequences as S
from bench5 import run, summary
from baselines import Baselines
from data import natural


def bootstrap_ratio(a, b, n=4000, seed=0):
    """95% CI of mean(a)/mean(b) over sequences (paired resampling)."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def gates(long_rows, short_rows):
    L, Sh = summary(long_rows), summary(short_rows)
    pooled = summary(long_rows + short_rows)
    g = {}
    # H1: the same last view, two histories, two right answers
    paper_twins = np.mean([r["paper"] for r in long_rows] + [r["twin_paper"] for r in long_rows])
    alone_twins = np.mean([r["alone"] for r in long_rows] + [r["twin_alone"] for r in long_rows])
    own = np.mean([r["own_closer_A"] for r in long_rows] + [r["own_closer_B"] for r in long_rows])
    g["H1_error_ratio"] = paper_twins / alone_twins
    g["H1_own_closer_rate"] = float(own)
    g["H1_frame_alone_identical_rate"] = float(np.mean([r["twin_alone_identical"] for r in long_rows]))
    g["H1_PASS"] = bool(g["H1_error_ratio"] <= .5 and own >= .9)
    # H1b: history changes WHICH local explanation a patch takes, towards the right one
    g["H1b_states_alone"] = L["states_alone"]
    g["H1b_states_paper"] = L["states_paper"]
    g["H1b_PASS"] = bool(L["states_paper"] - L["states_alone"] >= .10)
    # H2: a history from another scene is checked against the present and switched off
    g["H2_foreign_vs_alone"] = pooled["foreign_paper"] / pooled["alone"]
    g["H2_check_vs_nocheck_under_foreign"] = pooled["foreign_paper"] / pooled["foreign_paper_nocheck"]
    g["H2_PASS"] = bool(g["H2_foreign_vs_alone"] <= 1.10 and g["H2_check_vs_nocheck_under_foreign"] <= .80)
    # H2b: the check costs (almost) nothing when the history is right
    g["H2b_check_cost"] = pooled["paper"] / pooled["paper_nocheck"]
    g["H2b_PASS"] = bool(g["H2b_check_cost"] <= 1.02)
    # H3: memory inside the image energy beats pasting memory and filling the rest
    g["H3_long_ratio"] = L["paper"] / L["copy"]
    g["H3_short_ratio"] = Sh["paper"] / Sh["copy"]
    g["H3_long_CI"] = bootstrap_ratio([r["paper"] for r in long_rows], [r["copy"] for r in long_rows])
    g["H3_short_CI"] = bootstrap_ratio([r["paper"] for r in short_rows], [r["copy"] for r in short_rows])
    g["H3_long_wins"] = float(np.mean([r["paper"] < r["copy"] for r in long_rows]))
    g["H3_short_wins"] = float(np.mean([r["paper"] < r["copy"] for r in short_rows]))
    g["H3_PASS"] = bool(g["H3_long_ratio"] <= .97 and g["H3_short_ratio"] <= .97)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--name", default="natural")
    ap.add_argument("--folders", default=None, help="folder of your own photographs to pan over")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    p = json.loads(Path("params.json").read_text())["params"]
    eng = models.engine()
    train, _, _ = natural(n_train=1500, n_val=64, n_test=64, seed=0)
    base = Baselines(train, wiener_shrink=models.M4_PARAMS["wiener_shrink"])
    images = S.load_folder(a.folders) if a.folders else None
    sets = {"long": S.build_set("test", a.n, "long", sigma=p["sigma"], seed=101, twins=True, images=images),
            "short": S.build_set("test", a.n, "short", sigma=p["sigma"], seed=202, images=images)}
    rows, maps = {}, {}
    for reg, seqs in sets.items():
        print(f"{reg}: {len(seqs)} sequences", flush=True)
        rows[reg], maps[reg] = run(eng, seqs, p, baselines=base)
    out = {"params": p, "n": a.n, "summary": {r: summary(rows[r]) for r in rows},
           "gates": gates(rows["long"], rows["short"]), "rows": rows,
           "minutes": round((time.time() - t0) / 60, 1)}
    d = Path("results") / a.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(out["summary"], indent=2))
    print(json.dumps(out["gates"], indent=2))
    if not a.no_figures:
        import figures
        figures.all_figures(eng, p, sets, maps, rows, d)


if __name__ == "__main__":
    main()
