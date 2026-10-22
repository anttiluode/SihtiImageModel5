"""Choose beta, rho, kappa (paper) and rho, threshold (copy baseline) on VALIDATION only.

Validation strips come from the bottom quarter of the training photographs.
Writes params.json. No test photograph is touched.
"""
import json
import time
import numpy as np
import models
import sequences as S
from bench5 import make_paper, make_copy, final, mse_hidden

BASE = dict(beta=1e6, rho=1.0, kappa=16., sigma=.02, smooth=4., copy_rho=1.0, copy_threshold=.5)


def mean_mse(eng, seqs, build, history="own"):
    errs = []
    n = len(seqs)
    for i, q in enumerate(seqs):
        fr = S.frames(q) if history == "own" else S.foreign_history(q, seqs[(i + n // 2) % n], BASE["sigma"])
        x = final(build(), fr)["x"]
        errs.append(mse_hidden(x, q["clean"][-1], q["mask"]))
    return float(np.mean(errs))


def both(eng, val, build, history="own"):
    a = mean_mse(eng, val["long"], build, history)
    b = mean_mse(eng, val["short"], build, history)
    return a, b, float(np.sqrt(a * b))


def main(n=24):
    t0 = time.time()
    eng = models.engine()
    val = {r: S.build_set("val", n, r, sigma=BASE["sigma"], seed=11) for r in ("long", "short")}
    log = {}
    p = dict(BASE)

    best = None
    for beta in (1e5, 3e5, 1e6, 3e6, 1e7):
        for rho in (1.0, .8):
            q = dict(p, beta=beta, rho=rho)
            a, b, g = both(eng, val, lambda: make_paper(eng, q))
            log[f"paper|beta={beta:g}|rho={rho}"] = [a, b, g]
            print(f"paper beta={beta:g} rho={rho}: long {a:.6f} short {b:.6f}", flush=True)
            if best is None or g < best[0]:
                best = (g, beta, rho)
    p.update(beta=best[1], rho=best[2])

    nocheck = both(eng, val, lambda: make_paper(eng, p, kappa=np.inf))
    log["paper|nocheck"] = list(nocheck)
    best = None
    for kappa in (1., 4., 16., 64.):
        q = dict(p, kappa=kappa)
        own = both(eng, val, lambda: make_paper(eng, q))
        foreign = both(eng, val, lambda: make_paper(eng, q), history="foreign")
        log[f"paper|kappa={kappa:g}"] = {"own": list(own), "foreign": list(foreign)}
        print(f"kappa={kappa:g}: own {own[2]:.6f} (nocheck {nocheck[2]:.6f}) foreign {foreign[2]:.6f}", flush=True)
        if own[2] <= 1.02 * nocheck[2] and (best is None or foreign[2] < best[0]):
            best = (foreign[2], kappa)
    if best is not None:
        p["kappa"] = best[1]

    best = None
    for rho in (1.0, .8):
        for thr in (.1, .5, .9):
            q = dict(p, copy_rho=rho, copy_threshold=thr)
            a, b, g = both(eng, val, lambda: make_copy(eng, q))
            log[f"copy|rho={rho}|thr={thr}"] = [a, b, g]
            print(f"copy rho={rho} thr={thr}: long {a:.6f} short {b:.6f}", flush=True)
            if best is None or g < best[0]:
                best = (g, rho, thr)
    p.update(copy_rho=best[1], copy_threshold=best[2])

    out = {"params": p, "validation_log": log, "n_per_regime": n, "minutes": round((time.time() - t0) / 60, 1),
           "note": "chosen on validation strips (training photographs, bottom quarter) only"}
    with open("params.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(p, indent=2))


if __name__ == "__main__":
    main()
