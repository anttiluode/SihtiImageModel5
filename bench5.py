"""Every arm on the same sequences. Hidden truth is used only for scoring."""
import numpy as np
import sequences as S
from paper import LivingPaper, run_sequence


def mse_hidden(x, truth, mask):
    return float(np.mean((x[~mask] - truth[~mask]) ** 2))


def make_paper(eng, p, **over):
    kw = dict(beta=p["beta"], rho=p["rho"], kappa=p["kappa"], sigma=p["sigma"], smooth=p["smooth"])
    kw.update(over)
    return LivingPaper(eng, **kw)


def make_copy(eng, p, **over):
    kw = dict(mode="copy", rho=p["copy_rho"], kappa=p["kappa"], sigma=p["sigma"], smooth=p["smooth"],
              copy_threshold=p["copy_threshold"])
    kw.update(over)
    return LivingPaper(eng, **kw)


def final(paper, frames, shifts=None):
    res = run_sequence(paper, frames, shifts)
    return res[-1]


def state_agreement(eng, settled_R, obs, mask, clean):
    """Fraction of hole-touching patches whose chosen local state equals the state
    the same patch takes on the clean, fully known image (oracle states)."""
    prep = eng.prepare(obs, mask)
    oracle = eng.choose_states(prep, clean).argmax(1)
    return float(np.mean(settled_R.argmax(1) == oracle))


def evaluate(eng, seq, p, other=None, baselines=None):
    obs, mask = seq["obs"][-1], seq["mask"]
    truth = seq["clean"][-1]
    fr = S.frames(seq)
    out, maps = {}, {}

    alone = eng.infer(obs, mask, mode="settle")
    maps["alone"] = alone["map"]
    out["alone"] = mse_hidden(alone["map"], truth, mask)
    if baselines is not None:
        maps["wiener"] = baselines.wiener(obs, mask)
        out["wiener"] = mse_hidden(maps["wiener"], truth, mask)

    full = run_sequence(make_paper(eng, p), fr)
    r = full[-1]
    maps["paper"] = r["x"]
    out["paper"] = mse_hidden(r["x"], truth, mask)
    out["states_alone"] = state_agreement(eng, alone["base"]["R"], obs, mask, truth)
    out["states_paper"] = state_agreement(eng, r["settled"]["R"], obs, mask, truth)
    out["shift_errors"] = int(sum(tuple(x["pred"]["shift"]) != tuple(s) for x, s in zip(full[1:], seq["shifts"][1:])))

    for name, paper, shifts in (("paper_nocheck", make_paper(eng, p, kappa=np.inf), None),
                                ("paper_true_motion", make_paper(eng, p), seq["shifts"]),
                                ("copy", make_copy(eng, p), None),
                                ("copy_nocheck", make_copy(eng, p, kappa=np.inf), None)):
        x = final(paper, fr, shifts)["x"]
        maps[name] = x
        out[name] = mse_hidden(x, truth, mask)

    if other is not None:                      # a real history, of a different scene
        ff = S.foreign_history(seq, other, sigma=p["sigma"])
        for name, paper in (("foreign_paper", make_paper(eng, p)),
                            ("foreign_paper_nocheck", make_paper(eng, p, kappa=np.inf)),
                            ("foreign_copy", make_copy(eng, p)),
                            ("foreign_copy_nocheck", make_copy(eng, p, kappa=np.inf))):
            x = final(paper, ff)["x"]
            maps[name] = x
            out[name] = mse_hidden(x, truth, mask)

    if "twin" in seq:                          # same last view, different hidden truth
        tw = seq["twin"]
        t_truth = tw["clean"][-1]
        t_alone = eng.infer(tw["obs"][-1], mask, mode="settle")["map"]
        t_paper = final(make_paper(eng, p), S.frames(tw))["x"]
        maps["twin_alone"], maps["twin_paper"] = t_alone, t_paper
        out["twin_alone"] = mse_hidden(t_alone, t_truth, mask)
        out["twin_paper"] = mse_hidden(t_paper, t_truth, mask)
        out["twin_alone_identical"] = bool(np.allclose(t_alone, alone["map"]))
        # does each answer sit closer to its own truth than to its twin's?
        out["own_closer_A"] = mse_hidden(maps["paper"], truth, mask) < mse_hidden(maps["paper"], t_truth, mask)
        out["own_closer_B"] = mse_hidden(t_paper, t_truth, mask) < mse_hidden(t_paper, truth, mask)
        out["twin_gap"] = mse_hidden(truth, t_truth, mask)
    return out, maps


def run(eng, seqs, p, baselines=None, log_every=8):
    rows, all_maps = [], []
    n = len(seqs)
    for i, q in enumerate(seqs):
        other = seqs[(i + n // 2) % n] if n > 1 else None
        m, maps = evaluate(eng, q, p, other=other, baselines=baselines)
        m.update(index=i, photo=q["photo"], v=list(q["v"]), T=q["T"])
        rows.append(m)
        all_maps.append(maps)
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i+1}/{n}", flush=True)
    return rows, all_maps


def summary(rows):
    keys = [k for k, v in rows[0].items() if isinstance(v, float) and all(k in r for r in rows)]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
