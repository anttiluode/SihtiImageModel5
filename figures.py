"""Pictures of what the living paper does. Matplotlib, written into results/<name>/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import sequences as S
from bench5 import make_paper, make_copy
from paper import run_sequence, green_column, memory_influence, shift_image

GRAY = dict(cmap="gray", vmin=0, vmax=1, interpolation="nearest")


def _box(ax, mask):
    ys, xs = np.where(~mask)
    ax.add_patch(plt.Rectangle((xs.min() - .5, ys.min() - .5), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1,
                               fill=False, ec="#e8743b", lw=1.2))


def _off(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- 1 comparison
def comparison(rows, path):
    arms = [("alone", "Model 4, last frame only"), ("wiener", "Wiener, last frame only"),
            ("copy", "Paste memory + Model 4"), ("paper", "Living paper (Model 5)"),
            ("paper_true_motion", "Living paper, true motion")]
    arms = [a for a in arms if all(a[0] in r for reg in rows for r in rows[reg])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, reg in zip(axes, ("long", "short")):
        vals = [np.mean([r[k] for r in rows[reg]]) for k, _ in arms]
        y = np.arange(len(arms))
        ax.barh(y, vals, color=[{"alone": "#9aa5b1", "wiener": "#9aa5b1", "copy": "#5b8def", "paper": "#e8743b"}.get(k, "#f2b28f") for k, _ in arms])
        ax.set_yticks(y); ax.set_yticklabels([n for _, n in arms], fontsize=8)
        ax.invert_yaxis(); ax.set_xscale("log")
        ax.set_title(f"{reg} history ({'T=8, all of the hole seen before' if reg == 'long' else 'T=3, part seen'})", fontsize=9)
        ax.set_xlabel("hidden-pixel MSE (log)", fontsize=8)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.5f}", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------- 2 twins
def twins(seqs, maps, path, n=4):
    idx = [i for i in range(len(seqs)) if "twin" in seqs[i]][:n]
    cols = ["last view (same)", "truth A", "truth B", "last frame only\n(same answer)", "paper, history A", "paper, history B"]
    fig, axes = plt.subplots(len(idx), 6, figsize=(9, 1.65 * len(idx)))
    for row, i in enumerate(idx):
        q, mp = seqs[i], maps[i]
        ims = [q["obs"][-1], q["clean"][-1], q["twin"]["clean"][-1], mp["alone"], mp["paper"], mp["twin_paper"]]
        for c, im in enumerate(ims):
            ax = axes[row, c]
            ax.imshow(im, **GRAY); _off(ax); _box(ax, q["mask"])
            if row == 0:
                ax.set_title(cols[c], fontsize=7)
    fig.suptitle("Same last view, two different pasts: a single frame cannot tell them apart, the paper can", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------- 3 the paper over time
def paper_over_time(eng, p, seq, path):
    P = make_paper(eng, p)
    res = run_sequence(P, S.frames(seq))
    T = seq["T"]
    rows = ["observed", "paper prediction m-", "precision pi- x trust", "settled"]
    fig, axes = plt.subplots(4, T, figsize=(1.3 * T, 5.6))
    for t, r in enumerate(res):
        pr = r["pred"]
        ims = [seq["obs"][t], pr["m"], r["weight"], r["x"]]
        for k, im in enumerate(ims):
            ax = axes[k, t]
            if k == 2:
                ax.imshow(im, cmap="magma", vmin=0, vmax=max(T - 1, 1), interpolation="nearest")
            else:
                ax.imshow(im, **GRAY)
            _off(ax); _box(ax, seq["mask"])
            if t == 0:
                ax.set_ylabel(rows[k], fontsize=7)
            if k == 0:
                ax.set_title(f"t={t}  shift {tuple(pr['shift'])}", fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return res


def loop_gif(eng, p, seq, path):
    P = make_paper(eng, p)
    res = run_sequence(P, S.frames(seq))
    frames = []
    for t, r in enumerate(res):
        tiles = [seq["obs"][t], r["pred"]["m"], np.clip(r["weight"] / max(seq["T"] - 1, 1), 0, 1), r["x"], seq["clean"][t]]
        row = np.concatenate([np.pad(np.clip(x, 0, 1), 1, constant_values=1) for x in tiles], 1)
        im = Image.fromarray((row * 255).astype(np.uint8)).resize((row.shape[1] * 6, row.shape[0] * 6), Image.NEAREST)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:] + [frames[-1]] * 3, duration=600, loop=0)


# ---------------------------------------------------------------- 4 response operators
def response(eng, p, seq, other, path):
    obs, mask = seq["obs"][-1], seq["mask"]
    alone = eng.infer(obs, mask, mode="settle")
    alone["base"]["prep"] = alone["prep"]
    P = make_paper(eng, p)
    r = run_sequence(P, S.frames(seq))[-1]
    Pf = make_paper(eng, p)
    rf = run_sequence(Pf, S.foreign_history(seq, other, p["sigma"]))[-1]
    h = alone["prep"]["hidden"]
    centre = int(h[len(h) // 2])
    g0 = green_column(alone["base"], centre)
    g1 = green_column(r["settled"], centre)
    infl = memory_influence(P, r["settled"], r["weight"])
    inflf = memory_influence(Pf, rf["settled"], rf["weight"])
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6))
    for ax, im, t in ((axes[0], g0, "push one hidden pixel:\nresponse, last frame only"),
                      (axes[1], g1, "same push, with the paper")):
        vmax = max(np.abs(im).max(), 1e-12)
        ax.imshow(im, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"); _off(ax); _box(ax, mask)
        ax.set_title(t + f"\n(spread {np.abs(im).sum():.2e})", fontsize=7)
    for ax, im, t in ((axes[2], infl, "how much each hidden pixel\nis set by memory (own past)"),
                      (axes[3], inflf, "same, foreign past\n(checked against the present)")):
        ax.imshow(np.where(mask, np.nan, im), cmap="viridis", vmin=0, vmax=1, interpolation="nearest"); _off(ax); _box(ax, mask)
        ax.set_title(t, fontsize=7)
    axes[4].imshow(r["pred"]["trust"], cmap="viridis", vmin=0, vmax=1, interpolation="nearest"); _off(axes[4]); _box(axes[4], mask)
    axes[4].set_title("trust map, own past\n(prediction error on visible pixels)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"green_spread_alone": float(np.abs(g0).sum()), "green_spread_paper": float(np.abs(g1).sum()),
            "memory_influence_own": float(infl[~mask].mean()), "memory_influence_foreign": float(inflf[~mask].mean())}


# ---------------------------------------------------------------- 5 two kinds of residue
DEPTHS = (0, 1, 2, 4, 8, 16, 32)


def sieve(x, depths=DEPTHS, a=.2):
    """Linear diffusion sieve: bands r_k = x_{d_k} - x_{d_k+1}, plus the core. Sum = x exactly."""
    from scipy.ndimage import laplace
    levels, cur, d = [], x.astype(float), 0
    for target in depths:
        while d < target:
            cur = cur + a * laplace(cur, mode="reflect"); d += 1
        levels.append(cur.copy())
    return [levels[i] - levels[i + 1] for i in range(len(levels) - 1)] + [levels[-1]]


def two_residues(seq, path):
    """Illustration (not a test): what the sieve removes vs what the past failed to predict."""
    vy, vx = seq["v"]
    rng = np.random.default_rng(seq["noise_seed"] + 99)
    noisy = seq["clean"][-2:] + .02 * rng.standard_normal((2, 32, 32))     # no occluder here
    prev, cur = noisy[0], noisy[1]
    pred = shift_image(prev, (vy, vx), fill=np.nan)
    known = ~np.isnan(pred)
    yy, xx = np.mgrid[:32, :32]
    shadow = -.25 * np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / (2 * 7. ** 2))   # broad, slow, surprising
    cases = {"predictable motion": cur, "same + a broad shadow appears": cur + shadow}
    names = [f"r{DEPTHS[i]}-{DEPTHS[i+1]}" for i in range(len(DEPTHS) - 1)] + ["core"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), gridspec_kw={"width_ratios": [1, 1, 2.2]})
    axes[0].imshow(cases["predictable motion"], **GRAY); _off(axes[0]); axes[0].set_title("frame t", fontsize=8)
    axes[1].imshow(cases["same + a broad shadow appears"], **GRAY); _off(axes[1]); axes[1].set_title("frame t + broad shadow", fontsize=8)
    w = .2
    x = np.arange(len(names))
    out = {}
    for j, (label, frame) in enumerate(cases.items()):
        r = sieve(frame)
        e = sieve(np.where(known, frame - np.nan_to_num(pred), 0.))
        re = [float(np.mean(b[known] ** 2)) for b in r]
        ee = [float(np.mean(b[known] ** 2)) for b in e]
        out[label] = {"sihti_residue": dict(zip(names, re)), "prediction_error": dict(zip(names, ee))}
        axes[2].bar(x + (2 * j - 1.5) * w, re, w, color=["#9aa5b1", "#5b8def"][j], label=f"sieve residue, {label}")
        axes[2].bar(x + (2 * j - .5) * w, np.maximum(ee, 1e-9), w, color=["#e8743b", "#c0392b"][j], label=f"prediction error, {label}")
    axes[2].set_yscale("log"); axes[2].set_xticks(x); axes[2].set_xticklabels(names, fontsize=7)
    axes[2].set_ylabel("energy per pixel", fontsize=8); axes[2].legend(fontsize=6)
    axes[2].set_title("what the sieve removes is not what the past failed to predict", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return out


def all_figures(eng, p, sets, maps, rows, d):
    import json
    comparison(rows, d / "comparison.png")
    twins(sets["long"], maps["long"], d / "twins.png")
    q = sets["long"][0]
    paper_over_time(eng, p, q, d / "paper_over_time.png")
    loop_gif(eng, p, q, d / "loop.gif")
    qs = sets["short"][0]
    extra = {"response": response(eng, p, qs, sets["short"][len(sets["short"]) // 2], d / "response.png"),
             "two_residues": two_residues(q, d / "two_residues.png")}
    (d / "figure_numbers.json").write_text(json.dumps(extra, indent=2))
