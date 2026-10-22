"""The living paper: Model 4's image energy plus a state that survives between frames.

Model 4 settles ONE frame: patch states k_p (local explanations) and hidden pixels
x_h are chosen together by exact block-coordinate descent on

    E4(x, k) = fine patch term (GMM states) + gamma * coarse layout term.

Model 5 keeps that machinery unchanged and adds a retained state, the PAPER:
a mean image m and a per-pixel precision pi (how many observations, discounted,
stand behind each value). Between frames the paper is carried by the motion the
system itself estimates from visible pixels. In frame t the paper enters the SAME
energy as one more quadratic term on the hidden pixels:

    E5(x, k) = E4(x, k) + beta/2 * sum_{i hidden} trust_i * pi-_i * (x_i - m-_i)^2

so given the local states the hidden pixels are still an exact linear solve,
and the local states are still chosen against the image the memory pulls toward:
history can change WHICH local explanation a patch takes, not just the pixels.

Prediction error (Astra's e_t = x_t - x_hat_t) is used for one job: deciding how
far the paper is trusted. On visible pixels the transported memory makes a
prediction, and each prediction error is scored by how EXPECTED it is if the memory
belongs to this scene versus if it does not (a log-likelihood ratio). The verdict is
pooled near the hole (the hole itself can never be checked) and over the whole frame.
A history that agrees with what is visible is believed; a history from somewhere
else is switched off.

Update rule (only observations add precision, so the model cannot learn from its
own fill-ins):
    visible pixel : m = (pi- m- + y) / (pi- + 1),   pi = pi- + 1
    hidden pixel  : m = settled x,                   pi = pi-

No transformer, labels, pretrained network or hidden target pixels are used.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter
from sihti4 import Sihti4, gaussian_fill, SIDE


# ---------------------------------------------------------------- motion
def shift_image(a, s, fill=0.):
    """b(r, c) = a(r + sy, c + sx); pixels coming from outside get `fill`."""
    sy, sx = int(s[0]), int(s[1])
    b = np.full_like(a, fill, dtype=float)
    n = a.shape[0]
    r0, r1 = max(0, -sy), min(n, n - sy)
    c0, c1 = max(0, -sx), min(n, n - sx)
    if r1 > r0 and c1 > c0:
        b[r0:r1, c0:c1] = a[r0 + sy:r1 + sy, c0 + sx:c1 + sx]
    return b


def estimate_shift(prev, prev_mask, cur, cur_mask, max_shift=4, min_overlap=200):
    """Integer global translation s with cur(r, c) ~ prev(r + sy, c + sx),
    found from pixels visible in BOTH frames only (the occluder never votes)."""
    best, best_err = (0, 0), np.inf
    for sy in range(-max_shift, max_shift + 1):
        for sx in range(-max_shift, max_shift + 1):
            p = shift_image(prev, (sy, sx))
            pm = shift_image(prev_mask.astype(float), (sy, sx)) > .5
            v = pm & cur_mask
            if v.sum() < min_overlap:
                continue
            err = float(np.mean((cur[v] - p[v]) ** 2))
            if err < best_err - 1e-12 or (abs(err - best_err) <= 1e-12 and abs(sy) + abs(sx) < abs(best[0]) + abs(best[1])):
                best, best_err = (sy, sx), err
    return best, best_err


# ---------------------------------------------------------------- the paper
@dataclass
class LivingPaper:
    model: Sihti4
    beta: float = 1000.       # weight of one unit of memory precision (1 / sensor variance scale)
    rho: float = 1.0          # precision kept per frame (motion-model uncertainty)
    sigma: float = .02        # assumed sensor noise std, used only by the trust check
    kappa: float = 16.        # evidence discount in the trust check (see trust); inf = no check
    slack: float = 1.         # extra prediction-error variance allowed for a matching history, in sigma^2
    smooth: float = 4.        # px: how far a visible prediction error spreads into the hole
    max_shift: int = 4
    mode: str = "energy"      # "energy" (memory inside E5) or "copy" (memory pasted, rest by Model 4)
    copy_threshold: float = .5
    m: np.ndarray = field(default=None, repr=False)
    pi: np.ndarray = field(default=None, repr=False)
    prev: tuple = field(default=None, repr=False)

    def reset(self):
        self.m = np.zeros((SIDE, SIDE))
        self.pi = np.zeros((SIDE, SIDE))
        self.prev = None
        return self

    # .......................................................... prediction
    def predict(self, obs, mask, shift=None):
        """Transport the paper to the current frame and check it against what is visible.
        Returns the predicted mean m-, raw precision pi-, trust map, and the shift used."""
        if self.m is None:
            self.reset()
        if self.prev is None:
            z = np.zeros((SIDE, SIDE))
            return {"m": z, "pi": z, "trust": np.ones_like(z), "shift": (0, 0), "pred_err": z, "valid": z > 0}
        if shift is None:
            shift, _ = estimate_shift(self.prev[0], self.prev[1], obs, mask, self.max_shift)
        m = shift_image(self.m, shift)
        pi = self.rho * shift_image(self.pi, shift)
        valid = mask & (pi > 0)
        err = np.where(valid, (obs - m) ** 2, 0.)
        if np.isfinite(self.kappa) and valid.any():
            trust = self.trust(obs, mask, m, pi, valid, err)
        else:
            trust = np.ones((SIDE, SIDE))
        return {"m": m, "pi": pi, "trust": trust, "shift": tuple(shift), "pred_err": err, "valid": valid}

    def trust(self, obs, mask, m, pi, valid, err):
        """Probability that the transported memory describes THIS scene, from how
        expected each visible prediction error is under two stories:
            same scene : e ~ N(0, sigma^2 (1 + 1/pi) + slack)    (sensor + memory noise)
            unrelated  : e ~ N(0, 2 var(scene) + sigma^2)        (two independent views)
        The per-pixel log-likelihood ratios are summed near each pixel (Gaussian
        neighbourhood, so the hole borrows its neighbours' verdict) and over the whole
        frame, divided by kappa because neighbouring errors are not independent.
        trust = P(same | local errors) * P(same | all errors)."""
        s2 = self.sigma ** 2
        v_same = s2 * (1. + 1. / np.maximum(pi, 1e-6)) + self.slack * s2
        v_diff = 2. * float(np.var(obs[mask])) + s2
        llr = np.where(valid, .5 * (np.log(v_diff / v_same) + err / v_diff - err / v_same), 0.)
        local = gaussian_filter(llr, self.smooth, mode="constant") * (2 * np.pi * self.smooth ** 2)
        overall = float(llr.sum())
        sig = lambda z: 1. / (1. + np.exp(-np.clip(z, -60, 60)))
        return sig(local / self.kappa) * sig(overall / self.kappa)

    # .......................................................... one frame
    def step(self, obs, mask, shift=None, update=True):
        obs = np.asarray(obs, float)
        mask = np.asarray(mask, bool)
        pred = self.predict(obs, mask, shift)
        weight = pred["pi"] * pred["trust"]          # effective memory precision
        if self.mode == "copy":
            x, settled = self._copy_fill(obs, mask, pred["m"], weight)
        else:
            x, settled = self._energy_fill(obs, mask, pred["m"], weight)
        if update:
            m_new = np.where(mask, (pred["pi"] * pred["m"] + obs) / (pred["pi"] + 1.), x if self.mode == "energy" else pred["m"])
            pi_new = np.where(mask, pred["pi"] + 1., pred["pi"])
            self.m, self.pi = m_new, pi_new
            self.prev = (obs, mask)
        return {"x": x, "pred": pred, "weight": weight, "settled": settled}

    def memory_term(self, prep, m, weight):
        h = prep["hidden"]
        w = self.beta * weight.ravel()[h]
        return np.diag(w), w * m.ravel()[h]

    def _energy_fill(self, obs, mask, m, weight):
        prep = self.model.prepare(obs, mask)
        if weight.ravel()[prep["hidden"]].max() > 0:
            prep["extra"] = self.memory_term(prep, m, weight)
        start = gaussian_fill(obs, mask)
        known = (~mask) & (weight > self.copy_threshold)
        start[known] = m[known]
        settled = self.model.settle(prep, start)
        settled["prep"] = prep
        return settled["x"], settled

    def _copy_fill(self, obs, mask, m, weight):
        """Baseline: paste remembered pixels as if seen, let Model 4 fill only the rest."""
        known = (~mask) & (weight > self.copy_threshold)
        x = obs.copy()
        x[known] = m[known]
        vis = mask | known
        if vis.all():
            return x, None
        prep = self.model.prepare(x, vis)
        settled = self.model.settle(prep, gaussian_fill(x, vis))
        settled["prep"] = prep
        return settled["x"], settled


def run_sequence(paper, frames, shifts=None, final_only_metrics=True):
    """frames: list of (obs, mask). shifts: None (estimate) or list of true shifts
    (shift[t] maps frame t-1 to frame t). Returns the per-frame step results."""
    paper.reset()
    out = []
    for t, (obs, mask) in enumerate(frames):
        s = None if shifts is None or t == 0 else shifts[t]
        out.append(paper.step(obs, mask, shift=s))
    return out


# ---------------------------------------------------------------- response operators
def green_column(settled, pixel):
    """How a unit push at one hidden pixel spreads through the settled medium:
    the column A^-1 e_i of the Hessian (states held fixed). Returns a 32x32 map."""
    prep, A = settled["prep"], settled["A"]
    h = prep["hidden"]
    j = int(np.flatnonzero(h == pixel)[0])
    e = np.zeros(len(h))
    e[j] = 1.
    col = np.linalg.solve(A, e)
    out = np.zeros(SIDE * SIDE)
    out[h] = col
    return out.reshape(SIDE, SIDE)


def memory_influence(paper, settled, weight):
    """d x_h / d m-_h = A^-1 beta W: how much each hidden output pixel is set by
    the remembered value: row sums of the influence matrix, i.e. how far each hidden
    pixel moves when the whole memory moves by one unit (about 1 = set by memory,
    about 0 = set by the present frame; small overshoot above 1 is possible because
    the patch prior is not an M-matrix)."""
    prep, A = settled["prep"], settled["A"]
    h = prep["hidden"]
    w = paper.beta * weight.ravel()[h]
    infl = np.linalg.solve(A, np.diag(w)).sum(1)
    out = np.zeros(SIDE * SIDE)
    out[h] = infl
    return out.reshape(SIDE, SIDE)
