"""Moving views with a fixed occluder.

A camera pans over a photograph at a constant integer velocity (vy, vx) px/frame.
A fixed 10x10 occluder sits in the frame (a pole, dirt on the lens): it hides the
same frame pixels every frame, so what is behind it at the last frame was in
view, elsewhere in the frame, earlier. Every frame gets independent sensor noise.

frame_t(r, c) = W(r + oy(t), c + ox(t)),   so   frame_t(r, c) = frame_{t-1}(r + vy, c + vx)
(the shift that maps the paper from t-1 to t is s = (vy, vx)).

TWINS. For a long sequence we also build its twin: the same strip W with ONLY the
world region that sits behind the occluder at the last frame replaced by content
from a different photograph. The last frame's visible pixels (and their noise) are
identical in both twins; the hidden truths differ; earlier frames showed the
difference. Any method that sees only the last frame must give both twins the same
answer, so it cannot be right for both.

Splits follow Model 4: validation strips come from the training photographs'
bottom quarter (rows the Model 4 priors never saw), test strips from three
photographs never used for training or tuning.
"""
import numpy as np
from data import _photo, TRAIN_PHOTOS, TEST_PHOTOS

SIDE = 32
HOLE = 10


def offsets(T, v):
    """Window offsets o(t) = t*v shifted so the smallest is 0."""
    t = np.arange(T)
    o = t * v
    return o - o.min()


def strip_size(T, vy, vx):
    return SIDE + abs(vy) * (T - 1), SIDE + abs(vx) * (T - 1)


def cut_strip(img, T, vy, vx, rng, rows=None, min_std=.06, tries=4000):
    from skimage.transform import rescale
    H, W = strip_size(T, vy, vx)
    for _ in range(tries):
        s = float(rng.choice([.25, .35, .5]))
        small = rescale(img, s, anti_aliasing=True)
        h, w = small.shape
        lo, hi = (0, h) if rows is None else (int(rows[0] * h), int(rows[1] * h))
        if hi - lo < H or w < W:
            continue
        y = int(rng.integers(lo, hi - H + 1))
        x = int(rng.integers(0, w - W + 1))
        strip = small[y:y + H, x:x + W]
        if strip.std() >= min_std:
            return strip.astype(np.float64)
    return None


def render(strip, T, vy, vx):
    oy, ox = offsets(T, vy), offsets(T, vx)
    return np.array([strip[oy[t]:oy[t] + SIDE, ox[t]:ox[t] + SIDE] for t in range(T)])


def occluder(rng):
    r0, c0 = (int(v) for v in rng.integers(9, 14, size=2))
    mask = np.ones((SIDE, SIDE), bool)
    mask[r0:r0 + HOLE, c0:c0 + HOLE] = False
    return mask, (r0, c0)


def observe(clean, mask, sigma, noise_seed):
    rng = np.random.default_rng(noise_seed)
    obs = clean + sigma * rng.standard_normal(clean.shape)
    return np.where(mask[None], obs, .5)


def hidden_world_box(T, vy, vx, hole_rc):
    """Strip coordinates of what sits behind the occluder at the LAST frame."""
    oy, ox = offsets(T, vy), offsets(T, vx)
    r0, c0 = hole_rc
    return oy[-1] + r0, ox[-1] + c0


def make_sequence(strip, T, vy, vx, rng, sigma, noise_seed, mask=None, hole_rc=None):
    if mask is None:
        mask, hole_rc = occluder(rng)
    clean = render(strip, T, vy, vx)
    obs = observe(clean, mask, sigma, noise_seed)
    shifts = [(0, 0)] + [(vy, vx)] * (T - 1)
    return {"clean": clean, "obs": obs, "mask": mask, "hole": hole_rc, "v": (vy, vx), "T": T,
            "shifts": shifts, "strip": strip, "noise_seed": noise_seed}


def velocities(rng):
    vx = int(rng.choice([-3, -2, 2, 3]))
    vy = int(rng.choice([-1, 0, 1]))
    return vy, vx


def load_folder(folder, max_images=None):
    """Your own photographs as full-resolution grayscale arrays in [0, 1]."""
    from pathlib import Path
    from PIL import Image
    root = Path(folder).expanduser().resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No images found under {root}")
    out = {}
    for p in paths:
        with Image.open(p) as im:
            g = im.convert("L")
            g.thumbnail((512, 512))
            out[p.name] = np.asarray(g, dtype=np.float64) / 255.
    return out


def build_set(split, n, regime, sigma=.02, seed=0, twins=False, images=None):
    """split: "val" (training photos, bottom quarter) or "test" (unseen photos).
    regime: "long" (T=8: everything behind the occluder was seen) or
            "short" (T=3: only part of it was)."""
    rng = np.random.default_rng(seed)
    if images is not None:                      # your own folder (split ignored)
        imgs, rows = dict(images), None
        photos = list(imgs)
    else:
        photos = TRAIN_PHOTOS if split == "val" else TEST_PHOTOS
        rows = (.75, 1.) if split == "val" else None
        imgs = {p: _photo(p) for p in photos}
    T = 8 if regime == "long" else 3
    out = []
    i = 0
    while len(out) < n:
        name = photos[i % len(photos)]
        i += 1
        vy, vx = velocities(rng)
        strip = cut_strip(imgs[name], T, vy, vx, rng, rows)
        if strip is None:
            continue
        noise_seed = seed * 100000 + len(out)
        seq = make_sequence(strip, T, vy, vx, rng, sigma, noise_seed)
        seq["photo"] = name
        if twins:
            other = photos[(i + len(photos) // 2) % len(photos)]
            if other == name:
                other = photos[(i + 1) % len(photos)]
            d = cut_strip(imgs[other], 1, 0, 0, rng, rows, min_std=.06)
            if d is None:
                continue
            oy, ox = (int(v) for v in rng.integers(0, SIDE - HOLE + 1, size=2))
            donor = d[oy:oy + HOLE, ox:ox + HOLE]
            twin = make_sequence(_replace(strip, seq, donor), T, vy, vx, rng, sigma, noise_seed,
                                 mask=seq["mask"], hole_rc=seq["hole"])
            twin["photo"] = name + "+" + other
            if np.mean((twin["clean"][-1] - seq["clean"][-1]) ** 2) < 1e-4:
                continue                                   # donor too similar: not a real twin
            seq["twin"] = twin
        out.append(seq)
    return out


def _replace(strip, seq, donor):
    y, x = hidden_world_box(seq["T"], seq["v"][0], seq["v"][1], seq["hole"])
    s = strip.copy()
    s[y:y + HOLE, x:x + HOLE] = donor
    return s


def frames(seq):
    return [(seq["obs"][t], seq["mask"]) for t in range(seq["T"])]


def foreign_history(seq, other, sigma=.02):
    """Another scene's history, seen through THIS occluder, followed by this
    sequence's own last frame. The history is real video, just not of this scene."""
    T = min(seq["T"], other["T"])
    clean = other["clean"][other["T"] - T:other["T"] - 1]
    obs = observe(clean, seq["mask"], sigma, other["noise_seed"] + 7)
    return [(o, seq["mask"]) for o in obs] + [(seq["obs"][-1], seq["mask"])]
