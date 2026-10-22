# Model 5 preregistration — the living paper

Committed **before** any test sequence (photographs `camera`, `coffee`, `clock`) is generated or run.
Everything below was developed and tuned on validation strips only (bottom quarter of the
training photographs, which Model 4's priors never saw).

## Question

Model 3 and Model 4 explain one still image: local patch states negotiate with a coarse layout
until the hidden pixels settle. Model 5 asks whether an explanation that **survives between
observations** helps, and how:

> Can two different histories produce different, appropriate interpretations of the same,
> currently ambiguous view — and does putting the history *inside* the image energy do better
> than simply pasting remembered pixels?

## The model (paper.py)

Model 4's engine is vendored unchanged and frozen with Model 4's own natural-image parameters
(`params_m4_natural.json`, tuned and frozen in Model 4). Model 5 adds a retained state, the
**paper**: a mean image `m` and per-pixel precision `pi`.

* motion: integer global translation estimated each frame from pixels visible in both frames;
* transport: `m- = shift(m)`, `pi- = rho * shift(pi)`;
* trust: each visible prediction error `y - m-` is scored by a log-likelihood ratio, *same scene*
  `N(0, sigma^2(1+1/pi) + sigma^2)` vs *unrelated* `N(0, 2 var(scene) + sigma^2)`, pooled over a
  Gaussian neighbourhood and over the frame, divided by `kappa`; trust = P(same|local) * P(same|frame);
* inference: Model 4's energy **plus** `beta/2 sum_hidden trust*pi-*(x - m-)^2`, settled by Model 4's
  exact block-coordinate descent (states and hidden pixels together);
* update: visible pixels fuse `(pi- m- + y)/(pi- + 1)` with `pi = pi- + 1`; hidden pixels take the
  settled value and keep `pi-`. Only observations add precision.

`sigma = 0.02` is the true sensor noise of the generator and is given to the model (stated assumption).

## Data (sequences.py)

Camera pans at integer velocity `vx in {-3,-2,2,3}`, `vy in {-1,0,1}` px/frame over a photograph
(rescaled 0.25/0.35/0.5), 32x32 frames, sensor noise sigma 0.02, a **fixed** 10x10 occluder at a random
position in rows/cols 9..22.

* **long**: T = 8 frames; every pixel behind the occluder at the last frame was visible earlier.
* **short**: T = 3; only about half of it was (40–68% on the test draw checked during development).
* **twins** (long only): the same strip with only the world region behind the last frame's occluder
  replaced by a 10x10 patch of another test photograph. The last frame's observation is
  bit-identical for both twins; their hidden truths differ; earlier frames showed the difference.
* **foreign history**: the history of another sequence of the same set (n/2 away), seen through this
  sequence's occluder, followed by this sequence's own last frame.

Test: 48 long (seed 101, with twins) + 48 short (seed 202) from `camera`, `coffee`, `clock`.
Validation (tuning only): 24 + 24 from the 15 training photographs' bottom quarter (seed 11).

## Arms (bench5.py), all scored by MSE on the hidden pixels of the last frame

| name | what it sees |
|---|---|
| `alone` | Model 4 settle on the last frame only |
| `wiener` | full-image Gaussian conditional mean, last frame only |
| `copy` | same paper, motion and trust; remembered hidden pixels (weight > threshold) pasted as if seen, Model 4 fills the rest. **The strong temporal baseline.** |
| `paper` | Model 5 |
| `paper_nocheck`, `copy_nocheck` | trust switched off |
| `paper_true_motion` | true shifts instead of estimated ones (motion ceiling) |
| `foreign_*` | the arms above fed a foreign history |
| `twin_*` | the same arms on the twin |

Patch-state agreement: for hole-touching patches, the fraction whose settled local state equals the
state the same patch takes on the clean, fully known last frame.

## Tuning (tune.py, validation only)

1. `beta in {1e5,3e5,1e6,3e6,1e7} x rho in {1, 0.8}` minimising the geometric mean of long and short MSE (kappa 16).
2. `kappa in {1,4,16,64}`: lowest foreign-history MSE among values whose own-history MSE is <= 1.02x the no-check MSE.
3. copy baseline separately: `rho in {1,0.8} x threshold in {0.1,0.5,0.9}`, own-history MSE.

## Gates (decided by run5.py from results/natural/metrics.json)

| gate | claim | pass if |
|---|---|---|
| **H1** | the same last view, two histories, two right answers | mean `paper` MSE over both twins <= 0.5 x mean `alone` MSE over both twins, **and** each paper answer is closer to its own truth than to its twin's in >= 90% of cases |
| **H1b** | history changes *which* local explanation a patch takes, towards the right one | long: patch-state agreement `paper` − `alone` >= 0.10 |
| **H2** | a history from elsewhere is checked against the present and switched off | pooled long+short: `foreign_paper` <= 1.10 x `alone` **and** `foreign_paper` <= 0.80 x `foreign_paper_nocheck` |
| **H2b** | the check costs nothing when the history is right | pooled: `paper` <= 1.02 x `paper_nocheck` |
| **H3** | memory inside the energy beats pasting memory | `paper` <= 0.97 x `copy` in long **and** in short (bootstrap CIs and per-sequence win rates reported) |

## Expectations, said in advance

H1, H1b and H2b are **instrument checks**: memory of a seen region should obviously beat guessing it.
They verify the machine does what it says; they are not discoveries. H2 is a check that the trust
rule works out of sample (its first exponential form failed on validation and was replaced by the
likelihood-ratio form before this file was written). **H3 is the real question**: whether memory as a
soft prior inside the image energy (it denoises with the patch prior and pulls the never-seen part of
the hole through overlapping patches) beats the ordinary motion-compensated paste.

## Not claimed

No claim about cortex, qualia, or object permanence beyond this setting: global integer translation,
a static world, a fixed occluder, known sensor noise. Motion estimation, Kalman-style fusion and
flow-guided video inpainting are prior art; the test is of this particular combination with
Model 4's energy.
