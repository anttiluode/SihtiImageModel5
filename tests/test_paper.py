import numpy as np
import pytest
import sequences as S
from data import synthetic
from sihti4 import Sihti4, learn_priors, gaussian_fill
from paper import LivingPaper, run_sequence, shift_image, estimate_shift, green_column, memory_influence


@pytest.fixture(scope="module")
def eng():
    pri = learn_priors(synthetic(60, 0), K=4, patch=10, n_patches=4000)
    return Sihti4.from_priors(pri, gamma=3., noise=.0003, shrink=.05)


@pytest.fixture(scope="module")
def seqs():
    return S.build_set("val", 3, "long", sigma=.02, seed=2, twins=True)


def test_shift_convention_and_estimate(seqs):
    for q in seqs:
        for t in range(1, q["T"]):
            a = shift_image(q["clean"][t - 1], q["v"], fill=np.nan)
            ok = ~np.isnan(a)
            assert np.allclose(a[ok], q["clean"][t][ok])
            s, _ = estimate_shift(q["obs"][t - 1], q["mask"], q["obs"][t], q["mask"])
            assert tuple(s) == tuple(q["v"])


def test_twins_share_the_last_view_but_not_the_truth(eng, seqs):
    for q in seqs:
        tw, m = q["twin"], q["mask"]
        assert np.array_equal(q["obs"][-1], tw["obs"][-1])
        assert np.mean((q["clean"][-1][~m] - tw["clean"][-1][~m]) ** 2) > 1e-4
        a = eng.infer(q["obs"][-1], m, mode="settle")["map"]
        b = eng.infer(tw["obs"][-1], m, mode="settle")["map"]
        assert np.allclose(a, b)          # a frame-alone method cannot tell them apart
        earlier = [np.abs(q["obs"][t] - tw["obs"][t]).max() for t in range(q["T"] - 1)]
        assert max(earlier) > .05         # but the history can


def test_settle_with_memory_never_raises_energy_and_is_exact(eng, seqs):
    q = seqs[0]
    P = LivingPaper(eng, beta=1e5)
    res = run_sequence(P, S.frames(q)[:-1])
    obs, mask = q["obs"][-1], q["mask"]
    pred = P.predict(obs, mask)
    prep = eng.prepare(obs, mask)
    prep["extra"] = P.memory_term(prep, pred["m"], pred["pi"] * pred["trust"])
    out = eng.settle(prep, gaussian_fill(obs, mask), record=True)
    e = np.array(out["energies"] + [out["energy"]])
    assert np.all(np.diff(e) <= 1e-6 * np.abs(e).max())
    R = eng.choose_states(prep, out["x"])
    A, b = eng.system(prep, R)
    xh = np.linalg.solve(A, b)
    f = lambda v: eng.energy(prep, eng.compose(prep, v), R)
    rng = np.random.default_rng(0)
    for _ in range(4):
        d = rng.standard_normal(len(xh)) * 1e-3
        assert f(xh + d) >= f(xh) - 1e-9 and f(xh - d) >= f(xh) - 1e-9


def test_hidden_values_never_reach_inference(eng, seqs):
    q = seqs[1]
    fr = S.frames(q)
    poisoned = [(np.where(m, o, .9), m) for o, m in fr]
    a = run_sequence(LivingPaper(eng, beta=1e5), fr)[-1]["x"]
    b = run_sequence(LivingPaper(eng, beta=1e5), poisoned)[-1]["x"]
    assert np.allclose(a, b)


def test_only_observations_add_precision(eng, seqs):
    q = seqs[0]
    P = LivingPaper(eng, beta=1e5, rho=.8).reset()
    fr = S.frames(q)
    P.step(*fr[0])
    pi0 = P.pi.copy()
    assert np.allclose(pi0[q["mask"]], 1.) and np.allclose(pi0[~q["mask"]], 0.)
    pred = P.predict(*fr[1], shift=q["v"])
    P.step(*fr[1], shift=q["v"])
    m = q["mask"]
    assert np.allclose(P.pi[m], pred["pi"][m] + 1.)
    assert np.allclose(P.pi[~m], pred["pi"][~m])


def test_empty_paper_is_model4(eng, seqs):
    q = seqs[2]
    obs, mask = q["obs"][0], q["mask"]
    a = LivingPaper(eng).reset().step(obs, mask)["x"]
    b = eng.infer(obs, mask, mode="settle")["map"]
    assert np.allclose(a, b)


def test_trust_believes_own_history_and_rejects_foreign(eng, seqs):
    q, other = seqs[0], seqs[1]
    P = LivingPaper(eng, beta=1e5, kappa=16.)
    run_sequence(P, S.frames(q)[:-1])
    own = P.predict(q["obs"][-1], q["mask"])["trust"][~q["mask"]].mean()
    run_sequence(P, S.foreign_history(q, other)[:-1])
    foreign = P.predict(q["obs"][-1], q["mask"])["trust"][~q["mask"]].mean()
    assert own > .5 and foreign < .2


def test_copy_mode_pastes_memory_when_everything_was_seen(eng, seqs):
    q = seqs[0]
    r = run_sequence(LivingPaper(eng, mode="copy"), S.frames(q))[-1]
    m = q["mask"]
    assert np.mean((r["x"][~m] - q["clean"][-1][~m]) ** 2) < 2e-4


def test_response_operators(eng, seqs):
    q = seqs[0]
    P = LivingPaper(eng, beta=1e5)
    r = run_sequence(P, S.frames(q))[-1]
    h = r["settled"]["prep"]["hidden"]
    g = green_column(r["settled"], int(h[len(h) // 2]))
    assert g.ravel()[h[len(h) // 2]] > 0 and np.allclose(g[q["mask"]], 0.)
    infl = memory_influence(P, r["settled"], r["weight"])
    assert infl[~q["mask"]].min() > -1e-6 and infl[~q["mask"]].max() < 1.05
