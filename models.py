"""Fit (or load) the frozen Model 4 engine that every arm shares."""
import json
import pickle
from pathlib import Path
import numpy as np
from data import natural
from sihti4 import Sihti4, learn_priors

# Model 4's natural-image parameters, tuned on Model 4's validation set and frozen there.
M4_PARAMS = json.loads((Path(__file__).parent / "params_m4_natural.json").read_text())["params"]


def priors(cache="runs/priors_natural.pkl", train=None):
    path = Path(cache)
    if path.exists() and train is None:
        return pickle.loads(path.read_bytes())
    if train is None:
        train, _, _ = natural(n_train=1500, n_val=64, n_test=64, seed=0)
    pri = learn_priors(train, K=M4_PARAMS["K"], patch=M4_PARAMS["patch"], block=M4_PARAMS["block"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(pri))
    return pri


def engine(pri=None):
    pri = pri if pri is not None else priors()
    p = M4_PARAMS
    return Sihti4.from_priors(pri, gamma=p["gamma"], noise=p["noise"], shrink=p["shrink"], temp=p["temp"])
