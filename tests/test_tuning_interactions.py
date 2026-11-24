import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.tuning import bayes_optimise, decode_unit
from src.interactions import pairwise_h, make_bins, add_crosses


def test_bayes_optimise_beats_random_search():
    space = {"x": (0.0, 1.0, "float"), "lr": (1e-3, 1.0, "log"), "k": (1, 10, "int")}
    f = lambda p: (-(p["x"] - 0.3) ** 2 - 0.1 * (np.log10(p["lr"]) + 1) ** 2 - 0.01 * abs(p["k"] - 5), 0.0)
    trials, method = bayes_optimise(f, space, n_trials=20, seed=1)
    assert len(trials) == 20 and {"cv_auc", "cv_sd"} <= set(trials.columns)
    rs = np.random.RandomState(1)
    random_best = max(f(decode_unit(rs.rand(3), space))[0] for _ in range(20))
    assert trials.cv_auc.max() >= random_best


def test_decode_unit_respects_bounds_and_types():
    space = {"a": (1e-3, 10.0, "log"), "b": (2, 8, "int")}
    p = decode_unit([0.0, 1.0], space)
    assert abs(p["a"] - 1e-3) < 1e-9 and p["b"] == 8 and isinstance(p["b"], int)


def _planted(n=4000, seed=0):
    r = np.random.RandomState(seed)
    X = pd.DataFrame({"a": r.randint(0, 2, n), "b": r.randint(0, 2, n), "c": r.randint(0, 2, n)})
    z = -1 + 2.0 * (X.a ^ X.b)                       # pure XOR interaction between a and b; c is noise
    y = (r.rand(n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y


def test_h_statistic_finds_planted_interaction():
    X, y = _planted()
    m = HistGradientBoostingClassifier(max_depth=3, random_state=0).fit(X, y)
    h = pairwise_h(m, X.sample(200, random_state=0), ["a", "b", "c"])
    assert {h.feature_a[0], h.feature_b[0]} == {"a", "b"}
    assert h.strength[0] > 3 * h.strength.iloc[-1]


def test_h_statistic_is_near_zero_for_additive_model():
    X, y = _planted()
    stumps = HistGradientBoostingClassifier(max_depth=1, max_leaf_nodes=2, random_state=0).fit(X, y)
    h = pairwise_h(stumps, X.sample(200, random_state=0), ["a", "b", "c"])
    assert h.strength.max() < 0.02


def test_add_crosses_builds_expected_columns():
    X = pd.DataFrame({"age": [20, 30, 40, 50, 60, 70], "emp": ["E", "U", "E", "U", "E", "U"]})
    bins = make_bins(X, ["age"])
    out, names = add_crosses(X, [("age", "emp")], bins)
    assert names == ["age x emp"] and out["age x emp"].str.contains(r" \| ").all()
    assert out["age x emp"].nunique() > 2
