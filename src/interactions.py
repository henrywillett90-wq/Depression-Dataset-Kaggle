"""Pairwise interaction strength (Friedman H-statistic) and cross features for a linear model."""
import itertools
import numpy as np
import pandas as pd


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def partial_dependence_at_points(model, S, cols):
    """Partial dependence (on the log-odds scale) of `cols`, evaluated at each row of S.

    For row i, set `cols` to row i's values for every row of S, predict, and average.
    """
    n = len(S)
    base = S.reset_index(drop=True)
    T = base.iloc[np.tile(np.arange(n), n)].reset_index(drop=True)
    T[cols] = base[cols].iloc[np.repeat(np.arange(n), n)].values
    return _logit(model.predict_proba(T)[:, 1]).reshape(n, n).mean(axis=1)


def pairwise_h(model, S, features):
    """Return a DataFrame of pairwise interaction strength.

    strength = root-mean-square of the part of the joint effect that is NOT the sum of the two
    separate effects, in log-odds units (an absolute size; 0 = purely additive).
    h2       = that variance as a share of the joint effect's variance (Friedman's H squared).
    """
    single = {f: partial_dependence_at_points(model, S, [f]) for f in features}
    single = {f: v - v.mean() for f, v in single.items()}
    rows = []
    for a, b in itertools.combinations(features, 2):
        jk = partial_dependence_at_points(model, S, [a, b])
        jk = jk - jk.mean()
        resid = jk - single[a] - single[b]
        rows.append({"feature_a": a, "feature_b": b,
                     "strength": float(np.sqrt(np.mean(resid ** 2))),
                     "h2": float(np.sum(resid ** 2) / max(np.sum(jk ** 2), 1e-12))})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def make_bins(X_train, num_cols):
    """Quartile cut-points learned on training data only."""
    return {c: np.unique(np.quantile(X_train[c], [0.25, 0.5, 0.75])) for c in num_cols}


def add_crosses(X, pairs, bins):
    """Add one categorical 'A x B' column per pair (numeric variables are cut into quartiles)."""
    out = X.copy()
    names = []

    def lab(c):
        if c in bins:
            return pd.Series(np.digitize(X[c], bins[c]), index=X.index).astype(str)
        return X[c].astype(str)

    for a, b in pairs:
        nm = f"{a} x {b}"
        out[nm] = lab(a) + " | " + lab(b)
        names.append(nm)
    return out, names
