"""Bayesian hyper-parameter optimisation (no hard dependency).

Uses Optuna's TPE sampler when Optuna is installed. Otherwise falls back to a small
Gaussian-process optimiser with expected improvement, written with scikit-learn only,
so the notebook runs in any environment.
"""
import numpy as np
import pandas as pd


def decode_unit(u, space):
    """Map a point in the unit cube to parameter values. space = {name: (low, high, kind)}, kind in {'float','log','int'}."""
    params = {}
    for ui, (name, (lo, hi, kind)) in zip(u, space.items()):
        if kind == "log":
            params[name] = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))))
        elif kind == "int":
            params[name] = int(round(lo + ui * (hi - lo)))
        else:
            params[name] = float(lo + ui * (hi - lo))
    return params


def bayes_optimise(objective, space, n_trials, seed=42, n_init=6):
    """Maximise objective(params) -> (mean_score, sd_score).

    Returns (trials DataFrame with one row per trial, method name).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        rows = []

        def obj(trial):
            p = {}
            for name, (lo, hi, kind) in space.items():
                p[name] = (trial.suggest_int(name, lo, hi) if kind == "int"
                           else trial.suggest_float(name, lo, hi, log=(kind == "log")))
            m, s = objective(p)
            rows.append({**p, "cv_auc": m, "cv_sd": s})
            return m

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_init))
        study.optimize(obj, n_trials=n_trials)
        return pd.DataFrame(rows), "Optuna (TPE)"
    except ImportError:
        pass

    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    rng = np.random.RandomState(seed)
    d = len(space)
    U, rows = [], []
    for t in range(n_trials):
        if t < n_init:
            u = rng.rand(d)
        else:
            y = np.array([r["cv_auc"] for r in rows])
            kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(d)) + WhiteKernel(1e-3)
            gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed,
                                          n_restarts_optimizer=2).fit(np.array(U), y)
            cand = rng.rand(3000, d)
            mu, sd = gp.predict(cand, return_std=True)
            sd = np.maximum(sd, 1e-9)
            z = (mu - y.max()) / sd
            ei = (mu - y.max()) * norm.cdf(z) + sd * norm.pdf(z)          # expected improvement
            u = cand[int(np.argmax(ei))]
        p = decode_unit(u, space)
        m, s = objective(p)
        U.append(u)
        rows.append({**p, "cv_auc": m, "cv_sd": s})
    return pd.DataFrame(rows), "Gaussian process + expected improvement"
