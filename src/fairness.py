"""Group-level fairness screening."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_report(groups, y_true, p, threshold, min_n=100):
    """Per-group base rate, selection rate, TPR, FPR, precision, AUC and calibration gap."""
    d = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y_true), "p": np.asarray(p)})
    rows = []
    for g, s in d.groupby("g"):
        pred = s.p >= threshold
        pos, neg = (s.y == 1).sum(), (s.y == 0).sum()
        rows.append(dict(group=g, n=len(s), share=len(s) / len(d), base_rate=s.y.mean(),
                         selection_rate=pred.mean(),
                         TPR=(pred & (s.y == 1)).sum() / pos if pos else np.nan,
                         FPR=(pred & (s.y == 0)).sum() / neg if neg else np.nan,
                         precision=(s.y[pred] == 1).mean() if pred.sum() else np.nan,
                         AUC=roc_auc_score(s.y, s.p) if (pos > 0 and neg > 0 and len(s) >= min_n) else np.nan,
                         mean_pred=s.p.mean(), calib_gap=s.p.mean() - s.y.mean()))
    return pd.DataFrame(rows)


def gap_summary(audit: pd.DataFrame, y_true, p, threshold, min_n=100):
    """Screen every column of `audit` and flag disparate-impact < 0.8, TPR/FPR gaps > 5pp, calibration gap > 2pp."""
    out = []
    for a in audit.columns:
        r = group_report(audit[a].astype(str).values, y_true, p, threshold, min_n)
        r = r[r.n >= min_n]
        out.append(dict(attribute=a, groups=len(r),
                        disparate_impact_ratio=r.selection_rate.min() / r.selection_rate.max()
                        if r.selection_rate.max() > 0 else np.nan,
                        TPR_gap=r.TPR.max() - r.TPR.min(), FPR_gap=r.FPR.max() - r.FPR.min(),
                        max_abs_calib_gap=r.calib_gap.abs().max(), AUC_range=r.AUC.max() - r.AUC.min()))
    g = pd.DataFrame(out)

    def flags(r):
        f = [name for name, cond in [("DI<0.8", r.disparate_impact_ratio < 0.8), ("TPR gap>5pp", r.TPR_gap > 0.05),
                                     ("FPR gap>5pp", r.FPR_gap > 0.05), ("calib>2pp", r.max_abs_calib_gap > 0.02)] if cond]
        return ", ".join(f) or "none"
    g["flags"] = g.apply(flags, axis=1)
    return g
