"""Hàm tổn thất và kiểm định thống kê (Mục 3.6.2–3.6.3) — đúng notebook V5.1B."""
import math

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from scipy.stats import t as _tdist

from .config import CFG


def qlike_loss(y_true, y_pred, eps=1e-8):
    vt = np.maximum(np.asarray(y_true, float) ** 2, eps)
    vp = np.maximum(np.asarray(y_pred, float) ** 2, eps)
    r = vt / vp
    return r - np.log(r) - 1.0


def se_loss(y_true, y_pred):
    return (np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2


LOSSES = {"SE": se_loss, "QLIKE": qlike_loss}


def dm_test(loss1, loss2, h, small_sample=True):
    """Diebold–Mariano, hiệu chỉnh HLN. DM < 0: mô hình 1 có tổn thất nhỏ hơn."""
    d = np.asarray(loss1, float) - np.asarray(loss2, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    s = np.mean((d - dbar) ** 2)
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        s += 2 * (1 - k / h) * cov
    if s <= 0:
        return np.nan, np.nan
    dm = dbar / math.sqrt(s / n)
    if small_sample:
        dm *= math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
        p = 2 * (1 - _tdist.cdf(abs(dm), df=n - 1))
    else:
        p = 2 * (1 - _norm.cdf(abs(dm)))
    return float(dm), float(p)


def mcs_test(loss_dict, B=None, block=None, alpha=None, seed=42):
    """Model Confidence Set (Hansen, Lunde & Nason, 2011), thống kê T_max, bootstrap khối."""
    B = B or CFG["MCS_B"]
    block = block or CFG["MCS_BLOCK"]
    alpha = alpha or CFG["MCS_ALPHA"]
    names = list(loss_dict.keys())
    L = np.column_stack([np.asarray(loss_dict[k], float) for k in names])
    L = L[np.isfinite(L).all(axis=1)]
    n, m = L.shape
    if n < 20 or m < 2:
        return {k: float("nan") for k in names}
    rng = np.random.default_rng(seed)

    def boot_idx():
        idx = np.empty(n, dtype=int)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            ln = min(block, n - i)
            idx[i:i + ln] = np.arange(start, start + ln) % n
            i += ln
        return idx

    boots = [boot_idx() for _ in range(B)]
    alive = list(range(m))
    mcs_p = {k: None for k in names}
    pmax = 0.0
    while len(alive) > 1:
        sub = L[:, alive]
        Lbar = sub.mean(0)
        d = Lbar - Lbar.mean()
        dstar = np.empty((B, len(alive)))
        for b, bi in enumerate(boots):
            lb = sub[bi].mean(0)
            dstar[b] = lb - lb.mean()
        zeta = dstar.var(0) + 1e-12
        tstat = d / np.sqrt(zeta)
        Tmax = tstat.max()
        Tmax_star = ((dstar - d) / np.sqrt(zeta)).max(1)
        pval = float(np.mean(Tmax_star >= Tmax))
        pmax = max(pmax, pval)
        worst = alive[int(np.argmax(tstat))]
        mcs_p[names[worst]] = pmax
        if pval >= alpha:
            for a in alive:
                if mcs_p[names[a]] is None:
                    mcs_p[names[a]] = pmax
            break
        alive.remove(worst)
    for a in alive:
        if mcs_p[names[a]] is None:
            mcs_p[names[a]] = 1.0
    return {k: float(v) for k, v in mcs_p.items()}


def evaluate_frame(y, P, h):
    """y: Series biến động thực; P: DataFrame dự báo (cột = mã mô hình), cùng chỉ số ngày.
    Chỉ dùng các ngày mọi mô hình đều có dự báo để so sánh công bằng."""
    D = P.join(y.rename("y"), how="inner").dropna()
    if len(D) < 30:
        return None
    codes = [c for c in P.columns]
    yv = D["y"].values
    rows = []
    for c in codes:
        e = D[c].values - yv
        rows.append({"model": c, "RMSE": float(np.sqrt(np.mean(e ** 2))),
                     "MAE": float(np.mean(np.abs(e))),
                     "QLIKE": float(np.mean(qlike_loss(yv, D[c].values))), "N": int(len(D))})
    dm = {}
    mcs = {}
    for lname, fn in LOSSES.items():
        Ls = {c: fn(yv, D[c].values) for c in codes}
        dm[lname] = {a: {b: (None if a == b else dict(zip(["dm", "p"], dm_test(Ls[a], Ls[b], h))))
                         for b in codes} for a in codes}
        mcs[lname] = mcs_test(Ls)
    years = {}
    for yr, g in D.groupby(D.index.year):
        years[int(yr)] = {c: float(np.sqrt(np.mean((g[c] - g["y"]) ** 2))) for c in codes}
    return {"start": D.index.min().strftime("%Y-%m-%d"), "end": D.index.max().strftime("%Y-%m-%d"),
            "table": rows, "dm": dm, "mcs": mcs, "yearly_rmse": years}
