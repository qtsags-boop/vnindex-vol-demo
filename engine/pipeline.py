"""Các bước chạy: walk-forward (có cache), huấn luyện hằng tuần, dự báo hằng ngày, xuất JSON."""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .config import BT_DIR, CFG, LIVE_DIR, RAW_DIR, SITE_DATA_DIR, STATE_DIR
from .data import VN_TZ, load_raw
from .evaluate import evaluate_frame
from .features import BEHAV, FEAT_ALL, FEAT_LABELS, build_frame, feature_rows, valid_rows
from .models import MODELS, ML_MODELS, Fitted, ewma_series, feats_of, garch_series

H = CFG["HORIZONS"]
LOG_PATH = os.path.join(STATE_DIR, "data", "forecast_log.csv")


def log(*a):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


def load_frame():
    vn, gold, usd = load_raw()
    return build_frame(vn, gold, usd)


# ------------------------------------------------------------------ walk-forward
def wf_origins(idx, h):
    n = len(idx)
    emb = max(CFG["EMBARGO"], h)
    o = int(np.searchsorted(idx, pd.Timestamp(CFG["WF_OOS_START"])))
    out = []
    while o < n:
        hi = min(o + CFG["WF_REFIT"], n)
        te = o - emb
        if te > 60:
            out.append((te, o, hi))
        o = hi
    return out


def _bt_path(h, code):
    return os.path.join(BT_DIR, f"h{h}_{code}.csv")


def classic_series(df, h):
    """M0, M1 trên toàn bộ ngày từ đầu giai đoạn ngoài mẫu tới phiên cuối."""
    dates = df.loc[CFG["WF_OOS_START"]:].index
    m0 = ewma_series(df).reindex(dates)
    t0 = time.time()
    m1 = garch_series(df, dates, h)
    log(f"  GARCH h={h}: {len(dates)} ngày, {time.time()-t0:.0f}s")
    return pd.DataFrame({"M0": m0, "M1": m1})


def backtest_ml(df, h, code, force=False):
    """Walk-forward 126 phiên cho một mô hình học máy; dùng lại các khối đã đóng trong cache."""
    feats = feats_of(code)
    F, yv, idx = valid_rows(df, h, feats)
    origins = wf_origins(idx, h)
    path = _bt_path(h, code)
    cache = None
    if os.path.exists(path) and not force:
        cache = pd.read_csv(path, parse_dates=["date"])
    parts, sel_log = [], []
    for (te, lo, hi) in origins:
        key = idx[lo].strftime("%Y-%m-%d")
        full = (hi - lo) == CFG["WF_REFIT"]
        if cache is not None:
            c = cache[cache["origin"] == key]
            if len(c) and int(c["closed"].iloc[0]) == 1 and full:
                parts.append(c)
                continue
        t0 = time.time()
        fm = Fitted(code, h).fit(F, yv, te)
        yh, ps = fm.predict(F, np.arange(lo, hi))
        part = pd.DataFrame({"date": idx[ps], "pred": yh, "origin": key, "closed": int(full),
                             "sel": "|".join(fm.sel) if fm.kind == "hybrid" else ""})
        parts.append(part)
        log(f"  {code} h={h} mốc {key} ({hi-lo} phiên): {time.time()-t0:.0f}s")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["date", "pred", "origin", "closed", "sel"])
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    return out


def run_backtest(df, horizons=None, codes=None, force=False):
    for h in horizons or H:
        for code in codes or ML_MODELS:
            log(f"Walk-forward {code} h={h}")
            backtest_ml(df, h, code, force=force)


def backtest_table(df, h):
    """DataFrame dự báo ngoài mẫu (cột M0..M5) + cột y thực tế, trên các ngày có mục tiêu."""
    _, yv, idx = valid_rows(df, h, FEAT_ALL)
    y = pd.Series(yv, index=idx).loc[CFG["WF_OOS_START"]:]
    cl_path = os.path.join(BT_DIR, f"h{h}_classic.csv")
    cl = pd.read_csv(cl_path, parse_dates=["date"], index_col="date")
    P = cl[["M0", "M1"]].copy()
    for code in ML_MODELS:
        p = _bt_path(h, code)
        if os.path.exists(p):
            s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["pred"]
            P[code] = s
    P = P.reindex(y.index)
    return y, P


# ------------------------------------------------------------------ huấn luyện hằng tuần
def train_live(df, horizons=None, codes=None):
    for h in horizons or H:
        for code in codes or ML_MODELS:
            t0 = time.time()
            F, yv, idx = valid_rows(df, h, feats_of(code))
            fm = Fitted(code, h).fit(F, yv, len(idx))
            fm.info.update(trained_until=idx[-1].strftime("%Y-%m-%d"),
                           trained_at=datetime.now(VN_TZ).isoformat(timespec="minutes"))
            fm.save(os.path.join(LIVE_DIR, f"h{h}", code))
            log(f"Huấn luyện {code} h={h}: {len(idx)} dòng, {time.time()-t0:.0f}s"
                + (f", hành vi chọn {fm.behavioral_selected}" if fm.kind == "hybrid" else ""))


# ------------------------------------------------------------------ dự báo hằng ngày
def predict_today(df, n_recent=60):
    """Dự báo tại các phiên gần nhất cho mọi mô hình; ghi nhật ký dự báo."""
    out = {}
    for h in H:
        cl = classic_series(df, h)
        cl.to_csv(os.path.join(BT_DIR, f"h{h}_classic.csv"), index_label="date", date_format="%Y-%m-%d")
        recent = {"M0": cl["M0"].iloc[-n_recent:], "M1": cl["M1"].iloc[-n_recent:]}
        for code in ML_MODELS:
            folder = os.path.join(LIVE_DIR, f"h{h}", code)
            if not os.path.exists(os.path.join(folder, "meta.json")):
                continue
            fm = Fitted.load(folder)
            F, idx = feature_rows(df, fm.feats)
            pos = np.arange(max(0, len(idx) - n_recent), len(idx))
            yh, ps = fm.predict(F, pos)
            recent[code] = pd.Series(yh, index=idx[ps])
        out[h] = pd.DataFrame(recent)
    _append_log(df, out)
    return out


def _append_log(df, recent):
    T = df.index[-1]
    rows = []
    for h, R in recent.items():
        if T not in R.index:
            continue
        for code, v in R.loc[T].items():
            if pd.notna(v):
                rows.append({"date": T.strftime("%Y-%m-%d"), "h": h, "model": code, "pred": float(v),
                             "logged_at": datetime.now(VN_TZ).isoformat(timespec="minutes")})
    new = pd.DataFrame(rows)
    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH, dtype={"date": str})
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=["date", "h", "model"], keep="first")
    new.to_csv(LOG_PATH, index=False)


# ------------------------------------------------------------------ xuất JSON cho web
def _r(x, k=5):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def _dump(name, obj):
    with open(os.path.join(SITE_DATA_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _clean_nan(o):
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean_nan(v) for v in o]
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    return o


def export_all(df, recent=None, fetch_status=None):
    now = datetime.now(VN_TZ).isoformat(timespec="minutes")
    T = df.index[-1]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ---- lịch sử giá và biến động
    hist = df[["close", "volatility_20d", "volume"]].dropna(subset=["close"])
    _dump("history.json", {
        "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
        "close": [_r(v, 2) for v in hist["close"]],
        "vol20": [_r(v, 4) for v in hist["volatility_20d"]],
        "volume": [_r(v, 0) for v in hist["volume"]],
    })

    # ---- walk-forward và đánh giá
    metrics = {"thesis": {}, "all": {}}
    for h in H:
        try:
            y, P = backtest_table(df, h)
        except FileNotFoundError:
            continue
        _dump(f"backtest_h{h}.json", {
            "dates": [d.strftime("%Y-%m-%d") for d in y.index],
            "actual": [_r(v, 4) for v in y.values],
            "pred": {c: [_r(v, 4) for v in P[c].values] for c in P.columns},
        })
        cols = [c for c in MODELS if c in P.columns and P[c].notna().sum() > 30]
        P = P[cols]
        metrics["all"][str(h)] = evaluate_frame(y, P, h)
        yt = y.loc[:CFG["THESIS_OOS_END"]]
        metrics["thesis"][str(h)] = evaluate_frame(yt, P.loc[:CFG["THESIS_OOS_END"]], h)
    _dump("metrics.json", _clean_nan(metrics))

    # ---- dự báo mới nhất
    fc = {}
    rec_json = {}
    if recent:
        for h, R in recent.items():
            fc[str(h)] = {c: _r(R[c].dropna().iloc[-1], 4) if c in R and R[c].notna().any() else None
                          for c in MODELS}
            fc[str(h) + "_date"] = {c: (R[c].dropna().index[-1].strftime("%Y-%m-%d")
                                        if c in R and R[c].notna().any() else None) for c in MODELS}
            rec_json[str(h)] = {"dates": [d.strftime("%Y-%m-%d") for d in R.index],
                                **{c: [_r(v, 4) for v in R[c].values] for c in R.columns}}
    _dump("recent.json", rec_json)

    # ---- mô hình live: đặc trưng đã chọn
    live = {}
    for h in H:
        live[str(h)] = {}
        for code in ML_MODELS:
            p = os.path.join(LIVE_DIR, f"h{h}", code, "meta.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                imp = m.get("importance") or {}
                live[str(h)][code] = {
                    "sel": m["sel"], "behav_sel": [c for c in m["sel"] if c in BEHAV] if m["kind"] == "hybrid" else [],
                    "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])[:27]),
                    "info": m.get("info", {})}

    # ---- mức biến động so với lịch sử (phân vị biến động thực tế 20 phiên)
    v20 = df["volatility_20d"].dropna()
    pct = {str(q): _r(v20.quantile(q / 100), 4) for q in [10, 25, 50, 75, 90]}

    # ---- nhật ký dự báo live, kèm giá trị thực khi đã biết
    track = []
    if os.path.exists(LOG_PATH):
        L = pd.read_csv(LOG_PATH, dtype={"date": str})
        for _, r in L.iterrows():
            d = pd.Timestamp(r["date"])
            yt = df.at[d, f"y_h{int(r['h'])}"] if d in df.index else np.nan
            track.append({"date": r["date"], "h": int(r["h"]), "model": r["model"],
                          "pred": _r(r["pred"], 4), "actual": _r(yt, 4)})
    _dump("track.json", track)

    beh_now = {c: _r(last[c], 4) for c in BEHAV}
    beh_pct = {}
    for c in BEHAV:
        s = df[c].dropna()
        if c == "vol_anomaly":
            beh_pct[c] = _r(s.mean() * 100, 2) if len(s) else None   # tần suất phiên bất thường
        else:
            beh_pct[c] = _r((s <= last[c]).mean() * 100, 1) if len(s) and pd.notna(last[c]) else None

    status = {
        "generated_at": now,
        "last_session": T.strftime("%Y-%m-%d"),
        "n_sessions": int(len(df)),
        "data_start": df.index[0].strftime("%Y-%m-%d"),
        "fetch": fetch_status or {},
        "config": {k: CFG[k] for k in ["HORIZONS", "WF_OOS_START", "WF_REFIT", "EMBARGO", "LOOKBACK",
                                        "TOP_K_FEATURES", "N_SEEDS", "EWMA_LAMBDA", "THESIS_OOS_END"]},
    }
    latest = {
        "status": status,
        "market": {"date": T.strftime("%Y-%m-%d"), "close": _r(last["close"], 2),
                   "change": _r(last["close"] - prev["close"], 2),
                   "change_pct": _r((last["close"] / prev["close"] - 1) * 100, 2),
                   "volume": _r(last["volume"], 0), "vol20": _r(last["volatility_20d"], 4),
                   "vol5": _r(last["vol_5"], 4), "vol60": _r(last["vol_60"], 4)},
        "forecast": fc,
        "vol_percentiles": pct,
        "behavior_now": beh_now,
        "behavior_pct": beh_pct,
        "live_models": live,
        "models": MODELS,
        "feature_labels": FEAT_LABELS,
        "behav": BEHAV,
    }
    _dump("latest.json", _clean_nan(latest))
    log(f"Đã xuất JSON vào {SITE_DATA_DIR}")
