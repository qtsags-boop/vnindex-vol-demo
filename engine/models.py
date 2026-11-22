"""Sáu mô hình so sánh (Bảng 3.4) và cơ chế huấn luyện theo mốc (walk-forward).

M0 EWMA (λ=0,94) · M1 GARCH(1,1) skew-t cập nhật đệ quy · M2 XGBoost (27 biến)
M3 LSTM (19 biến) · M4 lai XGBoost→LSTM (19 biến) · M5 lai hành vi (27 biến, đề xuất)

Muốn thêm mô hình mới (HAR-RV, GJR-GARCH, M5B...), khai báo trong MODELS và viết hàm
huấn luyện/dự báo tương ứng — xem hướng dẫn trong README.
"""
import json
import math
import os
import warnings

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from .config import CFG, XGB_PARAMS
from .features import BEHAV, FEAT_ALL, FEAT_BASE

MODELS = {
    "M0": {"name": "EWMA", "short": "EWMA", "group": "Kinh tế lượng", "kind": "ewma",
           "input": "Chuỗi lợi suất quá khứ"},
    "M1": {"name": "GARCH(1,1)", "short": "GARCH", "group": "Kinh tế lượng", "kind": "garch",
           "input": "Chuỗi lợi suất quá khứ"},
    "M2": {"name": "XGBoost", "short": "XGBoost", "group": "Học máy đơn lẻ", "kind": "xgb",
           "feats": "ALL", "input": "27 biến (19 kỹ thuật + 8 hành vi)"},
    "M3": {"name": "LSTM", "short": "LSTM", "group": "Học máy đơn lẻ", "kind": "lstm",
           "feats": "BASE", "input": "19 đặc trưng kỹ thuật"},
    "M4": {"name": "Hybrid XGBoost-LSTM (không hành vi)", "short": "Hybrid",
           "group": "Mô hình lai", "kind": "hybrid", "feats": "BASE",
           "input": "Top-15 chọn từ 19 đặc trưng kỹ thuật"},
    "M5": {"name": "Behavioral Hybrid XGBoost-LSTM (đề xuất)", "short": "Behavioral Hybrid",
           "group": "Mô hình lai", "kind": "hybrid", "feats": "ALL",
           "input": "Top-15 chọn từ 27 biến (có 8 biến hành vi)"},
}
ML_MODELS = [k for k, v in MODELS.items() if v["kind"] in ("xgb", "lstm", "hybrid")]


def feats_of(code):
    return FEAT_ALL if MODELS[code].get("feats") == "ALL" else FEAT_BASE


# ------------------------------------------------------------------ M0, M1
def ewma_series(df):
    """Phương sai EWMA RiskMetrics: σ²_t = λσ²_{t-1} + (1-λ) r²_{t-1}, quy năm."""
    lam = CFG["EWMA_LAMBDA"]
    r = df["log_return"].fillna(0).values
    s2 = np.zeros(len(r))
    s2[0] = np.nanvar(r[:20]) if len(r) > 20 else np.nanvar(r)
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(s2 * CFG["ANNUALIZE"]), index=df.index)


def garch_series(df, dates, h):
    """GARCH(1,1) skew-t, ước lượng lại mỗi 22 phiên, cập nhật đệ quy giữa các lần ước lượng.
    Trả về dự báo biến động trung bình h phiên (quy năm) tại các ngày `dates`."""
    from arch import arch_model
    refit = CFG["GARCH_REFIT"]
    r_all = df["log_return"].ffill().fillna(0).values * 100
    pos_map = {d: i for i, d in enumerate(df.index)}
    pos = [pos_map[d] for d in dates if d in pos_map]
    out, par, s2_prev, e2_prev = [], None, None, None
    for k, t in enumerate(pos):
        if k % refit == 0 or par is None:
            try:
                am = arch_model(pd.Series(r_all[:t]), vol="GARCH", p=1, q=1, dist="skewt")
                res = am.fit(disp="off", show_warning=False)
                p = res.params
                par = {"omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
                       "beta": float(p["beta[1]"])}
                s2_prev = float(res.conditional_volatility.iloc[-1] ** 2)
                e2_prev = float(r_all[t - 1] ** 2)
            except Exception:
                par = None
        if par is None:
            out.append(np.nan)
            continue
        s2_t = par["omega"] + par["alpha"] * e2_prev + par["beta"] * s2_prev
        pers = min(par["alpha"] + par["beta"], 0.9999)
        s2_lr = par["omega"] / (1 - pers)
        tong, s2_k = 0.0, s2_t
        for _ in range(h):
            tong += s2_k
            s2_k = s2_lr + pers * (s2_k - s2_lr)
        out.append(math.sqrt((tong / h) / (100 ** 2) * CFG["ANNUALIZE"]))
        s2_prev, e2_prev = s2_t, float(r_all[t] ** 2)
    return pd.Series(out, index=[df.index[t] for t in pos])


def garch_last_params(df):
    from arch import arch_model
    r = df["log_return"].dropna().values * 100
    res = arch_model(pd.Series(r), vol="GARCH", p=1, q=1, dist="skewt").fit(disp="off", show_warning=False)
    p = res.params
    return {k: float(p[k]) for k in ["omega", "alpha[1]", "beta[1]"]}


# ------------------------------------------------------------------ học máy
def _xgb_model(h):
    import xgboost as xgb
    p = dict(XGB_PARAMS[h])
    p.update({"random_state": CFG["BASE_SEED"], "n_jobs": -1, "tree_method": "hist"})
    return xgb.XGBRegressor(**p, early_stopping_rounds=50)


def _build_lstm(shape, seed):
    import keras
    from keras import layers, regularizers
    keras.utils.set_random_seed(seed)
    u1, u2 = min(48, CFG["MAX_LSTM_UNITS"]), min(24, CFG["MAX_LSTM_UNITS"])
    m = keras.Sequential([
        keras.Input(shape=shape),
        layers.LSTM(u1, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.25),
        layers.LSTM(u2, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.25),
        layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def _callbacks():
    import keras
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)]


def _windows(Fs, positions, lookback):
    X, ps = [], []
    for i in positions:
        if i - lookback + 1 < 0:
            continue
        w = Fs[i - lookback + 1:i + 1]
        if np.isfinite(w).all():
            X.append(w)
            ps.append(i)
    return np.array(X, dtype="float32"), np.array(ps, dtype=int)


class Fitted:
    """Mô hình đã huấn luyện tại một mốc: giữ bộ chuẩn hoá, đặc trưng đã chọn và mô hình con."""

    def __init__(self, code, h):
        self.code, self.h = code, h
        self.kind = MODELS[code]["kind"]
        self.feats = feats_of(code)
        self.sel = list(self.feats)
        self.importance = None
        self.xgb = None
        self.lstms = []
        self.sx_mean = self.sx_scale = None
        self.sy_mean = self.sy_scale = None
        self.info = {}

    # -------- huấn luyện trên dòng hợp lệ [0, train_end) --------
    def fit(self, F, yv, train_end):
        lb = CFG["LOOKBACK"]
        tr = np.arange(0, train_end)
        n_val = max(1, int(len(tr) * CFG["WF_VAL_FRAC"]))
        itr, iva = tr[:-n_val], tr[-n_val:]
        y_learn = np.log(np.maximum(yv, 1e-8)) if CFG["USE_LOG_TARGET"] else yv
        self.sy_mean = float(y_learn[itr].mean())
        self.sy_scale = float(y_learn[itr].std()) or 1.0
        ys = (y_learn - self.sy_mean) / self.sy_scale

        def scale(cols):
            Fc = F[:, cols]
            mu, sd = Fc[itr].mean(0), Fc[itr].std(0)
            sd[sd == 0] = 1.0
            return (Fc - mu) / sd, mu, sd

        all_cols = list(range(len(self.feats)))
        if self.kind in ("xgb", "hybrid"):
            Fs, mu, sd = scale(all_cols)
            m = _xgb_model(self.h)
            m.fit(Fs[itr], ys[itr], eval_set=[(Fs[iva], ys[iva])], verbose=False)
            self.importance = pd.Series(m.feature_importances_, index=self.feats).sort_values(ascending=False)
            if self.kind == "xgb":
                self.xgb, self.sx_mean, self.sx_scale = m, mu, sd
            else:
                self.sel = self.importance.head(min(CFG["TOP_K_FEATURES"], len(self.feats))).index.tolist()
        if self.kind in ("lstm", "hybrid"):
            cols = [self.feats.index(c) for c in self.sel]
            Fs, mu, sd = scale(cols)
            self.sx_mean, self.sx_scale = mu, sd
            Xtr, ptr = _windows(Fs, itr, lb)
            Xva, pva = _windows(Fs, iva, lb)
            if len(Xtr) < 20:
                raise RuntimeError("quá ít mẫu huấn luyện LSTM")
            ytr, yva = ys[ptr].astype("float32"), ys[pva].astype("float32")
            for s in range(CFG["N_SEEDS"]):
                net = _build_lstm(Xtr.shape[1:], CFG["BASE_SEED"] + s)
                net.fit(Xtr, ytr, validation_data=(Xva, yva) if len(Xva) else None,
                        epochs=CFG["EPOCHS"], batch_size=CFG["BATCH_SIZE"],
                        callbacks=_callbacks(), verbose=0)
                self.lstms.append(net)
        self.info.update(n_train=int(len(itr)), n_val=int(len(iva)))
        return self

    # -------- dự báo tại các vị trí của ma trận đặc trưng thô F --------
    def predict(self, F, positions):
        cols = [self.feats.index(c) for c in self.sel]
        Fs = (F[:, cols] - self.sx_mean) / self.sx_scale
        positions = np.asarray(positions, dtype=int)
        if self.kind == "xgb":
            ys = self.xgb.predict(Fs[positions])
            ps = positions
        else:
            X, ps = _windows(Fs, positions, CFG["LOOKBACK"])
            if not len(X):
                return np.array([]), ps
            ys = np.stack([m.predict(X, verbose=0).ravel() for m in self.lstms]).mean(0)
        y = np.asarray(ys) * self.sy_scale + self.sy_mean
        return (np.exp(y) if CFG["USE_LOG_TARGET"] else y), ps

    @property
    def behavioral_selected(self):
        return [c for c in self.sel if c in BEHAV] if self.kind == "hybrid" else []

    # -------- lưu / nạp --------
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        meta = {"code": self.code, "h": self.h, "kind": self.kind, "feats": self.feats,
                "sel": self.sel, "sx_mean": list(map(float, self.sx_mean)),
                "sx_scale": list(map(float, self.sx_scale)), "sy_mean": self.sy_mean,
                "sy_scale": self.sy_scale, "n_lstm": len(self.lstms), "info": self.info,
                "importance": None if self.importance is None else
                {k: float(v) for k, v in self.importance.items()}}
        if self.xgb is not None:
            self.xgb.save_model(os.path.join(folder, "xgb.json"))
        for i, m in enumerate(self.lstms):
            m.save(os.path.join(folder, f"lstm_{i}.keras"))
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, folder):
        import keras
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        o = cls(meta["code"], meta["h"])
        o.feats, o.sel = meta["feats"], meta["sel"]
        o.sx_mean, o.sx_scale = np.array(meta["sx_mean"]), np.array(meta["sx_scale"])
        o.sy_mean, o.sy_scale = meta["sy_mean"], meta["sy_scale"]
        o.info = meta.get("info", {})
        if meta.get("importance"):
            o.importance = pd.Series(meta["importance"]).sort_values(ascending=False)
        if o.kind == "xgb":
            import xgboost as xgb
            o.xgb = xgb.XGBRegressor()
            o.xgb.load_model(os.path.join(folder, "xgb.json"))
        for i in range(meta["n_lstm"]):
            o.lstms.append(keras.models.load_model(os.path.join(folder, f"lstm_{i}.keras")))
        return o
