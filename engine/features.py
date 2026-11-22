"""Biến mục tiêu và bể đặc trưng (Bước 2) — đúng công thức notebook V5.1B.

- Biến mục tiêu y_h(t) = sqrt( (252/h) * Σ_{i=1..h} r_{t+i}^2 )   (công thức 3.2)
- 19 đặc trưng thị trường và kỹ thuật (Bảng 3.1)
- 8 biến hành vi (Bảng 3.2)
"""
import math

import numpy as np
import pandas as pd

from .config import CFG

FEAT_BASE = [
    "vol_5", "vol_10", "vol_20", "vol_60", "volatility_20d",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "hl_range", "hl_range_ma5", "close_pos",
    "rsi_14", "macd", "macd_sig", "macd_hist",
    "boll_pctb", "boll_width", "vol_norm",
]
BEHAV = ["vol_ratio", "vol_anomaly", "return_zscore_20", "up_streak", "down_streak",
         "vol_x_anom", "gold_return_lag1", "usdvnd_return_lag1"]
FEAT_ALL = FEAT_BASE + BEHAV

FEAT_LABELS = {
    "vol_5": "Biến động lăn 5 phiên", "vol_10": "Biến động lăn 10 phiên",
    "vol_20": "Biến động lăn 20 phiên", "vol_60": "Biến động lăn 60 phiên",
    "volatility_20d": "Biến động 20 phiên (tham chiếu)",
    "ret_lag1": "Lợi suất trễ 1 phiên", "ret_lag2": "Lợi suất trễ 2 phiên",
    "ret_lag3": "Lợi suất trễ 3 phiên", "ret_lag5": "Lợi suất trễ 5 phiên",
    "hl_range": "Biên độ cao–thấp trong phiên", "hl_range_ma5": "Biên độ trung bình 5 phiên",
    "close_pos": "Vị trí giá đóng cửa trong phiên", "rsi_14": "RSI 14 phiên",
    "macd": "MACD", "macd_sig": "Đường tín hiệu MACD", "macd_hist": "Histogram MACD",
    "boll_pctb": "Bollinger %b", "boll_width": "Độ rộng dải Bollinger",
    "vol_norm": "Khối lượng chuẩn hoá",
    "vol_ratio": "Tỷ lệ khối lượng (chú ý)", "vol_anomaly": "Khối lượng bất thường",
    "return_zscore_20": "Z-score lợi suất 20 phiên", "up_streak": "Chuỗi phiên tăng",
    "down_streak": "Chuỗi phiên giảm", "vol_x_anom": "Biến động × khối lượng",
    "gold_return_lag1": "Lợi suất vàng trễ 1 phiên",
    "usdvnd_return_lag1": "Lợi suất USD/VND trễ 1 phiên",
}


def _rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_frame(vn, gold=None, usdvnd=None):
    """Từ OHLCV ngày tạo DataFrame có log_return, mục tiêu y_h*, 27 đặc trưng."""
    df = vn[["open", "high", "low", "close", "volume"]].copy().astype(float)
    ann = math.sqrt(CFG["ANNUALIZE"])
    df["log_return"] = np.log(df["close"]).diff()
    df["volatility_20d"] = df["log_return"].rolling(CFG["VOL_WINDOW"]).std() * ann
    for w in [5, 10, 20, 60]:
        df[f"vol_{w}"] = df["log_return"].rolling(w).std() * ann

    # Biến mục tiêu nhìn tới trước
    r2 = df["log_return"] ** 2
    for h in CFG["HORIZONS"]:
        fwd = r2.shift(-1).rolling(h).sum().shift(-(h - 1))
        df[f"y_h{h}"] = np.sqrt(fwd * CFG["ANNUALIZE"] / h)

    # Nhóm A — thị trường và kỹ thuật
    for k in [1, 2, 3, 5]:
        df[f"ret_lag{k}"] = df["log_return"].shift(k)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["hl_range_ma5"] = df["hl_range"].rolling(5).mean()
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["rsi_14"] = _rsi(df["close"], 14)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    ma20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    df["boll_pctb"] = (df["close"] - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    df["boll_width"] = (4 * sd20) / ma20.replace(0, np.nan)
    df["vol_norm"] = df["volume"] / df["volume"].rolling(20).mean()

    # Nhóm hành vi
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_anomaly"] = (df["vol_ratio"] > 2).astype(float)
    df.loc[df["vol_ratio"].isna(), "vol_anomaly"] = np.nan
    mu20 = df["log_return"].rolling(20).mean()
    st20 = df["log_return"].rolling(20).std()
    df["return_zscore_20"] = (df["log_return"] - mu20) / st20.replace(0, np.nan)
    sign = np.sign(df["log_return"].fillna(0).values)
    up = np.zeros(len(sign))
    dn = np.zeros(len(sign))
    for i in range(1, len(sign)):
        up[i] = up[i - 1] + 1 if sign[i] > 0 else 0
        dn[i] = dn[i - 1] + 1 if sign[i] < 0 else 0
    df["up_streak"] = up
    df["down_streak"] = dn
    df["vol_x_anom"] = df["volatility_20d"] * df["vol_ratio"]
    for name, s in [("gold_return_lag1", gold), ("usdvnd_return_lag1", usdvnd)]:
        if s is not None and len(s):
            ret = np.log(s.astype(float)).diff()
            df[name] = ret.reindex(df.index).ffill().shift(1)
        else:
            df[name] = 0.0
    return df


def valid_rows(df, h, feats):
    """(F, yv, idx) của các dòng có đủ đặc trưng và có mục tiêu — như sub_hop_le()."""
    y = df[f"y_h{h}"]
    use = df[feats].replace([np.inf, -np.inf], np.nan)
    keep = use.notna().all(axis=1) & y.notna()
    sub = df.loc[keep]
    return sub[feats].values.astype(float), sub[y.name].values.astype(float), sub.index


def feature_rows(df, feats):
    """Các dòng có đủ đặc trưng (kể cả những phiên cuối chưa có mục tiêu)."""
    use = df[feats].replace([np.inf, -np.inf], np.nan)
    keep = use.notna().all(axis=1)
    sub = df.loc[keep]
    return sub[feats].values.astype(float), sub.index
