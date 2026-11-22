"""Thu thập dữ liệu (Bước 1 của quy trình nghiên cứu).

VN-Index theo ngày (OHLCV) được lấy TRỌN lịch sử từ MỘT nguồn mỗi lần chạy để
đơn vị khối lượng nhất quán. Nhiều nguồn được thử lần lượt; nguồn nào chạy được
thì dùng, nếu tất cả lỗi thì giữ dữ liệu đã lưu lần trước.

Giá vàng thế giới (GC=F) và tỷ giá USD/VND (VND=X) lấy từ Yahoo Finance như notebook,
có nguồn dự phòng.
"""
import io
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .config import CFG, RAW_DIR, SEED_DIR

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}
VN_TZ = timezone(timedelta(hours=7))
COLS = ["open", "high", "low", "close", "volume"]


def _log(*a):
    print("[data]", *a, flush=True)


def _ts(d):
    return int(pd.Timestamp(d).tz_localize(VN_TZ).timestamp())


def _from_udf(js, ts_unit="s"):
    """Chuyển định dạng TradingView UDF {t,o,h,l,c,v} sang DataFrame theo ngày."""
    t = js.get("t") or []
    if not t:
        return None
    idx = (pd.to_datetime(t, unit=ts_unit, utc=True).tz_convert(VN_TZ)
           .tz_localize(None).normalize())
    df = pd.DataFrame({"open": js["o"], "high": js["h"], "low": js["l"],
                       "close": js["c"], "volume": js.get("v", [np.nan] * len(t))}, index=idx)
    return df


# ---------------------------------------------------------------- nguồn VN-Index
def src_vnstock(start, end):
    """Thư viện vnstock (thử lần lượt các nguồn KBS, VCI, TCBS, MSN)."""
    from vnstock import Quote  # noqa: import trong hàm để không bắt buộc cài
    last = None
    for s in ["KBS", "VCI", "TCBS", "MSN"]:
        try:
            q = Quote(symbol="VNINDEX", source=s)
            d = q.history(start=start, end=end, interval="1D")
            if d is not None and len(d) > 500:
                d = d.rename(columns=str.lower)
                tcol = "time" if "time" in d.columns else d.columns[0]
                d.index = pd.to_datetime(d[tcol]).dt.tz_localize(None).dt.normalize()
                return d[[c for c in COLS if c in d.columns]], f"vnstock:{s}"
        except Exception as e:  # pragma: no cover - phụ thuộc mạng
            last = e
    raise RuntimeError(f"vnstock lỗi: {last}")


def src_dnse(start, end):
    url = "https://services.entrade.com.vn/chart-api/v2/ohlcs/index"
    r = requests.get(url, params={"from": _ts(start), "to": _ts(end) + 86400,
                                  "symbol": "VNINDEX", "resolution": "1D"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    return _from_udf(r.json()), "DNSE"


def src_vndirect(start, end):
    url = "https://dchart-api.vndirect.com.vn/dchart/history"
    r = requests.get(url, params={"resolution": "D", "symbol": "VNINDEX",
                                  "from": _ts(start), "to": _ts(end) + 86400},
                     headers=UA, timeout=30)
    r.raise_for_status()
    return _from_udf(r.json()), "VNDirect"


def src_ssi(start, end):
    url = "https://iboard-api.ssi.com.vn/statistics/charts/history"
    r = requests.get(url, params={"resolution": "1D", "symbol": "VNINDEX",
                                  "from": _ts(start), "to": _ts(end) + 86400},
                     headers=UA, timeout=30)
    r.raise_for_status()
    return _from_udf(r.json().get("data", {})), "SSI"


def src_tcbs(start, end):
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term"
    n = int((pd.Timestamp(end) - pd.Timestamp(start)).days * 0.75) + 50
    r = requests.get(url, params={"ticker": "VNINDEX", "type": "index", "resolution": "D",
                                  "to": _ts(end) + 86400, "countBack": n},
                     headers=UA, timeout=30)
    r.raise_for_status()
    rows = r.json().get("data") or []
    if not rows:
        return None, "TCBS"
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["tradingDate"]).dt.tz_localize(None).dt.normalize()
    return d[COLS], "TCBS"


VN_SOURCES = [src_vnstock, src_dnse, src_vndirect, src_ssi, src_tcbs]


def _clean(d, start=None):
    d = d.copy()
    for c in COLS:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d.dropna(subset=["close"])
    d = d[d["close"] > 0]
    if start:
        d = d.loc[start:]
    return d


def _read_seed():
    """CSV dự phòng định dạng TradingView (giống notebook) đặt trong data_seed/."""
    p = os.path.join(SEED_DIR, "HOSE_DLY_VNINDEX1D.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d.columns = [c.strip().lower() for c in d.columns]
    if "time" in d.columns:
        d.index = (pd.to_datetime(d["time"], unit="s", utc=True).dt.tz_convert(VN_TZ)
                   .dt.tz_localize(None).dt.normalize())
    else:
        d.index = pd.to_datetime(d.iloc[:, 0]).dt.normalize()
    return _clean(d)


def fetch_vnindex(start=None, end=None):
    """Trả về (DataFrame OHLCV, tên nguồn, danh sách cảnh báo)."""
    start = start or CFG["START_DATE"]
    end = end or datetime.now(VN_TZ).strftime("%Y-%m-%d")
    warns = []
    for fn in VN_SOURCES:
        try:
            d, name = fn(start, end)
            if d is None or len(d) < 500:
                raise RuntimeError(f"quá ít dòng ({0 if d is None else len(d)})")
            d = _clean(d, start)
            # kiểm tra hợp lý: VN-Index trong khoảng 300–5000 điểm
            if not d["close"].between(300, 5000).all():
                raise RuntimeError("giá đóng cửa ngoài khoảng hợp lý")
            _log(f"VN-Index: {name} {d.index.min().date()} → {d.index.max().date()} ({len(d)} phiên)")
            return d, name, warns
        except Exception as e:
            msg = f"{fn.__name__}: {str(e)[:160]}"
            warns.append(msg)
            _log("lỗi", msg)
    seed = _read_seed()
    if seed is not None:
        warns.append("Dùng dữ liệu dự phòng data_seed/ (không lấy được dữ liệu mới).")
        return seed.loc[start:], "seed", warns
    return None, None, warns


# ---------------------------------------------------------------- vàng, tỷ giá
def _yahoo(ticker, start):
    import yfinance as yf
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True, threads=False)
    if raw is None or not len(raw):
        raise RuntimeError("rỗng")
    s = raw["Close"]
    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna()


def _stooq(sym):
    r = requests.get("https://stooq.com/q/d/l/", params={"s": sym, "i": "d"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(d["Close"].values, index=pd.to_datetime(d["Date"]).dt.normalize())
    return s.dropna()


def _vnstock_fx(sym, start):
    from vnstock import Vnstock
    fx = Vnstock().fx(symbol=sym, source="MSN")
    d = fx.quote.history(start=start, end=datetime.now().strftime("%Y-%m-%d"), interval="1D")
    d.index = pd.to_datetime(d["time"]).dt.tz_localize(None).dt.normalize()
    return d["close"].astype(float).dropna()


def fetch_series(kind, start=None):
    """kind = 'gold' | 'usdvnd'. Trả về (Series giá đóng cửa, nguồn, cảnh báo)."""
    start = start or CFG["START_DATE"]
    plans = {
        "gold": [("Yahoo GC=F", lambda: _yahoo("GC=F", start)),
                 ("Stooq XAUUSD", lambda: _stooq("xauusd"))],
        "usdvnd": [("Yahoo VND=X", lambda: _yahoo("VND=X", start)),
                   ("vnstock MSN", lambda: _vnstock_fx("USDVND", start)),
                   ("Stooq USDVND", lambda: _stooq("usdvnd"))],
    }[kind]
    warns = []
    for name, fn in plans:
        for attempt in range(2):
            try:
                s = fn().loc[start:]
                if len(s) < 500:
                    raise RuntimeError(f"quá ít dòng ({len(s)})")
                _log(f"{kind}: {name} → {s.index.max().date()} ({len(s)})")
                return s, name, warns
            except Exception as e:
                if attempt == 0:
                    time.sleep(3)
                    continue
                warns.append(f"{kind} {name}: {str(e)[:120]}")
                _log("lỗi", kind, name, e)
    return None, None, warns


# ---------------------------------------------------------------- cập nhật kho
def _save(df, name):
    df.to_csv(os.path.join(RAW_DIR, name), index_label="date")


def _load(name):
    p = os.path.join(RAW_DIR, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, parse_dates=["date"], index_col="date")
    return d


def update_all(offline=False):
    """Cập nhật 3 chuỗi dữ liệu vào state/data/raw. Trả về dict trạng thái."""
    status = {"time": datetime.now(VN_TZ).isoformat(timespec="minutes"), "warnings": []}
    old = _load("vnindex.csv")
    if offline:
        status.update(vn_source="(offline)", gold_source="(offline)", usdvnd_source="(offline)")
        return status
    vn, src, w = fetch_vnindex()
    status["warnings"] += w if vn is None else []
    if vn is not None:
        if old is not None and len(old) > len(vn) + 5:
            status["warnings"].append(f"Nguồn {src} trả về ít phiên hơn dữ liệu cũ, giữ dữ liệu cũ.")
        else:
            _save(vn[COLS], "vnindex.csv")
            status["vn_source"] = src
    else:
        status["vn_source"] = "(giữ dữ liệu cũ)"
    for kind in ["gold", "usdvnd"]:
        s, src, w = fetch_series(kind)
        if s is not None:
            _save(s.rename("close").to_frame(), f"{kind}.csv")
            status[f"{kind}_source"] = src
        else:
            status["warnings"] += w
            status[f"{kind}_source"] = "(giữ dữ liệu cũ)"
    with open(os.path.join(RAW_DIR, "fetch_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=1)
    return status


def load_raw():
    """Đọc dữ liệu đã lưu: (vnindex DataFrame, gold Series, usdvnd Series)."""
    vn = _load("vnindex.csv")
    if vn is None:
        raise FileNotFoundError("Chưa có state/data/raw/vnindex.csv — chạy cập nhật dữ liệu trước.")
    g = _load("gold.csv")
    u = _load("usdvnd.csv")
    return (vn.loc[CFG["START_DATE"]:],
            None if g is None else g["close"],
            None if u is None else u["close"])
