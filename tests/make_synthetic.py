"""Sinh dữ liệu giả (GARCH) để kiểm thử pipeline khi không có mạng. KHÔNG dùng cho báo cáo."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import RAW_DIR  # noqa: E402

rng = np.random.default_rng(7)
idx = pd.bdate_range("2015-01-02", "2026-11-20")
n = len(idx)
w, a, b = 2e-6, 0.08, 0.9
s2 = np.full(n, 1e-4)
r = np.zeros(n)
for t in range(1, n):
    s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
    r[t] = np.sqrt(s2[t]) * rng.standard_t(5) / np.sqrt(5 / 3) + 0.0003
close = 600 * np.exp(np.cumsum(r))
rng_hl = np.abs(rng.normal(0, 1, n)) * np.sqrt(s2) * close
vn = pd.DataFrame({"open": close * (1 - r / 2), "high": close + rng_hl, "low": close - rng_hl,
                   "close": close, "volume": rng.lognormal(19, 0.3, n) * (1 + 30 * np.sqrt(s2))},
                  index=idx)
vn.to_csv(os.path.join(RAW_DIR, "vnindex.csv"), index_label="date")
g = pd.Series(1200 * np.exp(np.cumsum(rng.normal(0, 0.009, n))), index=idx, name="close")
u = pd.Series(22000 * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=idx, name="close")
g.to_frame().to_csv(os.path.join(RAW_DIR, "gold.csv"), index_label="date")
u.to_frame().to_csv(os.path.join(RAW_DIR, "usdvnd.csv"), index_label="date")
print("Đã tạo dữ liệu giả:", n, "phiên →", RAW_DIR)
