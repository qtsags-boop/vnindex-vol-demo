"""Điểm chạy dòng lệnh.

    python -m engine.run daily              # lấy dữ liệu mới, dự báo, xuất JSON
    python -m engine.run weekly             # + cập nhật walk-forward, huấn luyện lại mô hình
    python -m engine.run backtest --h 1 --models M3,M4 [--force]
    python -m engine.run export             # chỉ xuất lại JSON từ trạng thái hiện có
Thêm --offline để không tải dữ liệu (dùng dữ liệu đã lưu).
"""
import argparse
import json
import os
import sys
import time

from .config import RAW_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "daily", "weekly", "backtest", "train", "export"])
    ap.add_argument("--h", default="", help="các tầm, ví dụ 1,5")
    ap.add_argument("--models", default="", help="ví dụ M3,M4")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    from . import data
    hs = [int(x) for x in a.h.split(",") if x] or None
    ms = [x for x in a.models.split(",") if x] or None
    t0 = time.time()

    status = None
    if a.cmd in ("fetch", "daily", "weekly") and not a.offline:
        status = data.update_all()
        print(json.dumps(status, ensure_ascii=False, indent=1))
        if a.cmd == "fetch":
            return
    if status is None:
        p = os.path.join(RAW_DIR, "fetch_status.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                status = json.load(f)

    from . import pipeline as P
    df = P.load_frame()
    P.log(f"Dữ liệu: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} phiên)")

    if a.cmd == "backtest":
        P.run_backtest(df, hs, ms, force=a.force)
    elif a.cmd == "train":
        P.train_live(df, hs, ms)
    elif a.cmd == "weekly":
        P.run_backtest(df, hs, ms)
        P.train_live(df, hs, ms)
        recent = P.predict_today(df)
        P.export_all(df, recent, status)
    elif a.cmd == "daily":
        recent = P.predict_today(df)
        P.export_all(df, recent, status)
    elif a.cmd == "export":
        recent = P.predict_today(df)
        P.export_all(df, recent, status)
    P.log(f"Xong '{a.cmd}' sau {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
