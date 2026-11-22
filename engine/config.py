"""Cấu hình chung của ứng dụng — giữ đúng tham số của luận văn (notebook V5.1B, Chương 3).

Mọi đường dẫn trạng thái (dữ liệu, mô hình, kết quả walk-forward) nằm dưới STATE_DIR.
Trên GitHub, STATE_DIR là nhánh `state` được checkout vào thư mục ./state.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("APP_STATE_DIR", os.path.join(ROOT, "state"))
RAW_DIR = os.path.join(STATE_DIR, "data", "raw")
SITE_DATA_DIR = os.path.join(STATE_DIR, "site_data")       # JSON cho web
LIVE_DIR = os.path.join(STATE_DIR, "models", "live")       # mô hình huấn luyện hằng tuần
BT_DIR = os.path.join(STATE_DIR, "models", "backtest")     # dự báo walk-forward (cache)
SEED_DIR = os.path.join(ROOT, "data_seed")                 # CSV dự phòng (tuỳ chọn)

CFG = {
    # ---- Phạm vi dữ liệu (Mục 3.3.1) ----
    "START_DATE": "2015-01-01",
    # ---- Biến mục tiêu (Mục 3.3.2) ----
    "HORIZONS": [1, 5, 22],
    "ANNUALIZE": 252,
    "VOL_WINDOW": 20,
    "USE_LOG_TARGET": True,
    # ---- Chia dữ liệu, chống rò rỉ (Mục 3.6.1, 3.6.4) ----
    "TRAIN_END": "2019-12-31",
    "VAL_END": "2020-12-31",
    "EMBARGO": 22,
    # ---- Walk-forward (Mục 3.6.1) ----
    "WF_OOS_START": "2021-01-01",
    "WF_REFIT": 126,
    "WF_VAL_FRAC": 0.20,
    "THESIS_OOS_END": "2025-12-31",
    # ---- Mô hình (Mục 3.5) ----
    "EWMA_LAMBDA": 0.94,
    "GARCH_REFIT": 22,
    "LOOKBACK": 20,
    "MAX_LSTM_UNITS": 64,
    "EPOCHS": 150,
    "BATCH_SIZE": 32,
    "TOP_K_FEATURES": 15,
    "N_SEEDS": 3,          # luận văn dùng 5; ứng dụng dùng 3 để vừa thời gian chạy GitHub
    "BASE_SEED": 44,
    # ---- Kiểm định (Mục 3.6.3) ----
    "MCS_B": 1000,
    "MCS_BLOCK": 20,
    "MCS_ALPHA": 0.10,
}

# Siêu tham số XGBoost đã tinh chỉnh bằng Optuna trên tập kiểm định 2020 (notebook V5.1B).
XGB_PARAMS = {
    1: {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.13254556101349418,
        "subsample": 0.6595392252176876, "colsample_bytree": 0.7814749790794866,
        "min_child_weight": 4, "reg_alpha": 0.0007319385695073038,
        "reg_lambda": 0.0030817739050816024},
    5: {"n_estimators": 700, "max_depth": 3, "learning_rate": 0.036065724771929715,
        "subsample": 0.6396253098638202, "colsample_bytree": 0.5117037832346841,
        "min_child_weight": 20, "reg_alpha": 0.0001107252232849108,
        "reg_lambda": 0.012804329670264477},
    22: {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.027435886105921403,
         "subsample": 0.9910634404610816, "colsample_bytree": 0.999799041304074,
         "min_child_weight": 20, "reg_alpha": 0.00011082650272631263,
         "reg_lambda": 0.0011751634513180013},
}

# Cho phép ghi đè siêu tham số bằng file do engine/tune.py sinh ra
_tuned = os.path.join(STATE_DIR, "models", "xgb_params_tuned.json")
if os.path.exists(_tuned):
    with open(_tuned, encoding="utf-8") as f:
        XGB_PARAMS.update({int(k): v for k, v in json.load(f).items()})

# Chế độ nhanh để kiểm thử (APP_QUICK=1): ít seed, ít epoch, ít mốc
if os.environ.get("APP_QUICK"):
    CFG.update({"N_SEEDS": 1, "EPOCHS": 4, "MCS_B": 100, "WF_REFIT": 400})

if os.environ.get("APP_SEEDS"):
    CFG["N_SEEDS"] = int(os.environ["APP_SEEDS"])

for d in (RAW_DIR, SITE_DATA_DIR, LIVE_DIR, BT_DIR):
    os.makedirs(d, exist_ok=True)
