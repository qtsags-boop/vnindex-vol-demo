# VN-Index · Dự báo biến động (Behavioral Hybrid XGBoost-LSTM)

Ứng dụng minh hoạ cho luận văn thạc sĩ **“Dự báo biến động chỉ số VN-Index trên thị trường chứng khoán Việt Nam bằng mô hình học máy lai”** của Võ Quang Tuyến (Trường Đại học Ngân hàng TP.HCM, người hướng dẫn PGS.TS Đào Lê Kiều Oanh).

Sau mỗi phiên giao dịch, ứng dụng tự lấy dữ liệu VN-Index và chạy dự báo biến động cho ba tầm (1, 5 và 22 phiên) bằng sáu mô hình của luận văn. Người dùng có thể chọn mô hình, xem dự báo ngoài mẫu từ 2021 và so sánh các mô hình bằng RMSE, MAE, QLIKE, kiểm định Diebold-Mariano và Model Confidence Set.

| Mã | Mô hình | Đầu vào |
|---|---|---|
| M0 | EWMA (λ = 0,94) | chuỗi lợi suất |
| M1 | GARCH(1,1) skew-t | chuỗi lợi suất |
| M2 | XGBoost | 27 biến |
| M3 | LSTM | 19 đặc trưng kỹ thuật |
| M4 | Lai XGBoost → LSTM, không hành vi | top-15 từ 19 biến |
| M5 | **Behavioral Hybrid XGBoost-LSTM (đề xuất)** | top-15 từ 27 biến, có 8 biến hành vi |

> Ứng dụng phục vụ học thuật, không phải khuyến nghị đầu tư.

## Cách vận hành trên GitHub

Các nhánh của repo:

- `main` chứa mã nguồn.
- `state` chứa dữ liệu, mô hình đã huấn luyện và kết quả walk-forward. Bot tự ghi đè nhánh này sau mỗi lần chạy.
- `gh-pages` là trang web (GitHub Pages), cài được như ứng dụng trên điện thoại.
- `apk` chứa file APK Android mới nhất.

Các workflow trong tab **Actions**:

| Workflow | Khi nào chạy | Việc làm |
|---|---|---|
| Dự báo hằng ngày | 16:30 thứ Hai đến thứ Sáu | Lấy dữ liệu mới, dự báo M0–M5, ghi nhật ký, cập nhật web |
| Huấn luyện lại hằng tuần | 08:07 sáng thứ Bảy | Cập nhật walk-forward, huấn luyện lại M2–M5 trên toàn bộ dữ liệu, tính lại kiểm định |
| Khởi tạo toàn bộ | Chạy tay | Tính walk-forward từ 2021 bằng 12 việc chạy song song, dùng khi mới tạo repo |
| Cập nhật giao diện web | Khi sửa `web/` hoặc `pwa/` | Dựng lại trang web |
| Build APK | Khi sửa `app/`, `web/` hoặc `pwa/` | Tạo APK, đưa lên nhánh `apk` và mục Releases |

### Thiết lập lần đầu

1. Vào **Settings → Actions → General → Workflow permissions**, chọn *Read and write permissions*.
2. Vào **Actions → Khởi tạo toàn bộ → Run workflow**. Lần đầu mất khoảng 30–60 phút.
3. Vào **Settings → Pages**, chọn *Deploy from a branch*, nhánh `gh-pages`, thư mục `/ (root)`.
4. Mở `https://<tài-khoản>.github.io/<tên-repo>/`.

## Phương pháp

Mã trong `engine/` viết lại theo Chương 3 và dùng đúng công thức của notebook V5.1B:

- **Biến mục tiêu:** y_h(t) = √((252/h)·Σ r²_{t+i}), mô hình học trên log và đổi ngược khi báo cáo.
- **Đặc trưng:** 19 đặc trưng kỹ thuật (Bảng 3.1) và 8 biến hành vi (Bảng 3.2).
- **Walk-forward:** cửa sổ mở rộng từ 01/2021, tái huấn luyện mỗi 126 phiên, khoảng đệm max(22, h). Bộ chuẩn hoá chỉ khớp trên tập huấn luyện của từng mốc, và 20% cuối của tập này dùng để dừng sớm.
- **Siêu tham số XGBoost:** lấy bộ đã tinh chỉnh bằng Optuna trên tập kiểm định 2020 (`engine/config.py`). LSTM có hai tầng 48/24 đơn vị, cửa sổ 20 phiên.
- **Đánh giá:** RMSE, MAE, QLIKE; Diebold-Mariano có hiệu chỉnh HLN; MCS dùng bootstrap khối 1.000 lần, khối 20 phiên, mức 90%.

Khác biệt so với luận văn:

- Ứng dụng dùng 3 hạt giống LSTM thay vì 5 để vừa thời gian chạy trên GitHub.
- Dữ liệu lấy từ nguồn công khai, có thể khác file TradingView dùng trong luận văn.

Vì vậy số liệu do ứng dụng chạy lại có thể lệch nhẹ so với luận văn. Tab **So sánh** cho xem song song số liệu chạy lại và số liệu cố định của luận văn (bản V2.5).

### Nguồn dữ liệu

- **VN-Index ngày (OHLCV):** thử lần lượt vnstock (KBS, VCI, TCBS, MSN), DNSE, VNDirect, SSI rồi TCBS. Mỗi lần chạy lấy trọn lịch sử từ cùng một nguồn để đơn vị khối lượng nhất quán.
- **Giá vàng thế giới:** Yahoo GC=F, dự phòng Stooq.
- **Tỷ giá USD/VND:** Yahoo VND=X, dự phòng vnstock MSN và Stooq.
- **Dữ liệu dự phòng:** có thể đặt file `data_seed/HOSE_DLY_VNINDEX1D.csv` (định dạng TradingView như notebook).

## Chạy trên máy tính

```bash
pip install -r requirements.txt
python -m engine.run weekly        # lấy dữ liệu, walk-forward, huấn luyện, dự báo (lần đầu lâu)
python -m engine.run daily         # cập nhật hằng ngày
python pwa/build_site.py           # dựng trang vào _site/
python -m http.server -d _site     # mở http://localhost:8000
```

Muốn kiểm thử nhanh mà không cần mạng: `python tests/make_synthetic.py` rồi `APP_QUICK=1 python -m engine.run weekly --offline`. Dữ liệu giả này chỉ để kiểm thử.

## Mở rộng cho nghiên cứu sau

- **Thêm mô hình** (HAR-RV, GJR-GARCH, M5B có độ rộng thị trường, Transformer…): khai báo trong `MODELS` ở `engine/models.py`. Mô hình học máy dùng chung lớp `Fitted`, chỉ cần thêm nhánh `kind` mới. Mô hình kinh tế lượng thêm một hàm như `garch_series` rồi ghép vào `classic_series` trong `engine/pipeline.py`.
- **Thêm biến:** sửa `engine/features.py`, bổ sung tên vào `FEAT_BASE` hoặc `BEHAV` cùng nhãn tiếng Việt trong `FEAT_LABELS`.
- **Tinh chỉnh lại siêu tham số:** đặt file `state/models/xgb_params_tuned.json`. File này sẽ được dùng thay cho bộ mặc định.
- **Đổi cấu hình** (tầm dự báo, chu kỳ tái huấn luyện, số hạt giống, top-K…): sửa `CFG` trong `engine/config.py`.

## Cấu trúc thư mục

```
engine/      Python: dữ liệu, đặc trưng, mô hình, kiểm định, pipeline
web/         giao diện (HTML/CSS/JS, Chart.js, phông Be Vietnam Pro)
pwa/         dựng trang GitHub Pages, service worker, biểu tượng
app/         dự án Android (WebView) tạo APK
.github/     workflow và script
tests/       sinh dữ liệu giả để kiểm thử
```
