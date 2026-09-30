# Cập nhật phần GD và AGD

Thư mục này là vùng làm việc độc lập cho phần sửa theo nhận xét của giảng viên.
Mã nguồn gốc trong `bai-lam-taitt/` không bị ghi đè.

## Mục tiêu

1. Mở rộng lưới dò bước của GD và AGD, đánh giá ở nhiều ngân sách vòng lặp.
2. Với AGD, phân biệt rõ cấu hình lý thuyết và cấu hình dò thực nghiệm.
3. So sánh **chưa dò tối ưu** và **sau dò tối ưu** theo hai nhóm chỉ tiêu:
   - tối ưu: `f - f*`, số vòng, thời gian, số gradient;
   - dự đoán: Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC, F1, Precision,
     Recall và Log-loss trên tập test.
4. Không dùng tập test để chọn bước. Cấu hình được chọn trên tập train/validation,
   sau đó test đúng một lần.

## Cấu trúc

```text
tu_update/
├── README.md
├── config.py                 lưới tham số và ngân sách
├── gd_agd_runner.py          vòng lặp GD/AGD có lưu checkpoint
├── evaluation.py             chia dữ liệu và metric phân loại
├── run_experiments.py        driver chạy toàn bộ thí nghiệm
├── main_changes.md           kế hoạch cập nhật main.tex
└── artifacts/
    └── README.md             mô tả các kết quả sẽ sinh
```

## Chạy

Chạy từ thư mục gốc của repository:

```bash
python tu_update/run_experiments.py --method all
```

Chạy thử nhanh:

```bash
python tu_update/run_experiments.py --method all --quick
```

Mặc định script tìm ma trận theo thứ tự:

1. `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`;
2. `data/ridge.npz`.

Có thể chỉ định trực tiếp:

```bash
python tu_update/run_experiments.py --data-path C:/duong_dan/ridge.npz
```

## Quy ước báo cáo

- “Chưa dò tối ưu” = cấu hình lý thuyết: GD dùng `t=1/L`; AGD dùng
  `t=1/L` và momentum theo `L, mu`.
- “Sau dò tối ưu” = cấu hình tốt nhất **trong lưới và ngân sách đã xét**.
- Không gọi một cấu hình là tối ưu toàn cục nếu chỉ được chọn từ một lưới hữu hạn.
- Nếu file `.npz` không có tập test, script tạo một phép chia cố định. Đây là kết
  quả bổ sung; bản cuối nên tái sinh dữ liệu với holdout trước mọi phép fit tiền xử lý.

