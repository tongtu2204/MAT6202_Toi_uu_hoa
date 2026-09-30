# Cập nhật phần GD và AGD

Thư mục này là vùng làm việc độc lập cho phần sửa theo nhận xét của giảng viên.
Mã nguồn gốc trong `bai-lam-taitt/` không bị ghi đè.

## Mục tiêu

1. Thay lưới dò cũ bằng lưới GD 25 bước và lưới AGD 21 x 12 cặp
   `(t, beta)`, đánh giá ở nhiều ngân sách vòng lặp.
2. Với AGD, dò `t` và `beta` độc lập thay vì buộc `beta` thay đổi ngầm theo
   `L_eff=1/t` như thí nghiệm cũ.
3. So sánh **cấu hình độ cong chưa dò** và **cấu hình sau dò** theo hai nhóm chỉ tiêu:
   - tối ưu: `f - f*`, số vòng, thời gian, số gradient;
   - dự đoán: Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC, F1, Precision,
     Recall và Log-loss trên tập test.
4. Chọn cấu hình theo `f-f*` trên train tại cùng ngân sách 150 vòng, sau đó sàng
   lọc độ ổn định tới 500 vòng để loại nghiệm thắng tạm thời nhưng dao động dài
   hạn. Validation dùng để theo dõi; test được giữ ngoài quá trình dò.

## Cấu trúc

```text
tu_update/
├── README.md
├── config.py                 lưới tham số và ngân sách
├── gd_agd_runner.py          vòng lặp GD/AGD có lưu checkpoint
├── evaluation.py             chia dữ liệu và metric phân loại
├── run_experiments.py        driver chạy toàn bộ thí nghiệm
├── make_artifacts.py         sinh bảng, hình và bản tóm tắt tự động
├── main_changes.md           kế hoạch cập nhật main.tex
├── tests/                    kiểm tra chia dữ liệu và vòng lặp
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

- “Baseline chưa dò” = cấu hình theo độ cong: GD dùng `t=1/L`; AGD dùng
  `t=1/L` và momentum theo `L, mu`.
- “Sau dò tối ưu” = cấu hình tốt nhất **trong lưới và ngân sách đã xét**.
- Không gọi một cấu hình là tối ưu toàn cục nếu chỉ được chọn từ một lưới hữu hạn.
- Nếu file `.npz` không có tập test, script tạo một phép chia cố định. Đây là kết
  quả bổ sung; bản cuối nên tái sinh dữ liệu với holdout trước mọi phép fit tiền xử lý.
- Báo cáo kết quả mới nằm tại `tu_update/artifacts/summary.md`; chưa tự động sửa
  `main.tex`.
