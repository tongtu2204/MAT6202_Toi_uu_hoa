# GD bước cố định — dò tinh

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`; 75.026 × 415; không chia lại.
- SHA-256: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- f* (Newton) = 0.212258252066; λ=0,001; w0=0.
- Chuẩn dừng của lượt chạy cuối: ||gradient|| ≤ 1e-08; BLAS một luồng.
- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.
- Dò thô chỉ cần nhận diện vùng có dấu hiệu hội tụ, không yêu cầu đạt ngưỡng dừng.
- Dò tinh so sánh độ giảm sai số trong cùng 500 bước; ngưỡng dừng dùng cho lượt chạy cuối.
- Cấu hình chọn: `{'step': 0.5}`.

| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) | Gradient ≤ 1e-8 |
|---|---:|---:|---:|---:|---|
| {'step': 0.1} | 500 | 6.342e-03 | 1.298e-02 | 14.302 | Chưa |
| {'step': 0.2} | 500 | 2.471e-03 | 5.855e-03 | 15.310 | Chưa |
| {'step': 0.30000000000000004} | 500 | 1.366e-03 | 3.653e-03 | 14.431 | Chưa |
| {'step': 0.4} | 500 | 8.652e-04 | 2.591e-03 | 14.002 | Chưa |
| {'step': 0.5} | 500 | 5.836e-04 | 1.954e-03 | 13.707 | Chưa |
| {'step': 0.625} | 500 | 2.497e-02 | 3.946e-01 | 13.552 | Chưa |
| {'step': 0.75} | 500 | 7.097e-02 | 5.501e-01 | 13.435 | Chưa |
| {'step': 0.875} | 500 | 9.468e-02 | 6.239e-01 | 13.682 | Chưa |
| {'step': 1.0} | 500 | 1.601e-01 | 6.671e-01 | 14.536 | Chưa |
