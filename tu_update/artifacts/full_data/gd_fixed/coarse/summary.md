# GD bước cố định — dò thô

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`; 75.026 × 415; không chia lại.
- SHA-256: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- f* (Newton) = 0.212258252066; λ=0,001; w0=0.
- Chuẩn dừng: ||gradient|| ≤ 1e-08; BLAS một luồng.
- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.
- Chỉ báo cáo hội tụ khi chuẩn gradient đạt ngưỡng; đường giảm không đủ để kết luận hội tụ.
- Cấu hình chọn: `{'step': 0.5}`.

| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) | Đạt ngưỡng |
|---|---:|---:|---:|---:|---|
| {'step': 0.05} | 500 | 1.593e-02 | 2.902e-02 | 13.939 | Chưa |
| {'step': 0.1} | 500 | 6.342e-03 | 1.298e-02 | 13.783 | Chưa |
| {'step': 0.5} | 500 | 5.836e-04 | 1.954e-03 | 14.213 | Chưa |
| {'step': 1.0} | 500 | 1.601e-01 | 6.671e-01 | 13.748 | Chưa |
| {'step': 1.5} | 500 | 3.503e-01 | 7.435e-01 | 13.553 | Chưa |
| {'step': 2.0} | 500 | 3.804e-01 | 7.742e-01 | 13.602 | Chưa |
| {'step': 3.0} | 500 | 3.011e-01 | 4.455e-01 | 14.009 | Chưa |
| {'step': 5.0} | 500 | 9.886e-01 | 7.182e-01 | 14.910 | Chưa |
| {'step': 10.0} | 500 | 3.610e+00 | 7.603e-01 | 14.367 | Chưa |
