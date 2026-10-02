# GD backtracking — dò thô

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`; 75.026 × 415; không chia lại.
- SHA-256: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- f* (Newton) = 0.212258252066; λ=0,001; w0=0.
- Chuẩn dừng của lượt chạy cuối: ||gradient|| ≤ 1e-08; BLAS một luồng.
- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.
- Dò thô chỉ cần nhận diện vùng có dấu hiệu hội tụ, không yêu cầu đạt ngưỡng dừng.
- Dò tinh so sánh độ giảm sai số trong cùng 500 bước; ngưỡng dừng dùng cho lượt chạy cuối.
- Cấu hình chọn: `{'t0': 20.0, 'rho': 0.9, 'c': 0.5}`.

| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) | Gradient ≤ 1e-8 |
|---|---:|---:|---:|---:|---|
| {'t0': 20.0, 'rho': 0.1, 'c': 0.0001} | 500 | 1.473e-04 | 9.057e-04 | 22.515 | Chưa |
| {'t0': 20.0, 'rho': 0.1, 'c': 0.1} | 500 | 1.614e-04 | 8.878e-04 | 21.906 | Chưa |
| {'t0': 20.0, 'rho': 0.1, 'c': 0.5} | 500 | 1.713e-04 | 8.786e-04 | 21.580 | Chưa |
| {'t0': 20.0, 'rho': 0.5, 'c': 0.0001} | 500 | 2.095e-04 | 2.757e-03 | 39.519 | Chưa |
| {'t0': 20.0, 'rho': 0.5, 'c': 0.1} | 500 | 1.472e-04 | 2.766e-03 | 38.883 | Chưa |
| {'t0': 20.0, 'rho': 0.5, 'c': 0.5} | 500 | 7.997e-05 | 1.000e-03 | 38.162 | Chưa |
| {'t0': 20.0, 'rho': 0.9, 'c': 0.0001} | 500 | 6.539e-04 | 1.161e-02 | 187.619 | Chưa |
| {'t0': 20.0, 'rho': 0.9, 'c': 0.1} | 500 | 6.606e-04 | 6.242e-03 | 186.829 | Chưa |
| {'t0': 20.0, 'rho': 0.9, 'c': 0.5} | 500 | 1.690e-05 | 2.547e-04 | 162.039 | Chưa |
