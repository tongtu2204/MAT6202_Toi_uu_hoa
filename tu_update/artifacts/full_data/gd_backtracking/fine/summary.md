# GD backtracking — dò tinh

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
| {'t0': 20.0, 'rho': 0.5, 'c': 0.1} | 500 | 1.472e-04 | 2.766e-03 | 38.749 | Chưa |
| {'t0': 20.0, 'rho': 0.5, 'c': 0.5} | 500 | 7.997e-05 | 1.000e-03 | 37.168 | Chưa |
| {'t0': 20.0, 'rho': 0.5, 'c': 0.99} | 500 | 9.361e-04 | 2.754e-03 | 47.354 | Chưa |
| {'t0': 20.0, 'rho': 0.7, 'c': 0.1} | 500 | 6.284e-04 | 4.166e-03 | 65.804 | Chưa |
| {'t0': 20.0, 'rho': 0.7, 'c': 0.5} | 500 | 4.486e-06 | 2.951e-04 | 58.326 | Chưa |
| {'t0': 20.0, 'rho': 0.7, 'c': 0.99} | 500 | 6.049e-04 | 2.035e-03 | 76.750 | Chưa |
| {'t0': 20.0, 'rho': 0.9, 'c': 0.1} | 500 | 6.606e-04 | 6.242e-03 | 191.097 | Chưa |
| {'t0': 20.0, 'rho': 0.9, 'c': 0.5} | 500 | 1.690e-05 | 2.547e-04 | 165.659 | Chưa |
| {'t0': 20.0, 'rho': 0.9, 'c': 0.99} | 0 | 4.809e-01 | 8.637e-01 | 0.027 | Chưa |

Cấu hình `{'t0': 20.0, 'rho': 0.9, 'c': 0.99}` dừng sau 0 bước cập nhật: Armijo không tìm được bước trong giới hạn thử. Không đưa cấu hình này vào chọn tham số.
Dấu × ở k=0 trên hình là lần thử dừng trước bước cập nhật đầu tiên.

Theo tiêu chí chuẩn gradient cuối: chọn `{'t0': 20.0, 'rho': 0.9, 'c': 0.5}`, gradient=2.547e-04, gap=1.690e-05.
Cấu hình có gap nhỏ nhất là `{'t0': 20.0, 'rho': 0.7, 'c': 0.5}`: gap=4.486e-06, gradient=2.951e-04.
Hai tiêu chí cho hai cấu hình khác nhau; giữ nguyên tiêu chí chọn đã dùng ở các giai đoạn trước.
