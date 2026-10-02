# AGD momentum hằng — dò thô

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`; 75.026 × 415; không chia lại.
- SHA-256: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- f* (Newton) = 0.212258252066; λ=0,001; w0=0.
- Chuẩn dừng của lượt chạy cuối: ||gradient|| ≤ 1e-08; BLAS một luồng.
- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.
- Dò thô chỉ cần nhận diện vùng có dấu hiệu hội tụ, không yêu cầu đạt ngưỡng dừng.
- Dò tinh so sánh độ giảm sai số trong cùng 500 bước; ngưỡng dừng dùng cho lượt chạy cuối.
- Cấu hình chọn: `{'step': 0.1, 'beta': 0.9832713994805439}`.

| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) | Gradient ≤ 1e-8 |
|---|---:|---:|---:|---:|---|
| {'step': 0.05, 'beta': 0.9832713994805439} | 500 | 6.011e-04 | 1.821e-03 | 14.615 | Chưa |
| {'step': 0.1, 'beta': 0.9832713994805439} | 500 | 1.026e-04 | 1.041e-03 | 14.154 | Chưa |
| {'step': 0.5, 'beta': 0.9832713994805439} | 500 | 1.128e-02 | 1.678e-01 | 14.409 | Chưa |
| {'step': 1.0, 'beta': 0.9832713994805439} | 500 | 2.432e-01 | 2.528e-01 | 13.828 | Chưa |
| {'step': 1.5, 'beta': 0.9832713994805439} | 500 | 4.667e-01 | 1.396e-01 | 14.358 | Chưa |
| {'step': 2.0, 'beta': 0.9832713994805439} | 500 | 8.762e-01 | 3.048e-01 | 14.189 | Chưa |
| {'step': 3.0, 'beta': 0.9832713994805439} | 500 | 1.338e+00 | 3.098e-01 | 14.861 | Chưa |
| {'step': 5.0, 'beta': 0.9832713994805439} | 500 | 1.909e+00 | 1.356e-01 | 14.305 | Chưa |
| {'step': 10.0, 'beta': 0.9832713994805439} | 500 | 5.282e+00 | 9.723e-02 | 14.848 | Chưa |

## Nhận xét dò thô và kiểm tra

- t=0,05 và t=0,1 có dấu hiệu tiến về nghiệm; t=0,5 còn dao động, các bước lớn hơn dao động mạnh.
- t=0,1 tốt nhất theo cả gradient cuối và sai số hàm mục tiêu, chưa đạt ngưỡng 1e-8.
- Kiểm tra cấu trúc 4.500 bước ghi lại, tái dựng độc lập 90 bước đầu, tính lại objective/gradient từ trọng số cuối của cả 9 cấu hình: đạt; xem validation.json.
- Gradient trong trace được đo ở điểm momentum; final_grad_norm được đo ở trọng số trả về.
- Khoảng dò tinh tiếp theo [0,05; 0,5], giữ beta và t=0,1 ở giữa lưới 9 điểm.
