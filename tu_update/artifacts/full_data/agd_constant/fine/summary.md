# AGD momentum hằng — dò tinh

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`; 75.026 × 415; không chia lại.
- SHA-256: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- f* (Newton) = 0.212258252066; λ=0,001; w0=0.
- Chuẩn dừng của lượt chạy cuối: ||gradient|| ≤ 1e-08; BLAS một luồng.
- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.
- Dò thô chỉ cần nhận diện vùng có dấu hiệu hội tụ, không yêu cầu đạt ngưỡng dừng.
- Dò tinh so sánh độ giảm sai số trong cùng 500 bước; ngưỡng dừng dùng cho lượt chạy cuối.
- Cấu hình chọn: `{'step': 0.2, 'beta': 0.9832713994805439}`.

| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) | Gradient ≤ 1e-8 |
|---|---:|---:|---:|---:|---|
| {'step': 0.05, 'beta': 0.9832713994805439} | 500 | 6.011e-04 | 1.821e-03 | 14.098 | Chưa |
| {'step': 0.0625, 'beta': 0.9832713994805439} | 500 | 2.803e-04 | 1.374e-03 | 14.021 | Chưa |
| {'step': 0.07500000000000001, 'beta': 0.9832713994805439} | 500 | 1.697e-04 | 1.248e-03 | 15.127 | Chưa |
| {'step': 0.08750000000000001, 'beta': 0.9832713994805439} | 500 | 1.217e-04 | 1.099e-03 | 14.547 | Chưa |
| {'step': 0.1, 'beta': 0.9832713994805439} | 500 | 1.026e-04 | 1.041e-03 | 14.379 | Chưa |
| {'step': 0.2, 'beta': 0.9832713994805439} | 500 | 7.747e-05 | 9.304e-04 | 14.504 | Chưa |
| {'step': 0.30000000000000004, 'beta': 0.9832713994805439} | 500 | 1.074e-04 | 1.159e-03 | 14.516 | Chưa |
| {'step': 0.4, 'beta': 0.9832713994805439} | 500 | 2.145e-03 | 1.015e-01 | 13.924 | Chưa |
| {'step': 0.5, 'beta': 0.9832713994805439} | 500 | 1.128e-02 | 1.678e-01 | 14.021 | Chưa |

## Nhận xét dò tinh và kiểm tra

- t=0,2 tốt nhất trong lưới theo cả chuẩn gradient cuối và sai số hàm mục tiêu, beta giữ nguyên.
- So với cấu hình chọn ở dò thô t=0,1, gap giảm 24,49%, chuẩn gradient cuối giảm 10,63% sau cùng 500 bước; chưa đạt gradient 1e-8.
- t=0,4 và t=0,5 còn dao động rõ.
- Kiểm tra cấu trúc 4.500 bước, tái dựng độc lập 90 bước đầu, tính lại objective/gradient từ trọng số cuối của cả 9 cấu hình: đạt; xem validation.json.
- Ba cấu hình trùng dò thô t={0,05; 0,1; 0,5} tái lập trace số học và trọng số tới sai số 1e-12.
- Gradient trong trace ở điểm momentum; final_grad_norm ở đúng trọng số trả về.
