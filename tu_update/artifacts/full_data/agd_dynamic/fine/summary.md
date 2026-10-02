# AGD momentum biến thiên — dò tinh

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`; 75.026 × 415; không chia lại.
- SHA-256: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- f* (Newton) = 0.212258252066; λ=0,001; w0=0.
- BLAS một luồng.
- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.
- Dò thô chỉ cần nhận diện vùng có dấu hiệu hội tụ, không yêu cầu đạt ngưỡng dừng.
- Dò tinh so sánh độ giảm sai số trong cùng 500 bước; ngưỡng dừng dùng cho lượt chạy cuối.
- Cấu hình chọn: `{'step': 0.30000000000000004}`.

| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) |
|---|---:|---:|---:|---:|
| {'step': 0.05} | 500 | 1.747e-05 | 4.441e-04 | 13.736 |
| {'step': 0.0625} | 500 | 1.471e-05 | 3.549e-04 | 13.466 |
| {'step': 0.07500000000000001} | 500 | 1.165e-05 | 3.223e-04 | 13.367 |
| {'step': 0.08750000000000001} | 500 | 7.592e-06 | 2.633e-04 | 13.223 |
| {'step': 0.1} | 500 | 4.379e-06 | 2.056e-04 | 13.835 |
| {'step': 0.2} | 500 | 1.566e-06 | 1.044e-04 | 14.064 |
| {'step': 0.30000000000000004} | 500 | 6.205e-07 | 6.577e-05 | 14.961 |
| {'step': 0.4} | 500 | 2.009e-03 | 1.028e-01 | 14.229 |
| {'step': 0.5} | 500 | 1.128e-02 | 1.672e-01 | 14.103 |

## Nhận xét dò tinh và kiểm tra

- Chọn t=0,3 theo cả chuẩn gradient cuối và gap trong lưới 9 điểm. Giữ lịch momentum beta_j=(j-1)/(j+2), j=1,...,500.
- So với cấu hình chọn ở dò thô t=0,1, gap giảm 85.83%, chuẩn gradient cuối giảm 68.02% sau cùng 500 bước.
- t=0,4 và t=0,5 còn dao động rõ.
- Kiểm tra cấu trúc 4.500 bước, tái dựng độc lập 90 bước đầu, tính lại objective/gradient từ trọng số cuối của cả 9 cấu hình: đạt; xem validation.json.
- Ba cấu hình trùng dò thô t={0,05; 0,1; 0,5} tái lập trace số học và trọng số tới sai số 1e-12.
- Gradient trong trace ở điểm momentum; final_grad_norm ở đúng trọng số trả về.
- Cấu hình được khóa để chạy lại từ w0=0 tới hội tụ ở lượt final.
