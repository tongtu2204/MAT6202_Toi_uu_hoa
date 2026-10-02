# AGD momentum biến thiên — dò thô

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`; 75.026 × 415; không chia lại.
- SHA-256: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- f* (Newton) = 0.212258252066; λ=0,001; w0=0.
- BLAS một luồng.
- Dò thô/tinh: 500 bước/cấu hình; chọn chuẩn gradient cuối nhỏ nhất, sau đó gap và số objective.
- Dò thô chỉ cần nhận diện vùng có dấu hiệu hội tụ, không yêu cầu đạt ngưỡng dừng.
- Dò tinh so sánh độ giảm sai số trong cùng 500 bước; ngưỡng dừng dùng cho lượt chạy cuối.
- Cấu hình chọn: `{'step': 0.1}`.

| Cấu hình | Bước | f−f* cuối | Gradient cuối | Thời gian (s) |
|---|---:|---:|---:|---:|
| {'step': 0.05} | 500 | 1.747e-05 | 4.441e-04 | 14.491 |
| {'step': 0.1} | 500 | 4.379e-06 | 2.056e-04 | 13.387 |
| {'step': 0.5} | 500 | 1.128e-02 | 1.672e-01 | 14.120 |
| {'step': 1.0} | 500 | 3.534e-01 | 3.449e-01 | 13.794 |
| {'step': 1.5} | 500 | 7.597e-01 | 4.209e-01 | 14.677 |
| {'step': 2.0} | 500 | 8.568e-01 | 3.904e-01 | 14.205 |
| {'step': 3.0} | 500 | 1.534e+00 | 3.114e-01 | 14.350 |
| {'step': 5.0} | 500 | 2.480e+00 | 1.699e-01 | 13.636 |
| {'step': 10.0} | 500 | 3.890e+00 | 1.795e-01 | 13.409 |

## Nhận xét dò thô và kiểm tra

- Momentum beta_j=(j-1)/(j+2) ở bước cập nhật j=1,...,500; tương đương beta_k=(k-2)/(k+1) trong công thức gốc với k=j+1.
- t=0,05 và t=0,1 có dấu hiệu tiến về nghiệm; t=0,5 còn dao động quanh gap khoảng 0,01, các bước lớn hơn dao động mạnh.
- Chọn t=0,1 theo cả chuẩn gradient cuối và gap. Dò thô nhằm chọn khoảng để dò tinh, không yêu cầu chạy tới hội tụ.
- Kiểm tra cấu trúc 4.500 bước, tái dựng 90 bước đầu theo momentum biến thiên, tính lại objective/gradient từ trọng số cuối cho cả 9 cấu hình: đạt; xem validation.json.
- Gradient trong trace ở điểm momentum; final_grad_norm ở đúng trọng số trả về.
- Lưới tinh tiếp theo t={0,05; 0,0625; 0,075; 0,0875; 0,1; 0,2; 0,3; 0,4; 0,5}.
