# Tiến độ thí nghiệm dữ liệu đầy đủ

Cập nhật ngày 02/10/2026. Dữ liệu: ridge.npz của Tài, 75.026 × 415; không chia lại.
Dò thô/tinh dùng ngân sách 500 bước mỗi cấu hình. Dò thô chỉ cần dấu hiệu hội tụ;
ngưỡng gradient 1e-8 dành cho lượt chạy cuối. Hình không có tiêu đề.

| Phương pháp | Dò thô | Dò tinh | Chạy cuối |
|---|---|---|---|
| GD cố định | Hoàn tất; 9 cấu hình; chọn t=0,5 | Hoàn tất; 9 cấu hình; chọn t=0,5 | Chưa chạy |
| GD backtracking | Hoàn tất; 9 cấu hình; chọn rho=0,9, c=0,5 | Hoàn tất; 8 cấu hình đủ 500 bước, 1 bị loại | Chưa chạy |
| AGD momentum hằng | Hoàn tất; 9 cấu hình; chọn t=0,1 | Chưa chạy | Chưa chạy |
| AGD momentum biến thiên | Chưa chạy | Chưa chạy | Chưa chạy |

## GD backtracking dò tinh

- Giữ t0=20. Lưới: rho={0,5; 0,7; 0,9} × c={0,1; 0,5; 0,99}.
- Ở biên rho, dò một phía trong khoảng thô hợp lệ để giữ đủ 9 cặp.
- Cặp (0,9; 0,99) không tìm được bước Armijo trong 60 lần thử, dừng ở k=0 và bị loại.
- Đã kiểm tra 4.000 bước chấp nhận của 8 cấu hình, xác minh riêng 60 lần thử thất bại,
  tái tính objective và gradient từ trọng số; các cấu hình trùng dò thô tái lập đúng.
- Các lưới đã chạy trước đó và mã thực thi GD/AGD không thay đổi khi sửa cách xử lý biên rho.
- Bảy kiểm tra có sẵn/bổ sung đạt, gồm kiểm tra 9 cặp hợp lệ tại biên rho.

| Tiêu chí sau 500 bước | rho | c | f−f* | Gradient cuối |
|---|---:|---:|---:|---:|
| Gradient nhỏ nhất; cấu hình chọn theo giao thức hiện tại | 0,9 | 0,5 | 1,690e-5 | 2,547e-4 |
| Sai số hàm mục tiêu nhỏ nhất | 0,7 | 0,5 | 4,486e-6 | 2,951e-4 |

Cần phân biệt hai tiêu chí; không gọi một cấu hình là tốt nhất ở mọi chỉ số.
Checkpoint đã được lưu lên Git trong lúc chạy trước khi lưu giai đoạn hoàn chỉnh.

## AGD momentum hằng dò thô

- Lưới t={0,05; 0,1; 0,5; 1; 1,5; 2; 3; 5; 10}; beta=0,9832713994805439.
- Cả 9 cấu hình đủ 500 bước, chưa đạt ngưỡng gradient 1e-8.
- t=0,05 và t=0,1 có dấu hiệu tiến về nghiệm; t=0,5 còn dao động ở khoảng gap 0,011–0,016 trong 100 bước cuối. Các bước lớn hơn dao động mạnh.
- Chọn t=0,1 theo cả chuẩn gradient cuối và gap: f-f*=1,0258698819315404e-4; gradient=1,0410427659084478e-3.
- Đã kiểm tra cấu trúc 4.500 bước ghi lại, tái dựng độc lập 10 bước đầu/cấu hình (90 bước), tính lại objective và gradient từ trọng số cuối cho cả 9 cấu hình; validation.json đạt.
- Trace gradient của AGD ở điểm momentum; final_grad_norm ở đúng trọng số cuối.
- Hình không có tiêu đề. Đã lưu checkpoint lên Git trong lúc chạy.
- Dò tinh tiếp theo quanh t=0,1 trong khoảng [0,05; 0,5], giữ beta: t={0,05; 0,0625; 0,075; 0,0875; 0,1; 0,2; 0,3; 0,4; 0,5}.

## Bước tiếp theo

```bash
python tu_update/run_full_data_stage.py --method agd_constant --stage fine
```

Sau khi xong dò tinh, kiểm tra và lưu kết quả lên Git trước khi chuyển sang phương pháp tiếp theo.
Tập test gốc rỗng; chỉ số mô hình cuối cùng là in-sample. Chưa tính chỉ số phân loại;
chưa chạy lượt final và chưa sửa `.tex`.
