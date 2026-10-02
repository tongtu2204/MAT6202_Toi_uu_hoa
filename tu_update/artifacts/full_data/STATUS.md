# Tiến độ thí nghiệm dữ liệu đầy đủ

Cập nhật ngày 02/10/2026. Mỗi giai đoạn hoàn tất phải được lưu Git trước khi chạy giai đoạn tiếp theo.

| Phương pháp | Dò thô | Dò tinh | Chạy cuối |
|---|---|---|---|
| GD cố định | Hoàn tất; vùng có dấu hiệu hội tụ, tốt nhất t=0,5 | Hoàn tất; chọn t=0,5 | Chưa chạy |
| GD backtracking | Đang chạy; đã lưu 7/9 cấu hình | Chưa chạy lại | Chưa chạy |
| AGD momentum hằng | Chưa chạy | Chưa chạy | Chưa chạy |
| AGD momentum biến thiên | Chưa chạy | Chưa chạy | Chưa chạy |

GD thô: 9 cấu hình × 500 bước. Các bước 0,05; 0,1; 0,5 có dấu hiệu hội tụ; không yêu cầu đạt ngưỡng dừng trong dò thô.
GD tinh: 9 cấu hình × 500 bước; chọn t=0,5, gap=5,836e-4, gradient=1,954e-3. Từ 0,625 trở lên xuất hiện dao động kéo dài.
Hình GD thô và tinh không có tiêu đề; chỉ giữ nhãn trục và chú giải.
Đã kiểm tra cả 9 lịch sử objective khớp lượt GD thô đã lưu trong 6e279cb (sai số tuyệt đối ≤1e-12).
Sáu kiểm tra thuật toán có sẵn đã đạt; script từ chối dữ liệu 35.115 × 40 do hash không khớp.
Chạy lại lệnh GD thô tiếp tục dùng checkpoint, không chạy lại 9 cấu hình.

## Bước tiếp theo

```bash
python tu_update/run_full_data_stage.py --method gd_backtracking --stage coarse
```

Lưới GD tinh đã chạy: {0,1; 0,2; 0,3; 0,4; 0,5; 0,625; 0,75; 0,875; 1}.
Khi GD backtracking dò thô xong, kiểm tra và lưu `gd_backtracking/coarse/` lên Git trước khi dò tinh.

Tệp đầy đủ có X_test rỗng; bảng chất lượng sau cùng là in-sample. Chưa sửa `.tex`.
