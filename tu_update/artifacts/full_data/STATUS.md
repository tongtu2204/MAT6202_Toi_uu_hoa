# Tiến độ thí nghiệm dữ liệu đầy đủ

Cập nhật ngày 02/10/2026. Mỗi giai đoạn hoàn tất phải được lưu Git trước khi chạy giai đoạn tiếp theo.

| Phương pháp | Dò thô | Dò tinh | Chạy cuối |
|---|---|---|---|
| GD cố định | Hoàn tất; chọn t=0,5 | Chưa chạy | Chưa chạy |
| GD backtracking | Chưa chạy lại | Chưa chạy lại | Chưa chạy |
| AGD momentum hằng | Chưa chạy | Chưa chạy | Chưa chạy |
| AGD momentum biến thiên | Chưa chạy | Chưa chạy | Chưa chạy |

GD thô: 9 cấu hình × 500 bước, không cấu hình nào đạt chuẩn gradient 1e-8.
Đã kiểm tra cả 9 lịch sử objective khớp lượt GD thô đã lưu trong 6e279cb (sai số tuyệt đối ≤1e-12).
Sáu kiểm tra thuật toán có sẵn đã đạt; script từ chối dữ liệu 35.115 × 40 do hash không khớp.
Chạy lại lệnh GD thô tiếp tục dùng checkpoint, không chạy lại 9 cấu hình.

## Bước tiếp theo

```bash
python tu_update/run_full_data_stage.py --method gd_fixed --stage fine
```

Lưới tinh sinh từ lưới thô: {0,1; 0,2; 0,3; 0,4; 0,5; 0,625; 0,75; 0,875; 1}.
Khi xong, kiểm tra và lưu `gd_fixed/fine/` lên Git, rồi chuyển sang GD backtracking dò thô.

Tệp đầy đủ có X_test rỗng; bảng chất lượng sau cùng là in-sample. Chưa sửa `.tex`.
