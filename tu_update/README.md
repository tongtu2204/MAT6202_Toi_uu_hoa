# Cập nhật thí nghiệm GD và AGD

Thư mục độc lập để sửa phần GD/AGD theo nhận xét của giảng viên. Mã cũ và
`main.tex` không bị chỉnh sửa.

## Giao thức đã chốt

1. Dùng trực tiếp `data/ridge.npz`: `(X_train, y_train)` cho tối ưu và
   `(X_test, y_test)` chỉ để đánh giá sau khi khóa tham số.
2. Không chia lại tập train.
3. Bốn phép dò:
   - GD: độ dài bước cố định `t`;
   - GD backtracking: `(rho, c)`, giữ `t0=20`;
   - AGD momentum cố định: dò `t`, giữ beta tính từ `(L, mu)`;
   - AGD với `beta_k=(k-2)/(k+1)`: dò `t`.
4. Mỗi ứng viên ở lưới thô và lưới tinh chạy đúng 500 vòng.
5. Sau khi chọn tham số, khởi tạo lại từ `w0=0` và chạy tới
   `||grad|| <= 1e-8`. Ngưỡng này tránh đòi Armijo phân biệt mức giảm objective
   nhỏ hơn độ chính xác máy; trần 50.000 vòng chỉ là van an toàn.
6. So GD với GD trước; AGD với AGD sau; cuối cùng chỉ so hai người thắng.
7. Báo cáo Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC,
   PR-AUC và Log-loss trước/sau tối ưu.

## Chạy

Từ thư mục gốc repository:

```bash
python tu_update/run_experiments.py
```

Kết quả được ghi vào `tu_update/artifacts/`.
