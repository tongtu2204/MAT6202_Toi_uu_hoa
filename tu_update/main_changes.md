# Ghi chú cho lần sửa `main.tex` sau này

Hiện tại **chưa sửa `main.tex`**. Chỉ cập nhật sau khi kết quả trong
`artifacts/summary.md` được duyệt.

Thứ tự trình bày dự kiến:

1. GD bước cố định: dò thô và dò tinh, mỗi cấu hình 500 vòng.
2. GD backtracking: dò `(rho, c)` với cùng ngân sách.
3. So sánh nội bộ hai GD và chọn một phương án.
4. AGD momentum cố định: dò bước.
5. AGD `beta_k=(k-2)/(k+1)`: dò bước.
6. So sánh nội bộ hai AGD và chọn một phương án.
7. Chỉ sau đó so GD thắng với AGD thắng.
8. So sánh metric mô hình trước tối ưu, sau tối ưu và với scikit-learn.
