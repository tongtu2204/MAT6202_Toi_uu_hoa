# Kế hoạch cập nhật `main.tex`

Chưa sửa trực tiếp bài trình bày ở bước khởi tạo thư mục. Sau khi chạy và khóa số
liệu, phần GD/AGD sẽ được cập nhật theo thứ tự:

1. **GD — lưới dò mở rộng:** thay hình/bảng năm điểm bằng toàn bộ lưới rộng, các
   checkpoint 50/150/500/1000/3000 và một vòng dò tinh quanh vùng thắng.
2. **AGD — tách lý thuyết và thực nghiệm:** trình bày riêng `t=1/L, beta_theory`,
   dò bước với beta ghép cặp, rồi dò tinh `(t, beta)`.
3. **Bảng trước/sau dò tối ưu:** cùng ngân sách, báo cáo `f-f*`, gradient norm,
   thời gian và số vòng.
4. **Bảng chất lượng dự đoán:** Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC,
   F1, Recall, Precision và Log-loss trên test.
5. **Sửa diễn giải:**
   - thay “tối ưu” bằng “tốt nhất trong lưới và ngân sách đã xét”;
   - không dùng chứng nhận bước `1/L` cho AGD dò tay ngoài miền bảo đảm;
   - phân biệt `kappa_bound` với `kappa_star`;
   - sửa điều kiện Armijo thành `0 < c < 1`;
   - sửa các số mục và mâu thuẫn ở slide tổng hợp 4.16.

