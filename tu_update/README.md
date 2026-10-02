# Thí nghiệm GD/AGD trên dữ liệu đầy đủ của Tài

## Đầu vào và giao thức hiện tại

- Dùng đúng `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz`: **75.026 × 415**.
- SHA-256 bắt buộc: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- Tệp gốc: https://drive.google.com/file/d/1mehXGDef4XyBnGdZp4gG89EbDUnAbU6Q/view.
  Tải về, đặt đúng đường dẫn trên; script xác minh hash trước khi chạy.
- Tệp này có `X_test` rỗng. Chỉ số chất lượng là **in-sample**, không phải test độc lập.
- Không dùng `data/ridge.npz` thay thế: đó là dữ liệu khác, 35.115 × 40 cho train.
- Giữ hàm mục tiêu logistic của Tài, λ=0,001, intercept không phạt, w0=0 và công thức momentum.
- Bốn phương pháp: GD cố định; GD backtracking; AGD momentum hằng; AGD momentum biến thiên.
- Dò thô/tinh: 500 bước mỗi cấu hình. Chọn chuẩn gradient cuối nhỏ nhất, rồi objective gap và số objective.
- Chạy cuối: từ w0 đến `||gradient|| <= 1e-8`, trần 50.000 bước; 3 lần đo thời gian.
- BLAS một luồng. Không sửa `.tex` trong lượt thí nghiệm này.

## Chạy từng giai đoạn và lưu Git

Từ thư mục gốc repo, chạy **một lệnh mỗi lần**, kiểm tra kết quả và lưu Git trước bước tiếp theo:

```bash
python tu_update/run_full_data_stage.py --method gd_fixed --stage coarse
python tu_update/run_full_data_stage.py --method gd_fixed --stage fine
python tu_update/run_full_data_stage.py --method gd_backtracking --stage coarse
python tu_update/run_full_data_stage.py --method gd_backtracking --stage fine
python tu_update/run_full_data_stage.py --method agd_constant --stage coarse
python tu_update/run_full_data_stage.py --method agd_constant --stage fine
python tu_update/run_full_data_stage.py --method agd_dynamic --stage coarse
python tu_update/run_full_data_stage.py --method agd_dynamic --stage fine
```

Sau khi cả bốn phương pháp đã khóa cấu hình, chạy riêng `--stage final` cho từng phương pháp.
Sau đó tổng hợp so sánh cả bốn và bảng chất lượng mô hình.

Mỗi cấu hình hoàn thành được ghi ngay vào `results.json` bằng thay thế file nguyên tử.
Chạy lại cùng lệnh sẽ tiếp tục từ cấu hình kế tiếp; không chạy lại phần đã lưu.
Checkpoint kiểm tra hash dữ liệu, mã nguồn và lưới tham số. Nếu khác, cần lưu riêng
kết quả cũ trước khi tạo lượt mới; không tự trộn checkpoint.

Sau mỗi giai đoạn, lưu các file liên quan, ví dụ:

```bash
git add tu_update/artifacts/full_data/reference.json tu_update/artifacts/full_data/gd_fixed/coarse
git commit -m "Save full-data GD fixed coarse results"
git push origin main
```

## Kết quả

Kết quả mới ở `tu_update/artifacts/full_data/<method>/<stage>/`:

- `results.json`: dữ liệu/giao thức/môi trường, tham số, trọng số, lịch sử objective,
  gradient, thời gian và bước Armijo; `status=complete` khi xong giai đoạn.
- `search_results.csv`: bảng cấu hình và chỉ số tối ưu.
- `summary.md`: tóm tắt kết quả thật, cấu hình chọn và trạng thái hội tụ.
- `convergence.png`: **một biểu đồ riêng**, chỉ vẽ `f(w_k)-f*` trên trục log.

Tám giai đoạn dò tạo tám hình riêng. Sau lượt final sẽ bổ sung một hình so sánh
cả bốn phương pháp. Không dùng heatmap thay đường hội tụ backtracking.
`reference.json` lưu nghiệm Newton và hệ số momentum hằng tính theo giao thức cũ.

Các artifact ở trực tiếp `tu_update/artifacts/` là lượt cũ trên dữ liệu khác.
`artifacts/full_data_gd_coarse/` là GD thô đầy đủ đã lưu trong commit `6e279cb`;
lượt chạy mới có checkpoint và đường dẫn riêng như trên.
`run_experiments.py` là runner cũ chạy toàn bộ trên `data/ridge.npz`; không dùng
lệnh đó cho lượt chạy đầy đủ mới.
