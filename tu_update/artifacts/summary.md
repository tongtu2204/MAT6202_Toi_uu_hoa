# Kết quả cập nhật GD và AGD

> Kết quả này được sinh tự động từ `gd_agd_results.json`; chưa sửa `main.tex`.

## Giao thức

- Chọn cấu hình theo `f-f*` trên train tại 150 vòng, sau khi loại ứng viên không ổn định tới 500 vòng.
- Tập test được giữ ngoài quá trình dò và chỉ dùng để báo cáo sau khi chọn.
- Thời gian là trung vị của 3 lần chạy.
- Kích thước: train=28,092, validation=7,023, test=11,706, d=40.

## Tham số được chọn

| Thuật toán | Baseline | Sau dò | Số cấu hình dò |
|---|---|---|---:|
| GD | t=0.5114 | t=3.3 | 25 |
| AGD | t=0.5114, β=0.937571 | t=2.2, β=0.88 | 252 |

## So sánh tại cùng ngân sách 150 vòng

| Thuật toán | Cấu hình | f−f* | ‖∇f‖ | Giây | Accuracy | F1 | ROC-AUC | PR-AUC | Log-loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GD | baseline | 0.0057 | 0.0084 | 0.1674 | 0.8981 | 0.5323 | 0.9133 | 0.6319 | 0.2456 |
| GD | tuned | 3.389e-04 | 0.0013 | 0.1757 | 0.8991 | 0.5424 | 0.9158 | 0.6405 | 0.2384 |
| AGD | baseline | 1.743e-05 | 4.761e-04 | 0.1823 | 0.8995 | 0.5463 | 0.9166 | 0.6421 | 0.2372 |
| AGD | tuned | 4.084e-09 | 5.507e-06 | 0.1629 | 0.8993 | 0.5443 | 0.9167 | 0.6424 | 0.2371 |

## Mức thay đổi tại ngân sách chọn

- **GD**: giảm objective gap 94.08%; thời gian thay đổi 4.96%; Accuracy +0.103 điểm %, F1 +1.009 điểm %.
- **AGD**: giảm objective gap 99.98%; thời gian thay đổi -10.64%; Accuracy -0.026 điểm %, F1 -0.204 điểm %.

## Trạng thái ở 3,000 vòng và hội tụ

| Thuật toán | Cấu hình | f−f* ở mốc cuối | Đạt ‖∇f‖≤1e-10 | Vòng hội tụ | Giây hội tụ |
|---|---|---:|---|---:|---:|
| GD | baseline | 4.460e-06 | Không | — | — |
| GD | tuned | 0.0000 | Có | 2,551 | 3.0257 |
| AGD | baseline | 2.776e-17 | Có | 603 | 0.7049 |
| AGD | tuned | 0.0000 | Có | 304 | 0.3573 |

## Tệp kết quả

- `gd_agd_results.json`: toàn bộ lưới, checkpoint và metric.
- `comparison.csv`: bảng dài để kiểm tra hoặc vẽ lại.
- `gd_step_search.png`: lưới bước GD.
- `agd_joint_search.png`: heatmap dò đồng thời `(t, beta)`.
- `equal_budget_convergence.png`: baseline và sau dò tại cùng ngân sách.
- `test_metric_delta.png`: thay đổi metric trên test.
