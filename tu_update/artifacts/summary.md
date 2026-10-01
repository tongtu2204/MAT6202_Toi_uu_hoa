# Kết quả cập nhật GD và AGD

> Sinh tự động từ `results.json`. `main.tex` chưa được sửa.

## Giao thức

- Dữ liệu: train `[35115, 40]`, test `[11706, 40]` từ `data/ridge.npz`.
- Không chia lại train; test không tham gia dò tham số.
- Mỗi ứng viên ở lưới thô và lưới tinh chạy đúng **500 vòng**.
- Sau khi khóa tham số, chạy lại từ `w0=0` tới `||grad|| <= 1e-08`.
- Trần kỹ thuật: 50,000 vòng; thời gian lấy trung vị 3 lần.

## Tham số được chọn sau dò 500 vòng

| Biến thể | Tham số | Số cấu hình | Gradient norm cuối |
|---|---|---:|---:|
| GD bước cố định | `step=3` | 18 | 1.478e-04 |
| GD backtracking | `t0=20, rho=0.1, c=0.1` | 18 | 2.068e-06 |
| AGD momentum cố định | `step=1.625, beta=0.938678` | 18 | 3.770e-10 |
| AGD $\beta_k=(k-2)/(k+1)$ | `step=1.75` | 18 | 3.749e-06 |

## Chạy chính thức đến hội tụ

| Biến thể | Hội tụ | Số vòng | Thời gian trung vị (s) | Gradient eval | Objective eval | f−f* |
|---|---|---:|---:|---:|---:|---:|
| GD bước cố định | Có | 2.128 | 3.1557 | 2.129 | 2.129 | 2.554e-14 |
| GD backtracking | Có | 859 | 1.6723 | 860 | 1.457 | 1.413e-14 |
| AGD momentum cố định | Có | 432 | 0.5958 | 434 | 433 | 1.382e-14 |
| AGD $\beta_k=(k-2)/(k+1)$ | Có | 1.891 | 2.6163 | 1.893 | 1.892 | 8.549e-15 |

## So sánh theo ba tầng

1. **Nội bộ GD:** `GD backtracking` thắng theo thời gian tới hội tụ.
2. **Nội bộ AGD:** `AGD momentum cố định` thắng theo thời gian tới hội tụ.
3. **GD với AGD:** `AGD momentum cố định` là phương án nhanh hơn trong hai người thắng.

## Chất lượng mô hình trên test

| Mô hình | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Log-loss | ||w−w*|| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chưa tối ưu (w0=0) | 0.1369 | 0.5000 | 0.1369 | 1.0000 | 0.2409 | 0.5000 | 0.1369 | 0.6931 | 4.2156 |
| GD bước cố định | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 | 5.107e-06 |
| GD backtracking | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 | 3.791e-06 |
| AGD momentum cố định | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 | 3.017e-06 |
| AGD momentum (k-2)/(k+1) | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 | 2.499e-06 |
| LogisticRegression (scikit-learn) | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 | 5.885e-06 |

## Diễn giải chính

- So với `w0=0`, phương án cuối tăng Accuracy **76.28 điểm %**, F1 **30.58 điểm %** và ROC-AUC **41.68 điểm %**.
- Khi các thuật toán đều hội tụ về cùng nghiệm Ridge logistic, metric dự đoán gần như trùng nhau; khác biệt chính nằm ở số vòng, thời gian và số lần đánh giá hàm.
- Accuracy không được dùng một mình vì tỷ lệ churn chỉ khoảng 13,7%; cần đọc cùng Balanced Accuracy, F1, ROC-AUC, PR-AUC và Log-loss.

## Tệp kết quả

- `results.json`: toàn bộ lưới, quỹ đạo và metric.
- `search_results.csv`: toàn bộ ứng viên tham số.
- `optimization_results.csv`: bốn cấu hình chạy tới hội tụ.
- `model_metrics.csv`: metric train/test trước và sau tối ưu.
- `search_gd_fixed.png`: dò thô/tinh của GD bước cố định.
- `search_gd_backtracking.png`: dò thô/tinh của GD backtracking.
- `search_agd_constant.png`: dò thô/tinh của AGD momentum cố định.
- `search_agd_dynamic.png`: dò thô/tinh của AGD momentum động.
- `hierarchical_comparison.png`: so GD, so AGD, rồi GD–AGD.
- `all_methods_comparison.png`: cả bốn đường theo số vòng và thời gian.
- `model_comparison.png`: metric test trước và sau tối ưu.
