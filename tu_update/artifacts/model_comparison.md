# Chất lượng mô hình trên tập test

| Mô hình | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Log-loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chưa tối ưu (w0=0) | 0.1369 | 0.5000 | 0.1369 | 1.0000 | 0.2409 | 0.5000 | 0.1369 | 0.6931 |
| GD bước cố định | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 |
| GD backtracking | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 |
| AGD momentum cố định | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 |
| AGD momentum (k-2)/(k+1) | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 |
| LogisticRegression (scikit-learn) | 0.8997 | 0.7070 | 0.7173 | 0.4417 | 0.5467 | 0.9168 | 0.6416 | 0.2370 |

Các chỉ số được làm tròn đến bốn chữ số thập phân. Log-loss thấp hơn là tốt hơn; các chỉ số còn lại cao hơn là tốt hơn.
Các phương pháp sau hội tụ cho chất lượng dự đoán gần như bằng nhau; khác biệt chính nằm ở tốc độ hội tụ.
Giá trị đầy đủ của cả train và test được lưu trong `model_metrics.csv`.
