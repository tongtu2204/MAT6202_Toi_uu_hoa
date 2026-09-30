# Artifacts của phần cập nhật

Thư mục này sẽ chứa kết quả do `tu_update/run_experiments.py` sinh ra:

- `gd_agd_quick.json`: smoke test với lưới rút gọn;
- `gd_agd_results.json`: toàn bộ lưới, checkpoint và metric trước/sau;
- `summary.md` và `comparison.csv`: bảng kết quả để duyệt;
- `gd_step_search.png`, `agd_joint_search.png`: kết quả dò tham số;
- `equal_budget_convergence.png`: so sánh tại cùng ngân sách;
- `test_metric_delta.png`: thay đổi metric trên tập test.

Không chép kết quả cũ vào đây. Mọi số đưa vào `main.tex` mới phải truy được về
`gd_agd_results.json`.
