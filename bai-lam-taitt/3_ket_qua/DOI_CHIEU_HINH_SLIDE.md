# Đối chiếu hình vẽ ↔ slide ↔ lệnh tái lập

Mỗi hình trong `hinh_ve/` được liệt kê kèm slide dùng nó và lệnh sinh ra nó.
Chạy lệnh trong thư mục `../5_ma_nguon_tai_lap/`; hình ghi đè vào `artifacts/figures/`.

**25/38 hình được bài trình bày dùng.** 13 hình còn lại là kết quả thăm dò, giữ lại để đối chiếu.

## Hình dùng trong bài trình bày

| Slide | Hình | Lệnh sinh |
|---|---|---|
| **4.1** Dữ liệu chuẩn hóa và không chuẩn hóa | `standardization_contrast.png` | `python run_experiments.py std` |
| **4.3** Newton — dò bước cố định | `tune_newton_fixed.png` | `python run_tuning.py newton-fixed` |
| **4.4** Newton — dò (ρ, c) của backtracking | `tune_newton_bt_grid.png` | `python run_tuning.py newton-bt` |
| **4.5** Newton — so sánh ba cách chọn bước | `tune_cmp_newton.png` | `python run_tuning.py compare` |
| **4.6** Newton — pure và damped khi khởi tạo ở xa nghiệm | `newton_init_ridge.png` | `python run_stage.py newton-init` |
| **4.7** GD — dò độ dài bước cố định: thô rồi tinh | `tune_gd_fixed.png` | `python run_tuning.py gd-fixed` |
| **4.8** GD — dò (ρ, c) của backtracking: hai trục, hai câu trả lời | `tune_gd_bt_grid.png` | `python run_tuning.py gd-bt` |
| **4.9** GD — backtracking thật sự chọn bước nào? | `tune_step_trace_gd.png` | `python run_tuning.py gd-bt` |
| **4.10** AGD — dò độ dài bước cố định | `tune_gd_accel.png` | `python run_tuning.py gd-accel` |
| **4.11** AGD — lược đồ momentum (k-2)/(k+1): dò bước | `tune_agd_k.png` | `python run_tuning.py agd-k` |
| **4.12** AGD — hai lược đồ momentum: ai thắng tùy ngân sách | `agd_schemes.png` | `python run_tuning.py agd-k` |
| **4.13** GD và AGD — so sánh bốn cách chọn bước | `tune_cmp_gd.png` | `python run_tuning.py compare` |
| **4.14** SGD — dò lịch bước cho từng cỡ mini-batch | `tune_sgd_schedules.png` | `python run_tuning.py sgd` |
| **4.15** SGD — batch càng lớn càng tốt? | `tune_sgd_batches.png` | `python run_tuning.py sgd` |
| **4.17** So sánh cả bảy cấu hình — theo số bước | `money7_iter.png` | `python run_tuning.py money7` |
| **4.18** So sánh cả bảy cấu hình — theo thời gian | `money7_time.png` | `python run_tuning.py money7` |
| **5.3** Xác định hằng số trơn L — bốn đường, một con số | `tune_L.png` | `python run_tuning.py L` |
| **5.4** Ba vùng độ dài bước quanh 2/L và 2/λ | `three_regimes_ridge.png` | `python run_stage.py three-regimes` |
| **5.5** Kiểm chứng tốc độ lý thuyết: GD tỉ lệ κ, AGD tỉ lệ √κ | `kappa_sweep_ridge.png` | `python run_stage.py kappa-sweep` |
| **6.5** Dò η — minh họa với AMSGrad | `tune_ada_amsgrad.png` | `python run_tuning.py adaptive` |
| **6.6** Kết quả: họ Ada/Adam so với GD, AGD, Newton | `adaptive_family_ridge.png` | `python run_adaptive.py` |
| **7.2** Dò η_0 cho subgradient | `tune_l1_subgradient.png` | `python run_tuning.py l1` |
| **7.4** Dò bước cho ISTA — ba vòng mới tới | `tune_l1_ista.png` | `python run_tuning.py l1` |
| **7.5** Dò bước cho FISTA — vì sao không dùng lại 7/L của ISTA | `tune_l1_fista.png` | `python run_tuning.py l1` |
| **7.6** Kết quả trên bộ dữ liệu lasso (d=519) | `l1_lasso.png` | `python run_stage.py l1` |

## Hình không dùng trong bài trình bày

| Hình | Nội dung / lý do giữ | Lệnh sinh |
|---|---|---|
| `affine_invariance.png` | Newton bất biến affine — minh chứng phụ, không lên slide | `python run_stage.py affine` |
| `breakeven.png` | điểm hòa vốn d* ≈ 139 giữa Newton và L-BFGS — số đã đưa vào bảng, hình không dùng | `python run_stage.py breakeven` |
| `convergence_ridge.png` | đường hội tụ 4 thuật toán ở cấu hình mặc định (chưa dò tay) | `python run_benchmark.py` |
| `gd_step_sweep_ridge.png` | quét bước GD bản cũ, đã thay bằng tune_gd_fixed | `python run_experiments.py gd-sweep` |
| `kappa_honesty_ridge.png` | κ cận trên so với κ đo tại w* — số dùng ở mục 5, hình không dùng | `python run_stage.py kappa-honesty` |
| `money_iter_ridge.png` | money plot 4 đường bản cũ, đã thay bằng money7_iter | `python run_stage.py money` |
| `money_time_ridge.png` | money plot 4 đường bản cũ, đã thay bằng money7_time | `python run_stage.py money` |
| `sgd_sweep_ridge.png` | quét (η₀, batch) của SGD bản cũ, đã thay bằng tune_sgd_* | `python run_experiments.py sgd` |
| `tune_ada_adagrad.png` | dò η cho AdaGrad — slide 6.5 chỉ minh họa bằng AMSGrad | `python run_tuning.py adaptive` |
| `tune_ada_adam.png` | dò η cho Adam — như trên | `python run_tuning.py adaptive` |
| `tune_ada_adamw.png` | dò η cho AdamW — như trên | `python run_tuning.py adaptive` |
| `tune_ada_rmsprop.png` | dò η cho RMSprop — như trên | `python run_tuning.py adaptive` |
| `tune_step_trace_newton.png` | vết bước backtracking của Newton — slide 4.9 chỉ dùng bản GD | `python run_tuning.py newton-bt` |
