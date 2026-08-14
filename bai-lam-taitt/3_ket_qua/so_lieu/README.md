# Số liệu thô

Bảy file này là **nguồn duy nhất** cho mọi con số xuất hiện trên slide. Quy tắc của dự
án: nếu một con số không truy được về một file ở đây thì nó là số viết tay và phải bị
nghi ngờ.

Đây là bản sao để đọc; bản mà mã nguồn thực sự đọc và ghi nằm ở
`../../5_ma_nguon_tai_lap/artifacts/`.

---

## `diagnostics.txt` — điều kiện của bài toán

Dạng văn bản thuần, đọc trực tiếp được. Với mỗi biến thể ma trận: `n`, `d`, hạng,
`L`, `μ`, `κ_cận trên`, và trị riêng nhỏ nhất/lớn nhất của ma trận Gram kèm kết luận
"Newton chạy được / sẽ sập". Cuối file là cấu hình cửa sổ thời gian và tỉ lệ churn.

Đây là chỗ tra nhanh nhất cho các số của mục 2 và mục 5.

## `tuning.json` — toàn bộ quá trình dò tay (44 KB)

File quan trọng nhất. Mỗi khóa là một lần dò hai tầng:

```
L, gd_fixed, gd_accel, agd_k, agd_schemes, agd_schemes_cross,
newton_fixed, gd_backtracking, newton_backtracking, compare,
sgd_schedules, sgd_chosen, adaptive,
l1_subgradient, l1_ista, l1_fista, money7
```

Mỗi mục chứa:

| Trường | Nghĩa |
|---|---|
| `coarse.grid`, `coarse.rows` | lưới **thô** đã thử (20 vòng) và kết quả từng giá trị |
| `fine.grid`, `fine.rows` | lưới **tinh** (1 000 vòng) và kết quả từng giá trị |
| `chosen` | giá trị **người chọn** sau khi nhìn hình |
| `reason` | **lý do chọn**, viết tay — chính câu này hiện lên slide |

Mỗi dòng trong `rows` có: giá trị tham số, số vòng, số vòng để đạt ngưỡng, `f` cuối,
khoảng cách `f − f*`, có hữu hạn không, có giảm đơn điệu không, thời gian.

Khóa `money7` chứa bảy cấu hình cuối cùng của mục 4.17–4.18, mỗi cấu hình ở đúng tham
số đã dò tay của nó.

## `rev1_numbers.json` — các chặng dài của `run_stage.py`

```
newton_init, three_regimes, affine, sgd, breakeven, gd_step_sweep, money,
kappa_honesty, kappa_sweep, armijo_sweep, fixed_vs_backtracking,
sklearn_timing, l1
```

- `kappa_sweep` — độ dốc log–log của GD và AGD theo κ (kết quả **0,819** so với
  **0,413**, tỉ số **1,98 ≈ 2**). Chạy trên **mẫu con 15 000 dòng**, ghi trong `n_used`.
- `kappa_honesty` — κ cận trên so với κ đo tại `w*`, và dự đoán tốc độ GD từ mỗi cái.
- `sklearn_timing` — đối chiếu với `lbfgs`/`liblinear`/`saga` theo **giá trị hàm mục
  tiêu**, không theo accuracy.
- `breakeven` — điểm hòa vốn `d* ≈ 139` giữa Newton và L-BFGS.
- `l1` — Lasso: `f*`, khoảng cách cuối và số hệ số khác 0 của subgradient / ISTA /
  FISTA / CD / SAGA.

## `sgd_hyper.json` — dò siêu tham số SGD

Lưới 6×4 trên `(η₀, batch)` cộng quét 5 điểm trên `γ`, tất cả ở **cùng ngân sách 50
epoch**. Robbins–Monro chỉ ràng buộc *hình dạng* của lịch bước, không bao giờ ràng buộc
hằng số — nên hằng số phải đo. Ô thắng trong file này:
`η₀ = 0,05`, `batch = 64`, `γ = 2·10⁻⁴`, khoảng cách cuối `1,86·10⁻⁴`.

> ⚠️ **Đây KHÔNG phải cấu hình SGD mà bài trình bày dùng.** File này là kết quả của một
> bộ chọn **tự động**, đã bị rút khỏi lập luận chính của bài. Cấu hình lên slide là
> cấu hình **dò tay** ở `tuning.json → sgd_chosen`: `η₀ = 0,2`, `batch = 1024`, lịch
> **hằng**. Lý do chọn khác nhau vì tiêu chí khác nhau: bộ chọn tự động xếp hạng theo
> khoảng cách cuối sau 50 epoch, còn money plot đọc theo **trục thời gian** — `b = 1024`
> đạt ~2·10⁻³ sau 4,5 giây trong khi `b = 16384` cần 20 giây mới tới cùng mức. Giữ file
> này lại để đối chiếu, không để trích số.

## `adaptive_numbers.json` — họ Ada/Adam

`eta_grid`, `best_eta`, `sweep` (kết quả từng η) và `table` (bảng cuối) cho AdaGrad,
RMSprop, Adam, AdamW, AMSGrad. Chạy **toàn batch** theo mặc định để đường của chúng rơi
thẳng vào cùng một hệ trục với GD/AGD/Newton.

## `gate.json` — báo cáo điều kiện đầy đủ

Do `python -m optim.diagnostics` sinh. Chứa các phép kiểm (`gates`), backend BLAS đang
chạy (`blas`), κ lớn nhất theo từng biến thể, và biến thể được chọn.

## `metrics.json` — chỉ số phân loại (phụ lục)

`auc`, `accuracy`, `f1`, `precision`, `recall`. **Tính trong mẫu** — trường
`in_sample: true` ghi rõ điều đó. Các số này **không** thuộc lập luận chính của bài:
đối tượng nghiên cứu là thuật toán tối ưu, không phải chất lượng mô hình churn. Chúng
chỉ có mặt ở phụ lục để trả lời câu hỏi "mô hình có chạy không".
