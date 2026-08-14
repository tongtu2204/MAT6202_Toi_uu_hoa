# Mã nguồn — MAT6202

Thư mục này là **bản chạy được** của toàn bộ bài: mã nguồn, ma trận thiết kế đã xử lý,
số liệu, hình vẽ và nguồn LaTeX của cả hai PDF. Giữ nguyên cấu trúc thư mục — nhiều
đường dẫn là **tương đối** và sẽ hỏng nếu di chuyển các thư mục ra khỏi nhau.

Hướng dẫn chạy từng bước: [`../HUONG_DAN_TAI_LAP.md`](../HUONG_DAN_TAI_LAP.md).

```
5_ma_nguon_tai_lap/
├── churn_opt/          pipeline đặc trưng → ma trận thiết kế X
├── optim/              hàm mục tiêu + thuật toán tự cài + thí nghiệm + vẽ hình
├── tests/              19 test kiểm chứng
├── artifacts/          X (*.npz) + mọi số và hình bài trích dẫn
│   └── figures/        38 hình PNG
├── docs/               nguồn báo cáo xử lý dữ liệu (report.tex → report.pdf)
├── presentation_v2/    nguồn bài trình bày (main.tex → main.pdf, 68 trang)
├── data/               ĐỂ TRỐNG — chỗ đặt dữ liệu thô, xem ../4_du_lieu/README.md
├── run_*.py            các driver chạy thí nghiệm
├── blas_threads.py     ghim BLAS về 1 luồng và KIỂM CHỨNG điều đó
└── requirements.txt    phiên bản đã ghim
```

## Hàm mục tiêu

```
f(w) = (1/n) Σᵢ [ −yᵢ log pᵢ − (1−yᵢ) log(1−pᵢ) ] + (λ/2)‖w‖²,   pᵢ = σ(w·xᵢ)
∇f(w)  = (1/n) Xᵀ(p − y) + λ·diag(r)·w
∇²f(w) = (1/n) XᵀSX + λ·diag(r),   S = diag(pᵢ(1−pᵢ))
```

`r = (1,…,1,0)` — **hệ số chặn không bị phạt**. Newton trên hàm này chính là IRLS.
Ràng buộc khó nhất của đề bài là *tự cài Newton*, và chính nó quyết định mọi lựa chọn
còn lại: Hessian phải khả nghịch, nên ma trận thiết kế phải đủ hạng.

## Đọc mã theo thứ tự nào

**`churn_opt/`** — các mô-đun xếp trùng đúng thứ tự thực thi:

| File | Bước | Việc |
|---|---|---|
| `config.py` | — | `WindowConfig` (các cửa sổ thời gian), `FeatureConfig` (ngưỡng lọc, λ). **Mọi tham số nằm ở đây.** |
| `loaders.py` | — | nạp từng bảng + **chuẩn hóa mã khách hàng** (`_norm_id` đệm 0 về 8 ký tự) + phân tích ngày **theo từng file** |
| `labeling.py` | 0 | dân số đủ điều kiện + nhãn `y` không rò rỉ |
| `aggregate.py` | 1 | gấp mỗi bảng về một dòng/khách hàng trên cửa sổ quan sát; `_slope_momentum` sinh đặc trưng xu hướng theo tháng; pivot `TRANS_LV1` (`TXG_*`) là nguồn số chiều chính |
| `build.py` | 2–7 | ghép → `_clean` (điền khuyết + `log1p`) → `_encode` (one-hot `drop_first`) → `StandardScaler` → `_corr_prune` + `_vif_prune` → `_rank_prune` → `_add_polynomial` |
| `diagnostics.py` | — | `conditioning(X, λ)` trả về `L`, `μ`, `κ_cận trên`, hạng |

**`optim/`**:

| File | Việc |
|---|---|
| `objective.py` | `LogisticObjective`: `value` / `grad` / `hessian` / `sklearn_C` |
| `optimizers/gd.py` | GD bước cố định và GD backtracking |
| `optimizers/agd.py` | AGD Nesterov, **hai lược đồ momentum** (`scheme="const"` \| `"k"`) |
| `optimizers/newton.py` | Newton damped = IRLS; giải bằng Cholesky, **không bao giờ nghịch đảo ma trận** |
| `optimizers/sgd.py` | SGD mini-batch, lịch bước Robbins–Monro |
| `optimizers/ista.py` | subgradient / ISTA / FISTA / coordinate descent cho L1 |
| `optimizers/adaptive.py` | một driver `adaptive()` cho AdaGrad / RMSprop / Adam (+cờ AdamW, AMSGrad) |
| `optimizers/base.py` | `OptResult` + `Recorder` — đếm chi phí (gradient, Hessian, thời gian) |
| `linesearch.py` | `armijo_backtrack` |
| `reference.py` | `newton_reference` → `f*` **kèm chứng chỉ**; `best_known` cho bài L1 |
| `tuning.py` | bộ khung dò tay hai tầng (`Sweep`, `Choice`, `l_study`, `sgd_steps`) |
| `benchmark.py` | `run_benchmark`, `run_l1_benchmark`, và các hàm đọc tham số đã dò |
| `experiments.py` | các thí nghiệm dài: quét κ, ba vùng bước, bất biến affine, hòa vốn… |
| `plots.py` | mọi hình vẽ, ghi vào `artifacts/figures/` |
| `data.py` | nạp `artifacts/<biến thể>.npz` thành `Dataset` |

## Bốn biến thể ma trận thiết kế

| Biến thể | `d` | Hạng | Sinh ra để làm gì |
|---|---|---|---|
| `ridge` | 415 | 415 | Ridge + Newton. Lọc tương quan/VIF **rồi sửa hạng bằng QR xoay trục** |
| `lasso` | 519 | 506 | L1/ISTA/FISTA — **cố ý** giữ dư thừa để có cái mà đưa về 0 |
| `poly` | 428 | 428 | lõi `ridge` + 36 số hạng bậc 2. Minh họa cái giá của `d` lớn với Newton |
| `ridge_raw` | 415 | **376** | đối chứng **không chuẩn hóa**: κ ≈ 2,7·10¹⁹ vượt float64 ⇒ λI thành vô hình |

Lõi đa thức chỉ lấy **các cột liên tục**: bình phương hay nhân hai biến giả 0/1 sẽ cho
lại một hàm affine của chính nó → cộng tuyến chính xác → Hessian suy biến. Lỗi này đã
xảy ra thật và đã sửa; **đừng đưa lại vào**.

## Sáu ràng buộc đúng đắn không hiển nhiên

Đây là các cách hỏng mà bài đã gặp hoặc đã chặn trước:

1. **Bắt buộc chuẩn hóa `X` (z-score).** Đặc trưng thô trải ~6 bậc độ lớn (tuổi so với
   số tiền giao dịch tính bằng VND); không chuẩn hóa thì κ nổ tung và phép so
   GD–AGD–Newton mất hết ý nghĩa. Biến thể `ridge_raw` giữ lại để chứng minh điều đó.

2. **Bắt buộc one-hot với `drop_first=True`.** Giữ đủ `k` biến giả cộng thêm hệ số chặn
   cho cộng tuyến hoàn hảo → `XᵀX` suy biến → **Newton sập thật sự**. GD/AGD không báo
   lỗi nhưng nghiệm không xác định.

3. **`log1p` các cột tiền/đếm đuôi nặng** trước khi chuẩn hóa. Điểm ngoại lai thổi
   phồng `λ_max(XᵀX)` → `L` lớn → GD chậm một cách giả tạo.

4. **Không được sót NaN.** Một giá trị NaN duy nhất biến gradient và hàm mục tiêu thành
   NaN và phá mọi vòng lặp. Điền 0 cho số đếm/số tiền vắng mặt và thêm cờ `HAS_X`.

5. **Đủ hạng không tự nhiên mà có.** Lọc tương quan và VIF là công cụ *thống kê*: chúng
   bắt cộng tuyến **theo cặp** và **gần đúng**. Không cái nào loại được một đẳng thức
   **chính xác** giữa nhiều cột, và VIF thậm chí **không nhìn thấy** nó — ma trận tương
   quan suy biến sẽ đẩy `np.linalg.inv` sang nhánh `pinv`, cho ra đường chéo hữu hạn
   trông y hệt một VIF bình thường. Hai quy tắc:
   - **Không bao giờ tạo một đặc trưng là hàm chính xác của các đặc trưng đã có.**
     `SPAN = FIRST_AGE − RECENCY` và `DAYS_PER_MONTH = ACTIVE_DAYS / K` đều đã được
     viết ra và đều phải xóa đi: không thêm thông tin, chắc chắn thiếu hạng.
   - `_rank_prune` (QR xoay trục) chạy **cuối cùng** trên `ridge` và `poly` như một bảo
     đảm. Không có nó, lần chạy `d = 416` đầu tiên ra hạng 415 và Newton sẽ sập.

6. **`newton(strict=True)` (mặc định) NÉM LỖI khi Hessian không xác định dương.** Trước
   đây nó `break` im lặng, biến một ma trận thiếu hạng thành một `converged=False` trông
   vô hại. Chỉ `newton_init_study` truyền `strict=False`, vì ở đó Hessian suy biến
   **chính là kết quả cần trình bày**.

## Nguyên tắc đo đạc (đừng làm thoái lui)

- **`f*` là giá trị tại điểm CUỐI, không phải `min(f_history)`.** Lấy `min` trên lịch sử
  của chính lần chạy đó là báo cáo một điểm khác với điểm mà lần chạy trả về, và nó cắt
  bớt khoảng cách của mọi phương pháp khác theo hướng có lợi cho ta — đọc lên thành
  chọn lọc số liệu. `optim.reference.newton_reference` trả về `f*` kèm **chứng chỉ**
  `‖∇f(w*)‖²/(2μ)`. Bài L1 không lồi mạnh nên không có chứng chỉ; ở đó `best_known()`
  lấy giá trị nhỏ nhất mà **bất kỳ** lần chạy nào đạt được — hướng an toàn, chỉ làm
  khoảng cách báo cáo *lớn* hơn.

- **`μ` không phải `λ`.** Xem phần "Hàm mục tiêu" ở trên: `λI ⪯ ∇²f` chỉ đúng trên các
  tọa độ bị phạt. `newton_reference` vì thế dùng `λ_min(∇²f(w*))` **đo được**; ở
  `d = 415` giá trị đó là `1,001·10⁻³ ≈ 1,00λ`. Sự gần bằng này là **trùng hợp số học
  tại điểm `w*` cụ thể**, không phải một đồng nhất thức — luôn đo, đừng bao giờ thay.

- **Newton của money plot giữ line search MẶC ĐỊNH** của `newton_reference`
  (`t₀=1, ρ=0,5, c=10⁻⁴`), **không** dùng cặp đã dò tay `(t₀=2, ρ=0,5, c=0,2)` của mục
  4.4 — và đó là chủ ý, có đo đạc. Đưa cặp đã dò vào làm lần chạy chuẩn **chạm trần 100
  vòng trong 95 giây mà không đạt `tol = 10⁻¹⁴`**, so với 15 vòng / 10,4 giây ở mặc
  định: `c = 0,2` là phép thử Armijo ngặt, và gần nghiệm thì nhiễu dấu phẩy động làm nó
  thất bại, nên bước bị lùi dần về 0 trong khi `‖∇f‖` kẹt quanh 10⁻¹³. Vì chính lần chạy
  đó sinh ra `f*` và chứng chỉ của nó, lần chạy chuẩn **phải** là cấu hình thật sự
  chứng nhận được. Mục 4.15 nói thẳng điều này trên slide thay vì giấu đi.
  **Đừng "sửa" bằng cách nối `newton_backtracking` vào `run_benchmark`.**

- **AGD ghi log và dừng tại cùng một điểm.** Dòng `k` chứa `f(w_k)` cùng
  `‖∇f(y_{k−1})‖`; bổ đề giảm ở bước 1/L cộng PL cho
  `f(w_k) − f* ≤ ‖∇f(y_{k−1})‖²/(2μ)` — điều kiện dừng chứng nhận **đúng** đại lượng
  đang vẽ, trên **cùng một dòng**. Tính `‖∇f(w_k)‖` trực tiếp sẽ tốn thêm một gradient
  mỗi vòng và làm phép so thời gian với GD thành không trung thực.

- **`blas_threads` KIỂM CHỨNG chứ không giả định.** Đặt biến môi trường sau khi numpy
  đã nạp là một lệnh không có tác dụng, và nó âm thầm làm hỏng mọi phép đo thời gian.
  Mô-đun cảnh báo nếu bị import muộn, và `blas_threads.verify()` hỏi trực tiếp backend
  đang chạy qua `threadpoolctl`. Các driver đo thời gian đều gọi và in kết quả.

- **`Recorder.tick(grad=...)` nhận số thực** — SGD tính một mini-batch là `|B|/n`
  gradient đầy đủ, để trục chi phí so được với các phương pháp toàn batch.

- **Đối chiếu scikit-learn theo giá trị hàm mục tiêu, không theo accuracy.** sklearn tối
  thiểu hóa **tổng** (không có `1/n`) với phạt `(1/2)wᵀw`, nên `C = 1/(λn)` — *không*
  phải `1/(2λn)`; xem `optim.objective.LogisticObjective.sklearn_C`. Solver đối chiếu
  mặc định là `lbfgs`.

- **Không chia train/test theo mặc định.** `FeatureConfig.use_holdout = False`;
  `run_pipeline.py --holdout` khôi phục cách chia 75/25 cũ và chỉ cần khi muốn AUC/F1
  ngoài mẫu. Khi bật chia, nó **phải** xảy ra *trước* khi khớp scaler/encoder/imputer
  (rò rỉ). Với `use_holdout=False`, các lát `X_test`/`y_test` trong `.npz` rỗng nhưng
  vẫn được giữ, nên `optim.data.Dataset` và mọi nơi dùng nó không phải đổi.

## Một nguồn cho mỗi hằng số

Khiếm khuyết lặp đi lặp lại của dự án này là **driver tự đặt tham số riêng thay vì đọc
`artifacts/tuning.json`**. Đã chặn dứt điểm bằng ba hàm đọc trong `optim/benchmark.py`:

| Hàm | Đọc gì | Giá trị hiện tại |
|---|---|---|
| `gd_config(L)` | `tuning.json["gd_fixed"]["chosen"]` | `t = 0,4` (dự phòng `1/L`, và **in ra** đã dùng cái nào) |
| `agd_config(L)` | `tuning.json["gd_accel"]["chosen"]` | `t = 0,3`, trả về `L_eff = 1/t` để **β được tính lại theo đúng bước đó** |
| `l1_config()` | `tuning.json["l1_*"]` | `η₀ = 2,0`, `t_ista = 7/L`, `t_fista = 4/L` |

`agd_config` trả `L_eff` là chi tiết cốt lõi: β = (√κ−1)/(√κ+1) được suy ra **với giả
thiết** bước bằng 1/L, nên thay bước mà giữ nguyên β là phá cặp `(t, β)`. Truyền bước đã
dò vào dưới dạng `L_eff` giữ cho cặp đó vẫn khớp nhau.

`t_ista ≠ t_fista` cũng vậy, và **đừng dùng lẫn**: đưa 7/L của ISTA cho FISTA cho
1,17·10⁻² với 346 hệ số khác 0 — tệ hơn cả 1/L trong sách — vì momentum tích lũy đúng
phần vượt đà mà ISTA thuần hấp thụ được. Vách nằm giữa 5/L và 6/L.

## Chạy job dài

`run_stage.py <chặng>...` chạy **một** chặng rồi gộp vào `artifacts/rev1_numbers.json`,
nên một lần ngắt chỉ mất tối đa một chặng. Ưu tiên nó hơn `run_revision.py`, thứ chỉ ghi
ở cuối. Chặng dưới ~10 phút chạy foreground được. Chặng dài hơn thì tách tiến trình bằng
`subprocess.Popen(..., start_new_session=True)` — **macOS không có `setsid`**, nên câu
lệnh `setsid nohup ... &` sẽ thất bại âm thầm. Chi phí đo được của từng chặng nằm trong
docstring của `run_stage.py`.

## Ghi chú về `CLAUDE.md`

File `CLAUDE.md` trong thư mục này là **nhật ký kỹ thuật** của dự án (tiếng Anh): nó ghi
lại mọi kết luận đo được, mọi cách hỏng đã gặp và mọi quyết định không được phép thay
đổi. Nó cũng đóng vai trò file hướng dẫn cho công cụ hỗ trợ lập trình đã dùng trong quá
trình làm bài. Giữ lại vì đây là bản ghi trung thực nhất về *vì sao* mã nguồn có hình
dạng như hiện tại.
