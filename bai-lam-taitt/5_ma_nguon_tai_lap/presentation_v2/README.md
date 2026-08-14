# presentation_v2 — deck theo `cấu trúc bài thuyết trình mới.txt`

Nội dung theo 7 phần của cấu trúc mới; **văn phong và mức độ trình bày theo một bài
mẫu do giảng viên giới thiệu**: một ý / một slide, mỗi slide gồm tiêu đề mục,
vài gạch đầu dòng "Nhận xét", và một hình (hoặc một bảng). Bản PDF của bài mẫu là tài
liệu của nhóm khác nên **không** được kèm vào gói nộp bài.

## Biên dịch

```bash
cd presentation_v2
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
```

Yêu cầu **XeLaTeX** (tiếng Việt qua `fontspec`). Không dùng `titlesec`/`tcolorbox`/
`sectsty`/`mdframed` (không có trong TeX Live của máy này).

Hiện tại: `main.pdf`, không còn cảnh báo tràn khung, không có citation lỗi.

**Đây là bản chính và là bản duy nhất trong gói nộp bài.** Các bản nháp cũ (`presentation/`,
`main_v1_backup.tex.bak`) có số liệu lỗi thời nên đã bị loại khỏi gói.

## Bố cục

| Phần | Slide | Nội dung |
|---|---|---|
| 1. Giới thiệu | 3–7 | phát biểu bài toán · mô hình logistic & hàm tổn thất · gradient · Hessian và **chứng minh tính lồi** · hàm phạt Ridge và $L,\mu,\kappa$ |
| 2. Lý thuyết | 8–15 | bảng 4 thuật toán · GD và 3 cách chọn bước · $\kappa$ vs $\sqrt\kappa$ · Newton (pure/damped, giải hệ) · **Kantorovich** · SGD · **bất biến affine → phải chuẩn hóa** · **Newton cần ma trận đủ hạng** |
| 3. Thiết kế thí nghiệm | 16–19 | bộ dữ liệu & cách gán nhãn · 7 bước xây dựng $X$ · 3 bộ dữ liệu · danh sách thí nghiệm và tiêu chí so sánh |
| 4. Kết quả | 20–28 | chuẩn hóa vs không · GD quét bước · GD phân kỳ · bảng cố định vs backtracking · Newton pure vs damped · SGD mini-batch · GD/AGD theo $\kappa$ · so sánh 4 thuật toán (số bước \| thời gian) · đối chiếu scikit-learn |
| 5. Ada / Adam | 29–32 | họ Ada · họ Adam · quét $\eta$ · **kết quả so với GD/AGD/Newton** |
| 6. Lasso | 33–35 | bài toán + subgradient · proximal gradient (ISTA/FISTA) · kết quả |
| 7. Kết luận | 36–37 | điều rút ra về từng thuật toán · về phần mở rộng |

Phụ lục: **A** cài đặt lõi + kiểm chứng đạo hàm · **B** giao thức đo (quy tắc `f*` + chứng chỉ
lồi mạnh, tính nhất quán log/dừng của AGD, kiểm tra luồng BLAS) + chỉ số phân loại ·
**C** quét siêu tham số SGD (trả lời câu hỏi "vì sao $\eta_0$ bằng chừng đó?").

## So với bản đầu (giữ trong `main_v1_backup.tex.bak`)

Đã **bỏ** để bớt rối: slide "luận đề / 3 câu hỏi", slide cận trên $L/\lambda$ vs $\kappa_\star$,
slide điểm hòa vốn $d^\star$ + phân tích BLAS/GFLOP, bảng danh mục E1–E5, chứng minh
bất biến affine bằng covector, bất đẳng thức PL, và toàn bộ hộp màu
"Kinh nghiệm"/"Điểm chốt" trên mỗi slide.

Đã **giữ** (vì cấu trúc mới yêu cầu): chứng minh tính lồi, rates + step sizes của GD/AGD,
Kantorovich, phương sai của SGD, bất biến affine, yêu cầu đủ hạng.

Đổi giao diện: tiêu đề phẳng màu tím (không dải màu xanh), có tiêu đề mục con
(`\begin{frame}{Mục}{Mục con}`) như bài mẫu; bỏ các slide mục lục chen giữa các phần.

## Nguồn số liệu

Mọi con số lấy từ `../artifacts/` (K=3 tháng, xếp chồng 3 mốc, **không chia train/test**,
$n=75\,026$ dòng, $d=415$ cho `ridge`):

- `artifacts/diagnostics.txt` — $L,\mu,\kappa$, hạng của các bộ dữ liệu
- `artifacts/gate.json` — $\kappa_\star$ theo $\lambda$
- `artifacts/rev1_numbers.json` — bảng backtracking, fixed-vs-backtracking, sklearn, L1
- `artifacts/adaptive_numbers.json` — quét $\eta$ + bảng kết quả họ Ada/Adam (mục 5.3–5.4)
- `artifacts/sgd_hyper.json` — lưới $(\eta_0,b)$ + quét $\gamma$ của SGD (phụ lục C)
- `artifacts/metrics.json` — chỉ số phân loại in-sample (phụ lục B)
- `artifacts/figures/*.png` — mọi hình (`\graphicspath` đã trỏ sẵn)

Sinh lại toàn bộ (thứ tự bắt buộc):

```bash
python run_pipeline.py                       # ~90s   (thêm --holdout nếu muốn chia 75/25)
python -m optim.diagnostics                  # ~1min  -> gate.json
python run_experiments.py sgd-hyper          # ~25s   -> sgd_hyper.json (chốt eta0/batch/gamma)
python run_stage.py all                      # ~4h    -> figures + rev1_numbers.json
#   (nên chạy từng chặng — xem docstring của run_stage.py để biết chi phí từng chặng)
python run_experiments.py std sgd --lam 1e-3 # ~3min  -> standardization_contrast.png, sgd_sweep
python run_adaptive.py                       # ~3min  -> adaptive_family_ridge.png + adaptive_numbers.json
python run_classification_metrics.py         # ~5s    -> metrics.json
```

Lưu ý: `run_revision.py` **không** sinh `standardization_contrast.png` (mục 4.1) và
`sgd_sweep_*.png` — phải chạy `run_experiments.py std sgd` riêng, nếu không hai slide đó
sẽ dùng hình cũ.

`sgd-hyper` phải chạy **trước** `run_revision.py` nếu muốn đổi $(\eta_0,b,\gamma)$: hằng số
thắng cuộc được chép tay vào `optim/benchmark.py` (`SGD_ETA0`, `SGD_BATCH`, `SGD_GAMMA`), và
đó là bộ tham số mọi hình money-plot dùng.

Hai bản chạy cũ (K=6 một mốc chụp với $n=46\,821$, $d=39$; và bản có chia 75/25 với
$n=35\,115$) **không** được kèm vào gói nộp bài — deck không lấy số từ chúng.

## Còn thiếu (cần chạy thí nghiệm mới)

**Subgradient** đã cài (`optim/optimizers/ista.py::subgradient`) và có mặt trong
`l1_lasso.png` cùng ISTA / FISTA / Coordinate Descent. `η₀` của nó được **quét** (5 lần dò
400 vòng) chứ không đặt tay — một phương pháp subgradient trông tệ vì bị cho bước sai thì
không chứng minh được gì.

Phần 5 đã có số đo đầy đủ cho **AdaGrad / RMSprop / Adam**
(`optim/optimizers/adaptive.py`, chạy bằng `run_adaptive.py`). **AdaDelta** và **Nadam**
vẫn chỉ có công thức trên slide 5.1–5.2, chưa cài; **AdamW** và **AMSGrad** đã cài
(cờ của hàm `adaptive`) nhưng chưa đưa vào hình so sánh — thêm chúng vào tham số
`methods` của `optim.experiments.adaptive_family` là đủ.

## Cần điền thủ công

- `\author{...}` — tên + MSSV thật.
- `\date{\today}` — đổi thành ngày thuyết trình.
- Số trang trong slide "Nội dung trình bày" (đang khớp với bản dựng hiện tại: 3, 8, 16, 20, 29, 32, 35) — nếu thêm/bớt slide thì cập nhật lại.
