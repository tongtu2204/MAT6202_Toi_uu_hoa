# Hướng dẫn tái lập kết quả

Toàn bộ mã nguồn chạy được nằm trong `5_ma_nguon_tai_lap/`. **Mọi lệnh dưới đây đều
chạy từ thư mục đó.**

```bash
cd 5_ma_nguon_tai_lap
```

---

## 0. Có thể tái lập được tới đâu

Gói nộp bài **đã kèm sẵn ma trận thiết kế** (`artifacts/*.npz`, 127 MB) nhưng **không
kèm dữ liệu thô** (974 MB, xem `4_du_lieu/README.md`). Hệ quả:

| Muốn tái lập | Cần gì | Trạng thái |
|---|---|---|
| Toàn bộ mục 4, 5, 6, 7 (thuật toán, hình, số) | chỉ cần `artifacts/*.npz` | ✅ **chạy được ngay** |
| Kiểm chứng gradient/Hessian, prox, Armijo (19 test) | không cần gì thêm | ✅ **chạy được ngay** |
| Dựng lại bài trình bày và báo cáo (PDF) | XeLaTeX + biber | ✅ **chạy được ngay** |
| Sinh lại `artifacts/*.npz` từ đầu (`run_pipeline.py`) | dữ liệu thô VIB | ⚠️ phải xin lại dữ liệu |

---

## 1. Cài môi trường

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Phiên bản đã dùng để sinh mọi số liệu trong bài được ghim trong `requirements.txt` và
chép lại nguyên trạng ở `MOI_TRUONG.txt` (Python 3.12.7, numpy 1.26.4, scipy 1.13.1,
scikit-learn 1.5.1, OpenBLAS 0.3.21, macOS arm64).

Dựng PDF cần thêm **XeLaTeX + biber** ở mức hệ thống (TeX Live) — không phải gói pip.
Bài dùng font Times New Roman qua `fontspec` để hiển thị tiếng Việt.

> ⚠️ Các gói `titlesec` / `tcolorbox` / `sectsty` / `mdframed` **không** được dùng — phần
> mở đầu của `report.tex` cố ý tránh chúng và tự dựng khung ghi chú từ
> `xcolor` + `\colorbox`. Đừng "sửa" thành các gói đó.

## 2. Kiểm tra nhanh (30 giây)

```bash
pytest tests/ -q
```

Kỳ vọng: **19 passed**. Bộ test kiểm chứng gradient và Hessian bằng sai phân hữu hạn,
toán tử prox, điều kiện Armijo, tốc độ hội tụ, độ khớp với scikit-learn, chuẩn `f*`,
hành vi của Newton khi Hessian suy biến, và tính nhất quán của hai lược đồ AGD.

## 3. Tái lập từng phần của bài trình bày

Thời gian đo trên `ridge` (n = 75 026, d = 415), **BLAS ghim 1 luồng**.

> ⚠️ **Không chạy hai công việc đo thời gian cùng lúc.** BLAS đã bị ghim về 1 luồng để
> phép so thời gian giữa các thuật toán công bằng; chạy song song sẽ làm sai lệch mọi
> con số trên trục thời gian.

### Mục 4 — dò tham số bằng thử và sai (`run_tuning.py`)

Mỗi lệnh ghi hình vào `artifacts/figures/` và số vào `artifacts/tuning.json`.

```bash
python run_tuning.py --list          # liệt kê các mục

python run_tuning.py newton-fixed    # 4.3   hình tune_newton_fixed
python run_tuning.py newton-bt       # 4.4, 4.9(Newton)
python run_tuning.py gd-fixed        # 4.7   → chọn t = 0,4
python run_tuning.py gd-bt           # 4.8, 4.9 → chọn (ρ, c) = (0,8; 0,5)
python run_tuning.py gd-accel        # 4.10  → chọn t = 0,3
python run_tuning.py agd-k           # 4.11, 4.12 → chọn t = 0,3
python run_tuning.py compare         # 4.5, 4.13
python run_tuning.py sgd             # 4.14, 4.15
python run_tuning.py money7          # 4.17, 4.18  ← hình tổng kết của mục 4
python run_tuning.py adaptive        # 6.5
python run_tuning.py l1              # 7.2, 7.4, 7.5   (~60 phút)
python run_tuning.py L               # 5.3
```

**Cách dò là dò tay hai tầng, có chủ đích.** Tầng **thô** vẽ `f(w_k)` trên trục thẳng —
ở tầng này chỉ cần thấy bậc nào phân kỳ, bậc nào bò chậm. Tầng **tinh** chạy lưới quanh
giá trị thắng và vẽ `log(f − f*)` — ở tầng này các đường mới tách ra để so được. Giá
trị chọn được **viết tay** vào đầu `run_tuning.py` (`CHON_*`) kèm lý do, và lý do đó
hiện thẳng lên slide.

Ngân sách vòng lặp **khác nhau theo từng mục**, vì mỗi thuật toán cần một khoảng khác
nhau mới tách được các đường (số chính xác nằm trong `tuning.json`):

| Mục | Lưới thô | Vòng (thô) | Lưới tinh | Vòng (tinh) | Chọn |
|---|---|---:|---|---:|---|
| `gd-fixed` | 0,01·0,1·0,5·1·2·5 | 20 | 0,2·0,4·0,7·1·1,5 | 150 | `t = 0,4` |
| `gd-accel` | 0,01·0,1·0,5·1·2·5 | 20 | 0,1·0,2·0,3·0,5·0,8 | **1 000** | `t = 0,3` |
| `agd-k` | 0,01·0,1·0,5·1·2·5 | 20 | 0,1·0,2·0,3·0,5·0,8 | **1 000** | `t = 0,3` |
| `newton-fixed` | 0,01·0,1·0,5·1·2·5 | 20 | 0,8·0,9·1·1,1·1,2 | 40 | `t = 1` |
| `l1` subgradient | 4 điểm | 10 | 5 điểm | 150 | `η₀ = 2` |
| `l1` ISTA | 4 điểm (bội của 1/L) | 10 | 5 điểm | 150 | `7/L` |
| `l1` FISTA | 4 điểm (bội của 1/L) | 10 | 5 điểm | 600 | `4/L` |

Bốn mục ở mục 4 dùng **cùng một lưới thô**, nên bốn slide đó so trực tiếp được với nhau.
`newton-fixed` chỉ cần 40 vòng ở tầng tinh vì mọi `t ∈ [0,8; 1,2]` đều về tới độ chính
xác máy — ở đó phải so bằng **số vòng**, không so bằng khoảng cách cuối.

Không có hàm chấm điểm, không có bộ chọn tự động — vì tiêu chí thật sự là *"đường nào
vừa xuống nhanh vừa không dao động"*, nhìn hình thì rõ mà viết thành một con số thì
luôn thiếu. Một bộ chọn tự động cũng sẽ che mất chuyện đáng nói nhất: bước tốt nhất
nằm **xa** cận lý thuyết `2/L` đến mức nào.

### Mục 5, 6, 7 và các thí nghiệm dài (`run_stage.py`)

```bash
python run_stage.py --list
python run_stage.py newton-init      # ~2 phút   4.6
python run_stage.py three-regimes    # ~10 phút  5.4
python run_stage.py kappa-sweep      # ~35 phút  5.5   (lấy mẫu con 15 000 dòng)
python run_stage.py kappa-honesty    # ~55 phút  mục 5
python run_stage.py sklearn          # ~15 phút  4.19
python run_stage.py breakeven        # ~3 phút   d* ≈ 139
python run_stage.py l1               # ~60 phút  7.6
python run_stage.py armijo           # ~90 phút  4.16
python run_stage.py affine           # ~9 phút
python run_stage.py sgd-multiseed    # ~5 phút
python run_stage.py step-sweep       # ~15 phút
```

`kappa-sweep` và `kappa-honesty` **lấy mẫu con 15 000 dòng** một cách có chủ đích: đây
là các phát biểu về `κ`, mà `κ` là tính chất của **hình học đặc trưng**, không phải của
cỡ mẫu.

> ⚠️ **Dùng `run_stage.py`, đừng dùng `run_revision.py`.** Cả hai làm cùng một việc,
> nhưng `run_stage.py` gộp kết quả vào `artifacts/rev1_numbers.json` **sau mỗi chặng**
> nên một lần ngắt chỉ mất tối đa một chặng. `run_revision.py` chỉ ghi ở cuối — ở cỡ
> dữ liệu này đó là công việc nhiều giờ, và một công việc nhiều giờ chỉ ghi ở cuối là
> một công việc mất trắng khi bị ngắt (đã xảy ra thật).

### Các phần còn lại

```bash
python run_experiments.py std        # ~2 phút  4.1  standardization_contrast
python run_experiments.py sgd        # ~1 phút  sgd_sweep
python run_adaptive.py               # ~3 phút  6.6  adaptive_family + adaptive_numbers.json
python run_classification_metrics.py # ~5 giây  phụ lục (chỉ số phân loại, TRONG MẪU)
python -m optim.diagnostics          # ~1 phút  gate.json (L, μ, κ, hạng)
python replot_three_regimes.py       # vẽ lại 5.4 từ cache, không chạy lại thí nghiệm
```

> ⚠️ `run_revision.py` **không** sinh `standardization_contrast.png` và `sgd_sweep_*`.
> Phải chạy `run_experiments.py std sgd` riêng, nếu không slide 4.1 dùng hình cũ.

## 4. Dựng lại PDF

```bash
cd presentation_v2
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
# → main.pdf, 68 trang
```

```bash
cd docs
xelatex report.tex && xelatex report.tex     # chạy hai lần cho mục lục
# → report.pdf, 11 trang
```

`main.tex` khai báo `\graphicspath{{../artifacts/figures/}{images/}}`, nên thư mục
`presentation_v2/` phải nằm cạnh `artifacts/` — cấu trúc trong gói nộp đã đúng như vậy,
đừng di chuyển hai thư mục này ra khỏi nhau.

Sinh lại sơ đồ của báo cáo (matplotlib, không cần node/mermaid):

```bash
python docs/make_diagrams.py                 # → docs/diagrams/*.png
python docs/make_appendix_features.py        # → docs/appendix_features.tex
```

**Cả hai PDF trong gói này đã được dựng lại từ chính mã nguồn kèm theo** để kiểm chứng:
bài trình bày 68 trang, báo cáo 11 trang, 0 lỗi, 0 tham chiếu treo, 0 tràn khung.

## 5. Sinh lại ma trận thiết kế từ dữ liệu thô

Chỉ làm được nếu đã đặt 6 file dữ liệu thô vào `5_ma_nguon_tai_lap/data/`
(xem `4_du_lieu/README.md`).

```bash
python run_pipeline.py                        # ~90 giây
#   → artifacts/{ridge,lasso,poly,ridge_raw}.npz + *.features.txt + diagnostics.txt
python run_pipeline.py --lam 1e-3 --no-poly   # chỉnh λ của báo cáo điều kiện / bỏ poly
python run_pipeline.py --holdout              # khôi phục cách chia 75/25 cũ
```

Phần lớn 90 giây là để đọc file hoạt động 16 triệu dòng. Kết quả kỳ vọng:
**n = 75 026** dòng, churn 10,8 %, `ridge` `d = 415` đủ hạng, `lasso` `d = 519`
(hạng 506), `poly` `d = 428`, `ridge_raw` `d = 415` nhưng hạng số học chỉ **376**.

Thứ tự mô-đun trong `churn_opt/` trùng đúng thứ tự thực thi, nên đọc theo trình tự
này: `config.py` → `loaders.py` → `labeling.py` → `aggregate.py` → `build.py` →
`diagnostics.py`.

## 6. Chạy lại toàn bộ, đúng thứ tự

```bash
python run_pipeline.py                    # ~90 giây   (cần dữ liệu thô)
python -m optim.diagnostics               # ~1 phút
python run_experiments.py sgd-hyper       # ~25 giây
python run_tuning.py all                  # ~2 giờ
python run_stage.py all                   # ~5 giờ (tổng 12 chặng: 303 phút)
python run_experiments.py std sgd         # ~3 phút
python run_adaptive.py                    # ~3 phút
python run_classification_metrics.py      # ~5 giây
pytest tests/                             # 19 test
cd presentation_v2 && xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex
```

Tổng khoảng **7 giờ** trên máy đã dùng để sinh bài — con số này là **tổng của các chi
phí từng chặng đã đo**, không phải một lần chạy liên tục được bấm giờ. Chặng `armijo`
(90 phút), `l1` (60 phút) và `kappa-honesty` (55 phút) chiếm quá nửa.

Nếu chỉ muốn kiểm tra **một** con số cụ thể, đừng chạy cả gói: tra bảng trong
`3_ket_qua/DOI_CHIEU_HINH_SLIDE.md` để biết đúng một lệnh cần chạy.

## 7. Truy nguồn từng con số trên slide

Quy tắc của dự án: **mọi con số trên slide phải truy được về một file trong
`artifacts/`** (bản sao để đọc nằm ở `3_ket_qua/so_lieu/`). Nếu một con số không có
trong các file đó thì nó là số viết tay và phải bị nghi ngờ.

| File | Chứa gì |
|---|---|
| `diagnostics.txt` | `L`, `μ`, `κ`, hạng của cả bốn biến thể ma trận |
| `gate.json` | báo cáo điều kiện đầy đủ do `optim.diagnostics` sinh |
| `tuning.json` | **mọi** lưới dò tay: giá trị đã thử, kết quả từng giá trị, giá trị đã chọn và lý do |
| `rev1_numbers.json` | kết quả các chặng dài của `run_stage.py` |
| `sgd_hyper.json` | lưới 6×4 trên `(η₀, batch)` + quét `γ` của SGD |
| `adaptive_numbers.json` | họ AdaGrad / RMSprop / Adam / AdamW / AMSGrad |
| `metrics.json` | chỉ số phân loại ở phụ lục (**tính trong mẫu**, đã ghi rõ) |

## 8. Những khác biệt có thể gặp khi chạy lại

- **Thời gian tường** phụ thuộc máy. Các **tỉ số** mới là điều bài khẳng định (một vòng
  Newton ≈ 26× một vòng AGD), không phải giá trị tuyệt đối.
- **`f*` có thể lệch ở chữ số cuối** tùy phiên bản BLAS. Chứng chỉ `‖∇f(w*)‖²/(2μ*)`
  cho biết mức lệch đó bị chặn ở đâu — hãy đọc chứng chỉ thay vì so từng chữ số.
- **SGD dùng seed cố định** nên tái lập được từng bit trên cùng một phiên bản numpy.
- **Bước tốt nhất phụ thuộc ngân sách vòng lặp.** Nếu chạy lưới tinh với số vòng khác
  1 000 thì rất có thể chọn ra giá trị khác — đó là một kết quả của bài, không phải lỗi.
