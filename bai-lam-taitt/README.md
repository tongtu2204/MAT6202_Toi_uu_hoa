# MAT6202 — Tối ưu hóa nâng cao · Gói nộp bài

**So sánh từ đầu bốn thuật toán tối ưu trên Hồi quy Logistic có chính quy hóa L2**

> Đối tượng nghiên cứu là **thuật toán tối ưu**, không phải mô hình churn. Dữ liệu
> churn ngân hàng chỉ đóng vai trò **bàn thử nghiệm** để bốn thuật toán có cùng một
> hàm mục tiêu mà chạy. Mọi so sánh trong bài đều dựa trên **giá trị hàm mục tiêu**
> `f(w)`, không dựa trên AUC/accuracy (các chỉ số phân loại chỉ nằm ở phụ lục).

```
f(w) = (1/n) Σᵢ [ −yᵢ log pᵢ − (1−yᵢ) log(1−pᵢ) ] + (λ/2)‖w‖²,    pᵢ = σ(w·xᵢ)
∇f(w)  = (1/n) Xᵀ(p − y) + λ·diag(r)·w
∇²f(w) = (1/n) XᵀSX + λ·diag(r),    S = diag(pᵢ(1−pᵢ)),  r = (1,…,1,0)  ← hệ số chặn không bị phạt
```

Thuật toán tự cài: **GD** · **AGD (Nesterov, hai lược đồ momentum)** · **Newton/IRLS
(damped, Armijo backtracking)** · **SGD**. Mở rộng: **subgradient / ISTA / FISTA /
Coordinate Descent** cho Lasso và họ **AdaGrad / RMSprop / Adam / AdamW / AMSGrad**.

---

## Đọc theo thứ tự nào

| # | Thư mục | Nội dung | Bắt đầu từ file |
|---|---|---|---|
| **1** | `1_bai_trinh_bay/` | Bài trình bày, 68 trang | `MAT6202_bai_trinh_bay.pdf` |
| **2** | `2_bao_cao_xu_ly_du_lieu/` | Báo cáo xử lý dữ liệu / feature engineering | `bao_cao_xu_ly_du_lieu.pdf` |
| **3** | `3_ket_qua/` | 38 hình + toàn bộ số liệu thô (JSON) | `DOI_CHIEU_HINH_SLIDE.md` |
| **4** | `4_du_lieu/` | Mô tả dữ liệu nguồn và cách lấy lại | `README.md` |
| **5** | `5_ma_nguon_tai_lap/` | Mã nguồn chạy được + ma trận thiết kế | `README.md` |
| **6** | `6_tai_lieu_goc/` | Đặc tả gốc của đề tài | `outline.txt` |

Muốn **tái lập kết quả**: xem [`HUONG_DAN_TAI_LAP.md`](HUONG_DAN_TAI_LAP.md).
Danh sách file kèm mã băm kiểm tra: [`MANIFEST.txt`](MANIFEST.txt).

---

## Bài toán được đặt ra như thế nào

**Bàn thử nghiệm.** Dữ liệu VIB Hackathon: 6 bảng quan hệ (khách hàng, giao dịch
MyVIB, hoạt động MyVIB 16 triệu dòng, tiền gửi, cho vay, thẻ), khóa
`CUSTOMER_NUMBER`, toàn bộ nằm trong năm 2019.

**Gán nhãn chống rò rỉ.** Quanh mỗi mốc chụp `T` chia ba cửa sổ rời nhau: cửa sổ quan
sát `(T−K, T]` chỉ dùng để sinh đặc trưng, cửa sổ kết quả `(T, T+H]` chỉ dùng để gán
nhãn. `K = 3` tháng, `H = 3` tháng. Trong nhóm khách hàng còn hoạt động ở cửa sổ quan
sát, `yᵢ = 1` nếu **không** có bất kỳ giao dịch **và** hoạt động nào trong cửa sổ kết
quả. Xếp chồng ba mốc chụp (31/03, 30/06, 30/09/2019) → **n = 75 026** dòng
(khách hàng × mốc), tỉ lệ churn **10,8 %**.

**Ma trận thiết kế.** ~500 cột thô → lọc tương quan → lọc VIF → **sửa hạng bằng QR
xoay trục** → `d = 415`, đủ hạng. Ba biến thể được sinh ra có chủ đích:

| Biến thể | `d` | Hạng | Dùng cho |
|---|---|---|---|
| `ridge` | 415 | 415 (đủ) | Ridge + Newton — Hessian phải khả nghịch |
| `lasso` | 519 | 506 | L1/ISTA/FISTA — **cố ý** giữ dư thừa để có cái mà đưa về 0 |
| `poly` | 428 | 428 | minh họa cái giá của `d` lớn hơn với Newton |
| `ridge_raw` | 415 | **376 (số học)** | đối chứng: không chuẩn hóa thì κ ≈ 2,7·10¹⁹, vượt float64 |

**Điều kiện của bài toán** (đo trên `ridge`, λ = 10⁻³): `L = 14,07` ·
`μ* = 1,001·10⁻³` · `κ_cận trên = 1,41·10⁴` · `κ* = 3 646`.

---

## Các kết quả chính

Tất cả tham số dưới đây đều **dò tay hai tầng** (lưới thô → lưới tinh → nhìn hình rồi
chọn), không có bộ chọn tự động. Số liệu gốc nằm ở `3_ket_qua/so_lieu/tuning.json`.

**1 · Chuẩn nghiệm có chứng chỉ.** `f* = 0,212258252065506`, đạt bởi Newton damped chạy
sâu tới `‖∇f(w*)‖ = 7,9·10⁻¹⁵` (15 vòng, 10,4 s). Vì hàm lồi mạnh trên tập mức, bất
đẳng thức PL cho chặn `f* − f_thật ≤ ‖∇f‖²/(2μ*) = 3,1·10⁻²⁶` với `μ*` **đo được**
`= 1,001·10⁻³`. Sai số này nhỏ hơn sàn 10⁻¹⁶ của mọi hình **10 bậc**. Kiểm chứng độc
lập: `sklearn lbfgs` trên cùng hàm mục tiêu dừng ở `f* + 2,2·10⁻¹³` — hai cài đặt hoàn
toàn khác nhau gặp nhau ở chữ số thứ 13. `f*` là giá trị **tại điểm cuối**, không
phải `min` của lịch sử — lấy `min` sẽ báo cáo một điểm khác với điểm thuật toán trả về
và làm mọi khoảng cách khác đẹp lên một cách không trung thực.

**2 · So sánh bảy cấu hình đã dò tay** (cùng một điều kiện dừng `‖∇f‖ < 10⁻¹⁰`,
trần 3 000 vòng):

| Cấu hình | Số vòng | Thời gian | `f − f*` cuối | Đạt ngưỡng |
|---|---:|---:|---:|:--:|
| Newton backtracking (ρ = 0,5, c = 0,2) | **8** | 6,0 s | 0 | ✓ |
| Newton bước cố định t = 1 | 10 | 7,2 s | 0 | ✓ |
| SGD (η₀ = 0,2, b = 1024, lịch hằng) | 50 epoch | 11,4 s | 1,6·10⁻³ | ✗ |
| AGD β hằng, t = 0,3 | 1 202 | 37,5 s | 0 | ✓ |
| GD bước cố định t = 0,4 | > 3 000 | 77,7 s | 1,2·10⁻⁵ | ✗ |
| AGD β = (k−2)/(k+1), t = 0,3 | > 3 000 | 99,1 s | 6,3·10⁻¹⁰ | ✗ |
| GD backtracking (ρ = 0,8, c = 0,5) | > 3 000 | 583,2 s | 1,5·10⁻¹¹ | ✗ |

**Chỉ 3 trên 7 cấu hình đạt ngưỡng.** SGD tới *gần* nghiệm rất rẻ nhưng dừng lại ở
1,6·10⁻³ — sàn phương sai của mini-batch, không phải chuyện chạy chưa đủ lâu.

**Newton thắng trên cả hai trục dù mỗi vòng của nó đắt nhất bảng**: 0,75 giây một vòng,
gấp **24×** một vòng AGD — nhưng cần ít hơn **150×** số vòng. Đây chính là luận đề của
bài: *chi phí mỗi vòng* nhân *số vòng* mới ra thời gian, và hai thừa số đó đi ngược
chiều nhau.

Đổi trục là đổi hạng: GD-backtracking chính xác thứ ba (1,5·10⁻¹¹) nhưng **chậm nhất
bảng** (gấp 7,5× GD bước cố định); ngược lại SGD thứ ba về thời gian nhưng bét về độ
chính xác.

**3 · Luật κ so với √κ được kiểm chứng.** Quét λ để κ trải hai bậc: độ dốc log–log của
GD là **0,819**, của AGD là **0,413**; **tỉ số 1,98 ≈ 2**. Đúng như lý thuyết dự đoán.

**4 · Bước tốt nhất phụ thuộc ngân sách.** Bước tinh tốt nhất của AGD chuyển từ 0,2
(ở 150 vòng) sang **0,3** (ở 1 000 vòng) — chênh 180× về độ chính xác cuối. Lưới thô
`(0,01; 0,1; 0,5; 1; 2; 5)` ở 20 vòng **xếp hạng sai hai lần** (chọn t = 2 cho GD, t = 0,5
cho AGD-k); cả hai đều bị lưới tinh bác bỏ. Hai chỗ sai này được **giữ lại trong bài
một cách có chủ đích**, vì chúng là nội dung cần nói.

**5 · Backtracking giúp GD nhưng phá AGD.** GD: 3,3·10⁻⁴ → 7,8·10⁻⁸. AGD: từ hội tụ ở
2 131 vòng thành **không** hội tụ sau 5 000 vòng. Lý do: β = (√κ−1)/(√κ+1) được suy ra
với **giả thiết** bước bằng 1/L; đổi `t` từng vòng trong khi giữ β cố định là phá cặp
`(t, β)` mà chứng minh dựa vào.

**6 · Trần bước của FISTA THẤP hơn của ISTA** — 4/L so với 7/L, đo được. Dùng lại 7/L
của ISTA cho FISTA cho kết quả 1,17·10⁻² với 346 hệ số khác 0, **tệ hơn cả** 1/L trong
sách, vì momentum tích lũy chính phần vượt đà mà ISTA hấp thụ được.

**7 · L-BFGS của scikit-learn nay thắng Newton tự cài.** Đối chiếu theo **giá trị hàm
mục tiêu** (khớp đúng dạng: `C = 1/(nλ)`, hệ số chặn không bị phạt ở cả hai phía):

| Solver | `f − f*` | Số vòng | Thời gian |
|---|---:|---:|---:|
| `sklearn lbfgs` | 2,2·10⁻¹³ | 225 | **3,31 s** |
| `sklearn newton-cholesky` | 2,8·10⁻¹⁷ | 10 | 6,84 s |
| Newton (tự cài) | 0 | 11 | 10,21 s |
| SGD (tự cài) | 1,6·10⁻³ | 51 epoch | 10,60 s |
| AGD (tự cài) | 5,6·10⁻¹⁷ | 1 335 | 39,7 s |
| GD (tự cài) | 2,1·10⁻¹³ | 20 000 | 540,9 s |
| `sklearn saga` | 1,2·10⁻¹¹ | 5 000 | 852 s |

Newton tự cài (11 vòng) khớp `newton-cholesky` (10 vòng) tới độ chính xác máy — chênh
lệch 10,21 s so với 6,84 s chỉ đến từ **cài đặt** (BLAS/Cython), không từ thuật toán.
Ở `d = 39` Newton áp đảo; ở `d = 415` chi phí `O(nd² + d³)` của Hessian thua 225 vòng
quasi-Newton rẻ tiền. Điểm hòa vốn dự đoán `d* ≈ 139`, khớp với đo đạc.

**8 · Lasso** (`f* = 0,221818416`, cùng ngân sách 8 000 vòng, mỗi phương pháp ở bước
đã dò riêng): CD đạt đúng `f*`, FISTA 1,5·10⁻¹¹, ISTA 1,2·10⁻⁶, subgradient 1,4·10⁻⁴ —
thứ tự `O(1/√k) < O(1/k) < O(1/k²)` hiện ra sạch sẽ **ngay cả sau khi** mỗi phương pháp
đã được cho bước tốt nhất của nó.

Điểm cốt lõi nằm ở cột **số hệ số khác 0**: ISTA 126, FISTA 125, CD 125, SAGA 126 —
bốn phương pháp đồng thuận. Subgradient: **519 trên 519**, tức là *không có một số 0
nào*. Đó không phải chuyện chạy chưa đủ lâu. `sign(w)` bằng ±1 tại **mọi** tọa độ khác
0, nên mỗi tọa độ vọt qua 0 thay vì đáp xuống đúng 0, và số 0 chính xác không bao giờ
xuất hiện. Ngưỡng mềm (soft-thresholding) **không phải** một đường nhanh hơn tới cùng
một đáp án — nó là thứ tạo ra tính thưa ngay từ đầu.

---

## Nguyên tắc đo đạc đã tuân thủ

Đây là những chỗ dễ sai nhất và bài đã xử lý dứt điểm từng chỗ:

- **`μ` không phải `λ`.** Hệ số chặn cố ý không bị phạt, nên `λI ⪯ ∇²f` chỉ đúng trên
  các tọa độ bị phạt. Dọc hướng hệ số chặn, độ cong `(1/n)Σpᵢ(1−pᵢ)` tiến về 0 khi
  `‖w‖` lớn. Hàm lồi mạnh **trên tập mức bị chặn**, không lồi mạnh toàn cục. Vì vậy
  `μ` luôn được **đo** bằng `λ_min(∇²f(w*))`, không bao giờ thay bằng `λ`.
  Đây không phải chuyện lý thuyết suông: Newton thuần từ điểm xuất phát xa đi tới
  `‖w‖ ~ 10¹⁶`, `S` tràn dưới, Hessian suy biến **đúng** theo hướng hệ số chặn.
- **`f*` lấy tại điểm cuối kèm chứng chỉ**, không lấy `min` lịch sử. Bài toán L1 không
  lồi mạnh nên không có chứng chỉ; ở đó dùng `best_known` (giá trị nhỏ nhất mà **mọi**
  lần chạy đạt được) — quy tắc an toàn, chỉ làm khoảng cách báo cáo *lớn* hơn.
- **Trục tung luôn là `log(f − f*)`.** Thang tuyến tính che mất sự khác nhau giữa tốc
  độ tuyến tính và bậc hai.
- **BLAS ghim 1 luồng và được *kiểm chứng*, không giả định.** Đặt biến môi trường sau
  khi numpy đã nạp là một lệnh không có tác dụng, và nó âm thầm làm hỏng mọi phép đo
  thời gian. `blas_threads.verify()` hỏi trực tiếp backend đang chạy.
- **Chi phí SGD quy đổi phân số**: một mini-batch được tính là `|B|/n` gradient đầy đủ,
  để trục chi phí so được với các phương pháp toàn batch.
- **AGD ghi log và dừng tại cùng một điểm.** Dòng `k` chứa `f(w_k)` cùng
  `‖∇f(y_{k−1})‖`; bổ đề giảm ở bước 1/L cộng PL cho `f(w_k) − f* ≤ ‖∇f(y_{k−1})‖²/(2μ)`
  — điều kiện dừng chứng nhận đúng đại lượng đang vẽ, trên cùng một dòng. Tính
  `‖∇f(w_k)‖` trực tiếp sẽ tốn thêm một gradient mỗi vòng và làm phép so thời gian với
  GD thành không trung thực.
- **Đối chiếu scikit-learn theo giá trị hàm mục tiêu, không theo accuracy.** sklearn
  tối thiểu hóa **tổng** (không có `1/n`) với phạt `(1/2)wᵀw`, nên `C = 1/(λn)`
  — *không* phải `1/(2λn)`.
- **Không chia train/test theo mặc định.** Đây là nghiên cứu về tối ưu: `f(w)` được
  định nghĩa bởi đúng cặp `(X, y)` đưa vào, và không đại lượng nào được đo
  (`L`, `μ`, `κ`, số vòng, thời gian, `f − f*`) cần tập giữ lại. Chia chỉ tốn 25 % số
  quan sát. Các chỉ số phân loại ở phụ lục được tính **trong mẫu** và ghi rõ như vậy.

## Giới hạn đã biết

- Một khách hàng có thể xuất hiện ở nhiều mốc chụp, nên các dòng **không độc lập cùng
  phân phối**. Không ảnh hưởng tới nghiên cứu về tối ưu, nhưng đã nêu rõ trên slide.
- Lưới thô không phân giải được cực trị thật của bước AGD: nó nằm trong khoảng
  (0,3; 0,5) và bài chỉ khẳng định 0,3 là **giá trị lớn nhất trước vách**.
- `poly` **không còn** là núm điều khiển κ: ở `d = 415`, 36 cột tương tác không làm κ
  đổi tới bốn chữ số. Câu chuyện AGD-so-GD vì thế dựa vào **λ**, quét bằng
  `optim.experiments.kappa_sweep`.
- AdaDelta và Nadam chỉ được nêu công thức ở mục 6.1–6.2, **không** được cài đặt và
  **không** có số đo nào trong bài.
