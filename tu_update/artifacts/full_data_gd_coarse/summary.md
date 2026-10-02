# GD dò thô trên dữ liệu đầy đủ của Tài

- Dữ liệu: `bai-lam-taitt/5_ma_nguon_tai_lap/artifacts/ridge.npz` — 75.026 dòng × 415 đặc trưng; không chia lại.
- SHA-256 khớp MANIFEST: `0c54d5d6c88667ae067cbf3be55df2f749d2c89e1a1f37c3e07fef60855f56d4`.
- Nguồn: https://drive.google.com/file/d/1mehXGDef4XyBnGdZp4gG89EbDUnAbU6Q/view
- λ=0.001; w0=0; 500 bước/cấu hình; BLAS một luồng.
- Giữ lưới dò thô và mã GD của tu_update; chỉ thay đầu vào bằng ma trận đầy đủ của Tài.
- Nghiệm tham chiếu Newton: f*=0.212258252066, chuẩn gradient=1.117e-16.

| t | Số bước | f cuối | f−f* | Chuẩn gradient cuối | Thời gian (s) | Giảm đều |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 500 | 0.228190786 | 1.593e-02 | 2.902e-02 | 13.78 | Có |
| 0.1 | 500 | 0.218599920 | 6.342e-03 | 1.298e-02 | 13.47 | Có |
| 0.5 | 500 | 0.212841898 | 5.836e-04 | 1.954e-03 | 13.53 | Không |
| 1 | 500 | 0.372401704 | 1.601e-01 | 6.671e-01 | 13.21 | Không |
| 1.5 | 500 | 0.562571092 | 3.503e-01 | 7.435e-01 | 13.91 | Không |
| 2 | 500 | 0.592609018 | 3.804e-01 | 7.742e-01 | 13.37 | Không |
| 3 | 500 | 0.513365861 | 3.011e-01 | 4.455e-01 | 13.60 | Không |
| 5 | 500 | 1.200828468 | 9.886e-01 | 7.182e-01 | 13.84 | Không |
| 10 | 500 | 3.822403533 | 3.610e+00 | 7.603e-01 | 13.83 | Không |

Đây là kết quả dò thô; chưa chạy dò tinh, AGD hay đánh giá chất lượng phân loại.
