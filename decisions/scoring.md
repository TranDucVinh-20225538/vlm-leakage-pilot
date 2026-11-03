# Tính điểm OOD

## Quyết định
- Cả hai nhánh tính max softmax qua log-softmax / logsumexp, bằng float64:
  `score = exp(max_c z_c − logsumexp_c z_c)`; lưu thêm `log_score = max_c z_c − logsumexp_c z_c` (float64) và dùng log_score cho xếp hạng, quantile và AUROC (đơn điệu với score).
  - MCM: z_c = cos(f, w_c), T = 1, không nhân logit scale. Embedding ép sang float64 trước khi tính cosine.
  - Adapter: z = tip_logits theo adapter_alpha_beta.md (scale 100), tính bằng float64.
- Bước 0 phải in, cho mỗi nhánh (MCM, adapter) và mỗi bộ dữ liệu:
  - tỉ lệ điểm bị trùng (ties): 1 − (số giá trị log_score khác nhau / tổng số ảnh);
  - riêng nhánh adapter: tỉ lệ score (float64) đúng bằng 1.0.

## Lý do
- MCM với cosine, T = 1, trên 4 lớp cho softmax gần đều (~0.25 ± rất nhỏ); float32 dễ làm nhiều điểm trùng nhau.
- Adapter nhân 100 làm softmax bão hoà, nhiều điểm thành đúng 1.0 ⇒ quantile TPR 95% không xác định; log_score float64 giữ được thứ hạng.
