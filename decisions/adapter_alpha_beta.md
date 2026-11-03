# Tip-Adapter (training-free) — α, β, logit scale
Nguồn kiểm ngày 2026-11-03: repo gaopengcuhk/Tip-Adapter, nhánh main, commit d0e2d6f8c5feb8b6ce937b757810761f7155d4d5; bài arXiv 2111.03930 (v1 và v2).

## Quyết định
- α = 1.0, β = 5.5 (bài báo, ablation ImageNet, Table 4: "set the hyper-parameter β as 5.5" khi quét α; "α is set as 1.0" khi quét β).
- Logit scale = 100, đúng như repo `main.py` (`clip_logits = 100. * features @ clip_weights`). KHÔNG dùng logit_scale của BiomedCLIP, vì α, β được đặt tương đối với clip_logits×100; đổi scale là đổi trọng số nhánh cache.
- search_hp: TẮT. Không grid search, không tune trên dữ liệu pilot.

## Cách tạo logit (repo main.py dòng ~32–42; bài báo Eq. 6)
```
clip_logits  = 100. * f_test @ W_c                 # f, W_c chuẩn hoá L2
affinity     = f_test @ cache_keys                 # cosine
cache_logits = exp(-(beta - beta*affinity)) @ cache_values   # exp(-β(1-A)) L, L one-hot
tip_logits   = clip_logits + alpha * cache_logits
```
Điểm OOD nhánh adapter = max softmax(tip_logits) (T = 1 trên logit này), tính theo scoring.md. Chỉ so Δ trong nhánh, không so giá trị tuyệt đối với MCM.

## Giới hạn
- α, β được tune trên ImageNet / dataset ảnh tự nhiên few-shot; không bảo đảm hợp với X-quang hay mô bệnh học.
- Bài báo khuyên "α is set to be large if the domain gap between pre-trained and downstream tasks is large". Ảnh y tế là trường hợp khác biệt miền lớn, nên α = 1 có thể làm nhánh cache bị đánh trọng số thấp.

## Phương án đã cân nhắc
- α = 1.17, β = 1 (`configs/imagenet.yaml`: init_alpha, init_beta). Không chọn: repo dùng chúng làm điểm khởi đầu cho search_hp (`search_hp: True`, `search_scale: [7, 3]`, `search_step: [200, 20]`), và giá trị khác theo dataset (vd `caltech101.yaml`: α = 3), README ghi là "overall best performance on 1~16 shots for a dataset", tức đã tune theo dataset.
- Dùng logit_scale của BiomedCLIP: không chọn (lý do ở trên).
