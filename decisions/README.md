# decisions/
- nih_classes.md — lớp ID/OOD NIH, dự phòng.
- prompts_nih.json, prompts_camelyon.json — 50 bộ prompt mỗi file; hash trong prompts_hash.txt.
  K lồng theo TIỀN TỐ: K5 = bộ set_id 0–4, K20 = 0–19, K50 = 0–49 (5/20 bộ ĐẦU, không rút ngẫu nhiên).
- adapter_alpha_beta.md — α = 1.0, β = 5.5, logit scale 100, không search.
- scoring.md — max softmax float64 qua logsumexp; kiểm ties ở Bước 0.
- data_paths.md — đã có trong repo (commit 9a7672e).
