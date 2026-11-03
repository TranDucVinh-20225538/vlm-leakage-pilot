# STATUS — vlm-leakage-pilot

| Time (UTC+7) | Block | Result |
|---|---|---|
| 2026-11-03 17:21–17:28 | Bước -1 find | NIH + Camelyon located under toandq `DST-Skin/data` (read-only). STOP pending repo + image choice + Vinh decisions. |
| 2026-11-03 17:33–17:47 | Clone + NIH extract | Repo cloned to `/data2/shared/vinhtranduc/vlm-leakage-pilot` (branch `pilot-v1`). RAW `images_001..012.tar.gz` extracted to `/data2/shared/vinhtranduc/data/nih_cxr14/images/`. **VERIFY_OK**: 112120 PNG ≡ Image Index. |
| 2026-11-03 17:47 | decisions/data_paths.md | Paths, MD5s, counts written. train_val+test lists unused (112118). |
| 2026-11-03 | STOP | Waiting for Vinh to commit NIH ID/OOD classes, 50 nested prompts + hash, and α/β into `decisions/`. **Bước 0 not started.** |
