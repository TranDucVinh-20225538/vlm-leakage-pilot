# STATUS — vlm-leakage-pilot

| Time (UTC+7) | Block | Result |
|---|---|---|
| 2026-11-03 17:21–17:28 | Bước -1 find | NIH + Camelyon located under toandq `DST-Skin/data` (read-only). STOP pending repo + image choice + Vinh decisions. |
| 2026-11-03 17:33–17:47 | Clone + NIH extract | Repo cloned to `/data2/shared/vinhtranduc/vlm-leakage-pilot` (branch `pilot-v1`). RAW `images_001..012.tar.gz` extracted to `/data2/shared/vinhtranduc/data/nih_cxr14/images/`. **VERIFY_OK**: 112120 PNG ≡ Image Index. |
| 2026-11-03 17:47 | decisions/data_paths.md | Paths, MD5s, counts written. train_val+test lists unused (112118). |
| 2026-11-03 20:39 | decisions pull | HEAD `3d2eac0`; `sha256sum -c prompts_hash.txt` OK. |
| 2026-11-03 20:46 | Bước 0 eligibility | **PASS**. NIH eligible (≥8 ID films)=2147, mean ID films=14.74; min eligible/role over seeds 0–199: P_sel 651, P_cal 658, P_test 663 (≥200). Camelyon ID-hospital-only 1/3: 8/8/8 (≥8). See `results/step0_20261103_204638/eligibility.json`. |
| 2026-11-03 22:15–23:55 | Bước 0 embeddings + ties | Slurm `72239` on A100-80GB, 1h40m. Embeddings under `/data2/shared/vinhtranduc/data/embeddings/step0_step0_20261103_204638/`. Ties in `results/step0_20261103_204638/ties.json`. |
| 2026-11-04 | STOP | **Bước 0 done. V0/V1 not started.** |

## Bước 0 numbers (from code)

### Eligibility
- NIH single-label counts: No Finding 60361, Infiltration 9547, Atelectasis 4215, Effusion 3955; OOD as in `decisions/nih_classes.md`; Hernia 110 dropped.
- Eligible patients (≥8 ID): **2147**; mean ID films **14.744**.
- Role mins (eligible, 200 seeds): **651 / 658 / 663** → no STOP.
- Camelyon: 43 patients; pure ID centers 0–2 = 24; ID-only split mins **8/8/8** → no STOP.

### Embeddings (GPU A100-SXM4-80GB)
| Branch | n | dim | seconds |
|---|---:|---:|---:|
| BiomedCLIP × NIH (ID+OOD single-label) | 91214 | 512 | 702.6 |
| BiomedCLIP × Camelyon | 455954 | 512 | 933.1 |
| CONCH × Camelyon | 455954 | 512 | 4170.5 |

Camelyon interp: 96×96 → model input via preprocess **BICUBIC**.

### Tie rates (prompt set_id 0, seed 0; float64 log_score)
| Branch | MCM tie rate | Adapter tie rate | Adapter frac score==1.0 |
|---|---:|---:|---:|
| BiomedCLIP × NIH | 0.000197 | 0.000197 | **0.0** |
| BiomedCLIP × Camelyon | 0.0 | 0.0 | **0.0** |
| CONCH × Camelyon | 0.0 | **0.346** | **0.144** |

Note: Camelyon adapter cache for this diagnostic used all ID-hospital P_sel patches (strict m/2 pool only 24 images; too small for N=16/class).
