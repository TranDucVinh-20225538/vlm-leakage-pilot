# Data paths (Bước -1)

Recorded before Bước 0. Sources under `/data2/cmdir/home/toandq/` are **read-only** (no modify/delete).

## NIH ChestX-ray14

| Field | Value |
|---|---|
| Chosen image source | **RAW** `images_001..012.tar.gz` (not staged-256, not `chestmnist_224.npz`) |
| Source archives (read-only) | `/data2/cmdir/home/toandq/DST-Skin/data/raw/nih_cxr14/images_001.tar.gz` … `images_012.tar.gz` |
| Extracted images (Vinh) | `/data2/shared/vinhtranduc/data/nih_cxr14/images/` |
| Metadata CSV (copy in Vinh data; MD5 matches source) | `/data2/shared/vinhtranduc/data/nih_cxr14/Data_Entry_2017_v2020.csv` |
| Source metadata (read-only) | `/data2/cmdir/home/toandq/DST-Skin/data/raw/nih_cxr14/Data_Entry_2017_v2020.csv` |
| Metadata MD5 | `63a8ca1a1b1f7a0e77d3db6522191103` |
| CSV rows / unique `Image Index` | 112120 |
| Patients (`Patient ID`) | 30805 |
| Extracted PNG count | **112120** |
| Match vs `Image Index` | **VERIFY_OK** (missing 0, extra 0) — extract log `.../extract_logs/extract.log` |

### Split-list note

`train_val_list.txt` + `test_list.txt` = 112118 (2 short of the CSV). This pilot assigns roles by patient and **does not use** those two lists.

## Camelyon17-WILDS (read-only)

| Field | Value |
|---|---|
| Root | `/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/camelyon17_v1.0` |
| Metadata | `/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/camelyon17_v1.0/metadata.csv` |
| Metadata MD5 | `4d03cc1a6eea7de71b1868cbbbab29ea` |
| Metadata rows (patches) | 455954 |
| Patients | 43 |
| Centers / hospitals | 5 (`center` ∈ {0,1,2,3,4}) |
| Columns checked | `patient`, `node`, `center`, `tumor`, `slide`, `split` |
| Patches dir | `/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/camelyon17_v1.0/patches/` |

## Working tree

| Field | Value |
|---|---|
| Repo | `https://github.com/TranDucVinh-20225538/vlm-leakage-pilot` |
| Branch | `pilot-v1` |
| Clone path | `/data2/shared/vinhtranduc/vlm-leakage-pilot` |
| Commit author | `vinhtranduc` |
