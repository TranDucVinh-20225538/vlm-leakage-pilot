#!/usr/bin/env python3
"""Bước 0 eligibility (CPU). Does not change decisions/."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ID_NIH = ["No Finding", "Infiltration", "Effusion", "Atelectasis"]
OOD_NIH = [
    "Pneumothorax",
    "Mass",
    "Nodule",
    "Cardiomegaly",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Consolidation",
    "Edema",
    "Pneumonia",
]
DROP_NIH = ["Hernia"]
ID_SET = set(ID_NIH)
OOD_SET = set(OOD_NIH)


def assign_roles(patient_ids, seed: int):
    rng = random.Random(seed)
    pts = sorted(patient_ids)
    rng.shuffle(pts)
    n = len(pts)
    n1, n2 = n // 3, n // 3
    roles = {}
    for i, p in enumerate(pts):
        if i < n1:
            roles[p] = "P_sel"
        elif i < n1 + n2:
            roles[p] = "P_cal"
        else:
            roles[p] = "P_test"
    return roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-csv", required=True)
    ap.add_argument("--camelyon-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview-seeds", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.nih_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert "Patient ID" in rows[0], list(rows[0].keys())

    lab_counts = Counter()
    id_films = defaultdict(int)
    ood_films = defaultdict(int)
    all_patients = set()
    n_single = n_multi = 0
    for r in rows:
        pid = str(r["Patient ID"])
        all_patients.add(pid)
        labs = [x.strip() for x in r["Finding Labels"].split("|") if x.strip()]
        if len(labs) != 1:
            n_multi += 1
            continue
        n_single += 1
        lab = labs[0]
        lab_counts[lab] += 1
        if lab in ID_SET:
            id_films[pid] += 1
        elif lab in OOD_SET:
            ood_films[pid] += 1

    eligible = sorted(p for p, n in id_films.items() if n >= 8)
    mean_id = statistics.mean(id_films[p] for p in eligible)

    m_levels = {}
    for m in (2, 4, 8):
        # Fixed pool: patients with >=8 ID films; all can supply m in {2,4,8}.
        m_levels[str(m)] = {
            "n_eligible_patients": len(eligible),
            "mean_id_films_per_patient": mean_id,
            "note": "eligibility fixed at >=8 ID films; population unchanged across m",
        }

    # Across seeds 0..preview_seeds-1: min eligible counts per role (ALL-patient 1/3 split)
    role_elig_mins = {k: 10**9 for k in ("P_sel", "P_cal", "P_test")}
    role_elig_example = None
    for seed in range(args.preview_seeds):
        roles = assign_roles(all_patients, seed)
        counts = Counter()
        for p in eligible:
            counts[roles[p]] += 1
        for k in role_elig_mins:
            role_elig_mins[k] = min(role_elig_mins[k], counts[k])
        if seed == 0:
            role_elig_example = {
                role: {
                    "n_patients_all": sum(1 for p, r in roles.items() if r == role),
                    "n_eligible_id": counts[role],
                    "mean_id_films_eligible": statistics.mean(
                        id_films[p] for p in eligible if roles[p] == role
                    ),
                }
                for role in ("P_sel", "P_cal", "P_test")
            }

    nih_stop = any(role_elig_mins[k] < 200 for k in role_elig_mins)

    # Camelyon
    with open(args.camelyon_csv, newline="") as f:
        crows = list(csv.DictReader(f))
    patient_centers = defaultdict(set)
    for r in crows:
        patient_centers[r["patient"]].add(r["center"])
    pure_id = sorted(p for p, c in patient_centers.items() if c <= {"0", "1", "2"})
    pure_ood = sorted(p for p, c in patient_centers.items() if c <= {"3", "4"})
    all_cam = sorted(patient_centers)

    cam_role_mins = {k: 10**9 for k in ("P_sel", "P_cal", "P_test")}
    cam_id_mins = {k: 10**9 for k in ("P_sel", "P_cal", "P_test")}
    cam_example = None
    for seed in range(args.preview_seeds):
        roles = assign_roles(all_cam, seed)
        total = Counter(roles.values())
        id_c = Counter(roles[p] for p in pure_id)
        for k in cam_role_mins:
            cam_role_mins[k] = min(cam_role_mins[k], total[k])
            cam_id_mins[k] = min(cam_id_mins[k], id_c[k])
        if seed == 0:
            cam_example = {
                role: {
                    "n_patients_all": total[role],
                    "n_id_hospital_0_2": id_c[role],
                    "n_ood_hospital_3_4": sum(1 for p in pure_ood if roles[p] == role),
                }
                for role in ("P_sel", "P_cal", "P_test")
            }

    # Gate: patient counts in each role. Camelyon: all-patient roles (1/3 of 43).
    # Also report ID-hospital mins (can be <8 under all-patient split).
    cam_stop_total = any(cam_role_mins[k] < 8 for k in cam_role_mins)
    # Primary Camelyon ID pool split (hospitals 0-2 only), for protocol sample sizes
    cam_id_only_mins = {k: 10**9 for k in ("P_sel", "P_cal", "P_test")}
    for seed in range(args.preview_seeds):
        roles = assign_roles(pure_id, seed)
        c = Counter(roles.values())
        for k in cam_id_only_mins:
            cam_id_only_mins[k] = min(cam_id_only_mins[k], c[k])
    cam_stop_id_pool = any(cam_id_only_mins[k] < 8 for k in cam_id_only_mins)

    report = {
        "nih": {
            "patient_id_column": "Patient ID",
            "n_csv_rows": len(rows),
            "n_patients_all": len(all_patients),
            "n_single_label": n_single,
            "n_multi_label": n_multi,
            "single_label_counts": dict(lab_counts.most_common()),
            "id_classes": ID_NIH,
            "ood_classes": OOD_NIH,
            "drop_classes": DROP_NIH,
            "n_eligible_ge8_id_films": len(eligible),
            "mean_id_films_eligible": mean_id,
            "m_levels": m_levels,
            "split_rule": "roles on ALL patients 1/3; ID uses eligible (>=8 ID films)",
            "eligible_per_role_min_over_seeds_0_to_N-1": role_elig_mins,
            "preview_seeds": args.preview_seeds,
            "example_seed0_roles": role_elig_example,
            "stop_threshold": 200,
            "STOP": nih_stop,
        },
        "camelyon": {
            "n_metadata_rows": len(crows),
            "n_patients_all": len(all_cam),
            "n_pure_id_centers_0_2": len(pure_id),
            "n_pure_ood_centers_3_4": len(pure_ood),
            "split_rule_all_patients_1_3": True,
            "role_patient_mins_all_split": cam_role_mins,
            "role_id_hospital_mins_under_all_split": cam_id_mins,
            "role_mins_if_split_id_hospitals_only": cam_id_only_mins,
            "example_seed0_all_patient_roles": cam_example,
            "stop_threshold": 8,
            "STOP_if_all_patient_role_lt_8": cam_stop_total,
            "STOP_if_id_hospital_pool_role_lt_8": cam_stop_id_pool,
            "STOP": cam_stop_total or cam_stop_id_pool,
            "note": "STOP true if either all-patient roles or ID-hospital-only roles fall below 8 in any seed preview.",
        },
        "go_embeddings": (not nih_stop) and (not (cam_stop_total or cam_stop_id_pool)),
    }

    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["go_embeddings"]:
        print("ELIGIBILITY_PASS")
    else:
        print("ELIGIBILITY_STOP")


if __name__ == "__main__":
    main()
