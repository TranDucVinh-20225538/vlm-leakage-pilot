#!/usr/bin/env python3
"""Bước 0 tie-rate diagnostics per decisions/scoring.md + adapter_alpha_beta.md."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ALPHA = 1.0
BETA = 5.5
CLIP_SCALE = 100.0
N_CACHE = 16
M_DEFAULT = 4  # primary m


def log_softmax_score(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z: (N, C) float64 -> score, log_score float64."""
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))
    log_score = (z.max(axis=1, keepdims=True) - lse).ravel()
    score = np.exp(log_score)
    return score, log_score


def tie_rate(log_score: np.ndarray) -> float:
    n = len(log_score)
    if n == 0:
        return float("nan")
    return 1.0 - (len(np.unique(log_score)) / n)


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


def encode_texts_biomedclip(prompts: list[str], device):
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(device).eval()
    with torch.inference_mode():
        tok = tokenizer(prompts).to(device)
        t = model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy().astype(np.float64)


def encode_texts_conch(prompts: list[str], device, ckpt, hf_token):
    import sys
    import torch

    sys.path.insert(0, "/data2/shared/vinhtranduc/CONCH")
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

    if ckpt and Path(ckpt).is_file():
        model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt)
    else:
        model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH", hf_auth_token=hf_token
        )
    tokenizer = get_tokenizer()
    model = model.to(device).eval()
    with torch.inference_mode():
        tok = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
        t = model.encode_text(tok)
        # encode_text already normalizes by default
    return t.float().cpu().numpy().astype(np.float64)


def build_cache(embs, labels, class_names, patient_ids, rng: random.Random):
    """N=16 per class, spread across patients in P_sel_a."""
    by_class = {c: defaultdict(list) for c in class_names}
    for i, (y, p) in enumerate(zip(labels, patient_ids)):
        if y in by_class:
            by_class[y][p].append(i)
    keys = []
    values = []  # one-hot
    for ci, c in enumerate(class_names):
        # round-robin over patients
        patients = sorted(by_class[c].keys())
        rng.shuffle(patients)
        picked = []
        while len(picked) < N_CACHE and patients:
            progress = False
            for p in list(patients):
                if by_class[c][p]:
                    idx = by_class[c][p].pop()
                    picked.append(idx)
                    progress = True
                    if len(picked) >= N_CACHE:
                        break
                else:
                    patients.remove(p)
            if not progress:
                break
        if len(picked) < N_CACHE:
            raise RuntimeError(f"cache shortfall class={c} got={len(picked)}")
        for idx in picked:
            keys.append(embs[idx])
            oh = np.zeros(len(class_names), dtype=np.float64)
            oh[ci] = 1.0
            values.append(oh)
    return np.stack(keys).astype(np.float64), np.stack(values).astype(np.float64)


def tip_logits(f, W, cache_keys, cache_values):
    """f: (N,d), W: (d,C) columns are class text — we use (C,d) rows."""
    # W_c as (d, C)
    Wc = W.T  # (d, C)
    clip_logits = CLIP_SCALE * (f @ Wc)
    affinity = f @ cache_keys.T  # (N, Ncache)
    cache_logits = np.exp(-(BETA - BETA * affinity)) @ cache_values
    return clip_logits + ALPHA * cache_logits


def nih_ties(emb_dir: Path, prompts_path: Path, seed: int, device: str):
    embs = np.load(emb_dir / "embeddings.npy")
    with open(emb_dir / "index.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(embs)

    with open(prompts_path) as f:
        pdata = json.load(f)
    classes = pdata["classes"]
    # Bước 0 diagnostic: prompt set_id 0 (first of nested K5⊂K20⊂K50)
    prompts = [pdata["sets"][0]["prompts"][c] for c in classes]
    W = encode_texts_biomedclip(prompts, device)  # (C,d)

    id_films = defaultdict(list)
    for i, r in enumerate(rows):
        if r["is_id"] == "True" or r["is_id"] is True or r["is_id"] == "true":
            id_films[r["patient_id"]].append(i)
        # normalize
    # fix is_id parsing
    def is_id(r):
        v = r["is_id"]
        return v in (True, "True", "true", "1")

    id_films = defaultdict(list)
    all_patients = set()
    for i, r in enumerate(rows):
        all_patients.add(r["patient_id"])
        if is_id(r):
            id_films[r["patient_id"]].append(i)

    # Also need all NIH patients from CSV for roles — use patients appearing in extract index + 
    # roles on patients in this index is incomplete. Load full patient list from emb meta or CSV.
    # Use patients from index only for role assignment among those with any film in extract;
    # better: load Data_Entry for all patients.
    nih_csv = Path("/data2/shared/vinhtranduc/data/nih_cxr14/Data_Entry_2017_v2020.csv")
    with open(nih_csv) as f:
        all_p = sorted({str(r["Patient ID"]) for r in csv.DictReader(f)})
    roles = assign_roles(all_p, seed)

    eligible = {p for p, idxs in id_films.items() if len(idxs) >= 8}
    # P_sel_a: among eligible in P_sel, sample m=4, take half
    rng = random.Random(seed)
    sel_a_indices = []
    for p in sorted(eligible):
        if roles.get(p) != "P_sel":
            continue
        idxs = list(id_films[p])
        rng.shuffle(idxs)
        take = idxs[:M_DEFAULT]
        half = len(take) // 2  # m=4 -> 2
        sel_a_indices.extend(take[:half])

    labels = [r["label"] for r in rows]
    patient_ids = [r["patient_id"] for r in rows]
    f_all = embs.astype(np.float64)
    # L2 already from extract; ensure
    f_all = f_all / np.clip(np.linalg.norm(f_all, axis=1, keepdims=True), 1e-12, None)

    # MCM on all extracted images
    z_mcm = f_all @ W.T
    score_m, log_m = log_softmax_score(z_mcm)

    # Adapter cache from P_sel_a ID images only
    cache_labels = [labels[i] for i in sel_a_indices]
    cache_patients = [patient_ids[i] for i in sel_a_indices]
    cache_embs = f_all[sel_a_indices]
    cache_keys, cache_values = build_cache(
        cache_embs, cache_labels, classes, cache_patients, random.Random(seed + 1)
    )
    z_ad = tip_logits(f_all, W, cache_keys, cache_values)
    score_a, log_a = log_softmax_score(z_ad)

    return {
        "dataset": "NIH",
        "model": "BiomedCLIP",
        "prompt_set_id": 0,
        "seed": seed,
        "n_images": len(rows),
        "n_sel_a_cache_pool": len(sel_a_indices),
        "mcm": {
            "tie_rate_log_score": tie_rate(log_m),
            "n_unique_log_score": int(len(np.unique(log_m))),
        },
        "adapter": {
            "alpha": ALPHA,
            "beta": BETA,
            "clip_scale": CLIP_SCALE,
            "N_cache_per_class": N_CACHE,
            "tie_rate_log_score": tie_rate(log_a),
            "n_unique_log_score": int(len(np.unique(log_a))),
            "frac_score_eq_1": float(np.mean(score_a == 1.0)),
        },
    }


def camelyon_ties(emb_dir: Path, prompts_path: Path, seed: int, device: str, model_kind: str, conch_ckpt: str):
    embs = np.load(emb_dir / "embeddings.npy")
    with open(emb_dir / "index.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(embs)

    with open(prompts_path) as f:
        pdata = json.load(f)
    classes = pdata["classes"]  # normal, tumor
    prompts = [pdata["sets"][0]["prompts"][c] for c in classes]

    import os
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if model_kind == "biomedclip":
        W = encode_texts_biomedclip(prompts, device)
        model_name = "BiomedCLIP"
    else:
        W = encode_texts_conch(prompts, device, conch_ckpt, tok)
        model_name = "CONCH"

    # label from tumor
    label_of = []
    for r in rows:
        label_of.append("tumor" if int(r["tumor"]) == 1 else "normal")

    all_patients = sorted({r["patient"] for r in rows})
    roles = assign_roles(all_patients, seed)
    id_hospital = [i for i, r in enumerate(rows) if int(r["center"]) in (0, 1, 2)]

    # Camelyon has few patients: m/2 films in P_sel_a can be < N=16/class.
    # Bước 0 diagnostic cache pool = all ID-hospital patches of P_sel patients
    # (still patient-role–faithful; m-subsample kept for later V0/V1).
    by_p = defaultdict(list)
    for i in id_hospital:
        by_p[rows[i]["patient"]].append(i)
    rng = random.Random(seed)
    sel_a = []
    sel_a_strict_m = []
    for p, idxs in by_p.items():
        if roles.get(p) != "P_sel":
            continue
        idxs = list(idxs)
        rng.shuffle(idxs)
        sel_a.extend(idxs)  # full pool for diagnostic cache
        take = idxs[:M_DEFAULT]
        sel_a_strict_m.extend(take[: len(take) // 2])

    f_all = embs.astype(np.float64)
    f_all = f_all / np.clip(np.linalg.norm(f_all, axis=1, keepdims=True), 1e-12, None)

    z_mcm = f_all @ W.T
    score_m, log_m = log_softmax_score(z_mcm)

    cache_labels = [label_of[i] for i in sel_a]
    cache_patients = [rows[i]["patient"] for i in sel_a]
    cache_keys, cache_values = build_cache(
        f_all[sel_a], cache_labels, classes, cache_patients, random.Random(seed + 1)
    )
    z_ad = tip_logits(f_all, W, cache_keys, cache_values)
    score_a, log_a = log_softmax_score(z_ad)

    return {
        "dataset": "Camelyon17",
        "model": model_name,
        "prompt_set_id": 0,
        "seed": seed,
        "n_images": len(rows),
        "n_sel_a_cache_pool": len(sel_a),
        "n_sel_a_strict_m_half": len(sel_a_strict_m),
        "cache_pool_note": "step0 diagnostic uses all ID-hospital P_sel patches (strict m/2 too small for N=16)",
        "interp": "96x96 upscaled via model preprocess (BICUBIC)",
        "mcm": {
            "tie_rate_log_score": tie_rate(log_m),
            "n_unique_log_score": int(len(np.unique(log_m))),
        },
        "adapter": {
            "alpha": ALPHA,
            "beta": BETA,
            "clip_scale": CLIP_SCALE,
            "N_cache_per_class": N_CACHE,
            "tie_rate_log_score": tie_rate(log_a),
            "n_unique_log_score": int(len(np.unique(log_a))),
            "frac_score_eq_1": float(np.mean(score_a == 1.0)),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-emb", default="")
    ap.add_argument("--camelyon-biomedclip-emb", default="")
    ap.add_argument("--camelyon-conch-emb", default="")
    ap.add_argument("--prompts-nih", required=True)
    ap.add_argument("--prompts-camelyon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conch-ckpt", default="/data2/shared/vinhtranduc/checkpoints/conch/pytorch_model.bin")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {"branches": [], "errors": []}

    if args.nih_emb:
        try:
            results["branches"].append(
                nih_ties(Path(args.nih_emb), Path(args.prompts_nih), args.seed, device)
            )
        except Exception as e:
            results["errors"].append({"nih_biomedclip": f"{type(e).__name__}: {e}"})

    if args.camelyon_biomedclip_emb:
        try:
            results["branches"].append(
                camelyon_ties(
                    Path(args.camelyon_biomedclip_emb),
                    Path(args.prompts_camelyon),
                    args.seed,
                    device,
                    "biomedclip",
                    args.conch_ckpt,
                )
            )
        except Exception as e:
            results["errors"].append({"camelyon_biomedclip": f"{type(e).__name__}: {e}"})

    if args.camelyon_conch_emb:
        try:
            results["branches"].append(
                camelyon_ties(
                    Path(args.camelyon_conch_emb),
                    Path(args.prompts_camelyon),
                    args.seed,
                    device,
                    "conch",
                    args.conch_ckpt,
                )
            )
        except Exception as e:
            results["errors"].append({"camelyon_conch": f"{type(e).__name__}: {e}"})

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
