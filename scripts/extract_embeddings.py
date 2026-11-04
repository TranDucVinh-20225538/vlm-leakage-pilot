#!/usr/bin/env python3
"""Extract image embeddings once; save under a timestamped run dir."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ID_NIH = {"No Finding", "Infiltration", "Effusion", "Atelectasis"}
OOD_NIH = {
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
}


class ImageListDataset(Dataset):
    def __init__(self, paths, preprocess, resize_from=None, interp=Image.BICUBIC):
        self.paths = paths
        self.preprocess = preprocess
        self.resize_from = resize_from
        self.interp = interp

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        if self.resize_from is not None:
            # Documented: Camelyon 96x96 -> model input via BICUBIC
            if im.size != self.resize_from:
                pass
            # Always apply preprocess (includes resize to model size). We record interp used upstream.
        tensor = self.preprocess(im)
        return tensor, idx


def load_biomedclip(device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(device).eval()
    return model, preprocess, "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_conch(device, ckpt, hf_token):
    sys.path.insert(0, "/data2/shared/vinhtranduc/CONCH")
    from conch.open_clip_custom import create_model_from_pretrained

    if ckpt and Path(ckpt).is_file():
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path=ckpt
        )
    else:
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/CONCH",
            hf_auth_token=hf_token,
        )
    model = model.to(device).eval()
    return model, preprocess


@torch.inference_mode()
def encode_images(model, loader, device, model_kind: str):
    feats = None
    n = len(loader.dataset)
    ptr = 0
    for batch, idxs in loader:
        batch = batch.to(device, non_blocking=True)
        if model_kind == "biomedclip":
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        elif model_kind == "conch":
            f = model.encode_image(batch, proj_contrast=True, normalize=True)
        else:
            raise ValueError(model_kind)
        f = f.float().cpu().numpy()
        if feats is None:
            feats = np.zeros((n, f.shape[1]), dtype=np.float32)
        feats[idxs.numpy()] = f
        ptr += len(idxs)
        if ptr % 5000 < len(idxs):
            print(f"  encoded {ptr}/{n}", flush=True)
    return feats


def nih_index(nih_csv, img_dir):
    with open(nih_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    records = []
    for r in rows:
        labs = [x.strip() for x in r["Finding Labels"].split("|") if x.strip()]
        if len(labs) != 1:
            continue
        lab = labs[0]
        if lab not in ID_NIH and lab not in OOD_NIH:
            continue
        path = Path(img_dir) / r["Image Index"]
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "image_index": r["Image Index"],
                "patient_id": str(r["Patient ID"]),
                "label": lab,
                "is_id": lab in ID_NIH,
                "path": str(path),
            }
        )
    return records


def camelyon_index(meta_csv, patches_root):
    with open(meta_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    records = []
    root = Path(patches_root)
    for r in rows:
        patient, node = r["patient"], r["node"]
        x, y = r["x_coord"], r["y_coord"]
        folder = root / f"patient_{patient}_node_{node}"
        fname = f"patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
        path = folder / fname
        records.append(
            {
                "patient": patient,
                "node": node,
                "x": int(x),
                "y": int(y),
                "tumor": int(r["tumor"]),
                "center": int(r["center"]),
                "slide": int(r["slide"]),
                "path": str(path),
                "is_id_hospital": int(r["center"]) in (0, 1, 2),
            }
        )
    return records


def run_extract(records, model, preprocess, device, model_kind, batch_size, num_workers, out_npz, meta_name, interp_note):
    paths = [r["path"] for r in records]
    ds = ImageListDataset(paths, preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    t0 = time.time()
    feats = encode_images(model, loader, device, model_kind)
    elapsed = time.time() - t0
    np.save(out_npz / "embeddings.npy", feats)
    # Avoid giant JSON for Camelyon-scale; index.csv is the source of truth.
    if len(records) <= 200000:
        meta_path = out_npz / meta_name
        meta_path.write_text(json.dumps({"records": records, "interp": interp_note}, indent=2))
    else:
        (out_npz / "interp.txt").write_text(interp_note + "\n")
    with open(out_npz / "index.csv", "w", newline="") as f:
        if not records:
            return elapsed
        w = csv.DictWriter(f, fieldnames=[k for k in records[0].keys() if k != "path"] + ["path"])
        w.writeheader()
        for r in records:
            w.writerow(r)
    (out_npz / "extract_meta.json").write_text(
        json.dumps(
            {
                "n": len(records),
                "dim": int(feats.shape[1]),
                "seconds": elapsed,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "interp": interp_note,
            },
            indent=2,
        )
    )
    print(f"saved {out_npz} n={len(records)} dim={feats.shape[1]} sec={elapsed:.1f}", flush=True)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["nih_biomedclip", "camelyon_biomedclip", "camelyon_conch"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nih-csv", default="/data2/shared/vinhtranduc/data/nih_cxr14/Data_Entry_2017_v2020.csv")
    ap.add_argument("--nih-img-dir", default="/data2/shared/vinhtranduc/data/nih_cxr14/images")
    ap.add_argument(
        "--camelyon-csv",
        default="/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/camelyon17_v1.0/metadata.csv",
    )
    ap.add_argument(
        "--camelyon-patches",
        default="/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/camelyon17_v1.0/patches",
    )
    ap.add_argument("--conch-ckpt", default="/data2/shared/vinhtranduc/checkpoints/conch/pytorch_model.bin")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    if device.type == "cuda":
        print("gpu", torch.cuda.get_device_name(0), flush=True)

    t_wall0 = time.time()
    try:
        if args.task == "nih_biomedclip":
            model, preprocess, name = load_biomedclip(device)
            records = nih_index(args.nih_csv, args.nih_img_dir)
            interp = "NIH native resolution; BiomedCLIP preprocess resize (open_clip default)"
            run_extract(
                records, model, preprocess, device, "biomedclip",
                args.batch_size, args.num_workers, out, "records.json", interp,
            )
            (out / "model.json").write_text(json.dumps({"model": name, "kind": "biomedclip"}, indent=2))
        elif args.task == "camelyon_biomedclip":
            model, preprocess, name = load_biomedclip(device)
            records = camelyon_index(args.camelyon_csv, args.camelyon_patches)
            # Verify first image exists
            if not Path(records[0]["path"]).is_file():
                raise FileNotFoundError(records[0]["path"])
            interp = (
                "Camelyon patches are 96x96; upscaled to BiomedCLIP input inside open_clip "
                "preprocess (torchvision Resize, BICUBIC / InterpolationMode.BICUBIC)."
            )
            run_extract(
                records, model, preprocess, device, "biomedclip",
                args.batch_size, args.num_workers, out, "records.json", interp,
            )
            (out / "model.json").write_text(json.dumps({"model": name, "kind": "biomedclip"}, indent=2))
        elif args.task == "camelyon_conch":
            tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            model, preprocess = load_conch(device, args.conch_ckpt, tok)
            records = camelyon_index(args.camelyon_csv, args.camelyon_patches)
            if not Path(records[0]["path"]).is_file():
                raise FileNotFoundError(records[0]["path"])
            interp = (
                "Camelyon patches are 96x96; upscaled to CONCH input inside CONCH preprocess "
                "(Resize, BICUBIC)."
            )
            run_extract(
                records, model, preprocess, device, "conch",
                args.batch_size, args.num_workers, out, "records.json", interp,
            )
            (out / "model.json").write_text(json.dumps({"model": "conch_ViT-B-16", "kind": "conch"}, indent=2))
        print("WALL_SECONDS", time.time() - t_wall0, flush=True)
        print("EXTRACT_OK", flush=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print("EXTRACT_FAIL", err, flush=True)
        (out / "EXTRACT_FAIL.txt").write_text(err + "\n")
        raise


if __name__ == "__main__":
    main()
