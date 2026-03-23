"""
dataset_pipeline.py
====================
Assignment 3 — Drone Detection Dataset Pipeline
Combines:
  1. lgrzybowski/seraphim-drone-detection-dataset  (HuggingFace, YOLO, 83k images)
  2. tracker-qjlj1/drones_new                      (Roboflow, YOLO, ~9.5k images)

Output layout (YOLO-ready):
  drone_dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  ├── test/
  │   ├── images/
  │   └── labels/
  └── data.yaml

Usage:
  pip install huggingface_hub roboflow pyyaml tqdm
  python dataset_pipeline.py --roboflow-key YOUR_API_KEY
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import yaml
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SERAPHIM_REPO_ID  = "lgrzybowski/seraphim-drone-detection-dataset"
ROBOFLOW_WORKSPACE = "tracker-qjlj1"
ROBOFLOW_PROJECT   = "drones_new"
ROBOFLOW_VERSION   = 4          # latest published version
ROBOFLOW_FORMAT    = "yolov8"   # native YOLO TXT labels

OUTPUT_DIR  = Path("drone_dataset")
STAGING_DIR = Path("_staging")

# Only keep the "drone" class from Roboflow (multi-class dataset)
# After download, Roboflow puts class names in data.yaml — we re-map.
DRONE_CLASS_NAME = "drone"
FINAL_CLASS_ID   = 0            # merged dataset: drone == class 0


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def copy_files(src_img_dir: Path, src_lbl_dir: Path,
               dst_img_dir: Path, dst_lbl_dir: Path,
               prefix: str = ""):
    """Copy image+label pairs, adding an optional filename prefix."""
    images = sorted(src_img_dir.glob("*.*"))
    copied = 0
    for img_path in tqdm(images, desc=f"  Copying {prefix or src_img_dir.parent.name}", leave=False):
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue  # skip unannotated images
        new_stem = f"{prefix}{img_path.stem}" if prefix else img_path.stem
        shutil.copy2(img_path, dst_img_dir / (new_stem + img_path.suffix))
        shutil.copy2(lbl_path, dst_lbl_dir / (new_stem + ".txt"))
        copied += 1
    return copied


def remap_labels(label_dir: Path, old_id: int, new_id: int):
    """Rewrite YOLO .txt files replacing old_class_id with new_class_id."""
    for txt in tqdm(list(label_dir.glob("*.txt")),
                    desc=f"  Remapping class {old_id}→{new_id}", leave=False):
        lines = txt.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == old_id:
                parts[0] = str(new_id)
                new_lines.append(" ".join(parts))
            # lines with other class ids are dropped (multi-class → drone-only)
        txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))


def count_annotations(label_dir: Path) -> int:
    total = 0
    for txt in label_dir.glob("*.txt"):
        lines = [l for l in txt.read_text().splitlines() if l.strip()]
        total += len(lines)
    return total


# ─────────────────────────────────────────────
# STEP 1 — SERAPHIM (HuggingFace)
# ─────────────────────────────────────────────

def download_seraphim(staging: Path) -> Path:
    """Download and unzip the Seraphim dataset. Returns local repo path."""
    from huggingface_hub import snapshot_download

    dest = staging / "seraphim"
    if (dest / "train" / "images").exists():
        print("[Seraphim] Already downloaded — skipping.")
        return dest

    print("[Seraphim] Downloading from HuggingFace (this may take a while)…")
    repo_path = Path(
        snapshot_download(
            repo_id=SERAPHIM_REPO_ID,
            repo_type="dataset",
            local_dir=str(dest),
        )
    )

    # Unzip all batch archives
    zips = list(repo_path.rglob("*.zip"))
    print(f"[Seraphim] Extracting {len(zips)} zip archives…")
    for zp in tqdm(zips, desc="  Unzipping"):
        try:
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(zp.parent)
            zp.unlink()
        except zipfile.BadZipFile:
            print(f"  ⚠️  Skipping bad zip: {zp.name}")

    print("[Seraphim] ✅ Extraction complete.")
    return repo_path


def ingest_seraphim(repo_path: Path, out: Path):
    """
    Seraphim layout:  train/{images,labels}  and  test/{images,labels}
    We map:  train → train (80%) + val (20% via holdout)
             test  → test
    """
    print("[Seraphim] Ingesting into merged dataset…")

    # ── train split (we hold out last 20% for val) ──────────────────────
    s_train_img = repo_path / "train" / "images"
    s_train_lbl = repo_path / "train" / "labels"

    all_imgs = sorted(s_train_img.glob("*.*"))
    split_idx = int(len(all_imgs) * 0.80)
    train_imgs = all_imgs[:split_idx]
    val_imgs   = all_imgs[split_idx:]

    for img_path in tqdm(train_imgs, desc="  Seraphim → train"):
        lbl_path = s_train_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        shutil.copy2(img_path, out / "train" / "images" / ("ser_" + img_path.name))
        shutil.copy2(lbl_path, out / "train" / "labels" / ("ser_" + img_path.stem + ".txt"))

    for img_path in tqdm(val_imgs, desc="  Seraphim → val"):
        lbl_path = s_train_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        shutil.copy2(img_path, out / "val" / "images" / ("ser_" + img_path.name))
        shutil.copy2(lbl_path, out / "val" / "labels" / ("ser_" + img_path.stem + ".txt"))

    # ── test split ───────────────────────────────────────────────────────
    s_test_img = repo_path / "test" / "images"
    s_test_lbl = repo_path / "test" / "labels"
    if s_test_img.exists():
        copy_files(s_test_img, s_test_lbl,
                   out / "test" / "images", out / "test" / "labels",
                   prefix="ser_")

    # Seraphim is already single-class drone==0, no remapping needed.
    print("[Seraphim] ✅ Ingestion done.")


# ─────────────────────────────────────────────
# STEP 2 — DRONES_NEW (Roboflow)
# ─────────────────────────────────────────────

def download_roboflow(api_key: str, staging: Path) -> Path:
    """Download drones_new from Roboflow in YOLOv8 format."""
    from roboflow import Roboflow

    dest = staging / "drones_new"
    if (dest / "train" / "images").exists():
        print("[Roboflow] Already downloaded — skipping.")
        return dest

    print("[Roboflow] Downloading drones_new…")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version  = project.version(ROBOFLOW_VERSION)
    dataset  = version.download(ROBOFLOW_FORMAT, location=str(dest))
    print("[Roboflow] ✅ Download complete.")
    return Path(dataset.location)


def get_drone_class_id_roboflow(rf_path: Path) -> int:
    """
    Parse the Roboflow data.yaml to find which integer ID maps to 'drone'.
    drones_new is multi-class (bird, drone, helicopter, plane …).
    """
    yaml_path = rf_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {rf_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", [])
    # names can be a list or a dict {id: name}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    for idx, name in enumerate(names):
        if name.lower() == DRONE_CLASS_NAME:
            print(f"[Roboflow] '{DRONE_CLASS_NAME}' found at class id {idx} "
                  f"(total classes: {len(names)})")
            return idx

    raise ValueError(
        f"Class '{DRONE_CLASS_NAME}' not found in Roboflow data.yaml. "
        f"Available: {names}"
    )


def ingest_roboflow(rf_path: Path, drone_class_id: int, out: Path):
    """
    Roboflow layout: train/{images,labels}  valid/{images,labels}  test/{images,labels}
    We remap the drone class to 0 and drop all other classes.
    """
    print("[Roboflow] Ingesting into merged dataset…")

    split_map = {
        "train": "train",
        "valid": "val",
        "test":  "test",
    }

    for rf_split, out_split in split_map.items():
        src_img = rf_path / rf_split / "images"
        src_lbl = rf_path / rf_split / "labels"
        if not src_img.exists():
            continue

        # Copy files with prefix to avoid name collisions
        dst_img = out / out_split / "images"
        dst_lbl = out / out_split / "labels"
        copy_files(src_img, src_lbl, dst_img, dst_lbl, prefix="rf_")

    # Remap + filter: drone id → 0, drop non-drone lines
    for split in ("train", "val", "test"):
        lbl_dir = out / split / "labels"
        _filter_and_remap(lbl_dir, keep_id=drone_class_id, new_id=FINAL_CLASS_ID,
                          prefix="rf_")

    print("[Roboflow] ✅ Ingestion done.")


def _filter_and_remap(label_dir: Path, keep_id: int, new_id: int, prefix: str):
    """
    For files starting with `prefix`, keep only lines where class == keep_id,
    rewrite them with new_id.  Delete label files that end up empty.
    """
    rf_labels = [f for f in label_dir.glob("*.txt") if f.name.startswith(prefix)]
    removed_imgs = 0
    for txt in tqdm(rf_labels, desc=f"  Filtering {label_dir.parent.name} labels", leave=False):
        lines = txt.read_text().splitlines()
        kept = []
        for line in lines:
            parts = line.split()
            if parts and int(parts[0]) == keep_id:
                parts[0] = str(new_id)
                kept.append(" ".join(parts))

        if kept:
            txt.write_text("\n".join(kept) + "\n")
        else:
            # No drone annotations → remove label and corresponding image
            txt.unlink()
            img_dir = label_dir.parent / "images"
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                img = img_dir / (txt.stem + ext)
                if img.exists():
                    img.unlink()
                    break
            removed_imgs += 1

    if removed_imgs:
        print(f"  ℹ️  Removed {removed_imgs} non-drone frames from {label_dir.parent.name}.")


# ─────────────────────────────────────────────
# STEP 3 — WRITE data.yaml
# ─────────────────────────────────────────────

def write_data_yaml(out: Path):
    cfg = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["drone"],
    }
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[data.yaml] Written → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────
# STEP 4 — SUMMARY
# ─────────────────────────────────────────────

def print_summary(out: Path):
    print("\n" + "═" * 55)
    print("  MERGED DATASET SUMMARY")
    print("═" * 55)
    for split in ("train", "val", "test"):
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        n_imgs  = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_boxes = count_annotations(lbl_dir) if lbl_dir.exists() else 0
        print(f"  {split:<6}  images: {n_imgs:>7,}   boxes: {n_boxes:>8,}")
    print("═" * 55)
    print(f"  Output → {out.resolve()}")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Drone detection dataset pipeline")
    p.add_argument("--roboflow-key", required=True,
                   help="Roboflow API key (get free at app.roboflow.com)")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help=f"Where to write the merged dataset (default: {OUTPUT_DIR})")
    p.add_argument("--staging-dir", default=str(STAGING_DIR),
                   help=f"Temp download location (default: {STAGING_DIR})")
    p.add_argument("--skip-seraphim", action="store_true",
                   help="Skip Seraphim download (use if already staged)")
    p.add_argument("--skip-roboflow", action="store_true",
                   help="Skip Roboflow download (use if already staged)")
    return p.parse_args()


def main():
    args = parse_args()
    out     = Path(args.output_dir)
    staging = Path(args.staging_dir)

    # Create output skeleton
    for split in ("train", "val", "test"):
        ensure_dirs(out / split / "images", out / split / "labels")
    ensure_dirs(staging)

    # ── Dataset 1: Seraphim ──────────────────────────────────────────────
    if not args.skip_seraphim:
        seraphim_path = download_seraphim(staging)
        ingest_seraphim(seraphim_path, out)
    else:
        print("[Seraphim] Skipped.")

    # ── Dataset 2: Roboflow drones_new ───────────────────────────────────
    if not args.skip_roboflow:
        rf_path = download_roboflow(args.roboflow_key, staging)
        drone_id = get_drone_class_id_roboflow(rf_path)
        ingest_roboflow(rf_path, drone_id, out)
    else:
        print("[Roboflow] Skipped.")

    # ── Finalise ─────────────────────────────────────────────────────────
    write_data_yaml(out)
    print_summary(out)

    print("✅ Pipeline complete. Train YOLOv8 with:")
    print(f"   yolo detect train data={out}/data.yaml model=yolov8n.pt epochs=50 imgsz=640\n")


if __name__ == "__main__":
    main()
