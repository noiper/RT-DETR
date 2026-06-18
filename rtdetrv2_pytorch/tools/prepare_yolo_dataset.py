#!/usr/bin/env python3
"""Prepare COCO-format video detection data for Ultralytics YOLO training.

The RT-DETR configs in this repository use COCO JSON annotations. Ultralytics
YOLO expects image files under an ``images`` tree and normalized txt labels
under a parallel ``labels`` tree. This script converts the annotations and
symlinks images into that layout without copying the underlying frames.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PRESETS = {
    "mot17": {
        "train_img_root": "../dataset/mot17/train",
        "train_ann": "../dataset/mot17/train.json",
        "val_img_root": "../dataset/mot17/val",
        "val_ann": "../dataset/mot17/val.json",
        "out_dir": "datasets/yolo_mot17",
    },
    "virat30": {
        "train_img_root": "../dataset/VIRAT/train_30",
        "train_ann": "../dataset/VIRAT/train_30.json",
        "val_img_root": "../dataset/VIRAT/val_30",
        "val_ann": "../dataset/VIRAT/val_30.json",
        "out_dir": "datasets/yolo_virat30",
    },
}

IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO detection annotations to Ultralytics YOLO format."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Use repository defaults for a known dataset.",
    )
    parser.add_argument("--train-img-root", type=Path, help="Training image root.")
    parser.add_argument("--train-ann", type=Path, help="Training COCO annotation JSON.")
    parser.add_argument("--val-img-root", type=Path, help="Validation image root.")
    parser.add_argument("--val-ann", type=Path, help="Validation COCO annotation JSON.")
    parser.add_argument("--out-dir", type=Path, help="Output Ultralytics dataset root.")
    parser.add_argument(
        "--image-link",
        choices=("symlink", "hardlink", "copy", "none"),
        default="symlink",
        help="How to expose images under out-dir/images. Default: symlink.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing image links and label files.",
    )
    parser.add_argument(
        "--drop-crowd",
        action="store_true",
        help="Drop COCO annotations with iscrowd=1. Not recommended for MOT17.",
    )
    parser.add_argument(
        "--train-frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame per video in the train split.",
    )
    parser.add_argument(
        "--val-frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame per video in the val split.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit applied after frame stride filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read annotations and report the planned conversion without writing files.",
    )
    args = parser.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        for key, value in preset.items():
            attr = key.replace("-", "_")
            if getattr(args, attr, None) is None:
                setattr(args, attr, Path(value))

    required = ("train_img_root", "train_ann", "val_img_root", "val_ann", "out_dir")
    missing = [name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{name}" for name in missing))
    if args.train_frame_stride < 1 or args.val_frame_stride < 1:
        parser.error("--train-frame-stride and --val-frame-stride must be >= 1")
    if args.max_images_per_split is not None and args.max_images_per_split < 1:
        parser.error("--max-images-per-split must be >= 1")
    return args


def resolve_existing(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"{path} is missing required COCO key: {key}")
    return data


def category_mapping(*datasets: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, str]]:
    categories: Dict[int, str] = {}
    for data in datasets:
        for category in data["categories"]:
            cat_id = int(category["id"])
            categories.setdefault(cat_id, str(category.get("name", cat_id)))

    sorted_cat_ids = sorted(categories)
    cat_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    names = {cat_id_to_yolo[cat_id]: categories[cat_id] for cat_id in sorted_cat_ids}
    return cat_id_to_yolo, names


def video_key(image: Dict[str, Any]) -> str:
    if "video_name" in image:
        return str(image["video_name"])
    file_name = Path(str(image["file_name"]))
    if len(file_name.parts) > 1:
        return file_name.parts[0]
    return "default"


def frame_key(image: Dict[str, Any]) -> Tuple[int, str]:
    for field in ("frame_id", "frame_number"):
        if field in image:
            return int(image[field]), str(image["file_name"])

    stem = Path(str(image["file_name"])).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0, str(image["file_name"])


def select_images(
    images: Sequence[Dict[str, Any]],
    frame_stride: int,
    max_images: Optional[int],
) -> List[Dict[str, Any]]:
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for image in images:
        by_video[video_key(image)].append(image)

    selected: List[Dict[str, Any]] = []
    for _, video_images in sorted(by_video.items()):
        ordered = sorted(video_images, key=frame_key)
        selected.extend(image for idx, image in enumerate(ordered) if idx % frame_stride == 0)

    selected.sort(key=lambda image: str(image["file_name"]))
    if max_images is not None:
        selected = selected[:max_images]
    return selected


def grouped_annotations(annotations: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def yolo_label_lines(
    image: Dict[str, Any],
    anns: Sequence[Dict[str, Any]],
    cat_id_to_yolo: Dict[int, int],
    drop_crowd: bool,
) -> Tuple[List[str], int]:
    width = float(image["width"])
    height = float(image["height"])
    lines: List[str] = []
    skipped = 0

    for ann in anns:
        cat_id = int(ann.get("category_id", -1))
        if cat_id not in cat_id_to_yolo:
            skipped += 1
            continue
        if drop_crowd and int(ann.get("iscrowd", 0)) == 1:
            skipped += 1
            continue

        x, y, box_w, box_h = (float(value) for value in ann["bbox"])
        x1 = max(0.0, min(x, width))
        y1 = max(0.0, min(y, height))
        x2 = max(0.0, min(x + box_w, width))
        y2 = max(0.0, min(y + box_h, height))
        box_w = x2 - x1
        box_h = y2 - y1
        if width <= 0 or height <= 0 or box_w <= 0 or box_h <= 0:
            skipped += 1
            continue

        cx = (x1 + box_w / 2.0) / width
        cy = (y1 + box_h / 2.0) / height
        norm_w = box_w / width
        norm_h = box_h / height
        cls = cat_id_to_yolo[cat_id]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

    return lines, skipped


def replace_suffix(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if mode == "none":
        return
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported image link mode: {mode}")


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def convert_split(
    split: str,
    data: Dict[str, Any],
    img_root: Path,
    out_dir: Path,
    cat_id_to_yolo: Dict[int, int],
    drop_crowd: bool,
    image_link: str,
    overwrite: bool,
    frame_stride: int,
    max_images: Optional[int],
    dry_run: bool,
) -> Dict[str, int]:
    images = select_images(data["images"], frame_stride, max_images)
    selected_ids = {int(image["id"]) for image in images}
    anns_by_image = grouped_annotations(data["annotations"])

    stats = {
        "images": 0,
        "boxes": 0,
        "empty_labels": 0,
        "skipped_annotations": 0,
        "missing_images": 0,
    }

    for image in images:
        image_id = int(image["id"])
        rel_image = Path(str(image["file_name"]))
        if rel_image.suffix.lower() not in IMAGE_EXTS:
            stats["missing_images"] += 1
            continue
        src_image = img_root / rel_image
        if not src_image.exists():
            stats["missing_images"] += 1
            continue

        lines, skipped = yolo_label_lines(
            image,
            anns_by_image.get(image_id, []),
            cat_id_to_yolo,
            drop_crowd,
        )
        stats["images"] += 1
        stats["boxes"] += len(lines)
        stats["empty_labels"] += int(not lines)
        stats["skipped_annotations"] += skipped

        if dry_run:
            continue

        image_dst = out_dir / "images" / split / rel_image
        label_dst = replace_suffix(out_dir / "labels" / split / rel_image, ".txt")
        link_or_copy(src_image, image_dst, image_link, overwrite)
        write_text(label_dst, "\n".join(lines) + ("\n" if lines else ""), overwrite)

    skipped_unselected = sum(
        1 for ann in data["annotations"] if int(ann["image_id"]) not in selected_ids
    )
    stats["annotations_outside_stride"] = skipped_unselected
    return stats


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_data_yaml(out_dir: Path, names: Dict[int, str]) -> None:
    lines = [
        f"path: {yaml_quote(str(out_dir.resolve()))}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {idx}: {yaml_quote(name)}" for idx, name in sorted(names.items()))
    write_text(out_dir / "data.yaml", "\n".join(lines) + "\n", overwrite=True)


def print_stats(split: str, stats: Dict[str, int]) -> None:
    print(
        f"{split}: images={stats['images']} boxes={stats['boxes']} "
        f"empty_labels={stats['empty_labels']} skipped_annotations={stats['skipped_annotations']} "
        f"missing_images={stats['missing_images']} "
        f"annotations_outside_stride={stats['annotations_outside_stride']}"
    )


def main() -> None:
    args = parse_args()
    train_img_root = resolve_existing(args.train_img_root)
    val_img_root = resolve_existing(args.val_img_root)
    train_ann = resolve_existing(args.train_ann)
    val_ann = resolve_existing(args.val_ann)
    out_dir = args.out_dir.expanduser().resolve()

    train_data = load_coco(train_ann)
    val_data = load_coco(val_ann)
    cat_id_to_yolo, names = category_mapping(train_data, val_data)

    print("Class mapping:")
    for cat_id, yolo_id in sorted(cat_id_to_yolo.items(), key=lambda item: item[1]):
        print(f"  COCO category {cat_id} -> YOLO class {yolo_id} ({names[yolo_id]})")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_data_yaml(out_dir, names)

    train_stats = convert_split(
        "train",
        train_data,
        train_img_root,
        out_dir,
        cat_id_to_yolo,
        args.drop_crowd,
        args.image_link,
        args.overwrite,
        args.train_frame_stride,
        args.max_images_per_split,
        args.dry_run,
    )
    val_stats = convert_split(
        "val",
        val_data,
        val_img_root,
        out_dir,
        cat_id_to_yolo,
        args.drop_crowd,
        args.image_link,
        args.overwrite,
        args.val_frame_stride,
        args.max_images_per_split,
        args.dry_run,
    )

    print_stats("train", train_stats)
    print_stats("val", val_stats)
    if args.dry_run:
        print(f"Dry run only. Planned dataset root: {out_dir}")
    else:
        print(f"Wrote Ultralytics dataset: {out_dir}")
        print(f"Use data YAML: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
