import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import ADE20KDataConfig, ADE20KDataModule, colorize_mask, make_color_palette
from segmentation_metrics import compute_per_image_iou
from train import SegmentationLitModule


DEFAULT_CHECKPOINT = (
    "/home/ubuntu/Texas/personal/unet-segmentation/checkpoints/"
    "ade20k-unet-epoch=079-val_miou=0.3041.ckpt"
)
DEFAULT_DATA_ROOT = "/home/ubuntu/Texas/personal/unet-segmentation/data"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/Texas/personal/unet-segmentation/overlays"
DEFAULT_IMAGE_SIZE = 512
MAX_SCAN = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ADE20K segmentation panels from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, choices=("val", "test"), default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-iou", type=float, default=0.0)
    return parser.parse_args()


def resolve_checkpoint_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_file():
        return path

    if path_str == DEFAULT_CHECKPOINT:
        repo_candidate = REPO_ROOT / "checkpoints" / path.name
        if repo_candidate.is_file():
            print(f"Default checkpoint not found at {path}; using {repo_candidate} instead.")
            return repo_candidate

    available = sorted(
        (p for p in REPO_ROOT.rglob("*.ckpt") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    suggestions = ""
    if available:
        lines = "\n".join(f"  - {p.relative_to(REPO_ROOT)}" for p in available[:10])
        suggestions = f"\nAvailable checkpoints under {REPO_ROOT}:\n{lines}"

    raise FileNotFoundError(
        f"Checkpoint not found: {path}\n"
        f"Pass --checkpoint to a valid .ckpt path.{suggestions}"
    )


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: torch.Tensor | np.ndarray,
    palette: np.ndarray,
    ignore_index: int,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    color = colorize_mask(mask, palette, ignore_index)
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(0)
    valid = mask_np != ignore_index
    if not np.any(valid):
        return image.copy()
    image_f = image.astype(np.float32)
    color_f = color.astype(np.float32)
    blended = image_f.copy()
    blended[valid] = (1.0 - alpha) * image_f[valid] + alpha * color_f[valid]
    return blended.astype(np.uint8)


def format_panel_name(
    prefix: str,
    idx: int,
    width: int,
    filename: str | None,
    iou: float | None,
    dataset_index: int | None,
) -> str:
    stem = None
    if filename:
        stem = Path(str(filename)).stem
    name = f"{prefix}_{idx:0{width}d}"
    if dataset_index is not None:
        name = f"{name}_idx{dataset_index:0{width}d}"
    if iou is not None:
        name = f"{name}_iou{iou:.3f}"
    if stem:
        name = f"{name}_{stem}"
    return f"{name}.png"


def build_panel(image: np.ndarray, gt: np.ndarray | None, pred: np.ndarray) -> np.ndarray:
    if gt is None:
        return np.concatenate([image, pred], axis=1)
    return np.concatenate([image, gt, pred], axis=1)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be >= 1.")
    if args.min_iou < 0:
        raise ValueError("--min-iou must be >= 0.")

    min_iou = args.min_iou
    if args.split == "test" and min_iou > 0:
        print("Warning: --min-iou ignored for test split (no ground truth).")
        min_iou = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model = SegmentationLitModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.eval()
    model = model.to(device)

    config = ADE20KDataConfig(
        image_size=DEFAULT_IMAGE_SIZE,
        val_batch_size=1,
        num_workers=0,
        ignore_index=model.ignore_index,
        data_root=args.data_root,
        val_subset=(
            args.num_samples
            if args.split == "val" and min_iou == 0
            else MAX_SCAN if args.split == "val" else None
        ),
        test_subset=args.num_samples if args.split == "test" else None,
    )
    datamodule = ADE20KDataModule(config)
    datamodule.setup()

    if args.split == "val":
        dataset = datamodule.val_dataset
    else:
        dataset = datamodule.test_dataset

    if dataset is None:
        raise ValueError(f"Dataset split {args.split} not found.")
    if len(dataset) == 0:
        raise ValueError(f"Dataset split {args.split} is empty.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    palette = make_color_palette(model.num_classes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(args.num_samples)))

    if args.split == "val" and min_iou > 0:
        candidates = []
        scanned = 0
        max_scan = min(len(dataset), MAX_SCAN)
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if scanned >= max_scan:
                    break
                scanned += 1

                image, mask = batch
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                logits = model(image)
                preds = torch.argmax(logits, dim=1)
                iou_score = compute_per_image_iou(
                    preds[0], mask[0], model.num_classes, model.ignore_index
                )
                if iou_score < min_iou:
                    continue

                filename = None
                if hasattr(dataset, "image_paths"):
                    try:
                        filename = dataset.image_paths[idx].name
                    except (AttributeError, IndexError):
                        filename = None
                candidates.append({"idx": idx, "iou": iou_score, "filename": filename})

        if not candidates:
            raise ValueError("No samples matched --min-iou filter.")

        if len(candidates) < args.num_samples:
            print(
                f"Warning: found {len(candidates)}/{args.num_samples} samples after scanning "
                f"{scanned} images with --min-iou {min_iou:.3f} (cap={max_scan})."
            )

        selected = random.sample(candidates, k=min(args.num_samples, len(candidates)))

        panels = []
        saved = 0
        with torch.no_grad():
            for saved_idx, candidate in enumerate(selected):
                idx = candidate["idx"]
                filename = candidate["filename"]
                iou_score = candidate["iou"]

                image, mask = dataset[idx]
                image = image.unsqueeze(0).to(device, non_blocking=True)
                logits = model(image)
                preds = torch.argmax(logits, dim=1)

                image_np = denormalize(image[0])
                pred_overlay = overlay_mask(
                    image_np, preds[0], palette, model.ignore_index, args.alpha
                )
                gt_overlay = overlay_mask(
                    image_np, mask, palette, model.ignore_index, args.alpha
                )

                panel = build_panel(image_np, gt_overlay, pred_overlay)
                panels.append(panel)

                panel_name = format_panel_name(
                    args.split,
                    saved_idx,
                    width,
                    filename,
                    iou_score,
                    idx,
                )
                Image.fromarray(panel).save(output_dir / panel_name)
                saved += 1
    else:
        panels = []
        saved = 0
        scanned = 0
        max_scan = min(len(dataset), MAX_SCAN)
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if saved >= args.num_samples:
                    break
                if scanned >= max_scan:
                    break
                scanned += 1

                filename = None
                gt_overlay = None
                iou_score = None
                if args.split == "val":
                    image, mask = batch
                    if hasattr(dataset, "image_paths"):
                        try:
                            filename = dataset.image_paths[idx].name
                        except (AttributeError, IndexError):
                            filename = None
                    mask = mask.to(device, non_blocking=True)
                else:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        image, filename = batch
                        if isinstance(filename, (list, tuple)):
                            filename = filename[0] if filename else None
                    else:
                        image = batch

                image = image.to(device, non_blocking=True)
                logits = model(image)
                preds = torch.argmax(logits, dim=1)

                image_np = denormalize(image[0])
                pred_overlay = overlay_mask(
                    image_np, preds[0], palette, model.ignore_index, args.alpha
                )
                if args.split == "val":
                    gt_overlay = overlay_mask(
                        image_np, mask[0], palette, model.ignore_index, args.alpha
                    )
                    iou_score = compute_per_image_iou(
                        preds[0], mask[0], model.num_classes, model.ignore_index
                    )

                panel = build_panel(image_np, gt_overlay, pred_overlay)
                panels.append(panel)

                panel_name = format_panel_name(
                    args.split,
                    saved,
                    width,
                    filename,
                    iou_score,
                    idx,
                )
                Image.fromarray(panel).save(output_dir / panel_name)
                saved += 1

        if saved < args.num_samples and min_iou > 0:
            print(
                f"Warning: saved {saved}/{args.num_samples} samples after scanning {scanned} "
                f"images with --min-iou {min_iou:.3f} (cap={max_scan})."
            )

    if not panels:
        raise ValueError("No samples matched --min-iou filter.")

    grid = np.concatenate(panels, axis=0)
    grid_path = output_dir / f"{args.split}_grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"Saved {len(panels)} panels to {output_dir}")
    print(f"Saved grid to {grid_path}")


if __name__ == "__main__":
    main()
