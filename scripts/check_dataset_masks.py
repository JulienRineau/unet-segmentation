import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import ADE20KDataConfig, ADE20KDataModule, ADE20K_IGNORE_INDEX


NUM_CLASSES = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ADE20K mask semantics and ignore ratios after transforms."
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, choices=("train", "val"), default="train")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--train-resize-only", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iter_raw_masks(dataset, indices: list[int]) -> list[np.ndarray]:
    if hasattr(dataset, "mask_paths") and dataset.mask_paths is not None:
        arrays = []
        for idx in indices:
            mask_path = dataset.mask_paths[idx]
            arrays.append(np.array(Image.open(mask_path)))
        return arrays
    if hasattr(dataset, "dataset"):
        arrays = []
        for idx in indices:
            item = dataset.dataset[idx]
            arrays.append(np.array(item["annotation"]))
        return arrays
    raise ValueError("Unsupported dataset type for raw mask inspection.")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    config = ADE20KDataConfig(
        image_size=args.image_size,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        hflip_prob=args.hflip_prob,
        data_root=args.data_root,
        train_resize_only=args.train_resize_only,
    )
    datamodule = ADE20KDataModule(config)
    datamodule.setup()
    dataset = datamodule.train_dataset if args.split == "train" else datamodule.val_dataset

    if dataset is None:
        raise ValueError(f"Dataset split {args.split} not found.")

    sample_count = min(args.num_samples, len(dataset))
    indices = list(range(sample_count))

    raw_masks = iter_raw_masks(dataset, indices)
    raw_mins = [int(mask.min()) for mask in raw_masks]
    raw_maxs = [int(mask.max()) for mask in raw_masks]
    raw_unique = set()
    for mask in raw_masks:
        if len(raw_unique) < 1000:
            raw_unique.update(np.unique(mask).tolist())

    mapped_mins = []
    mapped_maxs = []
    ignore_ratios = []
    invalid_counts = []

    for idx in indices:
        _, mask = dataset[idx]
        ignore_mask = mask == config.ignore_index
        ignore_ratio = ignore_mask.float().mean().item()
        ignore_ratios.append(ignore_ratio)
        valid = ~ignore_mask
        if valid.any():
            mapped_mins.append(int(mask[valid].min().item()))
            mapped_maxs.append(int(mask[valid].max().item()))
        invalid = mask[valid & ((mask < 0) | (mask >= NUM_CLASSES))]
        invalid_counts.append(int(invalid.numel()))

    print(f"Split: {args.split}  Samples: {sample_count}")
    print(f"Raw mask min/max (sampled): {min(raw_mins)} / {max(raw_maxs)}")
    unique_sorted = sorted(raw_unique)
    unique_preview = unique_sorted[:10]
    unique_tail = unique_sorted[-10:] if len(unique_sorted) > 10 else unique_sorted
    print(
        f"Raw mask unique values (sampled): {unique_preview} ... {unique_tail} "
        f"(count={len(raw_unique)})"
    )

    if mapped_mins and mapped_maxs:
        print(f"Mapped mask min/max (valid): {min(mapped_mins)} / {max(mapped_maxs)}")
    else:
        print("Mapped mask min/max (valid): <no valid pixels>")

    ignore_mean = float(np.mean(ignore_ratios))
    ignore_p95 = float(np.percentile(ignore_ratios, 95))
    print(
        f"Ignore ratio after transforms: mean={ignore_mean:.4f} p95={ignore_p95:.4f}"
    )

    total_invalid = sum(invalid_counts)
    if total_invalid > 0:
        raise ValueError(
            f"Found {total_invalid} invalid mapped labels outside [0,{NUM_CLASSES - 1}] "
            f"and ignore_index={config.ignore_index}."
        )


if __name__ == "__main__":
    main()
