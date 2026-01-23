import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import ADE20KDataConfig, ADE20KDataModule, colorize_mask, make_color_palette


NUM_CLASSES = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save image+GT panels after dataset transforms for alignment checks."
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, choices=("train", "val"), default="train")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--train-resize-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default="debug_panels")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)


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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = make_color_palette(NUM_CLASSES)

    count = min(args.num_samples, len(dataset))
    for idx in range(count):
        image, mask = dataset[idx]
        image_np = denormalize(image)
        mask_np = colorize_mask(mask, palette, config.ignore_index)
        panel = np.concatenate([image_np, mask_np], axis=1)

        name = None
        if hasattr(dataset, "image_paths"):
            name = dataset.image_paths[idx].stem
        filename = f"{idx:03d}.png" if name is None else f"{idx:03d}_{name}.png"
        Image.fromarray(panel).save(out_dir / filename)

    print(f"Saved {count} panels to {out_dir}")


if __name__ == "__main__":
    main()
