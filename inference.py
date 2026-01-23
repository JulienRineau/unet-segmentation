import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset import ADE20KDataConfig, ADE20KDataset, colorize_mask, make_color_palette
from train import SegmentationLitModule


def create_side_by_side(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.hstack([image, gt, pred])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADE20K inference on a small split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="inference_outputs")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_raw = load_dataset("scene_parse_150", split=f"validation[:{args.num_samples}]")
    config = ADE20KDataConfig(image_size=args.image_size)
    dataset = ADE20KDataset(dataset_raw, config, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SegmentationLitModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model = model.to(device)

    palette = make_color_palette(150)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, (image, mask) in enumerate(dataloader):
            image = image.to(device)
            mask = mask.to(device)
            logits = model(image)
            pred = torch.argmax(logits, dim=1)

            img = image[0].cpu()
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
            img = (img * 255).astype(np.uint8)

            gt_color = colorize_mask(mask[0], palette, config.ignore_index)
            pred_color = colorize_mask(pred[0], palette, config.ignore_index)
            panel = create_side_by_side(img, gt_color, pred_color)

            out_path = output_dir / f"sample_{idx:03d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
