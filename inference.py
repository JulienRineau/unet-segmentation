import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
import cv2
import numpy as np
from unet import UNET, UnetConfig
from dataset import HuggingFacePILImageDataset, custom_collate


class UNETLightning(pl.LightningModule):
    def __init__(self, model_config: UnetConfig):
        super().__init__()
        self.model = UNET(model_config)

    def forward(self, x):
        return self.model(x)


def create_color_palette(num_classes):
    np.random.seed(42)
    return np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)


def create_segmentation_image(image, mask, color_palette, alpha=0.5):
    """
    Creates a segmentation overlay image using a consistent color palette.
    """
    # Ensure image is in the correct format (H, W, C) and uint8
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:  # (C, H, W) format
        image = np.transpose(image, (1, 2, 0))
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:  # (1, H, W) format
        mask = mask.squeeze(0)

    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(color_palette):
        colored_mask[mask == class_id] = color

    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    return overlay


def create_side_by_side_comparison(image, ground_truth, prediction, color_palette):
    gt_overlay = create_segmentation_image(image, ground_truth, color_palette)
    pred_overlay = create_segmentation_image(image, prediction, color_palette)

    # Create side-by-side comparison
    comparison = np.hstack((gt_overlay, pred_overlay))

    return comparison


def main():
    # Load the dataset
    dataset = load_dataset(
        "scene_parse_150", "instance_segmentation", split="train[:100]"
    )
    dataset = HuggingFacePILImageDataset(dataset)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4, collate_fn=custom_collate
    )

    # Load the model
    model_config = UnetConfig(
        out_channels=150
    )  # Adjust as per your model configuration
    model = UNETLightning.load_from_checkpoint(
        "U-Net-overfit/uv75ll4h/checkpoints/epoch=199-step=1400.ckpt",
        model_config=model_config,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    color_palette = create_color_palette(150)

    comparisons = []
    with torch.no_grad():
        for i, (image, mask) in enumerate(dataloader):
            if i >= 20:
                break

            image = image.to(device)
            mask = mask.to(device)

            prediction = model(image)
            prediction = prediction.argmax(dim=1)

            comparison = create_side_by_side_comparison(
                image.squeeze(0).cpu(),
                mask.squeeze(0).cpu(),
                prediction.squeeze(0).cpu(),
                color_palette,
            )
            comparisons.append(comparison)

            cv2.imwrite(
                f"comparison_{i}.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            )
            print(f"Saved comparison_{i}.png")

    if comparisons:
        all_comparisons = np.hstack(comparisons)
        cv2.imwrite(
            "all_comparisons.png", cv2.cvtColor(all_comparisons, cv2.COLOR_RGB2BGR)
        )
        print("Saved all_comparisons.png")


if __name__ == "__main__":
    main()
