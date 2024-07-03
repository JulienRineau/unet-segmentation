import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms.functional import pil_to_tensor


class HuggingFacePILImageDataset(Dataset):
    def __init__(self, dataset, image_size=(512, 512)):
        super(HuggingFacePILImageDataset, self).__init__()
        self.dataset = dataset

        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: pil_to_tensor(img)),
                transforms.Resize(image_size),
                transforms.Lambda(lambda t: t.float() / 255.0),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: pil_to_tensor(img)),
                transforms.Resize(image_size),
            ]
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        image = self.image_transform(item["image"])
        mask = self.mask_transform(item["annotation"])

        # Ensure mask is a single-channel tensor
        mask = mask[0] if mask.shape[0] > 1 else mask

        return image.numpy(), mask.numpy()

    def __len__(self):
        return len(self.dataset)


def create_segmentation_image(image, mask, alpha=0.2):
    """
    Creates a batch of images with a semi-transparent segmentation mask overlay.

    Args:
        image (numpy array or torch.Tensor): The original images, shape (N, H, W, C).
        mask (numpy array or torch.Tensor): The segmentation masks, shape (N, H, W) or (N, H, W, 1).
        alpha (float): Transparency factor for the overlay (0 is fully transparent, 1 is opaque).

    Returns:
        tuple of numpy arrays: (The images with the segmentation mask overlays, Full mask overlays)
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Check and adjust the image data type and scale
    if image.dtype == np.float32:
        # Assuming float32 image is in the range [0, 1]
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    unique_classes = np.unique(mask)
    colors = np.random.randint(0, 255, (len(unique_classes), 3), dtype=np.uint8)
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    output_images = image.copy()  # Start with a copy of the original images
    full_masks_overlay = np.zeros_like(
        output_images, dtype=np.uint8
    )  # Initialize full masks overlay

    for cls in unique_classes:
        class_mask = mask == cls
        class_mask = mask[:, None, :, :]
        class_mask = np.repeat(class_mask, 3, axis=1)
        if cls != 0:
            full_masks_overlay[class_mask] = class_to_color[cls]
        else:
            full_masks_overlay[class_mask] = image[class_mask]

    # Blend overlay with the image using a custom alpha
    cv2.addWeighted(
        full_masks_overlay, alpha, output_images, 1 - alpha, 0, output_images
    )

    return output_images, full_masks_overlay


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset(
        "scene_parse_150",
        "instance_segmentation",
        split="train",
    )

    train_dataset = HuggingFacePILImageDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

    for i, (image, mask) in enumerate(train_dataloader):
        image = image.to(device)
        mask = mask.to(device)
        seg_img, mask = create_segmentation_image(image, mask, 0.5)
        cv2.imwrite(f"img{i}.png", cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
        if i == 10:
            break
