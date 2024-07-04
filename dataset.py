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
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(image_size),
                transforms.Lambda(lambda img: pil_to_tensor(img)),
                transforms.Lambda(lambda t: t.float() / 255.0),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.NEAREST),
                transforms.Lambda(lambda img: pil_to_tensor(img)),
            ]
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        image = self.image_transform(item["image"])
        mask = self.mask_transform(item["annotation"])
        # Ensure mask is a single-channel tensor
        mask = mask[0] if mask.shape[0] > 1 else mask
        return image, mask

    def __len__(self):
        return len(self.dataset)


def custom_collate(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]

    # Pad images and masks to the same size
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])

    padded_images = []
    padded_masks = []

    for img, mask in zip(images, masks):
        p_img = torch.zeros((3, max_h, max_w), dtype=torch.float32)
        p_mask = torch.zeros((1, max_h, max_w), dtype=torch.long)

        p_img[:, : img.shape[1], : img.shape[2]] = img
        p_mask[:, : mask.shape[0], : mask.shape[1]] = mask

        padded_images.append(p_img)
        padded_masks.append(p_mask)

    return torch.stack(padded_images), torch.stack(padded_masks)


def create_segmentation_image(image, mask, alpha=0.2):
    """
    Creates a batch of images with a semi-transparent segmentation mask overlay.
    Args:
    image (torch.Tensor): The original images, shape (N, C, H, W).
    mask (torch.Tensor): The segmentation masks, shape (N, H, W) or (N, 1, H, W).
    alpha (float): Transparency factor for the overlay (0 is fully transparent, 1 is opaque).
    Returns:
    tuple of numpy arrays: (The images with the segmentation mask overlays, Full mask overlays)
    """
    # Move tensors to CPU and convert to numpy
    image = image.cpu().numpy()
    mask = mask.cpu().numpy()

    # Transpose image from (N, C, H, W) to (N, H, W, C)
    image = np.transpose(image, (0, 2, 3, 1))

    # Ensure mask is (N, H, W)
    if mask.ndim == 4:
        mask = np.squeeze(mask, axis=1)

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
        for i in range(image.shape[0]):  # Iterate over each image in the batch
            full_masks_overlay[i][class_mask[i]] = class_to_color[cls]

    # Blend overlay with the image using a custom alpha
    for i in range(image.shape[0]):
        cv2.addWeighted(
            full_masks_overlay[i],
            alpha,
            output_images[i],
            1 - alpha,
            0,
            output_images[i],
        )

    return output_images, full_masks_overlay


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    train_dataset_raw = load_dataset(
        "scene_parse_150",
        "instance_segmentation",
        split="train",
        trust_remote_code=True,
    )

    print(f"Raw dataset size: {len(train_dataset_raw)}")
    print(f"Raw dataset features: {train_dataset_raw.features}")
    print(f"Sample raw data item: {train_dataset_raw[0]}")

    train_dataset = HuggingFacePILImageDataset(train_dataset_raw)

    print(f"\nTransformed dataset size: {len(train_dataset)}")
    sample_image, sample_mask = train_dataset[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample mask shape: {sample_mask.shape}")
    print(f"Image value range: ({sample_image.min():.2f}, {sample_image.max():.2f})")
    print(f"Unique mask values: {torch.unique(sample_mask)}")

    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    print(f"\nNumber of batches in dataloader: {len(train_dataloader)}")

    num_samples = 3
    for i, (image, mask) in enumerate(train_dataloader):
        image = image.to(device)
        mask = mask.to(device)

        # Move tensors back to CPU before passing to create_segmentation_image
        seg_img, mask_overlay = create_segmentation_image(image.cpu(), mask.cpu(), 0.5)

        for j in range(seg_img.shape[0]):
            output_filename = f"sample_img_{i*batch_size+j+1}.png"
            cv2.imwrite(output_filename, cv2.cvtColor(seg_img[j], cv2.COLOR_RGB2BGR))
            print(f"  Saved {output_filename}")

        if i == num_samples - 1:
            break

    print("\nImage generation and dataset analysis complete.")
