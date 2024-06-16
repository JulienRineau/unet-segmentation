import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms.functional import pil_to_tensor


class PILImageProcessor:
    def __init__(self, output_size=(512, 512), scale_pixels=True):
        self.output_size = output_size
        self.scale_pixels = scale_pixels
        self.transform = transforms.Resize(self.output_size)
        self.scale_pixel = scale_pixels

    def __call__(self, image):
        tensor_img = pil_to_tensor(image)
        if self.scale_pixels:
            tensor_img = tensor_img.float() / 255.0
        return self.transform(tensor_img)


class HuggingFacePILImageDataset(IterableDataset):
    def __init__(self, dataset):
        super(HuggingFacePILImageDataset, self).__init__()
        self.dataset = dataset
        self.image_processor = PILImageProcessor(scale_pixels=True)
        self.annotation_processor = PILImageProcessor(scale_pixels=False)

    def __iter__(self):
        for item in self.dataset:
            for key, value in item.items():
                if key == "image":
                    image = self.image_processor(value)
                    image = image.cpu().numpy()
                elif key == "annotation":
                    mask = self.annotation_processor(value)
                    mask = mask[1].cpu().numpy()
            yield image, mask


def create_segmentation_image(image, mask, alpha=0.2):
    """
    Creates a batch of images with a semi-transparent segmentation mask overlay, and returns both the modified images and the full mask overlays. Ensures background (class 0) remains unaltered in color.

    Args:
        image (numpy array): The original images, shape (N, H, W, C), can be float32 or uint8.
        mask (numpy array): The segmentation masks, shape (N, H, W) or (N, H, W, 1).
        alpha (float): Transparency factor for the overlay (0 is fully transparent, 1 is opaque).

    Returns:
        tuple of numpy arrays: (The images with the segmentation mask overlays, Full mask overlays)
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

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
    train_dataset = load_dataset(
        "scene_parse_150",
        "instance_segmentation",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    train_dataset = HuggingFacePILImageDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

    for i, (image, mask) in enumerate(train_dataloader):
        seg_img, mask = create_segmentation_image(image, mask, 0.5)
        cv2.imwrite(f"img{i}.png", cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
        if i == 10:
            break
