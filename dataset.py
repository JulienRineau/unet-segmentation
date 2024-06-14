import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms.functional import pil_to_tensor
import torch


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
            transformed_item = {}
            for key, value in item.items():
                if key == "image":
                    image_tensor = self.image_processor(value)
                    transformed_item[key] = image_tensor
                elif key == "annotation":
                    mask_tensor = self.annotation_processor(value)
                    transformed_item[key] = mask_tensor
                else:
                    transformed_item[key] = value
            yield transformed_item


def create_segmentation_image(mask, image, alpha=0.2):
    """
    Creates an image with a semi-transparent segmentation mask overlay, and returns both the modified image and the full mask overlay. Ensures background (class 0) remains unaltered in color.

    Args:
        mask (numpy array): The segmentation mask, shape (H, W) or (H, W, 1).
        image (numpy array): The original image, shape (H, W, C), can be float32 or uint8.
        alpha (float): Transparency factor for the overlay (0 is fully transparent, 1 is opaque).

    Returns:
        tuple of numpy arrays: (The image with the segmentation mask overlay, Full mask overlay)
    """
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = np.squeeze(mask)  # Reduce (H, W, 1) to (H, W)

    # Check and adjust the image data type and scale
    if image.dtype == np.float32:
        # Assuming float32 image is in the range [0, 1]
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("Image dtype must be uint8 or float32")

    # Identify unique classes present in the mask
    unique_classes = np.unique(mask)
    num_classes = len(unique_classes)  # The number of unique values in the mask

    np.random.seed(42)  # Fixing the seed for color consistency
    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)

    # Create a dictionary to map class values to colors
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    output_image = image.copy()  # Start with a copy of the original image
    full_mask_overlay = np.zeros_like(
        output_image, dtype=np.uint8
    )  # Initialize full mask overlay

    for cls in unique_classes:
        class_mask = mask == cls
        if cls != 0:  # Apply colors only for non-background classes
            color = class_to_color[cls]
            full_mask_overlay[class_mask] = color  # Update full mask overlay with color
        else:
            full_mask_overlay[class_mask] = image[
                class_mask
            ]  # Copy original image color for background

    # Blend overlay with the image using a custom alpha
    cv2.addWeighted(full_mask_overlay, alpha, output_image, 1 - alpha, 0, output_image)

    return output_image, full_mask_overlay


if __name__ == "__main__":
    train_dataset = load_dataset(
        "scene_parse_150", "instance_segmentation", split="train", streaming=True
    )
    train_dataset = HuggingFacePILImageDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

    validation_dataset = load_dataset(
        "scene_parse_150", "instance_segmentation", split="validation", streaming=True
    )
    validation_dataset = HuggingFacePILImageDataset(validation_dataset)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=4
    )

    test_dataset = load_dataset(
        "scene_parse_150", "instance_segmentation", split="test", streaming=True
    )
    test_dataset = HuggingFacePILImageDataset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    for i, data in enumerate(train_dataloader):
        mask = data["annotation"][0][0].cpu().numpy()
        mask = mask.reshape((512, 512, 1))  # Adjust shape if necessary
        image = data["image"][0].cpu().numpy()
        image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

        seg_img, mask = create_segmentation_image(mask, image, 0.5)
        cv2.imwrite(f"img{i}.png", cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
        if i == 10:
            break
