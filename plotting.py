import cv2
import numpy as np
import os


def create_horizontal_image_stack(image_paths, output_path, target_height=500):
    """
    Create a horizontal stack of images and save it.

    Args:
    image_paths (list): List of paths to the images.
    output_path (str): Path where the output image will be saved.
    target_height (int): Height to which all images will be resized, maintaining aspect ratio.

    Returns:
    bool: True if successful, False otherwise.
    """
    # List to store resized images
    images = []

    for path in image_paths:
        # Read the image
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read image {path}")
            continue

        # Calculate aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]

        # Resize image
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, target_height))

        # Add to list
        images.append(resized_img)

    if not images:
        print("Error: No valid images found")
        return False

    # Concatenate images horizontally
    result = np.vstack(images)

    # Save the result
    cv2.imwrite(output_path, result)

    print(f"Horizontal stack saved to {output_path}")
    return True


# Example usage
if __name__ == "__main__":
    image_paths = [
        "comparison_12.png",
        "comparison_14.png",
        "comparison_17.png",
    ]
    output_path = "stacked_result.png"

    create_horizontal_image_stack(image_paths, output_path)
