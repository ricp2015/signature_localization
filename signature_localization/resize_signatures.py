import os
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np

def resize_signatures():
    input_dir = "data/interim/cropped_signatures/"
    output_dir = "data/interim/resized_signatures/"
    os.makedirs(output_dir, exist_ok=True)

    # Collect dimensions of all images
    widths, heights = [], []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)

    # Calculate average dimensions
    avg_width = int(np.mean(widths))
    avg_height = int(np.mean(heights))

    print(f"Resizing all images to average dimensions: {avg_width}x{avg_height}")

    # Resize and save all images
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                continue
            with Image.open(img_path) as img:
                # Resize image to the average dimensions
                resized_img = img.resize((avg_width, avg_height), Image.Resampling.LANCZOS)

                # Save the resized image
                resized_img.save(output_path)

    print(f"All images resized to average dimensions ({avg_width}x{avg_height}) and saved to '{output_dir}'")
