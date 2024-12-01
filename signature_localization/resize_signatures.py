import os
from PIL import Image
import numpy as np

def resize_signatures():
    input_dir = "data/interim/cropped_signatures/"
    output_dir = "data/interim/resized_signatures/"
    os.makedirs(output_dir, exist_ok=True)

    # Collect dimensions of all images
    heights = []
    widths = []

    # Loop through all images in the input folder
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)

    # Compute average height and width
    avg_width = int(np.mean(widths))
    avg_height = int(np.mean(heights))

    # Resize images and save to the output folder
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                # Resize the image
                resized_img = img.resize((avg_width, avg_height))
                
                # Save to the output folder
                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)

    print(f"All images resized to average dimensions ({avg_width}x{avg_height}) and saved to '{output_dir}'")