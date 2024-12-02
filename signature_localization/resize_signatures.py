import os
from PIL import Image, ImageOps
from tqdm import tqdm

def resize_signatures():
    input_dir = "data/interim/cropped_signatures/"
    output_dir = "data/interim/resized_signatures/"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all images in the input folder to determine max dimensions
    max_width = 0
    max_height = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                max_width = max(max_width, img.width)
                max_height = max(max_height, img.height)

    # Loop through images again to add padding and save them
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                continue
            with Image.open(img_path) as img:
                # Calculate padding to match max dimensions
                padding = (
                    (max_width - img.width) // 2,  # Left padding
                    (max_height - img.height) // 2, # Top padding
                    (max_width - img.width + 1) // 2, # Right padding
                    (max_height - img.height + 1) // 2 # Bottom padding
                )

                # Add padding to the image (fill with white pixels)
                padded_img = ImageOps.expand(img, padding, fill=0xffffff)  # White color

                padded_img.save(output_path)

    print(f"All images padded to maximum dimensions ({max_width}x{max_height}) and saved to '{output_dir}'")
