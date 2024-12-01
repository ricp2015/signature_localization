import os
import ast
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd


def create_nonsig_dataset():
    images_dir = "data/raw/signverod_dataset"
    output_dir = "data/interim/nonsig_dataset"
    annotations_path = "data/raw/fixed_dataset/full_data.csv"
    image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"
    resized_signatures_dir = "data/interim/resized_signatures/"
    os.makedirs(output_dir, exist_ok=True)

    # Load bounding box annotations and image metadata
    annotations = pd.read_csv(annotations_path)
    image_info = pd.read_csv(image_info_path)

    # Get the size of the first image in resized_signatures_dir
    first_image_path = os.listdir(resized_signatures_dir)[0]
    first_image = Image.open(os.path.join(resized_signatures_dir, first_image_path))
    crop_width, crop_height = first_image.size  # Set crop size to match the first image

    # Process each document
    for _, image_row in image_info.iterrows():
        image_id = image_row['id']
        file_name = image_row['file_name']
        img_width = int(image_row['width'])
        img_height = int(image_row['height'])
        img_path = os.path.join(images_dir, file_name)
        
        # Open the image
        if not os.path.exists(img_path):
            print(f"Image {file_name} not found, skipping.")
            continue
        img = Image.open(img_path).convert("RGB")

        # Create a white mask for the signature regions
        mask = Image.new("L", (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask)

        # Find signature bounding boxes for this image
        bboxes = annotations[annotations['image_id'] == image_id]
        for _, row in bboxes.iterrows():
            if row['category_id'] == 1:  # Signature category
                bbox = ast.literal_eval(row['bbox'])
                x_min = int(bbox[0] * img_width)
                y_min = int(bbox[1] * img_height)
                x_max = int((bbox[0] + bbox[2]) * img_width)
                y_max = int((bbox[1] + bbox[3]) * img_height)
                draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

        # Mask the image
        img_array = np.array(img)
        mask_array = np.array(mask)
        img_array[mask_array == 255] = [255, 255, 255]  # Colorize masked areas white
        masked_img = Image.fromarray(img_array)

        # Extract sections based on the dimensions of the first image (crop_width x crop_height)
        for top in range(0, img_height, crop_height):
            for left in range(0, img_width, crop_width):
                bottom = min(top + crop_height, img_height)
                right = min(left + crop_width, img_width)
                
                # Crop the region
                crop = masked_img.crop((left, top, right, bottom))
                
                # Skip crops containing too much white (masked signature regions)
                crop_array = np.array(crop)
                white_ratio = np.sum(crop_array == 255) / crop_array.size
                if white_ratio > 0.9:  # Skip if more than 90% of the crop is white
                    continue
                
                # Save the crop
                crop_filename = f"{file_name.split('.')[0]}_{left}_{top}.png"
                crop.save(os.path.join(output_dir, crop_filename))

    print(f"Non-signature crops saved to '{output_dir}'")


if __name__ == "__main__":
    create_nonsig_dataset()