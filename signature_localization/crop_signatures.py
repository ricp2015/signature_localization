import os
import pandas as pd
import ast
from PIL import Image
from tqdm import tqdm

def crop_signatures():
    image_dir = "data/raw/signverod_dataset/images/"
    output_dir = "data/interim/cropped_signatures/"
    os.makedirs(output_dir, exist_ok=True)
    image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"
    annotations_path = "data/raw/fixed_dataset/full_data.csv"
    image_info = pd.read_csv(image_info_path)
    annotations = pd.read_csv(annotations_path)

    # Filter annotations to only include category_id 1 (signatures)
    signature_annotations = annotations[annotations['category_id'] == 1]

    # Iterate over each signature annotation
    print("Cropping signatures ...")
    for _, row in tqdm(signature_annotations.iterrows(), total=len(signature_annotations)):
        # Extract the bounding box and related info
        bbox = ast.literal_eval(row['bbox'])
        image_id = row['image_id']
        
        # Find the corresponding image info
        image_info_row = image_info[image_info['id'] == image_id].iloc[0]
        file_name = image_info_row['file_name']
        img_height = image_info_row['height']
        img_width = image_info_row['width']
        
        # Calculate bounding box in pixel coordinates
        x_min = int(bbox[0] * img_width)
        y_min = int(bbox[1] * img_height)
        x_max = int((bbox[0] + bbox[2]) * img_width)
        y_max = int((bbox[1] + bbox[3]) * img_height)
        
        # Load the image
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping.")
            continue
        
        with Image.open(img_path) as img:
            # Crop the signature bounding box
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            
            # Save the cropped signature
            output_path = os.path.join(output_dir, f"{file_name.split('.')[0]}_signature_{row['id']}.png")
            cropped_img.save(output_path)

    print(f"Signatures cropped and saved to {output_dir}")
