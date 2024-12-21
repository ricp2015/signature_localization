import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import shutil
import uuid
import random
import pandas as pd
import ast
import localize_signatures



def crop_and_resize_center(image, x1, y1, x2, y2, target_size=(734, 177)):
    """
    Crop a region from the image, centered at the given bounding box, 
    and resize it to the target size if necessary.

    Parameters:
    - image: The input grayscale image.
    - x1, y1, x2, y2: Bounding box coordinates.
    - target_size: The desired size of the output image (width, height).

    Returns:
    - The cropped and resized image.
    """
    crop_width = x2 - x1
    crop_height = y2 - y1
    target_width, target_height = target_size

    # Calculate the center of the bounding box
    center_x = x1 + crop_width // 2
    center_y = y1 + crop_height // 2

    # Ensure the crop dimensions match the target size
    half_width = target_width // 2
    half_height = target_height // 2

    # Define new crop coordinates, centered around the bounding box
    new_x1 = max(0, center_x - half_width)
    new_y1 = max(0, center_y - half_height)
    new_x2 = new_x1 + target_width
    new_y2 = new_y1 + target_height

    # Adjust if the new crop exceeds image boundaries
    if new_x2 > image.shape[1]:  # Exceeds width
        new_x2 = image.shape[1]
        new_x1 = max(0, new_x2 - target_width)
    if new_y2 > image.shape[0]:  # Exceeds height
        new_y2 = image.shape[0]
        new_y1 = max(0, new_y2 - target_height)

    # Crop the region
    cropped_region = image[new_y1:new_y2, new_x1:new_x2]

    # Resize if the cropped region does not match the target size
    if cropped_region.shape[1] != target_width or cropped_region.shape[0] != target_height:
        cropped_region = cv2.resize(cropped_region, target_size)

    return cropped_region / 255.0  # Normalize to [0, 1]





def preprocess_image(image, target_size=(734, 177)):
    """
    Preprocess the input image for the model.
    """
    image = cv2.resize(image, target_size)
    return image 


def initialize_model_and_directories(model_path, output_dir):
    """
    Initialize the trained model and prepare directories for output.
    """
    # Load the trained model
    model = load_model(model_path)

    # Prepare the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    return model


def apply_canny_edge_detection(image_path):
    """
    Apply Canny edge detection to an input image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    #cv2.imwrite("signature_localization/debugging_ss/img.png", edges)  # Convert to uint8 for saving

    return edges

def initialize_bounding_boxes_with_oscillation(edges, num_boxes, image_info, avg_width_ratio, avg_height_ratio, file_name):
    """
    Randomly select N edge coordinates and create bounding boxes with oscillating sizes.

    Parameters:
    - edges: Edge-detected image.
    - num_boxes: Number of bounding boxes to initialize.
    - image_info: DataFrame containing image metadata.
    - avg_width_ratio: Average width ratio (relative to document width).
    - avg_height_ratio: Average height ratio (relative to document height).
    - file_name: Name of the current image file being processed.

    Returns:
    - List of bounding boxes (x1, y1, x2, y2).
    """
    # Get the image dimensions
    img_width = image_info[image_info['file_name'] == file_name]['width'].values[0]
    img_height = image_info[image_info['file_name'] == file_name]['height'].values[0]

    # Calculate average bounding box size based on ratios
    avg_width = int(avg_width_ratio * img_width)
    avg_height = int(avg_height_ratio * img_height)

    # Get the coordinates of all edge pixels
    edge_coords = np.column_stack(np.where(edges > 0))

    # Randomly select N edge coordinates
    selected_coords = edge_coords[np.random.choice(edge_coords.shape[0], num_boxes, replace=False)]

    # Initialize bounding boxes based on selected coordinates
    bounding_boxes = []
    for coord in selected_coords:
        y, x = coord

        # Oscillate width and height
        oscillated_width = int(random.uniform(0.75 * avg_width, 1.25 * avg_width))
        oscillated_height = int(random.uniform(0.85 * avg_height, 1.15 * avg_height))

        # Calculate bounding box coordinates
        x1 = max(0, x - (oscillated_width // 2))  # Ensure x1 is non-negative
        y1 = max(0, y - (oscillated_height // 2))  # Ensure y1 is non-negative
        x2 = x1 + oscillated_width
        y2 = y1 + oscillated_height

        # Adjust the bounding box to remain within the image dimensions
        if x2 > img_width:
            x2 = img_width
            x1 = max(0, x2 - oscillated_width)  # Adjust x1 to maintain box size
        if y2 > img_height:
            y2 = img_height
            y1 = max(0, y2 - oscillated_height)  # Adjust y1 to maintain box size
        bounding_boxes.append((x1, y1, x2, y2))

    return bounding_boxes

def merge_similar_regions(bounding_boxes, iou_threshold=0.3):
    """
    Merge regions based on similarity using IoU.
    """
    merged_boxes = []
    while bounding_boxes:
        box = bounding_boxes.pop(0)
        to_merge = []
        for other_box in bounding_boxes:
            if localize_signatures.calculate_iou(box, other_box) > iou_threshold:
                to_merge.append(other_box)

        # Merge all overlapping boxes into one
        if to_merge:
            for b in to_merge:
                bounding_boxes.remove(b)
            x1 = min([box[0]] + [b[0] for b in to_merge])
            y1 = min([box[1]] + [b[1] for b in to_merge])
            x2 = max([box[2]] + [b[2] for b in to_merge])
            y2 = max([box[3]] + [b[3] for b in to_merge])
            merged_boxes.append((x1, y1, x2, y2))
        else:
            merged_boxes.append(box)
    return merged_boxes
'''
def classify_regions_with_model(image_path, model, bounding_boxes, output_dir):
    """
    Use the trained model to classify regions as signatures or not,
    and save preprocessed regions for debugging.
    """
    # Load the full image in grayscale

    results=[]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Create a folder for debugging preprocessed regions
    debugging_dir = "signature_localization/debugging_ss_modelinput"
    if not os.path.exists(debugging_dir):
        os.makedirs(debugging_dir)

    for i, box in enumerate(bounding_boxes):
        # Ensure the box coordinates are integers
        x1, y1, x2, y2 = map(int, box)

        # Crop the region from the image
        cropped_region = image[y1:y2, x1:x2]

        # Preprocess the cropped region for the mode
        # Convert preprocessed_region back to uint8 for Canny edge detection
        preprocessed_region_uint8 = preprocess_image((cropped_region).astype(np.uint8))

        # Apply Canny edge detection
        canny_cropped_region = cv2.Canny(preprocessed_region_uint8, 100, 200)

        # Save the preprocessed region for debugging
        debug_path = os.path.join(debugging_dir, f"region_{i}.png")
        cv2.imwrite(debug_path, preprocessed_region_uint8)  # Save the uint8 preprocessed region
        print(f"Saved preprocessed region to {debug_path}")

        # Predict with the trained model
        prediction = model.predict(np.expand_dims(canny_cropped_region/255, axis=0))[0][0]
        results.append([box,prediction])
        # Save the region with its probability in the filename
        prob_str = f"{prediction:.4f}"
        save_path = os.path.join(output_dir, f"{prob_str}_{uuid.uuid4().hex[:8]}.png")

        cv2.imwrite(save_path, canny_cropped_region)
        
        print(f"Saved region with probability {prob_str} to {save_path}")

    return results
'''
def classify_regions_with_model(image_path, model, bounding_boxes, output_dir):
    """
    Use the trained model to classify regions as signatures or not,
    and save preprocessed regions for debugging.
    """
    # Load the full image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = []
    # Create a folder for debugging preprocessed regions
    debugging_dir = "signature_localization/debugging_ss_modelinput"
    if not os.path.exists(debugging_dir):
        os.makedirs(debugging_dir)

    for i, box in enumerate(bounding_boxes):
        # Ensure the box coordinates are integers
        x1, y1, x2, y2 = map(int, box)

        # Crop the region from the image
        cropped_region = image[y1:y2, x1:x2]

        # Apply Canny edge detection to match training preprocessing
        canny_cropped_region = cv2.Canny(cropped_region, 100, 200)

        # Crop and resize to match the model's input size
        preprocessed_region = crop_and_resize_center(canny_cropped_region, 0, 0, canny_cropped_region.shape[1], canny_cropped_region.shape[0])

        # Save the preprocessed region for debugging
        debug_path = os.path.join(debugging_dir, f"region_{i}.png")
        cv2.imwrite(debug_path, (preprocessed_region.squeeze() * 255).astype(np.uint8))  # Convert to uint8 for saving
        print(f"Saved preprocessed region to {debug_path}")

        # Predict with the trained model
        prediction = model.predict(np.expand_dims(preprocessed_region, axis=0))[0][0]
        results.append([box,prediction])
        # Save the region with its probability in the filename
        prob_str = f"{prediction:.4f}"
        save_path = os.path.join(output_dir, f"{prob_str}_{uuid.uuid4().hex[:8]}.png")

        cv2.imwrite(save_path, canny_cropped_region)

        print(f"Saved region with probability {prob_str} to {save_path}")
    return results


def detect_signatures_with_selective_search(image_path, output_dir, img_preprocessing, threshold, num_boxes=2000):
    # Prepare the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    image_info, avg_width_ratio, avg_height_ratio, model = localize_signatures.initialize_test(img_preprocessing)

    # Apply Canny edge detection
    print("Applying Canny edge detection...")
    edges = apply_canny_edge_detection(image_path)

    # Initialize bounding boxes based on calculated average ratios
    print("Initializing bounding boxes...")
    file_name = os.path.basename(image_path)
    bounding_boxes = initialize_bounding_boxes_with_oscillation(edges, num_boxes, image_info, avg_width_ratio, avg_height_ratio, file_name)

    # Load the original image for drawing
    original_image = cv2.imread(image_path)
    debugging_dir = "signature_localization/debugging_ss"
    if not os.path.exists(debugging_dir):
        os.makedirs(debugging_dir)


    # Save an image with the initial bounding boxes drawn in blue
    debug_image = original_image.copy()
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle
    debug_path = os.path.join(debugging_dir, "initial_bounding_boxes.png")
    cv2.imwrite(debug_path, debug_image)
    print(f"Saved initial bounding boxes image to {debug_path}")


    # Merge similar regions (if necessary)
    print("Merging similar regions...")
    merged_boxes = merge_similar_regions(bounding_boxes)
    # Save an image with the merged bounding boxes drawn in red
    debug_image_merged = original_image.copy()
    for box in merged_boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        cv2.rectangle(debug_image_merged, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle
    debug_path_merged = os.path.join(debugging_dir, "merged_bounding_boxes.png")
    cv2.imwrite(debug_path_merged, debug_image_merged)
    print(f"Saved merged bounding boxes image to {debug_path_merged}")

    # Classify regions with the trained model
    print("Classifying regions with the trained model...")
    regions = classify_regions_with_model(image_path, model, merged_boxes, output_dir)
    return [r for r, prob in regions if prob >= threshold]



if __name__ == "__main__":
    # Paths and parameters
    image_path = "data/raw/signverod_dataset/images/gsa_LAZ02206-SLA-3-_Z-01.png"
    model_path = "models/canny_signature_classifier_model.h5"  # Make sure to use the correct path
    annotations_path = "data/raw/fixed_dataset/full_data.csv"  # Path to annotations CSV
    image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"  # Path to image metadata CSV
    output_dir = "signature_localization/ss_test_pieces"
    threshold = 0.5
    img_preprocessing = "canny"
    # Detect signatures
    print(detect_signatures_with_selective_search(image_path, output_dir, img_preprocessing, threshold, num_boxes=2000))