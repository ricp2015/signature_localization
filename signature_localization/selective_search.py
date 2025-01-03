import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import shutil
import uuid
import pandas as pd
import ast
import localize_signatures
import binary_classifier_CNN


def crop_and_resize_center(image, x1, y1, x2, y2, target_size=(734, 177)):
    """
    Crop a region from the image, centered at the given bounding box, 
    and pad it with black pixels to reach the target size if necessary.

    Parameters:
    
    image: The input grayscale image.
    x1, y1, x2, y2: Bounding box coordinates.
    target_size: The desired size of the output image (width, height).

        Returns:
        
    The cropped and padded image."""
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
    new_x2 = min(image.shape[1], new_x1 + target_width)
    new_y2 = min(image.shape[0], new_y1 + target_height)

    # Crop the region
    cropped_region = image[new_y1:new_y2, new_x1:new_x2]

    # Calculate padding needed to reach the target size
    top_pad = max(0, half_height - (center_y - new_y1))
    bottom_pad = max(0, target_height - cropped_region.shape[0] - top_pad)
    left_pad = max(0, half_width - (center_x - new_x1))
    right_pad = max(0, target_width - cropped_region.shape[1] - left_pad)

    # Pad the cropped region with black pixels
    padded_region = cv2.copyMakeBorder(
        cropped_region,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=0  # Black padding
    )

    return padded_region / 255.0, new_x1 - x1, new_y1 - y1  # Normalize to [0, 1]


def initialize_bounding_boxes_with_oscillation(edges, preprocessing, num_boxes, image_info, avg_width_ratio, avg_height_ratio, file_name):
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
    if preprocessing == "canny" or preprocessing == "sobel" or preprocessing =="laplacian":
        edge_coords = np.column_stack(np.where(edges > 0))
    else:
        edge_coords = np.column_stack(np.where(edges < 90))
    # Randomly select N edge coordinates
    selected_coords = edge_coords[np.random.choice(edge_coords.shape[0], num_boxes, replace=False)]

    # Initialize bounding boxes based on selected coordinates
    bounding_boxes = []
    for coord in selected_coords:
        y, x = coord

        # Oscillate width and height
        oscillated_width = int(random.uniform(0.75 * avg_width, 2.3 * avg_width))
        oscillated_height = int(random.uniform(0.85 * avg_height, 1.6 * avg_height))

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

def classify_regions_with_model(image_path,preprocessing, model, bounding_boxes, output_dir):
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

    target_size = (734, 177)  # Target size used for resizing cropped regions

    for i, box in enumerate(bounding_boxes):
        # Ensure the box coordinates are integers
        x1, y1, x2, y2 = map(int, box)

        # Crop the region from the image
        cropped_region = image[y1:y2, x1:x2]
        preprocessed_cropped_region = binary_classifier_CNN.preprocess_image(cropped_region, method=preprocessing)

        # Crop and resize to match the model's input size
        preprocessed_region, a, b = crop_and_resize_center(preprocessed_cropped_region, 0, 0, preprocessed_cropped_region.shape[1], preprocessed_cropped_region.shape[0])

        # Save the preprocessed region for debugging
        debug_path = os.path.join(debugging_dir, f"region_{i}.png")
        cv2.imwrite(debug_path, (preprocessed_region.squeeze() * 255).astype(np.uint8))  # Convert to uint8 for saving
        print(f"Saved preprocessed region to {debug_path}")

        # Predict with the trained model
        prediction = model.predict(np.expand_dims(preprocessed_region, axis=0))[0][0]
        # Rescale the bounding box coordinates
        rescaled_box = (
            int(x1 + a),
            int(y1 + b),
            int(x2 - a),
            int(y2 - b)
        )


        # Append rescaled bounding box and prediction to results
        results.append([rescaled_box, prediction])

        # Save the region with its probability in the filename
        prob_str = f"{prediction:.4f}"
        save_path = os.path.join(output_dir, f"{prob_str}_{uuid.uuid4().hex[:8]}.png")
        cv2.imwrite(save_path, image[rescaled_box[1]:rescaled_box[3], rescaled_box[0]:rescaled_box[2]])
        print(f"Saved region with probability {prob_str} to {save_path}")
    
    return results


def filter_non_overlapping_regions(results, iou_threshold=0.6):
    """
    Filter the list of results to keep only non-overlapping regions based on IoU.
    
    Parameters:
    - results: List of [bounding_box, prediction_score].
    - iou_threshold: IoU threshold above which regions are considered overlapping.
    
    Returns:
    - Filtered list of results.
    """
    filtered_results = []
    while results:
        # Sort results by prediction score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        current = results.pop(0)
        filtered_results.append(current)
        # Remove overlapping regions
        results = [r for r in results if localize_signatures.calculate_iou(current[0], r[0]) < iou_threshold]
    return filtered_results

def detect_signatures_with_selective_search(image_path, output_dir, img_preprocessing, threshold, num_boxes=2000):
    # Prepare the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    image_info, avg_width_ratio, avg_height_ratio, model = localize_signatures.initialize_test(img_preprocessing)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = binary_classifier_CNN.preprocess_image(image,method=img_preprocessing)

    # Initialize bounding boxes based on calculated average ratios
    print("Initializing bounding boxes...")
    file_name = os.path.basename(image_path)
    bounding_boxes = initialize_bounding_boxes_with_oscillation(edges, img_preprocessing, num_boxes, image_info, avg_width_ratio, avg_height_ratio, file_name)

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
    #merged_boxes = merge_similar_regions(bounding_boxes)
    merged_boxes = bounding_boxes
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
    regions = classify_regions_with_model(image_path, img_preprocessing, model, merged_boxes, output_dir)

    # Filter results based on threshold
    results_above_threshold = [r for r in regions if r[1] >= threshold]

    # Filter non-overlapping results based on IoU
    print("Filtering non-overlapping regions...")
    filtered_results = filter_non_overlapping_regions(results_above_threshold)

    # Save debug images after filtering
    debug_image_filtered = original_image.copy()
    for box, prob in filtered_results:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(debug_image_filtered, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
    debug_path_filtered = os.path.join(debugging_dir, "filtered_bounding_boxes.png")
    cv2.imwrite(debug_path_filtered, debug_image_filtered)
    print(f"Saved filtered bounding boxes image to {debug_path_filtered}")

    return [r[0] for r in filtered_results]



if __name__ == "__main__":
    # Paths and parameters
    image_path = "data/raw/signverod_dataset/images/nist_r0876_01.png"
    annotations_path = "data/raw/fixed_dataset/full_data.csv"  # Path to annotations CSV
    image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"  # Path to image metadata CSV
    output_dir = "signature_localization/ss_test_pieces"
    threshold = 0.8
    img_preprocessing = "canny"
    # Detect signatures
    print(detect_signatures_with_selective_search(image_path, output_dir, img_preprocessing, threshold, num_boxes=200))