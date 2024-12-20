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

def preprocess_image(image, target_size=(734, 177)):
    """
    Preprocess the input image for the model.
    """
    image = cv2.resize(image, target_size)
    return image / 255.0


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
        oscillated_width = int(random.uniform(0.8 * avg_width, 2 * avg_width))
        oscillated_height = int(random.uniform(0.8 * avg_height, 1.4 * avg_height))

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


def classify_regions_with_model(image_path, model, bounding_boxes, output_dir):
    """
    Use the trained model to classify regions as signatures or not,
    and save preprocessed regions for debugging. Returns a list of results with probabilities.
    """
    # Load the full image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a folder for debugging preprocessed regions
    debugging_dir = "signature_localization/debugging_ss_modelinput"
    if not os.path.exists(debugging_dir):
        os.makedirs(debugging_dir)

    results = []  # List to store results: (bounding_box, probability)

    for i, box in enumerate(bounding_boxes):
        # Ensure the box coordinates are integers
        x1, y1, x2, y2 = map(int, box)

        # Crop the region from the image
        cropped_region = image[y1:y2, x1:x2]

        # Apply Canny edge detection to match training preprocessing
        canny_cropped_region = cv2.Canny(cropped_region, 100, 200)

        # Preprocess the cropped region for the model
        preprocessed_region = preprocess_image(canny_cropped_region)

        # Save the preprocessed region for debugging
        debug_path = os.path.join(debugging_dir, f"region_{i}.png")
        cv2.imwrite(debug_path, (preprocessed_region.squeeze() * 255).astype(np.uint8))  # Convert to uint8 for saving
        print(f"Saved preprocessed region to {debug_path}")

        # Predict with the trained model
        prediction = model.predict(np.expand_dims(preprocessed_region, axis=0))[0][0]
        prediction = round(prediction, 4)  # Limit probability to 4 decimal places
        results.append((box, prediction))  # Append the bounding box and its probability

        # Save the region with its probability in the filename
        prob_str = f"{prediction:.4f}"
        save_path = os.path.join(output_dir, f"{prob_str}_{uuid.uuid4().hex[:8]}.png")
        cv2.imwrite(save_path, canny_cropped_region)
        print(f"Saved region with probability {prob_str} to {save_path}")

    return results



def filter_results_by_threshold(results, threshold):
    """
    Filter the results based on a probability threshold.
    
    Parameters:
    - results: List of tuples [(bounding_box, probability), ...].
    - threshold: Float, the probability threshold.

    Returns:
    - Filtered results: List of bounding boxes that exceed the threshold.
    """
    return [box for box, prob in results if prob >= threshold]


def detect_signatures_with_selective_search(image_path, output_dir, img_preprocessing, num_boxes=2000, threshold=0.5):
    """
    Main function to detect signatures using selective search and apply a threshold.
    """
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
    results = classify_regions_with_model(image_path, model, merged_boxes, output_dir)

    # Filter results based on threshold
    filtered_boxes = filter_results_by_threshold(results, threshold)
    print(f"Filtered bounding boxes: {filtered_boxes}")

    return filtered_boxes


if __name__ == "__main__":
    # Paths and parameters
    image_path = "data/raw/signverod_dataset/images/zkd43f00_2.png"
    annotations_path = "data/raw/fixed_dataset/full_data.csv"  # Path to annotations CSV
    image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"  # Path to image metadata CSV
    output_dir = "signature_localization/ss_test_pieces"
    img_preprocessing = "canny"
    threshold = 0.5

    # Detect signatures
    detected_boxes = detect_signatures_with_selective_search(image_path, output_dir, img_preprocessing, num_boxes=2000, threshold=threshold)
    print(f"Detected bounding boxes with probability >= {threshold}: {detected_boxes}")
