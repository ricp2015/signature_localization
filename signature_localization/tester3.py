import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import ast
import cv2

# Directories and file paths
documents_dir = "data/raw/signverod_dataset/images"
model_path = "models/signature_classifier_model.h5"
annotations_path = "data/raw/fixed_dataset/full_data.csv"
image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"

# Load model
model = load_model(model_path)

# Load CSV files
annotations = pd.read_csv(annotations_path)
image_info = pd.read_csv(image_info_path)

# Filtra le firme (category_id == 1)
signatures = annotations[annotations['category_id'] == 1]

# Unisci i due DataFrame sui rispettivi ID immagine
merged_data = signatures.merge(image_info, left_on='image_id', right_on='id', suffixes=('_annotation', '_image'))


def calculate_average_ratios(df):
    """
    Calculates the average width and height ratios of the bounding boxes relative to document dimensions.

    Parameters:
    - df: DataFrame containing information about signatures and images.

    Returns:
    - width_ratios: List of average width ratios for each document.
    - height_ratios: List of average height ratios for each document.
    """
    width_ratios = []
    height_ratios = []

    # Group data by image ID to process each document separately
    for image_id, group in df.groupby('image_id'):
        # Extract document dimensions (height and width)
        height = group.iloc[0]['height']
        width = group.iloc[0]['width']

        # Compute the normalized width and height for each bounding box
        bbox_width_ratios = group['bbox'].apply(
            lambda bbox: calculate_bbox_width_ratio(ast.literal_eval(bbox), width)
        )
        bbox_height_ratios = group['bbox'].apply(
            lambda bbox: calculate_bbox_height_ratio(ast.literal_eval(bbox), height)
        )

        # Calculate the average width and height ratios for the document
        avg_width_ratio = bbox_width_ratios.mean()
        avg_height_ratio = bbox_height_ratios.mean()

        # Store the ratios for this document
        width_ratios.append(avg_width_ratio)
        height_ratios.append(avg_height_ratio)
    
    # Return the lists of width and height ratios
    return width_ratios, height_ratios


def calculate_bbox_width_ratio(bbox, img_width):
    """
    Calculates the normalized width of a bounding box relative to the document width.

    Parameters:
    - bbox: List [x_min, y_min, width, height] in normalized coordinates.
    - img_width: Width of the image in pixels.

    Returns:
    - Normalized width ratio.
    """
    bbox_width = bbox[2] * img_width  # Scale the normalized width to pixel units
    return bbox_width / img_width  # Compute the normalized ratio


def calculate_bbox_height_ratio(bbox, img_height):
    """
    Calculates the normalized height of a bounding box relative to the document height.

    Parameters:
    - bbox: List [x_min, y_min, width, height] in normalized coordinates.
    - img_height: Height of the image in pixels.

    Returns:
    - Normalized height ratio.
    """
    bbox_height = bbox[3] * img_height  # Scale the normalized height to pixel units
    return bbox_height / img_height  # Compute the normalized ratio


# Compute the average width and height ratios for each document
width_ratios, height_ratios = calculate_average_ratios(merged_data)

# Compute the global average ratios across all documents
global_average_width_ratio = sum(width_ratios) / len(width_ratios)
global_average_height_ratio = sum(height_ratios) / len(height_ratios)

# Print the global average ratios
print(f"Global average width ratio: {global_average_width_ratio:.6f}")
print(f"Global average height ratio: {global_average_height_ratio:.6f}")


def split_image(image, piece_size):
    """
    Split the image into non-overlapping pieces of the given size and apply edge detection.

    Parameters:
    - image: PIL.Image object.
    - piece_size: Tuple (width, height) specifying the size of each piece.

    Returns:
    - pieces: List of image pieces (as numpy arrays).
    - coords: List of top-left coordinates of each piece in the original image.
    """
    pieces = []
    coords = []
    img_width, img_height = image.size
    piece_width, piece_height = piece_size

    for top in range(0, img_height, piece_height // 4):
        for left in range(0, img_width, piece_width // 4):
            box = (left, top, left + piece_width, top + piece_height)
            crop = image.crop(box)
            
            # Convert to grayscale and apply edge detection
            crop = crop.convert("L")  # Grayscale
            crop_array = np.array(crop)
            #edges = cv2.Canny(crop_array, threshold1=100, threshold2=200)  # Edge detection
            
            # Resize to (734, 177) for the model
            #edges_resized = cv2.resize(edges, (734, 177))
            edges_resized = cv2.resize(crop_array, (734, 177))
            pieces.append(np.expand_dims(edges_resized / 255.0, axis=-1))  # Normalize
            coords.append((left, top))

    return pieces, coords


def get_signature_size(file_name, width_ratio, height_ratio):
    """
    Get the maximum size of the signature bounding boxes for the given document,
    scaled by the global width and height ratios.

    Parameters:
    - file_name: The name of the image file.
    - width_ratio: Global average width ratio.
    - height_ratio: Global average height ratio.

    Returns:
    - scaled_width: Scaled width of the signature bounding boxes.
    - scaled_height: Scaled height of the signature bounding boxes.
    """
    # Find the image ID from the image info CSV
    image_id = image_info[image_info['file_name'] == file_name]['id'].values[0]

    # Get the document's width and height
    img_width = image_info[image_info['id'] == image_id]['width'].values[0]
    img_height = image_info[image_info['id'] == image_id]['height'].values[0]

    # Scale the dimensions using the global average ratios
    scaled_width = int(img_width * width_ratio)
    scaled_height = int(img_height * height_ratio)

    print(f"Document dimensions: {img_width}x{img_height}")
    print(f"Scaled signature size: {scaled_width}x{scaled_height}")
    return scaled_width, scaled_height


def detect_signature():
    """
    Detect signatures in a random document by dividing the document into pieces, 
    predicting the probability of each piece containing a signature, 
    and drawing rectangles around detected regions.

    Parameters:
    - documents_dir: Directory containing document images.
    - global_average_width_ratio: Average width ratio of signature pieces.
    - global_average_height_ratio: Average height ratio of signature pieces.
    - model: Trained model to predict signature probability.
    """
    # Load the list of test files
    with open("data/splits/test_files.txt", "r") as f:
        test_files = f.read().splitlines()

    # Ensure the test files list is not empty
    if not test_files:
        print("No test files found in the annotation file.")
        return

    # Select a random document from the test files
    random_doc_file = random.choice(test_files)
    doc_path = os.path.join(documents_dir, random_doc_file)

    if not os.path.exists(doc_path):
        print(f"Document {random_doc_file} not found in the directory.")
        return

    # Load the document
    document = Image.open(doc_path).convert("RGB")
    img_width, img_height = document.size

    # Determine the signature piece size from CSV
    piece_width, piece_height = get_signature_size(
        random_doc_file, global_average_width_ratio, global_average_height_ratio
    )

    # Split the document into pieces based on signature size
    pieces, coords = split_image(document, (piece_width, piece_height))

    # Predict signature probability for each piece
    probabilities = [model.predict(np.expand_dims(piece, axis=0))[0][0] for piece in pieces]

    # Find the piece with the highest probability
    max_prob = max(probabilities)
    threshold = max_prob - 0.1  # Define the threshold for drawing rectangles

    # Define the percentage for top results (e.g., 2 for top 2%)
    top_percentage = 2  # Change this value to use a different percentage

    # Sort the probabilities in descending order and get the top N% indices
    sorted_probs = sorted(probabilities, reverse=True)  # Sort probabilities in descending order
    top_n_percent_index = int(len(sorted_probs) * (top_percentage / 100))  # Get the index for top N%

    # Select the top N% probabilities
    top_probs = sorted_probs[:top_n_percent_index]

    # Highlight only the pieces with top N% highest probabilities
    draw = ImageDraw.Draw(document)
    for prob, (left, top) in zip(probabilities, coords):
        if prob in top_probs:
            right = left + piece_width
            bottom = top + piece_height
            draw.rectangle([left, top, right, bottom], outline="blue", width=5)

    # Highlight the piece with the maximum probability
    max_index = probabilities.index(max_prob)
    max_coord = coords[max_index]
    left, top = max_coord
    right, bottom = left + piece_width, top + piece_height
    draw.rectangle([left, top, right, bottom], outline="red", width=5)

    # Show results
    print(f"Document: {random_doc_file}")
    print(f"Highest Probability: {max_prob:.4f} at location {max_coord}")
    plt.figure(figsize=(10, 10))
    plt.imshow(document)
    plt.axis("off")
    plt.show()

