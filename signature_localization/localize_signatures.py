import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import ast
import cv2
import binary_classifier_CNN


def initialize_test(img_preprocessing):
    preproc = img_preprocessing
    if img_preprocessing == None:
        preproc = "no_pre"

    # Directories and file paths
    model_path = "models/" + preproc + "_signature_classifier_model.h5"
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
    
    # Compute the average width and height ratios for each document
    width_ratios, height_ratios = calculate_average_ratios(merged_data)

    # Compute the global average ratios across all documents
    global_average_width_ratio = sum(width_ratios) / len(width_ratios)
    global_average_height_ratio = sum(height_ratios) / len(height_ratios)

    # Print the global average ratios
    print(f"Global average width ratio: {global_average_width_ratio:.6f}")
    print(f"Global average height ratio: {global_average_height_ratio:.6f}")
    return image_info, global_average_width_ratio, global_average_height_ratio, model

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

def process_rectangle(box, image, img_preprocessing, left, top):
    crop = image.crop(box)
    crop = crop.convert("L")  # Grayscale
    # Convert the PIL Image to a NumPy array
    crop_array = np.array(crop)
    # Preprocess the NumPy array image
    img = binary_classifier_CNN.preprocess_image(crop_array, img_preprocessing)
    edges_resized = cv2.resize(img, (734, 177))
    piece = np.expand_dims(edges_resized / 255.0, axis=-1) # Normalize
    coord = (left, top)
    return piece, coord



def get_signature_size(file_name, width_ratio, height_ratio, image_info):
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


def calculate_iou(box1, box2):
    """
    Calcola l'Intersection over Union (IoU) tra due rettangoli.

    Parameters:
    - box1: (left, top, right, bottom) del primo rettangolo
    - box2: (left, top, right, bottom) del secondo rettangolo

    Returns:
    - IoU: il rapporto di sovrapposizione tra i due rettangoli
    """
    # Calcolare le coordinate di intersezione
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Se non c'è sovrapposizione
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # Area di intersezione
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Aree dei due rettangoli
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Area di unione
    union_area = box1_area + box2_area - intersection_area

    # IoU
    return intersection_area / union_area


def detect_signature(img_preprocessing=None, doc_path=None, plot_results=True):
    """
    Detect signatures in a specified document or a random document iteratively by masking detected regions,
    and highlight only one signature per iteration based on a probability threshold.

    Parameters:
    - img_preprocessing: Preprocessing method to apply to the images (optional).
    - doc_path: Path to the specific document to process (optional). If not provided, a random document is selected.
    - plot_results: Boolean indicating whether to plot the detected regions on the document (optional, default: True).

    Returns:
    - detected_regions: List of detected regions as bounding boxes in the format [x_min, y_min, width, height].
    """
    documents_dir = "data/raw/signverod_dataset/images"

    # Initialize parameters and load models/data
    image_info, global_average_width_ratio, global_average_height_ratio, model = initialize_test(img_preprocessing)

    preproc = img_preprocessing or "no_pre"

    # Load the list of test files
    with open(f"data/splits/{preproc}_test_files.txt", "r") as f:
        test_files = f.read().splitlines()

    if not test_files:
        print("No test files found in the annotation file.")
        return []

    # Determine the document path
    if doc_path is None:
        # Select a random document
        random_doc_file = random.choice(test_files)
        doc_path = os.path.join(documents_dir, random_doc_file)
    else:
        random_doc_file = os.path.basename(doc_path)

    if not os.path.exists(doc_path):
        print(f"Document {doc_path} not found in the directory.")
        return []

    # Load the document as an RGB image
    document = Image.open(doc_path).convert("RGB")
    original_document = document.copy()  # Keep a copy of the original document

    # Determine the signature piece size
    piece_width, piece_height = get_signature_size(
        random_doc_file, global_average_width_ratio, global_average_height_ratio, image_info
    )

    # Prepare variables for iterative detection
    max_iterations = 10  # Stop after a certain number of signatures
    min_probability_threshold = 0.9
    highlighted_boxes = []

    # Create a working version of the document as a NumPy array
    document_array = np.array(document)
    img_width, img_height = document.size

    # Generate initial pieces and coordinates
    pieces = []
    coords = []
    for top in range(0, img_height, piece_height // 4):
        for left in range(0, img_width, piece_width // 4):
            box = (left, top, left + piece_width, top + piece_height)
            if box[2] > img_width or box[3] > img_height:  # Skip boxes that exceed image boundaries
                continue
            piece, coord = process_rectangle(box, document, img_preprocessing, left, top)
            pieces.append(piece)
            coords.append(coord)

    candidate_coords = coords

    for iteration in range(max_iterations):
        if not candidate_coords:
            print("No more candidate regions to evaluate.")
            break

        # Predict signature probability for each piece
        probabilities = [model.predict(np.expand_dims(piece, axis=0))[0][0] for piece in pieces]

        # Filter pieces with probability >= threshold
        valid_indices = [i for i, prob in enumerate(probabilities) if prob >= min_probability_threshold]
        if not valid_indices:
            print(f"No signatures detected with probability >= {min_probability_threshold:.2f}")
            break

        # Find the piece with the highest probability
        max_index = max(valid_indices, key=lambda i: probabilities[i])
        max_prob = probabilities[max_index]
        max_coord = candidate_coords[max_index]
        left, top = max_coord
        right = left + piece_width
        bottom = top + piece_height
        new_box = (left, top, right, bottom)

        # Non-Maximum Suppression: elimina rettangoli che si sovrappongono troppo
        keep_boxes = []
        for box in highlighted_boxes:
            if calculate_iou(new_box, box) < 0.3:  # Sovrapposizione minore del 30%
                keep_boxes.append(box)
        keep_boxes.append(new_box)

        # Aggiorna la lista dei rettangoli evidenziati
        highlighted_boxes = keep_boxes

        # Mask the detected region (set to black)
        document_array[top:bottom, left:right] = 0  # Black out the detected region

        # Update candidate coordinates and pieces for the next iteration
        candidate_coords = [candidate_coords[i] for i in valid_indices if i != max_index]
        pieces = [pieces[i] for i in valid_indices if i != max_index]

        print(f"Iteration {iteration + 1}: Detected signature with probability {max_prob:.4f}")
        print(f"Location: {max_coord}")

    # Optionally plot the results
    if plot_results:
        draw = ImageDraw.Draw(original_document)
        for box in highlighted_boxes:
            draw.rectangle(box, outline="red", width=5)

        # Show the results
        plt.figure(figsize=(10, 10))
        plt.imshow(original_document)
        plt.axis("off")
        plt.title("Detected Signatures")
        plt.show()

    return highlighted_boxes