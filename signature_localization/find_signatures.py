import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from keras import models
import matplotlib.pyplot as plt
import ast

def get_signature_size(file_name, image_info, annotations):
    """
    Get the average size of the signature bounding boxes for the given document.

    Parameters:
    - file_name: The name of the image file.

    Returns:
    - avg_width: Average width of the signature bounding boxes.
    - avg_height: Average height of the signature bounding boxes.
    """
    # Find the image ID from the image info CSV
    image_id = image_info[image_info['file_name'] == file_name]['id'].values[0]
    
    # Get all bounding boxes for the image
    bboxes = annotations[annotations['image_id'] == image_id]

    # Compute average dimensions of bounding boxes
    avg_width = int(bboxes['bbox'].apply(lambda x: ast.literal_eval(x)[2]).mean() * image_info[image_info['id'] == image_id]['width'].values[0])
    avg_height = int(bboxes['bbox'].apply(lambda x: ast.literal_eval(x)[3]).mean() * image_info[image_info['id'] == image_id]['height'].values[0])

    return avg_width, avg_height

def split_image(image, piece_size):
    """
    Split the image into non-overlapping pieces of the given size.

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

    for top in range(0, img_height, piece_height):
        for left in range(0, img_width, piece_width):
            box = (left, top, left + piece_width, top + piece_height)
            crop = image.crop(box)
            coords.append((left, top))
            
            # Resize to (734, 177) for the model
            crop = crop.resize((734, 177)).convert("L")  # Convert to grayscale, provare anche Canny: crop = cv2.Canny(crop_array, threshold1=100, threshold2=200)
            #crop = cv2.Canny(crop_array, threshold1=100, threshold2=200)
            pieces.append(np.expand_dims(np.array(crop) / 255.0, axis=-1))  # Normalize

    return pieces, coords

def detect_signatures():

    documents_dir = "data/raw/signverod_dataset/images"
    model_path = "models/signature_classifier_model.h5"
    annotations_path = "data/raw/fixed_dataset/full_data.csv"
    image_info_path = "data/raw/fixed_dataset/updated_image_ids.csv"

    # Load model
    model = models.load_model(model_path)

    # Load CSV files
    annotations = pd.read_csv(annotations_path)
    image_info = pd.read_csv(image_info_path)
    # Select a random document
    doc_files = os.listdir(documents_dir)

    #random_file = random.choice(doc_files)
    random_file = "nist_r0113_01.png"

    doc_path = os.path.join(documents_dir, random_file)

    # Load the document
    document = Image.open(doc_path)
    img_width, img_height = document.size

    # Determine the signature piece size from CSV
    piece_width, piece_height = get_signature_size(random_file, image_info, annotations)

    # Split the document into pieces based on signature size
    pieces, coords = split_image(document, (piece_width, piece_height))

    # Predict signature probability for each piece
    probabilities = [model.predict(np.expand_dims(piece, axis=0))[0][0] for piece in pieces]

    # Find the piece with the highest probability
    max_prob = max(probabilities)
    max_index = probabilities.index(max_prob)
    max_coord = coords[max_index]

    # Highlight the piece with the highest probability
    draw = ImageDraw.Draw(document)
    left, top = max_coord
    right, bottom = left + piece_width, top + piece_height
    draw.rectangle([left, top, right, bottom], outline="red", width=5)

    # Show results
    print(f"Document: {random_file}")
    print(f"Highest Probability: {max_prob:.4f} at location {max_coord}")
    plt.figure(figsize=(10, 10))
    plt.imshow(document)
    plt.axis("off")
    plt.show()

