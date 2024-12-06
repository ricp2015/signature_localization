import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import ast
import cv2

#localizza una firma passando una finestra grande quanto la firma più grande nell'immagine
#determina prima l'accuracy massima e da quella scala un offset arbitrario per individuare ulteriori firme

# Directories and file paths
documents_dir = "dataset/images"
model_path = "signature_classifier_model.h5"
annotations_path = "fixed_dataset/full_data.csv"
image_info_path = "fixed_dataset/updated_image_ids.csv"

# Load model
model = load_model(model_path)

# Load CSV files
annotations = pd.read_csv(annotations_path)
image_info = pd.read_csv(image_info_path)

def get_signature_size(file_name):
    """
    Get the maximum size of the signature bounding boxes for the given document.

    Parameters:
    - file_name: The name of the image file.

    Returns:
    - max_width: Maximum width of the signature bounding boxes.
    - max_height: Maximum height of the signature bounding boxes.
    """
    # Find the image ID from the image info CSV
    image_id = image_info[image_info['file_name'] == file_name]['id'].values[0]
    
    # Get all bounding boxes for the image
    bboxes = annotations[annotations['image_id'] == image_id]

    # Compute the maximum dimensions of bounding boxes
    max_width = int(bboxes['bbox'].apply(lambda x: ast.literal_eval(x)[2]).max() * image_info[image_info['id'] == image_id]['width'].values[0])
    max_height = int(bboxes['bbox'].apply(lambda x: ast.literal_eval(x)[3]).max() * image_info[image_info['id'] == image_id]['height'].values[0])
    print(max_height, max_width)
    return max_width, max_height


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
            edges = cv2.Canny(crop_array, threshold1=100, threshold2=200)  # Edge detection
            
            # Resize to (734, 177) for the model
            edges_resized = cv2.resize(edges, (734, 177))
            pieces.append(np.expand_dims(edges_resized / 255.0, axis=-1))  # Normalize
            coords.append((left, top))

    return pieces, coords



def detect_signature():
    # Select a random document
    doc_files = os.listdir(documents_dir)

    #scegli un file specifico o random

    #random_file = random.choice(doc_files)
    random_file = "gfg5aa00.png"

    doc_path = os.path.join(documents_dir, random_file)

    # Load the document
    document = Image.open(doc_path)
    img_width, img_height = document.size

    # Determine the signature piece size from CSV
    piece_width, piece_height = get_signature_size(random_file)

    # Split the document into pieces based on signature size
    pieces, coords = split_image(document, (piece_width, piece_height))

    # Predict signature probability for each piece
    probabilities = [model.predict(np.expand_dims(piece, axis=0))[0][0] for piece in pieces]

    # Find the piece with the highest probability
    max_prob = max(probabilities)
    threshold = max_prob - 0.1  # Define the threshold for drawing rectangles

    # Highlight all pieces with probability >= threshold
    draw = ImageDraw.Draw(document)
    for prob, (left, top) in zip(probabilities, coords):
        if prob >= threshold:
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
    print(f"Document: {random_file}")
    print(f"Highest Probability: {max_prob:.4f} at location {max_coord}")
    plt.figure(figsize=(10, 10))
    plt.imshow(document)
    plt.axis("off")
    plt.show()


# Run the detection
detect_signature()
