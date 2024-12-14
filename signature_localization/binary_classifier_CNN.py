import os
import random
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def preprocess_image(image, method=None):
    """
    Apply a pre-processing method to the image.

    Parameters:
    - image: Grayscale image as a NumPy array.
    - method: Pre-processing method. Options: 'canny', 'sobel', 'laplacian', 'gaussian', 'threshold', None.

    Returns:
    - Pre-processed image as a NumPy array.
    """
    if method is None:
        return image  # No pre-processing

    if method == 'canny':
        return cv2.Canny(image, 100, 200)  # Simple Canny Edge Detection

    if method == 'sobel':
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sobel_x, sobel_y).astype(np.uint8)

    if method == 'laplacian':
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        return cv2.Laplacian(blurred, cv2.CV_64F).astype(np.uint8)

    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)

    if method == 'threshold':
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    raise ValueError(f"Unknown pre-processing method: {method}")


def create_dataset(signature_dir, nonsig_dir, raw_documents_dir,
                   output_file="data/splits/test_files.txt", img_size=(734, 177),
                   preprocessing=None):
    """
    Create dataset from signature and non-signature images with optional pre-processing.

    Parameters:
    - signature_dir: Directory containing signature images.
    - nonsig_dir: Directory containing non-signature images.
    - raw_documents_dir: Directory containing raw document images.
    - output_file: File to save the test filenames.
    - img_size: Tuple specifying the target size for resizing images.
    - preprocessing: Pre-processing method to apply (e.g., 'canny', 'sobel').

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing datasets with labels.
    """
    data = []
    labels = []
    filenames = []

    def process_images(file_list, label, directory):
        for file in file_list:
            img_path = os.path.join(directory, file)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(img_size)
            img_array = np.array(img)

            # Apply pre-processing if specified
            img = preprocess_image(img_array, method=preprocessing)
            data.append(img)
            labels.append(label)
            filenames.append(img_path)

    # Process signature images
    sig_files = os.listdir(signature_dir)
    process_images(sig_files, label=1, directory=signature_dir)

    # Process non-signature images
    nonsig_files = os.listdir(nonsig_dir)
    random.shuffle(nonsig_files)  # Shuffle to ensure randomness
    process_images(nonsig_files[:len(sig_files)], label=0, directory=nonsig_dir)

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Normalize data
    data = data / 255.0

    # Expand dimensions to match CNN input requirements
    data = np.expand_dims(data, axis=-1)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
        data, labels, filenames, test_size=0.2, random_state=42)

    # Extract the base document names from test files
    test_docs = {os.path.splitext(os.path.basename(file).split("_signature_")[0])[0] for file in test_files}

    # Find all documents in the raw documents directory
    all_docs = {os.path.splitext(file)[0] for file in os.listdir(raw_documents_dir)}

    # Map base document names to their extensions in the raw documents directory
    doc_extensions = {os.path.splitext(file)[0]: os.path.splitext(file)[1] for file in os.listdir(raw_documents_dir)}

    # Add to valid docs for random file selection from the test set
    valid_docs = set()
    for t_file in test_docs:
        for a_file in all_docs:
            if a_file in t_file:  # Match base document names
                valid_docs.add(a_file)

    # Save valid document filenames to the output file
    with open(output_file, "w") as f:
        for doc in valid_docs:
            extension = doc_extensions.get(doc, "")  # Get the extension or default to empty
            f.write(f"{doc}{extension}\n")

    return X_train, X_test, y_train, y_test

import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def build_model(input_shape):
    """
    Build a CNN model for binary classification.

    Parameters:
    - input_shape: Shape of the input images (height, width, channels).

    Returns:
    - model: Compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train the CNN model.

    Parameters:
    - model: Compiled CNN model.
    - X_train: Training data.
    - y_train: Training labels.
    - X_val: Validation data.
    - y_val: Validation labels.
    - epochs: Number of training epochs.
    - batch_size: Size of training batches.

    Returns:
    - history: Training history object.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val))
    return history

def main(img_preprocessing = None, plot = True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, False)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7900)])  # 8GB
        except RuntimeError as e:
            print(e)

    # Directories containing the images
    signature_dir = 'data/interim/resized_signatures'
    nonsig_dir = 'data/interim/nonsig_dataset'
    documents_dir = "data/raw/signverod_dataset/images"
    preproc = img_preprocessing
    if img_preprocessing == None:
        preproc = "no_pre"

    # Create the dataset with optional pre-processing
    X_train, X_test, y_train, y_test = create_dataset(
        signature_dir, nonsig_dir, documents_dir, output_file = "data/splits/"+ preproc +"_test_files.txt",
        preprocessing=img_preprocessing
    )

    # Build the model
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)

    '''
    ep=10
    for i in range(ep):
        history = train_model(model, X_train, y_train, X_test, y_test, epochs=1)
        X_train, X_test, y_train, y_test = 0,0,0,0
        X_train, X_test, y_train, y_test = create_dataset(
            signature_dir, nonsig_dir, documents_dir, output_file = "data/splits/"+ preproc +"_test_files.txt",
            preprocessing=img_preprocessing
        )
    '''
    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=2)

    if plot:
        plot_training_history(history)


    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # Save the model
    model.save('models/'+preproc+'_signature_classifier_model.h5')