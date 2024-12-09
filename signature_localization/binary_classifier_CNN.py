import os
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset(signature_dir, nonsig_dir, output_file="data/splits/test_files.txt", img_size=(734, 177)):
    """
    Create dataset from signature and non-signature images.

    Parameters:
    - signature_dir: Directory containing signature images.
    - nonsig_dir: Directory containing non-signature images.
    - output_file: File to save the test filenames.
    - img_size: Tuple specifying the target size for resizing images.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing datasets with labels.
    """
    data = []
    labels = []
    filenames = []

    # Process signature images
    sig_files = os.listdir(signature_dir)
    for file in sig_files:
        img_path = os.path.join(signature_dir, file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(img_size)
        img_array = np.array(img)
        data.append(img_array)
        labels.append(1)  # Label for signatures
        filenames.append(img_path)

    # Process non-signature images
    nonsig_files = os.listdir(nonsig_dir)
    random.shuffle(nonsig_files)  # Shuffle to ensure randomness
    for file in nonsig_files[:len(sig_files)]:  # Match the number of signature images
        img_path = os.path.join(nonsig_dir, file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(img_size)
        img_array = np.array(img)
        data.append(img_array)
        labels.append(0)  # Label for non-signatures
        filenames.append(img_path)

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

    # Save test filenames to the specified file
    with open(output_file, "w") as f:
        for file in test_files:
            f.write(f"{file}\n")

    return X_train, X_test, y_train, y_test

import tensorflow as tf
from keras import layers, models

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

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Disable memory growth to pre-allocate GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, False)
                # Allocate a large chunk of GPU memory upfront
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7900)])  # 8GB
        except RuntimeError as e:
            print(e)
    # Directories containing the images
    signature_dir = 'data/interim/resized_signatures'
    nonsig_dir = 'data/interim/nonsig_dataset'

    # Create the dataset
    X_train, X_test, y_train, y_test = create_dataset(signature_dir, nonsig_dir)

    # Build the model
    input_shape = X_train.shape[1:]  # Shape of a single image
    model = build_model(input_shape)

    # Train the model
    #for i in range(2):
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=2)
        #X_train, X_test, y_train, y_test = create_dataset(signature_dir, nonsig_dir)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # Save the model
    model.save('models/signature_classifier_model.h5')
