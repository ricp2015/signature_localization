import os
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models


def split_file_names(signature_dir, nonsig_dir, test_size=0.2, val_size=0.1):
    """
    Split the dataset into training, validation, and testing sets based on file names.

    Parameters:
    - signature_dir: Directory containing signature images.
    - nonsig_dir: Directory containing non-signature images.
    - test_size: Fraction of data to reserve for the test set.
    - val_size: Fraction of training data to reserve for the validation set.

    Returns:
    - splits: Dictionary containing file name splits for train, validation, and test sets.
    """
    # Get file names
    sig_files = os.listdir(signature_dir)
    nonsig_files = os.listdir(nonsig_dir)

    # Match the number of non-signatures to the number of signatures
    random.shuffle(nonsig_files)
    nonsig_files = nonsig_files[:len(sig_files)]

    # Create labels for splitting
    sig_labels = [1] * len(sig_files)
    nonsig_labels = [0] * len(nonsig_files)

    # Combine signatures and non-signatures
    all_files = sig_files + nonsig_files
    all_labels = sig_labels + nonsig_labels

    # Split into train and test sets
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=test_size, stratify=all_labels, random_state=42
    )

    # Further split train into train and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=val_size, stratify=train_labels, random_state=42
    )

    # Organize splits into a dictionary
    splits = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }

    return splits


def load_images(file_list, dir_path, img_size):
    """
    Load and preprocess images from a given directory.

    Parameters:
    - file_list: List of file names to load.
    - dir_path: Directory containing the images.
    - img_size: Tuple specifying the target size for resizing images.

    Returns:
    - data: Numpy array of preprocessed images.
    """
    data = []
    for file in file_list:
        img_path = os.path.join(dir_path, file)
        if not os.path.exists(img_path):
            continue  # Skip files not present in the directory
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(img_size)
        img_array = np.array(img)
        data.append(img_array)

    # Convert list to numpy array and normalize
    data = np.array(data) / 255.0

    # Expand dimensions to match CNN input requirements
    data = np.expand_dims(data, axis=-1)

    return data


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
    # Configure GPU memory settings (optional)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Directories containing the images
    signature_dir = 'data/interim/resized_signatures'
    nonsig_dir = 'data/interim/nonsig_dataset'

    # Split the file names
    splits = split_file_names(signature_dir, nonsig_dir)

    # Save the splits for reproducibility
    for split_name, file_list in splits.items():
        split_path = f'data/splits/{split_name}_files.csv'
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, 'w') as f:
            f.write("\n".join(file_list))

    # Load images for each split
    img_size = (734, 177)
    X_train = np.concatenate([
        load_images([f for f in splits['train'] if f in os.listdir(signature_dir)], signature_dir, img_size),
        load_images([f for f in splits['train'] if f in os.listdir(nonsig_dir)], nonsig_dir, img_size)
    ])
    y_train = np.concatenate([
        [1] * len([f for f in splits['train'] if f in os.listdir(signature_dir)]),
        [0] * len([f for f in splits['train'] if f in os.listdir(nonsig_dir)])
    ])

    X_val = np.concatenate([
        load_images([f for f in splits['validation'] if f in os.listdir(signature_dir)], signature_dir, img_size),
        load_images([f for f in splits['validation'] if f in os.listdir(nonsig_dir)], nonsig_dir, img_size)
    ])
    y_val = np.concatenate([
        [1] * len([f for f in splits['validation'] if f in os.listdir(signature_dir)]),
        [0] * len([f for f in splits['validation'] if f in os.listdir(nonsig_dir)])
    ])

    X_test = np.concatenate([
        load_images([f for f in splits['test'] if f in os.listdir(signature_dir)], signature_dir, img_size),
        load_images([f for f in splits['test'] if f in os.listdir(nonsig_dir)], nonsig_dir, img_size)
    ])
    y_test = np.concatenate([
        [1] * len([f for f in splits['test'] if f in os.listdir(signature_dir)]),
        [0] * len([f for f in splits['test'] if f in os.listdir(nonsig_dir)])
    ])

    # Build the model
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=2)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # Save the model
    model.save('models/signature_classifier_model.h5')


if __name__ == "__main__":
    main()
