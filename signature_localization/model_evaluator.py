import csv
import localize_signatures
import os


def evaluate_all_documents(test_dir, output_csv, img_preprocessing=None):
    """
    Evaluate all documents in the test directory and calculate IoU for each.

    Parameters:
    - test_dir: Directory containing test document images.
    - output_csv: Path to the output CSV file.
    - img_preprocessing: Preprocessing method to apply to images.
    """
    preproc = img_preprocessing
    if img_preprocessing == None:
        preproc = "no_pre"

    # Load the list of test files
    with open("data/splits/" + preproc + "_test_files.txt", "r") as f:
        test_files = f.read().splitlines()

    results = []

    for test_file in test_files:
        doc_path = os.path.join(test_dir, test_file)

        if not os.path.exists(doc_path):
            print(f"Document {test_file} not found in the directory.")
            continue

        # Call detect_signature for this document
        detected_regions = localize_signatures.detect_signature(img_preprocessing=img_preprocessing, doc_path=doc_path)

        # Calculate IoU (placeholder for actual IoU calculation function)
        iou = calculate_iou(detected_regions, test_file)  # Define calculate_iou function later
        results.append((test_file, iou))

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "IoU"])  # Write header
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

# Placeholder for IoU calculation function
def calculate_iou(detected_regions, ground_truth_file):
    """
    Calculate Intersection over Union (IoU) between detected regions and ground truth.

    Parameters:
    - detected_regions: List of detected regions (e.g., bounding boxes).
    - ground_truth_file: Path to the ground truth data.

    Returns:
    - IoU: Intersection over Union value.
    """
    # This function should compare detected regions with ground truth data
    # and compute the IoU. Implementation depends on the format of ground truth data.
    pass