import csv
import os
import pandas as pd
import ast
import localize_signatures

def evaluate_all_documents(test_dir, annotations_path, image_info_path, output_csv, img_preprocessing=None, max_files=None):
    """
    Evaluate all documents in the test directory and calculate IoU for each.

    Parameters:
    - test_dir: Directory containing test document images.
    - annotations_path: Path to the annotations CSV.
    - image_info_path: Path to the image info CSV.
    - output_csv: Path to the output CSV file.
    - img_preprocessing: Preprocessing method to apply to images.
    - max_files: Optional limit to the number of files to process.
    """
    preproc = img_preprocessing or "no_pre"

    # Load the list of test files
    with open(f"data/splits/{preproc}_test_files.txt", "r") as f:
        test_files = f.read().splitlines()

    # Load annotations and image info
    annotations = pd.read_csv(annotations_path)
    image_info = pd.read_csv(image_info_path)

    # Filter annotations for category_id == 1 (signatures)
    signature_annotations = annotations[annotations['category_id'] == 1]

    # Merge annotations with image info
    merged_data = signature_annotations.merge(
        image_info, left_on='image_id', right_on='id', suffixes=('_annotation', '_image')
    )

    results = []
    cont = 0

    for test_file in test_files:
        doc_path = os.path.join(test_dir, test_file)

        if not os.path.exists(doc_path):
            print(f"Document {test_file} not found in the directory.")
            continue

        # Get the ground truth annotations for the document
        ground_truth_data = merged_data[merged_data['file_name'] == test_file]
        ground_truth_boxes = []
        for bbox in ground_truth_data['bbox']:
            # Convert string representation of list to actual list
            ground_truth_boxes.append(ast.literal_eval(bbox))

        # Call detect_signature for this document
        detected_regions = localize_signatures.detect_signature(
            img_preprocessing=img_preprocessing, doc_path=doc_path, plot_results=False
        )

        # Calculate IoU for each detected region against ground truth
        ious = []
        for detected_box in detected_regions:
            for gt_box in ground_truth_boxes:
                iou = localize_signatures.calculate_iou(detected_box, gt_box)
                ious.append(iou)

        # Use the maximum IoU as the result for this document
        max_iou = max(ious) if ious else 0.0
        results.append((test_file, max_iou))
        print(f"{test_file}: Max IoU = {max_iou:.4f}")

        cont += 1
        if max_files is not None and cont >= max_files:
            break

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "Max IoU"])  # Write header
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

evaluate_all_documents(
    test_dir="data/raw/signverod_dataset/images",
    annotations_path="data/raw/fixed_dataset/full_data.csv",
    image_info_path="data/raw/fixed_dataset/updated_image_ids.csv",
    output_csv="output/evaluation_results.csv",
    img_preprocessing=None,
    max_files=100
)
