import csv
import os
import pandas as pd
import ast
from PIL import Image
import localize_signatures

def convert_normalized_to_absolute(bbox, img_width, img_height):
    """
    Convert normalized bounding box coordinates to absolute pixel values
    and format as [x_min, y_min, x_max, y_max].
    """
    x_min = bbox[0] * img_width
    y_min = bbox[1] * img_height
    x_max = x_min + bbox[2] * img_width
    y_max = y_min + bbox[3] * img_height
    return [x_min, y_min, x_max, y_max]


def evaluate_all_documents_with_metrics(test_dir, annotations_path, image_info_path, img_preprocessing=None, max_files=None):
    """
    Evaluate all documents in the test directory and calculate metrics for signature detection.

    Parameters:
    - test_dir: Directory containing test document images.
    - annotations_path: Path to the annotations CSV.
    - image_info_path: Path to the image info CSV.
    - output_csv: Path to the output metrics CSV file.
    - iou_csv: Path to the output IoU CSV file.
    - img_preprocessing: Preprocessing method to apply to images.
    - max_files: Optional limit to the number of files to process.
    """
    preproc = img_preprocessing or "no_pre"
    iou_csv="reports/" + preproc + "_iou_results.csv"
    output_csv="reports/" + preproc + "_evaluation_metrics.csv"
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
    iou_results = []
    cont = 0

    total_gt_signatures = 0
    total_detected_signatures = 0
    true_positives = 0

    for test_file in test_files:
        doc_path = os.path.join(test_dir, test_file)

        if not os.path.exists(doc_path):
            print(f"Document {test_file} not found in the directory.")
            continue

        # Load image dimensions
        with Image.open(doc_path) as img:
            img_width, img_height = img.size

        # Get the ground truth annotations for the document
        ground_truth_data = merged_data[merged_data['file_name'] == test_file]
        ground_truth_boxes = []
        for idx, bbox in enumerate(ground_truth_data['bbox']):
            # Convert string representation of list to actual list
            bbox_norm = ast.literal_eval(bbox)
            bbox_abs = convert_normalized_to_absolute(bbox_norm, img_width, img_height)
            ground_truth_boxes.append((idx, bbox_abs))

        total_gt_signatures += len(ground_truth_boxes)

        # Call detect_signature for this document
        detected_regions = localize_signatures.detect_signature(
            img_preprocessing=img_preprocessing, doc_path=doc_path, plot_results=False
        )

        total_detected_signatures += len(detected_regions)

        # Match detected regions with ground truth
        matched_gt = set()
        ious = []
        for signature_id, gt_box in ground_truth_boxes:
            max_iou = 0.0
            for detected_box in detected_regions:
                iou = localize_signatures.calculate_iou(detected_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                if iou > 0.4:
                    matched_gt.add(signature_id)

            # Append IoU results for the CSV
            iou_results.append([test_file, signature_id, max_iou, img_preprocessing])
            ious.append(max_iou)

        # Count true positives
        true_positives += len(matched_gt)

        # Log IoUs for debugging
        for iou in ious:
            print(f"{test_file}: IoU = {iou:.4f}")
        print(detected_regions)
        print([gt_box for _, gt_box in ground_truth_boxes])

        cont += 1
        if max_files is not None and cont >= max_files:
            break

    # Calculate precision, recall, accuracy, F1-score
    precision = true_positives / total_detected_signatures if total_detected_signatures > 0 else 0
    recall = true_positives / total_gt_signatures if total_gt_signatures > 0 else 0
    accuracy = true_positives / total_gt_signatures if total_gt_signatures > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Write metrics results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Precision", "Recall", "Accuracy", "F1-Score", "method"])
        writer.writerow([precision, recall, accuracy, f1_score, img_preprocessing])

    # Write IoU results to CSV
    with open(iou_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["document_name", "signature_id", "IoU", "method"])
        writer.writerows(iou_results)

    print(f"Metrics saved to {output_csv}")
    print(f"IoU results saved to {iou_csv}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1_score:.4f}")

# Example call
evaluate_all_documents_with_metrics(
    test_dir="data/raw/signverod_dataset/images",
    annotations_path="data/raw/fixed_dataset/full_data.csv",
    image_info_path="data/raw/fixed_dataset/updated_image_ids.csv",
    img_preprocessing=None,
    max_files=100
)
