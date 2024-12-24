from dataset_downloader import download_dataset
from crop_signatures import crop_signatures
from resize_signatures import resize_signatures
from create_nonsig_dataset import create_nonsig_dataset
import localize_signatures
import binary_classifier_CNN
import find_signatures
import selective_search_evaluator

# Pre-processing methods
methods = ['canny', 'sobel', 'laplacian', 'gaussian', 'threshold', None]

def create_data():
    download_dataset()
    crop_signatures()
    resize_signatures()
    create_nonsig_dataset()
    return
    

def main(method):
    if method not in methods:
        raise ValueError(f"Unknown pre-processing method: {method}")
    create_data()
    binary_classifier_CNN.main(method)
    selective_search_evaluator.evaluate_all_documents_with_metrics(
        test_dir="data/raw/signverod_dataset/images",
        annotations_path="data/raw/fixed_dataset/full_data.csv",
        image_info_path="data/raw/fixed_dataset/updated_image_ids.csv",
        img_preprocessing=method,
        max_files=100
    )

if __name__ == "__main__":
    preprocessing_method = "canny"
    main(preprocessing_method)