from dataset_downloader import download_dataset
from crop_signatures import crop_signatures
from resize_signatures import resize_signatures
from create_nonsig_dataset import create_nonsig_dataset
import localize_signatures
import binary_classifier_CNN
import find_signatures

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
    #create_data() # (!TODO checker to verify if already created) For now, just comment this line if already done.
    binary_classifier_CNN.main(method) #!TODO builds and trains a binary classifier (CNN based). For now, comment this line if already trained.
    #localize_signatures.detect_signature(method)

if __name__ == "__main__":
    preprocessing_method = "canny"
    main(preprocessing_method)