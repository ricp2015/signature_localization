from dataset_downloader import download_dataset
from crop_signatures import crop_signatures
from resize_signatures import resize_signatures
from create_nonsig_dataset import create_nonsig_dataset
import binary_classifier_CNN
import find_signatures

def create_data():
    download_dataset()
    crop_signatures()
    resize_signatures()
    create_nonsig_dataset()

def main():
    create_data() # (!TODO checker to verify if already created) For now, just comment this line if already done.
    binary_classifier_CNN.main() #!TODO builds and trains a binary classifier (CNN based). For now, comment this line if already trained.
    find_signatures.detect_signatures()

if __name__ == "__main__":
    main()