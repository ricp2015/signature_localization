# Handwritten Signature Detection and Localization

This repository contains the implementation of a pipeline for detecting and localizing handwritten signatures in scanned documents. The project explores three distinct approaches for this task: a fixed grid-based method, a selective search-based approach, and a state-of-the-art Faster R-CNN model. Each method leverages machine learning and preprocessing techniques to propose and classify regions of interest.

## Table of Contents
- [Dataset](#dataset)
- [Methods](#methods)
- [Installation](#installation)
- [Code Structure](#code-structure)

---

## Dataset
The dataset, [SignverOD](https://www.kaggle.com/datasets/victordibia/signverod), consists of scanned document images annotated with bounding boxes for signatures.

## Methods
1. **Fixed Grid-Based Method**: Divides documents into a fixed grid of regions, each classified using a CNN.
2. **Selective Search-Based Method**: Proposes regions based on image features like gradients and textures.
3. **Faster R-CNN**: A state-of-the-art object detection model fine-tuned for handwritten signature localization.

## Installation

Clone the repository:
```
git clone https://github.com/yourusername/signature_localization.git
cd signature_localization
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Code Structure

The repository follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template:

- **`docs/`**: Includes the project report.
- **`data/`**: Where the dataset is stored during the first phase of the pipeline.
- **`signature_localization/`**: Contains Python files. Run `main.py` to execute the pipeline.
- **`reports/`**: Stores evaluation metrics, visualizations, and results generated during experiments.
- **`models/`**: Stores trained models.