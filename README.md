# signature_localization

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

### Dataset Preparation and Image Preprocessing
* **Image Cropping**:
  * Extract sections containing signatures based on annotated bounding boxes.
* **Dimensional Normalization**:
  * Apply padding to standardize image sizes.
* **Creation of Non-Signature Images**:
  * Extract random sections from documents that do not overlap with the annotated bounding boxes.
  * *Extra*: Generate synthetic false positives with artificial noise.
* **Edge Detection Application**:
  * Create images with edge detection applied. Various techniques can be tested, with each version given to the CNN for benchmarking (e.g., no edge detection vs. Canny vs. Sobel vs. Laplacian).

### Model Training
* **Dataset Splitting**:
  * Divide the dataset into training, validation, and test sets.
* **Defining and Configuring the CNN**:
  * Set up a CNN for binary classification (signature/no signature).
* **Training the Model**:
  * Monitor metrics during training:
    * Loss, accuracy, precision, recall, F1-score.
* **Save the Trained Model Parameters**.

### Document Segmentation
* **Splitting Documents into Sections**:
  * Apply a sliding window with fixed dimensions and overlap.
* **Testing the Classifier on Generated Sections**.

### Signature Localization
* **Identify the Window with the Highest Classification Score**.
* **Optional**:
  * Use a secondary model or algorithm to refine the bounding box precision.

### Final Validation and Testing
* **Evaluate CNN Performance with Metrics**:
  * Precision, recall, F1-score for classification.
  * Intersection over Union (IoU) for localization.

### Extra: Feature Extraction for SVMs + SVM Model Development (Refer to Relevant Papers)
* **Feature Extraction**:
  * Histogram of Oriented Gradients (HOG).
  * Local Binary Patterns (LBP).
  * Edge detection and edge statistics.
