# Brain Tumor Detection

## Overview
This project focuses on brain tumor detection using deep learning techniques. The provided Jupyter Notebook (`brain-tumor-detection.ipynb`) implements a model that classifies brain MRI images as tumorous or non-tumorous. The goal is to assist in early detection and diagnosis.

## Features
- Uses Convolutional Neural Networks (CNN) for classification.
- Trained on a dataset of brain MRI images.
- Performs image preprocessing including resizing, normalization, and augmentation.
- Provides accuracy metrics and visualizations.

## Installation
To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, manually install the necessary libraries:

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras opencv-python scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/sharmaniraj009/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook brain-tumor-detection.ipynb
   ```
3. Run all the cells sequentially to train and test the model.

## Dataset
- The dataset consists of brain MRI images categorized as having tumors or being normal.
- If the dataset is not included, download it from a publicly available source and update the file paths accordingly.

## Model Architecture
- CNN-based architecture with convolutional, pooling, and dense layers.
- Activation functions: ReLU and Softmax.
- Loss function: Categorical Cross-Entropy.
- Optimizer: Adam.

## Results
- Displays accuracy and loss graphs.
- Confusion matrix and classification report for performance evaluation.

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests.


## Contact
For questions or feedback, reach out via GitHub issues or the repository owner.

---

*Note:* Ensure that the dataset is correctly linked before running the notebook.

