# 🍚 Rice-Type-Classification

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w_V9dyQ_cAaMNQ3lTtkObkDpcvBNXohn)

This project focuses on classifying rice grain types based on their geometric features using machine learning techniques. The goal is to perform exploratory data analysis (EDA), preprocess the data, and compare classification models to achieve high predictive performance.

The project was developed in Python using Google Colab and demonstrates a complete workflow from data exploration to model evaluation.

## Dataset
This project uses the [Rice Type Classification Dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) from Kaggle, consisting of 18,185 samples with numerical features describing the geometric properties and shape characteristics of rice grains.

The target variable represents two rice types:
- **0 – Gonen**
- **1 – Jasmine**

## Project Objectives

- Perform **Exploratory Data Analysis (EDA)** to understand data distribution and feature relationships.
- Analyze correlations and reduce redundant features.
- Apply **data preprocessing**:
  - Removing non-informative columns
  - Feature reduction based on correlation
  - Standardization using StandardScaler
- Train and compare classification models:
  - **k-Nearest Neighbors (kNN)** with hyperparameter tuning
  - **Naive Bayes (GaussianNB)** as a baseline model
- Evaluate models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix

## Models
- **k-Nearest Neighbors (kNN)** – Hyperparameter k tuned on a validation set,  Best value: k = 5
- **Naive Bayes (GaussianNB)** – Used as a comparison baseline model

## Results

The final selected model was **kNN (k = 5)** based on validation performance.

Test set performance:
- **Accuracy** ≈ 99.1%
- **Precision** ≈ 98.9%
- **Recall** ≈ 99.5%
- **F1-score** ≈ 99.2%

The confusion matrix confirms a very small number of misclassifications and good generalization performance.
Feature reduction, standardization, and hyperparameter tuning significantly improved model stability and accuracy.

## Visual Results
### **Model Comparison on Validation Set**
  ![Model Metrics](model_metrics.png)
  Comparison of accuracy, precision, recall, and F1-score for kNN and Naive Bayes models.
### **Confusion Matrix – kNN (Test Set)**
  ![Confusion Matrix](confusion_matrix.png)
  The confusion matrix shows very few misclassifications between the two rice classes (Gonen and Jasmine).
### **Model Performance Comparison**
  ![Accuracy Comparison](accuracy_comparison.png)
  This chart compares validation accuracy for both models and test accuracy for the selected kNN model.

## Technologies
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Repository Structure
```bash
├── EWD.ipynb # Jupyter Notebook with full analysis and modeling 
├── ewd_presentation.pdf # Project presentation with visual results 
└── README.md # Project description
```

## How to Run
You can run the notebook in:
- **Google Colab**
- **Jupyter Notebook / JupyterLab**

Steps:

1. Clone the repository or download the files.
2. Open EWD.ipynb in your preferred environment.
3. Install required libraries if needed:
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
4. Run all cells sequentially.
