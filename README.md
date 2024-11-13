# Learning-Machine-Learning
## Assignment 1

# **Mango Leaf Disease Classification Using Decision Tree and Random Forest**

## Overview
This project aims to classify various mango leaf diseases using machine learning techniques, specifically **Decision Tree** and **Random Forest** classifiers. The dataset consists of images of mango leaves, each labeled with a specific disease or healthy status. The goal is to build a model that can accurately identify the disease based on the leaf image.

## Dataset
The dataset contains images of mango leaves with different diseases, including:
- **Anthracnose**
- **Bacterial Canker**
- **Cutting Weevil**
- **Die Back**
- **Gall Midge**
- **Healthy**
- **Powdery Mildew**
- **Sooty Mould**

The dataset was extracted from the [MangoLeafBD Dataset](https://data.mendeley.com/datasets/hxsnvwty3r/1). The images were resized to **128x128 pixels** and normalized for preprocessing. The data was split into **80% training** and **20% testing**.

## Libraries Used
This project uses the following libraries:
- `matplotlib`: For data visualization.
- `seaborn`: For enhanced data visualization.
- `scikit-learn`: For machine learning algorithms and evaluation.
- `opencv-python-headless`: For image processing.
- `pandas`: For handling and processing data.
- `numpy`: For numerical computations.
- `joblib`: For saving and loading models.

To install the necessary dependencies, run:

```bash
pip install matplotlib seaborn scikit-learn opencv-python-headless pandas numpy joblib
```

## Model Training and Evaluation

### **Decision Tree Classifier**
A **Decision Tree** was trained to classify the mango leaf diseases. The model showed varying performance across different disease categories.

- **Accuracy**: 59.38%
- **Classification Report**:

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| **Anthracnose**     | 0.45      | 0.50   | 0.47     | 90      |
| **Bacterial Canker**| 0.55      | 0.58   | 0.57     | 91      |
| **Cutting Weevil**  | 0.92      | 0.91   | 0.92     | 93      |
| **Die Back**        | 0.67      | 0.77   | 0.72     | 86      |
| **Gall Midge**      | 0.47      | 0.48   | 0.48     | 104     |
| **Healthy**         | 0.65      | 0.63   | 0.64     | 123     |
| **Powdery Mildew**  | 0.47      | 0.44   | 0.45     | 101     |
| **Sooty Mould**     | 0.57      | 0.49   | 0.53     | 112     |
| **Accuracy**        |           |        | 0.59     | 800     |
| **Macro avg**       | 0.60      | 0.60   | 0.60     | 800     |
| **Weighted avg**    | 0.59      | 0.59   | 0.59     | 800     |

### **Random Forest Classifier**
The **Random Forest** model, an ensemble method consisting of multiple decision trees, performed significantly better, achieving higher accuracy and more consistent results across the disease categories.

- **Accuracy**: 82.38%
- **Classification Report**:

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| **Anthracnose**     | 0.70      | 0.81   | 0.75     | 90      |
| **Bacterial Canker**| 0.76      | 0.84   | 0.80     | 91      |
| **Cutting Weevil**  | 0.98      | 0.98   | 0.98     | 93      |
| **Die Back**        | 0.93      | 0.88   | 0.90     | 86      |
| **Gall Midge**      | 0.83      | 0.82   | 0.82     | 104     |
| **Healthy**         | 0.84      | 0.81   | 0.83     | 123     |
| **Powdery Mildew**  | 0.78      | 0.71   | 0.75     | 101     |
| **Sooty Mould**     | 0.81      | 0.77   | 0.79     | 112     |
| **Accuracy**        |           |        | 0.82     | 800     |
| **Macro avg**       | 0.83      | 0.83   | 0.83     | 800     |
| **Weighted avg**    | 0.83      | 0.82   | 0.82     | 800     |

### **Comparison of Decision Tree and Random Forest**

| Metric             | Decision Tree   | Random Forest   |
|--------------------|-----------------|-----------------|
| **Accuracy**       | 59.38%          | 82.38%          |
| **Precision**      | Varies by class | Higher overall  |
| **Recall**         | Varies by class | Higher overall  |
| **F1-Score**       | Varies by class | Higher overall  |

The **Random Forest** outperformed **Decision Tree** across all evaluation metrics, with higher accuracy and more consistent results.

## Visualizations
The following visualizations are included in the project:
1. **Class Distribution**: A count plot of the class distribution of mango leaf diseases.
2. **Sample Images**: A random sample of images from the dataset.
3. **Confusion Matrices**: Visual representation of model predictions versus true labels.
4. **Learning Curves**: A comparison of the learning curves for both the Decision Tree and Random Forest classifiers.
5. **ROC Curve**: A receiver operating characteristic (ROC) curve for each class with AUC values.

## Future Work
- **Model Optimization**: Hyperparameter tuning and feature engineering to improve model performance.
- **Deep Learning**: Exploring deep learning models like Convolutional Neural Networks (CNNs) for potentially better results.
- **Web Deployment**: Deploying the trained models to a web application where users can upload images of mango leaves and receive predictions in real-time.

## License
This project is licensed under the MIT License.

## Author
Md. Abir Hasan Khan

## GitHub Repository
You can access the full code, dataset, and additional resources on my GitHub repository:  
[Assignment 1](https://github.com/abirhasankhan/Learning-Machine-Learning/tree/main/Assignment%201)
```

This updated `README.md` file reflects the new accuracy and detailed classification report for both models, providing more precise information about your project's results.
