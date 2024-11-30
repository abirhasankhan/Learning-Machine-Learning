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

---
## Assignment 2

```markdown
# Model Performance and Explainability on Classification Tasks

This project implements multiple machine learning models to classify a dataset and evaluate their performance using various metrics. The models used include:

- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Stacking Classifier (Logistic Regression, Decision Tree, Random Forest)**
- **Voting Classifiers (Hard and Soft Voting)**

In addition to the classification models, the project includes model explainability tools such as **SHAP** and **LIME** for interpreting the results.

## Dataset

The dataset used in this project can be downloaded from Kaggle:

- **Dataset Source:** [zeesolver/uhygtttt](https://www.kaggle.com/datasets/zeesolver/uhygtttt)
- **Description:** The dataset contains features relevant for classification. The target variable is used for the classification task.

## Setup Instructions

### Prerequisites

You need to have Python installed. Additionally, the following libraries are required:

```bash
pip install kagglehub shap lime scikit-learn pandas numpy matplotlib seaborn
```

### Data Loading

This project uses the `kagglehub` library to download the dataset. You can access the dataset using the following code:

```python
import kagglehub
path = kagglehub.dataset_download("zeesolver/uhygtttt")
data = pd.read_csv(f"{path}/output.csv")
```

Make sure to update the dataset path if the actual dataset name differs.

### Data Preprocessing

The data preprocessing involves:
- Handling missing values (rows with missing data are removed).
- Encoding categorical features using `LabelEncoder`.
- Splitting the data into features (`X`) and the target variable (`y`).

### Train-Test Split

The dataset is split into training and testing sets with 70% of the data used for training and 30% used for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Model Training and Evaluation

#### 1. **Random Forest Classifier**
A Random Forest Classifier is trained and evaluated using accuracy, precision, recall, and F1 score.

```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
```

#### 2. **Gradient Boosting Classifier**
Gradient Boosting is trained and evaluated similarly.

```python
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
```

#### 3. **Stacking Classifier**
A Stacking Classifier combining Logistic Regression, Decision Tree, and Random Forest is trained.

```python
stacking_model = VotingClassifier(
    estimators=[('lr', LogisticRegression(random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('rf', RandomForestClassifier(random_state=42))],
    voting='soft'
)
```

#### 4. **Voting Classifiers**
Both hard and soft voting classifiers are trained using Random Forest, Gradient Boosting, and Decision Tree models.

```python
hard_voting = VotingClassifier(
    estimators=[('rf', RandomForestClassifier(random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42))],
    voting='hard'
)
```

#### 5. **Model Evaluation**
For each model, the following performance metrics are evaluated:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (visualized using a heatmap)

#### Cross-Validation

Cross-validation is performed to evaluate the models' generalization capability. For each model, cross-validation accuracy is calculated.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
```

### Model Explainability

#### 1. **SHAP (Shapley Additive Explanations)**
SHAP is used to explain the model’s predictions. It provides a summary plot to understand feature importance.

```python
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, feature_names=X_test.columns)
```

#### 2. **LIME (Local Interpretable Model-Agnostic Explanations)**
LIME is used to explain individual predictions by approximating the model locally.

```python
from lime.lime_tabular import LimeTabularExplainer
lime_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Class 0', 'Class 1'], mode='classification')
lime_exp = lime_explainer.explain_instance(X_test.iloc[0].values, rf_model.predict_proba)
lime_exp.show_in_notebook()
```

## Results and Visualizations

After training and evaluation, the performance of each model is displayed in the form of a bar chart comparing the accuracy, precision, recall, and F1 score.

Additionally, confusion matrices for each model are visualized to understand the true positives, false positives, true negatives, and false negatives.

## Conclusion

This project demonstrates the application of several popular ensemble learning methods (Random Forest, Gradient Boosting, Stacking, and Voting) for classification tasks. Model evaluation and explanation techniques (SHAP and LIME) are used to ensure interpretability and transparency.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is provided by [zeesolver](https://www.kaggle.com/datasets/zeesolver/uhygtttt).
- The machine learning models and evaluation metrics are implemented using [scikit-learn](https://scikit-learn.org/).

```

### Notes:
1. **Dataset**: Ensure that the dataset link (`zeesolver/uhygtttt`) is correct and available on Kaggle.
2. **Libraries**: If any additional libraries are needed (e.g., for visualization or other tasks), make sure to include them in the setup instructions.
3. **Visualization**: The confusion matrix and performance comparison chart will help to compare models effectively.
