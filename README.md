# Learning-Machine-Learning
## Assignment 1
### **Report on Mango Leaf Disease Classification Using Decision Tree and Random Forest**

#### **1. Introduction**
This project aimed to classify mango leaf diseases using two machine learning classifiers: **Decision Tree** and **Random Forest**. The dataset consists of images of mango leaves, each labeled with one of several disease categories. Our objective was to evaluate and compare the performance of these models on the classification task.

#### **2. Dataset Overview**
The dataset contains images of mango leaves, categorized into several disease types. These categories include:
- **Cutting Weevil**
- **Gall Midge**
- **Anthracnose**
- **Bacterial Canker**
- **Sooty Mould**
- **Powdery Mildew**
- **Die Back**
- **Healthy**

Each image was resized to **128x128** pixels for consistency, and the images were flattened into 1D arrays for model training. The data was split into **80% training** and **20% testing**.

#### **3. Exploratory Data Analysis (EDA)**
- **Data Loading**: We successfully loaded images and labels from the dataset and preprocessed the images to a consistent size.
- **Data Preprocessing**: All images were resized to 128x128 pixels, flattened into 1D vectors, and normalized. The labels were encoded numerically for use in machine learning models.
- **Data Split**: We split the dataset into training and testing sets, with 80% of the data used for training and 20% for testing.

#### **4. Model Training and Evaluation**

##### **4.1 Decision Tree Classifier**
We trained a **Decision Tree Classifier** using scikit-learn, which builds a tree-like structure to make decisions based on input features. After training the model, we evaluated it using accuracy and a detailed classification report.

**Decision Tree Performance**:
- **Accuracy**: 69.63%
- **Precision, Recall, and F1-Score**:
  - **Anthracnose**: Precision: 0.69, Recall: 0.69, F1-Score: 0.69
  - **Bacterial Canker**: Precision: 0.75, Recall: 0.81, F1-Score: 0.78
  - **Cutting Weevil**: Precision: 0.89, Recall: 0.89, F1-Score: 0.89
  - **Die Back**: Precision: 0.85, Recall: 0.77, F1-Score: 0.81
  - **Gall Midge**: Precision: 0.52, Recall: 0.57, F1-Score: 0.54
  - **Healthy**: Precision: 0.69, Recall: 0.66, F1-Score: 0.67
  - **Powdery Mildew**: Precision: 0.56, Recall: 0.62, F1-Score: 0.59
  - **Sooty Mould**: Precision: 0.58, Recall: 0.50, F1-Score: 0.54

The **Decision Tree** classifier performed well on certain classes like **Cutting Weevil**, but struggled with others, such as **Gall Midge** and **Sooty Mould**.

##### **4.2 Random Forest Classifier**
We trained a **Random Forest Classifier**, which is an ensemble model consisting of multiple decision trees. The Random Forest method reduces overfitting by averaging the predictions from several trees, improving accuracy.

**Random Forest Performance**:
- **Accuracy**: 88.63%
- **Precision, Recall, and F1-Score**:
  - **Anthracnose**: Precision: 0.92, Recall: 0.88, F1-Score: 0.89
  - **Bacterial Canker**: Precision: 0.88, Recall: 0.90, F1-Score: 0.89
  - **Cutting Weevil**: Precision: 0.98, Recall: 0.94, F1-Score: 0.96
  - **Die Back**: Precision: 0.89, Recall: 0.93, F1-Score: 0.91
  - **Healthy**: Precision: 0.89, Recall: 0.91, F1-Score: 0.90
  - **Powdery Mildew**: Precision: 0.90, Recall: 0.88, F1-Score: 0.89
  - **Sooty Mould**: Precision: 0.85, Recall: 0.87, F1-Score: 0.86

The **Random Forest** classifier significantly outperformed the Decision Tree in terms of accuracy and other classification metrics. It showed strong performance across almost all categories, particularly **Cutting Weevil** and **Die Back**.

#### **5. Comparison of Decision Tree and Random Forest**

| Metric             | Decision Tree   | Random Forest   |
|--------------------|-----------------|-----------------|
| **Accuracy**       | 69.63%          | 88.63%          |
| **Precision**      | Varies by class | Higher overall  |
| **Recall**         | Varies by class | Higher overall  |
| **F1-Score**       | Varies by class | Higher overall  |

- **Random Forest** outperformed **Decision Tree** in terms of overall accuracy, precision, recall, and F1-score. The **Decision Tree** showed more variability across different disease categories.

#### **6. Conclusion**
The **Random Forest Classifier** demonstrated superior performance compared to the **Decision Tree Classifier**. The Random Forest model showed higher accuracy and more consistent performance across the various disease categories. This is attributed to its ensemble approach, which reduces overfitting and improves generalization.

- **Future Work**: To further improve performance, exploring **Convolutional Neural Networks (CNNs)** for image-based classification could yield even better results, as CNNs are specialized for handling image data.

#### **7. GitHub Repository**
You can access the full code, dataset, and additional resources on my GitHub repository:  
[Assignment 1](https://github.com/abirhasankhan/Learning-Machine-Learning/tree/main/Assignment%201)

---
