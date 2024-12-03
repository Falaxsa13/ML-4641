---
title: Machine Learning Methods 🤖
description: Detailed description of the ML models used in the heart disease predictor project.
---

## 🛠️ **Data Preprocessing Methods**

Our group used various data cleaning methods to adjust the database into a more model-friendly format. Below are the techniques we applied:

---

#### 🗓️ **Age Conversion**

We adjusted the data to account for **leap years**, ensuring the accuracy of our dataset. While this change does not affect the results, it adds precision to the data representation.

---

#### 🔢 **Feature Encoding**

- Initially applied **one-hot encoding** to separate the features of **Gender, Cholesterol**, and **Glucose** into individual columns. This approach simplifies clustering models like **K-Means**, which benefit from well-separated data [8].
- Later, we discovered this was unsuitable for **LogReg**, where distance matters (e.g., between normal and high cholesterol). Instead, we encoded these features as a range (1, 2, 3) for better compatibility.
- Gender was encoded as **0 (female)** or **1 (male)** for simplicity, while still treating it as categorical rather than binary.

---

#### 🚫 **Data Filtering**

We applied filters to ensure **realistic values** for:

- **Height, Weight**, and **Systolic/Diastolic Blood Pressures**.
- Added human-based limits (e.g., Diastolic Blood Pressure set between **60 and 140**).  
  Some rows contained impossible values (e.g., blood pressure over **1000**), which we excluded.

---

#### ✅ **Data Validation**

In cases where the **Diastolic pulse** was higher than the **Systolic pulse**, we swapped the values (if valid). This ensures the dataset conforms to expected physiological norms, preventing issues during model training.

---

#### 🗑️ **Duplicate and Missing Row Deletions**

- **Duplicates** were removed as unnecessary.
- Rows with missing values would have been dropped, though the dataset fortunately had none.

---

#### 🔄 **Binary Conversion**

The **Smoking, Alcohol**, and **Active** fields, originally represented as **0s and 1s**, were converted to **true/false** values to optimize processing and model performance.

---

#### 📏 **Scaling**

We scaled features such as **Age, Height, Weight, and Systolic/Diastolic Blood Pressures** to enhance compatibility with models like **LogReg**.

---

#### 🛠️ **Feature Engineering**

We engineered new features:

- **BMI** (Body Mass Index).
- **Pulse Pressure** (difference between Systolic and Diastolic pressures).

These features were added based on proposal suggestions to improve the dataset's predictive power.

---

title: Machine Learning Methods 🤖
description: Detailed description of the ML models used in the heart disease predictor project.

---

## 🤖 **Machine Learning Methods**

---

### **Logistic Regression 📈**

For our first methodology, we implemented **Logistic Regression** on the cleaned data. **LogReg** is a good choice for checking if someone has a **CVD** because:

- **LogReg is specifically designed for binary classification problems**, where the goal is to predict a binary problem. In our project, we identify the two outcomes as such:

  1. **The person has the condition**, or
  2. **The person does not have the condition**.

- **LogReg outputs probabilities** (values between 0 and 1) that represent the likelihood of an observation belonging to a certain class (e.g., the probability that someone has the condition). This makes it well-suited for decision-making where you might want to know not just the predicted class, but also the **confidence level of the prediction**. For instance:

  - A confidence level of **40%** may mean the doctors may want to do further screening and/or discuss preventive measures to ensure the patient’s well-being.
  - A confidence level of **90%** means the doctor definitely wants to start taking steps for the patient’s health.

- **LogReg uses the logistic function** to map any real-valued number into a probability. The logistic function outputs values between **0 and 1**, which works great for classification because it means we can interpret the output as a probability.

- **LogReg models are relatively easy to interpret**, especially when considering the coefficients. They represent the change in the odds of the outcome for a change in a variable we could modify, holding all other variables constant. This allows us to understand the influence of each feature on the outcome. For example, if we wanted to see how **glucose levels** were correlated with **alcohol**, we can use a visualizer like a **heatmap** to see how close the two variables are.

### **Implementation 💻**

We implemented **two codes** for our logistic regression to predict cardiovascular disease (CVD), but with distinct methodologies to improve model performance and validation.

- **The first code** uses **PCA** with two components for dimensionality reduction, enabling basic visualization but without handling class imbalance or hyperparameter tuning.
- **In contrast, the second code** uses **SMOTE** to address class imbalance, applies **GridSearchCV** for hyperparameter tuning, and includes an **SGD classifier** to compare model performance. Additionally, **PCA** in the second code dynamically retains **95% variance** for efficiency.

These variations provide a more comprehensive evaluation of logistic regression and alternative techniques, aiding in model validation and robustness.

#### **Part A - Logistic Regression with Fixed PCA for CVD Prediction**

This code performs logistic regression to predict cardiovascular disease (CVD) using various health-related features from a dataset. Initially, it preprocesses the data by:

- **Removing unnecessary columns**.
- **Splitting** it into features and target variables.
- **Converting boolean columns**.
- **Standardizing the data** with **StandardScaler** to prepare it for model training.

For dimensionality reduction, it applies **PCA (Principal Component Analysis)**, reducing the data to **two components**, which aids in visualization and helps compare model performance with and without dimensionality reduction.

The code then trains **two logistic regression models**:

1. **One on the data reduced by PCA**.
2. **Another on the complete set of standardized features**.

To evaluate model performance, it calculates several metrics, including:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 score**
- **Confusion Matrix**
- **ROC-AUC score**

For visualization, the code plots:

- **PCA results**
- **Heatmaps** to show feature correlations
- **Confusion matrices**
- **Receiver-Operating-Characteristics (ROC) Curves**

This makes it easier to interpret results.

In addition, the code conducts **feature analysis** by calculating and plotting feature importance using the logistic regression coefficients, odds ratios, and permutation importance, helping identify the impact of individual features on the prediction of cardiovascular disease.

#### **Part B - Advanced CVD Prediction with SMOTE, Tuned PCA, and SGD 🏋️‍♂️**

This code uses logistic regression and an **SGD (Stochastic Gradient Descent) classifier** to predict cardiovascular disease (CVD) based on a range of health-related features. It begins by:

- **Loading a dataset**, defining the target variable (**cardio**), and excluding unnecessary columns.
- The selected features include **age, gender, blood pressure measures, and lifestyle indicators**.
- After ensuring boolean columns are integer-based, the data is **split into training and test sets**.

To handle class imbalance, **SMOTE (Synthetic Minority Oversampling Technique)** is applied to the training set, creating a balanced sample for the minority class. The features are then:

- **Standardized** using **StandardScaler**.
- Followed by **PCA** to reduce dimensionality while retaining **95% of the variance**.

A logistic regression model is trained with **hyperparameter tuning** using **GridSearchCV** to optimize the regularization parameter. Model performance is assessed on the test set, calculating:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 score**
- **Confusion Matrix**

Visualization via a **heatmap** and **ROC curve** provides deeper insights.

The code then trains an **SGD classifier** with a logistic loss function as an alternative, followed by similar performance evaluation and visualization steps, including:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 score**
- **ROC-AUC scores**

Additional visualizations include:

- **Heatmaps** of feature correlations
- **Confusion matrices** for both models
- **ROC curves**

This provides a comprehensive view of model performance and feature relationships.
