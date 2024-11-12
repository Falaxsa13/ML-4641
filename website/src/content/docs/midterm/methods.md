---
title: Methods
description: Methods used in the ML model 
---

# Data Preprocessing Methods

Our group used some data cleaning methods to adjust the database given to a friendlier format for the Logistic Regression (LogReg) model:

- **Age Conversion**: We adjusted the data to account for leap years. This is to ensure accuracy of our data even if it does not change the results.
- **Feature Encoding**: We initially used one-hot encoding to split the features of Gender, Cholesterol, and Glucose into separate columns. The idea was to simplify the process for models like K-Means, which works best the more data is well separated [8]. Later, we realized this method is not appropriate for LogReg since we need to consider the distance between, for example, normal cholesterol against high cholesterol. Keeping it as a range from 1, 2, or 3 helped with that.
We also encoded Gender to be 0 (female) or 1 (male) instead of 1 or 2 for simplicity. We are still treating it as categorical data, however, and not a binary category.
- **Data Filtering**: We added filters for realistic values in Height, Weight, and Systolic and Diastolic Blood Pressures. We also added limits based on human ranges - for example, having the Diastolic Blood Pressure range between 60 and 140. Some rows in the dataset had values as high as 1000, which is clinically impossible.
- **Data Validation**: We swapped the Systolic and Diastolic blood pressures if both values are valid and if the Diastolic pulse was higher than the Systolic pulse. This is to ensure the model does not run into issues where it expected the Systolic pulse to be larger when it was not.
- **Duplicate and Missing Row Deletions**: We dropped any duplicate rows, as they are not necessary, and would have dropped rows with any missing values, although our database did not have any such rows.
- **Binary Conversion**: We converted the Smoking, Alcohol, and Active fields to true or false. They were handled in the databases as 0s and 1s, so we set them as a binary to speed the model.
- **Scaling**: We applied scaling to the Age, Height, Weight, Systolic/Diastolic Blood Pressures to help the LogReg model.
- **Feature Engineering**: We added BMI and Pulse Pressure as new features, based on the proposal suggestions.

# ML Methods

## Logistic Regression
For the **midterm report**, we have implemented Logistic Regression on the cleaned data. LogReg is a good choice for checking if someone has a CVD because:
- LogReg is specifically designed for binary classification problems, where the goal is to predict a binary problem. In our project, we identify the two outcomes as such:
  - The person has the condition, or
  - The person does not have the condition.
- LogReg outputs probabilities (values between 0 and 1) that represent the likelihood of an observation belonging to a certain class (e.g., the probability that someone has the condition). This makes it well-suited for decision-making where you might want to know not just the predicted class, but also the confidence level of the prediction. For instance, a confidence level of 40% may mean the doctors may want to do further screening and/or discuss preventive measures to ensure the patient’s well-being, while a confidence level of 90% means the doctor definitely wants to start taking steps for the patient’s health..
- LogReg uses the logistic function to map any real-valued number into a probability. The logistic function outputs values between 0 and 1, which works great for classification because it means we can interpret the output as a probability.
- LogReg models are relatively easy to interpret, especially when considering the coefficients. They represent the change in the odds of the outcome for a change in a variable we could modify, holding all other variables constant. This allows us to understand the influence of each feature on the outcome. For example, if we wanted to see how glucose levels were correlated with alcohol we can use a visualizer like a heatmap to see how close the two variables are.


## Implementation
We implemented two codes for our logistic regression to predict cardiovascular disease (CVD), but with distinct methodologies to improve model performance and validation. The first code uses PCA with two components for dimensionality reduction, enabling basic visualization but without handling class imbalance or hyperparameter tuning. In contrast, the second code uses SMOTE to address class imbalance, applies GridSearchCV for hyperparameter tuning, and includes an SGD classifier to compare model performance. Additionally, PCA in the second code dynamically retains 95% variance for efficiency. These variations provide a more comprehensive evaluation of logistic regression and alternative techniques, aiding in model validation and robustness.

**Part A - Logistic Regression with Fixed PCA for CVD Prediction**

This code performs logistic regression to predict cardiovascular disease (CVD) using various health-related features from a dataset. Initially, it preprocesses the data by removing unnecessary columns, splitting it into features and target variables, converting boolean columns, and standardizing the data with StandardScaler to prepare it for model training. For dimensionality reduction, it applies PCA (Principal Component Analysis), reducing the data to two components, which aids in visualization and helps compare model performance with and without dimensionality reduction.
The code then trains two logistic regression models: one on the data reduced by PCA and another on the complete set of standardized features. To evaluate model performance, it calculates several metrics, including accuracy, precision, recall, F1 score, the confusion matrix, and the ROC-AUC score for both models, providing a comprehensive view of prediction quality. For visualization, the code plots PCA results, heatmaps to show feature correlations, confusion matrices, and Receiver-Operating-Characteristics (ROC) Curves, making it easier to interpret results.
In addition, the code conducts feature analysis by calculating and plotting feature importance using the logistic regression coefficients, odds ratios, and permutation importance, helping identify the impact of individual features on the prediction of cardiovascular disease.

**Part B - Advanced CVD Prediction with SMOTE, Tuned PCA, and SGD**

This code uses logistic regression and an SGD (Stochastic Gradient Descent) classifier to predict cardiovascular disease (CVD) based on a range of health-related features. It begins by loading a dataset, defining the target variable (cardio), and excluding unnecessary columns. The selected features include age, gender, blood pressure measures, and lifestyle indicators. After ensuring boolean columns are integer-based, the data is split into training and test sets.
To handle class imbalance, SMOTE (Synthetic Minority Oversampling Technique) is applied to the training set, creating a balanced sample for the minority class. The features are then standardized using StandardScaler, followed by PCA (Principal Component Analysis) to reduce dimensionality while retaining 95% of the variance. A logistic regression model is trained with hyperparameter tuning using GridSearchCV to optimize the regularization parameter. Model performance is assessed on the test set, calculating accuracy, precision, recall, F1 score, and the confusion matrix, with visualization via a heatmap and ROC curve for deeper insights.
The code then trains an SGD classifier with a logistic loss function as an alternative, followed by similar performance evaluation and visualization steps, including accuracy, precision, recall, F1 score, and ROC-AUC scores. Additional visualizations include heatmaps of feature correlations and the confusion matrix for both models, along with ROC curves, providing a comprehensive view of model performance and feature relationships.
