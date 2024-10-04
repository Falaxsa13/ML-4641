---
title: Methods
description: Data preprocessing and ML algorithms that will be applied
---

## Preprocessing

#### ✂️ Dimensionality Reduction
Given the extensive entries and features, dimensionality reduction can massively reduce the amount of time needed for the model to process. It may be that some of the features will be strongly correlated (i.e. BMI and physical activity), and we can use that to reduce our feature size.

#### ✅ Data Cleaning
Data cleaning involves handling missing values by imputing an appropriate correction. Outliers or unrealistic values, like extreme blood pressure readings, should be corrected or removed. Categorical variables need to be numerically encoded, and duplicate entries must be eliminated to ensure the dataset is ready for model training.

#### 🎯 Data Sampling
Given the large dataset, data sampling is useful for reducing model training time. It can also address class imbalance between CVD and non-CVD cases. Additionally, sampling allows training on a subset of data and testing on the remaining portion to ensure model performance.

#### 🛠️ Feature Engineering
Feature engineering is essential for deriving meaningful data from existing features. Key examples include creating age group bins combined with blood pressure to account for higher CVD risk in older individuals and calculating BMI from height and weight data to improve predictive performance.

---

## ML Algorithms

#### 🌳 Decision Tree
A decision tree is an algorithm that splits data based on informative features, forming a tree structure. Internal nodes represent decisions, and leaf nodes give the final classification. For CVD detection, decision trees highlight which factors contribute to classifying a patient as having CVD. They handle both continuous and categorical data, such as blood pressure, cholesterol, and smoking status.

#### 🌲🌲 Random Forest
A random forest consists of many decision trees. It is an ensemble learning method, using multiple models to improve the accuracy of the decisions, thus being an ideal case for medical studies. Random forests can be useful because medical relations are complex and our dataset may not be accurately interpreted by one method alone. The tradeoff of a random forest is that it is harder to interpret due to its ensemble nature, as well as its slow runtime due to it needing to run multiple trees per test.

#### 📉 Logistic Regression
Logistic regression is ideal for binary classification, offering probabilistic outputs to assess the likelihood of CVD. It provides insights into key risk factors and works well with both continuous and categorical data, making it versatile for medical datasets.
