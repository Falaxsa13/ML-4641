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
