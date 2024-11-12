# Data Cleaning Changes

- **Age Conversion**: Adjusted age calculation to account for leap years.

- **Feature Encoding**:
  - Initially used one-hot encoding for "gender," "cholesterol," and "gluc" but reverted to numerical labels (1, 2, 3) for "cholesterol" and "gluc" to reflect the ordinal nature of the values.
  - Mapped "gender" to binary (0 for female, 1 for male) instead of (1, 2).

- **Filtering**:
  - Applied filters for realistic values in "height," "weight," "ap_hi," and "ap_lo." Set limits based on human ranges (e.g., diastolic pressure `ap_lo` between 60 and 140).

- **Blood Pressure Validation**: Checked that "ap_hi" (systolic) is higher than "ap_lo" (diastolic) and swapped values if necessary within a realistic range.

- **Duplicate and Missing Values**:
  - Dropped duplicate rows.
  - Dropped rows with missing values, though none were found in this dataset.

- **Binary Conversion**: Converted "smoke," "alco," and "active" fields to boolean (True/False).

- **Scaling**: Applied scaling to "age," "height," "weight," "ap_hi," "ap_lo," and added "pulse_pressure" to aid models like Logistic Regression.

- **Feature Engineering**: Added "BMI" and "pulse pressure" as new features based on the proposal's suggestions.
