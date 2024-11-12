import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# Load the dataset
data = pd.read_csv('cleaned_cardio_train_dataset.csv')

# The target column from the file that corresponds to having cvd
target_column = 'cardio'

# Drop unnecessary columns for X and set Y to the target from before
X = data.drop(columns=['id', 'height', 'weight', target_column])
y = data[target_column]

# Features to use
features = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure']
X = X[features]

# Convert boolean columns to integers (ay not be necessary but did it just in case)
X[['smoke', 'alco', 'active']] = X[['smoke', 'alco', 'active']].astype(int)

# Splitting the dataset into training and testing
# This value can be tuned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying PCA to reduce dimensionality
# You can manipulate the n_components
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plotting the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Training Data')
plt.colorbar(label='Cardio Disease Presence')
plt.show()

# Training the logistic regression model with PCA
model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_train_pca, y_train)

# Training the model without PCA
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction with PCA
y_pred_pca = model_pca.predict(X_test_pca)

# Prediction without PCA
y_pred = model.predict(X_test)

# Performance metrics with PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca)
recall_pca = recall_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca)
conf_matrix_pca = confusion_matrix(y_test, y_pred_pca)

# Performance metrics without PCA
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Results with PCA
print(f'Accuracy PCA: {accuracy_pca:.2f}')
print(f'Precision PCA: {precision_pca:.2f}')
print(f'Recall PCA: {recall_pca:.2f}')
print(f'F1 Score PCA: {f1_pca:.2f}')
print('Confusion Matrix PCA:')
print(conf_matrix_pca)

# Results without PCA
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# PLOTS

# Heat map of the correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = pd.DataFrame(X, columns=features).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Confusion matrix PCA
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_pca, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix PCA Heatmap')
plt.show()

# Confusion matrix no PCA
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# ROC Curve PCA
y_pred_prob_pca = model.predict_proba(X_test)[:, 1]
fpr_pca, tpr_pca, _ = roc_curve(y_test, y_pred_prob_pca)
roc_auc_pca = roc_auc_score(y_test, y_pred_prob_pca)

plt.figure(figsize=(10, 6))
plt.plot(fpr_pca, tpr_pca, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_pca:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve PCA')
plt.legend(loc='lower right')
plt.show()

# ROC Curve no PCA
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# Feature Analysis
feature_importance = model.coef_[0]
importance_df = pd.DataFrame({'Feature': features, 'Coefficient': feature_importance})
importance_df['AbsCoefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='AbsCoefficient', ascending=False)

# Feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

# Odds Ratio
odds_ratios = np.exp(model.coef_[0])
odds_ratios_df = pd.DataFrame({'Feature': features, 'Odds Ratio': odds_ratios}).sort_values(by='Odds Ratio', ascending=False)
print(odds_ratios_df)

# Permutation Feature Importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({'Feature': features, 'Importance': result.importances_mean}).sort_values(by='Importance', ascending=False)

# Plot of permutation feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_importance_df, palette='viridis')
plt.title('Permutation Feature Importance')
plt.xlabel('Mean Importance Score')
plt.ylabel('Feature')
plt.show()