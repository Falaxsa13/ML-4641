import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

# Dataset
data = pd.read_csv('cleaned_cardio_train_dataset.csv')

# The target column from the file that corresponds to having cvd
target_column = 'cardio'

# Drop unnecessary columns for X and set Y to the target from before
X = data.drop(columns=['id', 'height', 'weight', target_column])
y = data[target_column]

# Features to use
features = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure']
X = X[features]

# Boolean columns to integers (just in case)
X[['smoke', 'alco', 'active']] = X[['smoke', 'alco', 'active']].astype(int)

# Training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying PCA to reduce dimensionality (retaining 95% of variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initializing and training the logistic regression model with hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_pca, y_train)
model = grid_search.best_estimator_

# Predictions
y_pred = model.predict(X_test_pca)

# Calculating the performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Results
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Heat map of correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = pd.DataFrame(X, columns=features).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# ROC Curve
y_pred_prob = model.predict_proba(X_test_pca)[:, 1]
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


# Trying an SGD model with Logistic Regression and optimal learning rate
# The learning rate can other things can be tuned
sgd_model = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1000, eta0=0.01, random_state=42)
sgd_model.fit(X_train, y_train)

# Predictions for SGDClassifier
y_pred_sgd = sgd_model.predict(X_test)

# Performance metrics for SGDClassifier
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
precision_sgd = precision_score(y_test, y_pred_sgd)
recall_sgd = recall_score(y_test, y_pred_sgd)
f1_sgd = f1_score(y_test, y_pred_sgd)
conf_matrix_sgd = confusion_matrix(y_test, y_pred_sgd)

# Results for SGDClassifier
print(f'SGDClassifier Accuracy: {accuracy_sgd:.2f}')
print(f'SGDClassifier Precision: {precision_sgd:.2f}')
print(f'SGDClassifier Recall: {recall_sgd:.2f}')
print(f'SGDClassifier F1 Score: {f1_sgd:.2f}')
print('SGDClassifier Confusion Matrix:')
print(conf_matrix_sgd)

#PLOTS#

# Heat map of correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = pd.DataFrame(X, columns=features).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Confusion matrix as a heatmap for logistic regression
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Logistic Regression Confusion Matrix Heatmap')
plt.show()

# Confusion matrix as a heatmap for SGDClassifier
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_sgd, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SGDClassifier Confusion Matrix Heatmap')
plt.show()

# ROC Curve for logistic regression
y_pred_prob = model.predict_proba(X_test_pca)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Logistic Regression ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ROC Curve for SGDClassifier
y_pred_prob_sgd = sgd_model.decision_function(X_test)
fpr_sgd, tpr_sgd, _ = roc_curve(y_test, y_pred_prob_sgd)
roc_auc_sgd = roc_auc_score(y_test, y_pred_prob_sgd)

plt.figure(figsize=(10, 6))
plt.plot(fpr_sgd, tpr_sgd, color='darkgreen', lw=2, label=f'SGDClassifier ROC curve (area = {roc_auc_sgd:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SGDClassifier ROC Curve')
plt.legend(loc='lower right')
plt.show()
