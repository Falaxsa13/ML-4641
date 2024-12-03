#Importing basic math and plotting tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing scikit-learn's trees
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Importing scikit-learn's measurement tools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

#Get Dataset
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
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# List of depths to evaluate
depths = [3, 6, 9, 12]

# Store results for each model
results = {}

# Loop through each depth
for depth in depths:
    print(f"Training Random Forest with max_depth = {depth}")
    
    # Train Random Forest with PCA (Reduced Dimensionality)
    tree_pca = RandomForestClassifier(max_depth=depth)
    tree_pca.fit(X_train_pca, y_train)
    
    # Train Random Forest without PCA
    tree = RandomForestClassifier(max_depth=depth)
    tree.fit(X_train, y_train)
    
    # Predictions with PCA
    y_pred_pca = tree_pca.predict(X_test_pca)
    
    # Predictions without PCA
    y_pred = tree.predict(X_test)
    
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

    #Feature importances
    importances_pca = tree_pca.feature_importances_
    importances = tree.feature_importances_
    
    # Storing results for current depth
    results[depth] = {
        'accuracy_pca': accuracy_pca, 
        'precision_pca': precision_pca,
        'recall_pca': recall_pca,
        'f1_pca': f1_pca,
        'conf_matrix_pca': conf_matrix_pca,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'importances_pca': importances_pca,
        'importances': importances,
        'tree_pca': tree_pca,
        'tree': tree
    }
    
    # Print performance for PCA and no PCA
    print(f"Results for max_depth = {depth}:")
    print(f"  Accuracy PCA: {accuracy_pca:.2f}")
    print(f"  Precision PCA: {precision_pca:.2f}")
    print(f"  Recall PCA: {recall_pca:.2f}")
    print(f"  F1 Score PCA: {f1_pca:.2f}")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print('Confusion Matrix PCA:')
    print(conf_matrix_pca)
    print('Confusion Matrix:')
    print(conf_matrix)

    print("-" * 50)

# PLOT COMPARISONS for each depth

# Confusion matrix plots for each depth
for depth in depths:
    conf_matrix_pca = results[depth]['conf_matrix_pca']
    conf_matrix = results[depth]['conf_matrix']
    
    # Confusion matrix with PCA
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_pca, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (max_depth = {depth}) PCA')
    plt.show()

    # Confusion matrix without PCA
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (max_depth = {depth}) No PCA')
    plt.show()

# ROC Curve comparisons for each depth
for depth in depths:
    # ROC Curve with PCA
    y_pred_prob_pca = results[depth]['tree_pca'].predict_proba(X_test_pca)[:, 1]
    fpr_pca, tpr_pca, _ = roc_curve(y_test, y_pred_prob_pca)
    roc_auc_pca = roc_auc_score(y_test, y_pred_prob_pca)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_pca, tpr_pca, color='darkorange', lw=2, label=f'ROC curve (depth={depth}) PCA (AUC = {roc_auc_pca:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (max_depth = {depth}) PCA')
    plt.legend(loc='lower right')
    plt.show()

    # ROC Curve without PCA
    y_pred_prob = results[depth]['tree'].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (depth={depth}) No PCA (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (max_depth = {depth}) No PCA')
    plt.legend(loc='lower right')
    plt.show()

    #Feature Importance with PCA
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance with PCA depth = {depth}")
    plt.barh(range(X.shape[1]), importances_pca[np.argsort(importances_pca)], align="center")
    plt.yticks(range(X.shape[1]), np.array(features)[np.argsort(importances_pca)])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    #Feature Importance without PCA
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance without PCA depth = {depth}")
    plt.barh(range(X.shape[1]), importances[np.argsort(importances)], align="center")
    plt.yticks(range(X.shape[1]), np.array(features)[np.argsort(importances)])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()
