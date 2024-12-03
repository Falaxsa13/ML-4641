# ML 2024 - Team 108

# Decision Tree Model

# Imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = 'cleaned_cardio_train_dataset.csv'
data = pd.read_csv(file_path)

# Selecting features and creating a copy to avoid SettingWithCopyWarning
features = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure']
X = data[features].copy()  # Explicit copy of the slice
y = data['cardio']

# Convert boolean columns to integers (for training)
X[['smoke', 'alco', 'active']] = X[['smoke', 'alco', 'active']].astype(int)

# Split the dataset into training and testing sets
# Testing 20% - Training 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
# Initialize the decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20)

# Train the model
dt_model.fit(X_train, y_train)

# Evaluate the model
# Predictions
y_pred = dt_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}
print(metrics)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=features, class_names=["No CVD", "CVD"], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()
