# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # Added precision and recall

# Load your dataset
data = pd.read_csv(r'.\weights.csv')  # Use raw string for Windows file path

# Step 2: Exclude T3 class and keep only T1 and T2
# Assuming 'Task' column has textual labels like 'Relax', 'Talking', 'Arithmetic'
data = data[data['Task'].isin(['Relax', 'Talking'])]  # Keep only 'Relax' (T1) and 'Talking' (T2) rows

# Step 3: Map textual labels to numeric labels for two-class classification
label_mapping = {
    'Relax': 0,  # T1
    'Talking': 1  # T2
}

# Map the textual labels to numeric labels
data['Task'] = data['Task'].map(label_mapping)

# Display the updated dataset to verify that T3 is excluded
print(data.head())

# Step 4: Prepare the data for training
X = data.drop(columns=['Subject', 'Video', 'Task']).values  # Features
y = data['Task'].values  # Labels

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to NumPy arrays
X_array = np.array(X)
y_array = np.array(y)

# Display the shapes of X and y
print(f"Features shape: {X_array.shape}, Labels shape: {y_array.shape}")

# Step 5: Implement Stratified K-Fold Cross Validation (without shuffling)
kf = StratifiedKFold(n_splits=7, shuffle=False)  # shuffle=False ensures consistent splits
accuracies = []
f1_scores = []
precisions = []
recalls = []

# To ensure consistent parts, we manually print subjects per fold
for fold, (train_index, test_index) in enumerate(kf.split(X_array, y_array), start=1):  # y_array ensures stratified splits
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]

    # Print fold number and which part of the data is training vs testing
    print(f"Fold {fold}:")

    # Step 6: Initialize the TabNet model
    model = TabNetClassifier()

    # Step 7: Fit the model
    model.fit(X_train=X_train, y_train=y_train,
              eval_set=[(X_test, y_test)],
              max_epochs=300,
              batch_size=32,
              patience=300,
              virtual_batch_size=16,
              num_workers=0,
              drop_last=False)

    # Step 8: Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    accuracies.append(accuracy)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    # Print the metrics for each fold
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Print the average accuracy, precision, recall, and F1 score across folds
print(f'\nAverage Accuracy across folds: {np.mean(accuracies):.4f}')
print(f'Average Precision across folds: {np.mean(precisions):.4f}')
print(f'Average Recall across folds: {np.mean(recalls):.4f}')
print(f'Average F1 Score across folds: {np.mean(f1_scores):.4f}')
