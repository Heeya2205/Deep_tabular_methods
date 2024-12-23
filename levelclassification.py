# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv(r'.\levelweights.csv')  # Use raw string for Windows file path

# Step 2: Map textual labels to numeric labels for three classes
label_mapping = {
    'Relax': 0,      # Relax
    'test': 1,      # Test (Arithmetic for specified subjects)
    'control': 2     # Control (Arithmetic for remaining subjects)
}

# Map the textual labels to numeric labels
data['Task'] = data['Task'].map(label_mapping)

# Display the updated dataset to verify the mapping
print(data.head())

# Step 3: Prepare the data for training
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

# Step 4: Implement Stratified K-Fold Cross Validation (with 5 splits)
kf = StratifiedKFold(n_splits=5, shuffle=False)  # shuffle=False ensures consistent splits

# Metrics per class for each fold
accuracies, f1_scores, precisions, recalls = [], [], [], []
class_accuracies, class_f1s, class_precisions, class_recalls = [], [], [], []

# To ensure consistent parts, we manually print subjects per fold
for fold, (train_index, test_index) in enumerate(kf.split(X_array, y_array), start=1):  # y_array ensures stratified splits
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]

    # Print fold number and which part of the data is training vs testing
    print(f"Fold {fold}:")

    # Step 5: Initialize the TabNet model
    model = TabNetClassifier()

    # Step 6: Fit the model
    model.fit(X_train=X_train, y_train=y_train,
              eval_set=[(X_test, y_test)],
              max_epochs=300,
              batch_size=32,
              patience=300,
              virtual_batch_size=16,
              num_workers=0,
              drop_last=False)

    # Step 7: Make predictions and calculate overall metrics
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class
    recall = recall_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class

    accuracies.append(accuracy)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    # Step 8: Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for Fold {fold}:\n", conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Relax', 'Test', 'Control'], yticklabels=['Relax', 'Test', 'Control'])
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Step 9: Per-class metrics
    class_f1 = f1_score(y_test, y_pred, average=None)
    class_precision = precision_score(y_test, y_pred, average=None)
    class_recall = recall_score(y_test, y_pred, average=None)
    
    # Calculate per-class accuracy from confusion matrix
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Store per-class metrics
    class_accuracies.append(class_accuracy)
    class_f1s.append(class_f1)
    class_precisions.append(class_precision)
    class_recalls.append(class_recall)

    print(f'Fold {fold} - Per Class Metrics:')
    for idx, task_name in enumerate(['Relax', 'Test', 'Control']):
        print(f'{task_name}: Accuracy: {class_accuracy[idx]:.4f}, Precision: {class_precision[idx]:.4f}, Recall: {class_recall[idx]:.4f}, F1: {class_f1[idx]:.4f}')

# Step 10: Calculate and display average metrics across all folds
avg_accuracy = np.mean(accuracies)
avg_f1 = np.mean(f1_scores)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)

print(f'\nAverage Accuracy across folds: {avg_accuracy:.4f}')
print(f'Average Precision across folds: {avg_precision:.4f}')
print(f'Average Recall across folds: {avg_recall:.4f}')
print(f'Average F1 Score across folds: {avg_f1:.4f}')

# Step 11: Average per-class metrics across folds
avg_class_accuracies = np.mean(class_accuracies, axis=0)
avg_class_f1s = np.mean(class_f1s, axis=0)
avg_class_precisions = np.mean(class_precisions, axis=0)
avg_class_recalls = np.mean(class_recalls, axis=0)

print('\nAverage Per Class Metrics Across Folds:')
for idx, task_name in enumerate(['Relax', 'Test', 'Control']):
    print(f'{task_name}: Accuracy: {avg_class_accuracies[idx]:.4f}, Precision: {avg_class_precisions[idx]:.4f}, Recall: {avg_class_recalls[idx]:.4f}, F1: {avg_class_f1s[idx]:.4f}')
