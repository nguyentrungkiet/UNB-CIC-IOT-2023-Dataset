import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, log_loss
import os
import time
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectFromModel

# Set output paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\logistic\\bieudo_ketqua'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# Read dataset
print("Reading dataset...")
df = pd.read_csv(input_file)
print(f"Dataset shape: {df.shape}")

# Check if 'label' column exists
if 'label' not in df.columns:
    print("Error: 'label' column not found in dataset.")
    exit(1)

# Display label distribution
print("Original label distribution:")
print(df['label'].value_counts())

# Convert labels to binary classification (BenignTraffic vs Attack)
print("\nConverting to binary classification...")
df['binary_label'] = df['label'].apply(lambda x: 'BenignTraffic' if x == 'BenignTraffic' else 'Attack')
print("Binary label distribution:")
print(df['binary_label'].value_counts())
print(f"BenignTraffic: {df['binary_label'].value_counts()['BenignTraffic']} ({df['binary_label'].value_counts()['BenignTraffic']/len(df)*100:.2f}%)")
print(f"Attack: {df['binary_label'].value_counts()['Attack']} ({df['binary_label'].value_counts()['Attack']/len(df)*100:.2f}%)")

# Save binary label distribution
binary_label_counts = df['binary_label'].value_counts()
binary_label_df = pd.DataFrame({
    'label': binary_label_counts.index,
    'count': binary_label_counts.values,
    'percentage': [(count/len(df)*100) for count in binary_label_counts.values]
})
binary_label_df.to_csv(os.path.join(output_folder, 'binary_label_counts.csv'), index=False)

# Prepare data for modeling
print("\nPreparing data for modeling...")
# Drop non-numeric columns and the original label
X = df.drop(['label', 'binary_label'], axis=1)

# Find and drop any other non-numeric columns (like timestamp, IP addresses, etc.)
non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
if non_numeric_cols:
    print(f"Dropping non-numeric columns: {non_numeric_cols}")
    X = X.drop(non_numeric_cols, axis=1)

# Drop columns with NaN values
nan_cols = X.columns[X.isna().any()].tolist()
if nan_cols:
    print(f"Dropping columns with NaN values: {nan_cols}")
    X = X.drop(nan_cols, axis=1)

# Convert target to numeric (0 for BenignTraffic, 1 for Attack)
y = (df['binary_label'] == 'Attack').astype(int)

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Save feature names to CSV
pd.DataFrame({'feature_name': X.columns}).to_csv(os.path.join(output_folder, 'feature_names.csv'), index=False)

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
# This gives an approximate 60% train, 20% validation, 20% test split

print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Feature selection to prevent overfitting
print("\nPerforming feature selection...")
# Use L1 regularization for feature selection
selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42))
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature indices and names
selected_features = selector.get_support()
selected_feature_names = X.columns[selected_features].tolist()
print(f"Selected {len(selected_feature_names)} features out of {X.shape[1]}")

# Save selected features to CSV
pd.DataFrame({'selected_feature': selected_feature_names}).to_csv(
    os.path.join(output_folder, 'selected_features.csv'), index=False)

# Train logistic regression model with regularization to prevent overfitting
print("\nTraining logistic regression model...")
start_time = time.time()
# Using L2 regularization (Ridge) to prevent overfitting
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train_selected, y_train)
training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Evaluate model on training, validation, and test sets
print("\nEvaluating model...")
# Calculate predictions and probabilities
y_train_pred = model.predict(X_train_selected)
y_train_proba = model.predict_proba(X_train_selected)

y_val_pred = model.predict(X_val_selected)
y_val_proba = model.predict_proba(X_val_selected)

y_test_pred = model.predict(X_test_selected)
y_test_proba = model.predict_proba(X_test_selected)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate loss (log loss / cross-entropy)
train_loss = log_loss(y_train, y_train_proba)
val_loss = log_loss(y_val, y_val_proba)
test_loss = log_loss(y_test, y_test_proba)

# Calculate additional metrics
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba[:, 1])

# Print results
print(f"Training Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': ['Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 
              'Training Loss', 'Validation Loss', 'Test Loss',
              'Test Precision', 'Test Recall', 'Test F1 Score', 'Test AUC-ROC',
              'Training Time (seconds)'],
    'Value': [train_accuracy, val_accuracy, test_accuracy, 
             train_loss, val_loss, test_loss,
             test_precision, test_recall, test_f1, test_auc,
             training_time]
})
metrics_df.to_csv(os.path.join(output_folder, 'model_metrics.csv'), index=False)

# Save accuracy and loss data for plotting
accuracy_loss_df = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Accuracy': [train_accuracy, val_accuracy, test_accuracy],
    'Loss': [train_loss, val_loss, test_loss]
})
accuracy_loss_df.to_csv(os.path.join(output_folder, 'accuracy_loss.csv'), index=False)

# Detailed classification report
class_report = classification_report(y_test, y_test_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv(os.path.join(output_folder, 'classification_report.csv'))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm, 
    index=['Actual BenignTraffic', 'Actual Attack'], 
    columns=['Predicted BenignTraffic', 'Predicted Attack']
)
cm_df.to_csv(os.path.join(output_folder, 'confusion_matrix.csv'))

# Generate learning curves to visualize overfitting
print("\nGenerating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42),
    X_train_selected, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calculate mean and std for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Save learning curve data
learning_curve_df = pd.DataFrame({
    'Train_Size': train_sizes,
    'Train_Mean_Accuracy': train_mean,
    'Train_Std_Accuracy': train_std,
    'Val_Mean_Accuracy': val_mean,
    'Val_Std_Accuracy': val_std
})
learning_curve_df.to_csv(os.path.join(output_folder, 'learning_curve_data.csv'), index=False)

# Create visualizations
print("\nCreating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, annot_kws={"size": 30}, fmt='d', cmap='Blues', 
            xticklabels=['BenignTraffic', 'Attack'],
            yticklabels=['BenignTraffic', 'Attack'])
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
plt.close()

# 2. ROC Curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'roc_curve.png'))
plt.close()

# 3. Feature Importance
# Get absolute values of coefficients
coef_abs = np.abs(model.coef_[0])
# Get feature importances for selected features
feature_importance = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': coef_abs
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.to_csv(os.path.join(output_folder, 'feature_importance.csv'), index=False)

# Plot top 20 features (or fewer if less than 20 were selected)
plt.figure(figsize=(12, 10))
top_n = min(20, len(selected_feature_names))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
plt.title('Top Features by Importance', fontsize=16)
plt.xlabel('Absolute Coefficient Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_importance.png'))
plt.close()

# 4. Learning Curves
plt.figure(figsize=(12, 8))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Accuracy')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
plt.title('Learning Curves - Logistic Regression', fontsize=16)
plt.xlabel('Training Examples', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curves.png'))
plt.close()

# 5. Accuracy and Loss Comparison
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.barplot(x='Dataset', y='Accuracy', data=accuracy_loss_df)
plt.title('Accuracy Comparison', fontsize=16)
plt.xlabel('')
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1.0)

plt.subplot(1, 2, 2)
sns.barplot(x='Dataset', y='Loss', data=accuracy_loss_df)
plt.title('Loss Comparison', fontsize=16)
plt.xlabel('')
plt.ylabel('Log Loss', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'accuracy_loss_comparison.png'))
plt.close()

print(f"\nAll results and visualizations saved to {output_folder}")