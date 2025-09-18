import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, log_loss
import os
import time
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier
import xgboost as xgb

# Set output paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\XGBoost\\bieudo_ketqua'

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

# Train XGBoost model with parameters to prevent overfitting
print("\nTraining XGBoost model...")
start_time = time.time()

# Create list to store training and validation metrics for learning curves
train_metrics = {'accuracy': [], 'loss': []}
val_metrics = {'accuracy': [], 'loss': []}

# Initialize XGBoost classifier with parameters to prevent overfitting
model = XGBClassifier(
    n_estimators=100,             # Number of trees
    max_depth=5,                  # Maximum depth of trees (to prevent overfitting)
    learning_rate=0.1,            # Learning rate (smaller values help prevent overfitting)
    subsample=0.8,                # Use 80% of data for each tree (helps prevent overfitting)
    colsample_bytree=0.8,         # Use 80% of features for each tree
    min_child_weight=3,           # Minimum sum of instance weight needed in a child
    gamma=0.1,                    # Minimum loss reduction for partition
    reg_alpha=0.1,                # L1 regularization
    reg_lambda=1.0,               # L2 regularization
    objective='binary:logistic',  # Binary classification
    use_label_encoder=False,      # No need for label encoding
    random_state=42               # For reproducibility
)

# Create an evaluation set to monitor performance during training
eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

# Train the model with simple approach - remove early_stopping_rounds
model.fit(
    X_train_scaled, y_train,
    eval_set=eval_set,
    verbose=True
)

training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Set best_iter to the total number of iterations since we don't have early stopping
best_iter = model.get_booster().num_boosted_rounds()
print(f"Model trained for {best_iter} iterations")

# Extract the result history from the model
evals_result = model.evals_result()

# Calculate metrics for each iteration to create learning curves
# We'll manually create the data for learning curves after training
n_iters = len(evals_result['validation_0']['logloss'])
print(f"Model trained for {n_iters} iterations")

# Get loss values directly from evals_result
train_metrics['loss'] = evals_result['validation_0']['logloss']
val_metrics['loss'] = evals_result['validation_1']['logloss']

# For accuracy, we need to calculate it at different thresholds by using the model
train_metrics['accuracy'] = []
val_metrics['accuracy'] = []

# Generate a few key points for the learning curve visualization
sample_points = list(range(0, n_iters, max(1, n_iters // 10)))
if n_iters-1 not in sample_points:
    sample_points.append(n_iters-1)

print("Calculating accuracy metrics for learning curves...")
for i in sample_points:
    # Create a model with limited number of trees
    temp_model = XGBClassifier(
        n_estimators=i+1,
        max_depth=model.max_depth,
        learning_rate=model.learning_rate,
        subsample=model.subsample,
        colsample_bytree=model.colsample_bytree,
        min_child_weight=model.min_child_weight,
        gamma=model.gamma,
        reg_alpha=model.reg_alpha,
        reg_lambda=model.reg_lambda,
        objective=model.objective,
        random_state=model.random_state
    )
    temp_model.fit(X_train_scaled, y_train, verbose=False)
    
    # Calculate accuracy
    train_pred = temp_model.predict(X_train_scaled)
    val_pred = temp_model.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    train_metrics['accuracy'].append(train_acc)
    val_metrics['accuracy'].append(val_acc)

# Evaluate model on training, validation, and test sets
print("\nEvaluating model...")
# Calculate predictions and probabilities
y_train_pred = model.predict(X_train_scaled)
y_train_proba = model.predict_proba(X_train_scaled)

y_val_pred = model.predict(X_val_scaled)
y_val_proba = model.predict_proba(X_val_scaled)

y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)

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
              'Training Time (seconds)', 'Number of Iterations'],
    'Value': [train_accuracy, val_accuracy, test_accuracy, 
             train_loss, val_loss, test_loss,
             test_precision, test_recall, test_f1, test_auc,
             training_time, best_iter]
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

# Extract the validation loss from the model's evaluation history
evals_result = model.evals_result()

# Create learning curves - adjusted for sampled points
print("\nCreating learning curves...")

# Plot both accuracy and loss learning curves
plt.figure(figsize=(16, 7))

# Plot accuracy learning curves
plt.subplot(1, 2, 1)
iterations = [i+1 for i in sample_points]
plt.plot(iterations, train_metrics['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(iterations, val_metrics['accuracy'], 'ro-', label='Validation Accuracy')
plt.grid(True)
plt.title('Learning Curves - Accuracy', fontsize=16)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.ylim([0, 1.0])

# Plot loss learning curves using full data available
plt.subplot(1, 2, 2)
plt.plot(range(1, n_iters+1), train_metrics['loss'], 'b-', label='Training Loss')
plt.plot(range(1, n_iters+1), val_metrics['loss'], 'r-', label='Validation Loss')
plt.grid(True)
plt.title('Learning Curves - Loss', fontsize=16)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Log Loss', fontsize=12)
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curves.png'))
plt.close()

# Also create the existing loss curve
plt.figure(figsize=(10, 6))
plt.plot(evals_result['validation_0']['logloss'], label='Train')
plt.plot(evals_result['validation_1']['logloss'], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'xgboost_learning_curve.png'))
plt.close()

# Create visualizations
print("\nCreating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(15, 12))  # Increased figure size to fit larger text
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['BenignTraffic', 'Attack'],
            yticklabels=['BenignTraffic', 'Attack'],
            annot_kws={"size": 36},  # Make the numbers 3x larger
            cbar=False)  # Remove colorbar for more space

# Increase tick label font size (x3)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)

plt.title('Confusion Matrix', fontsize=48)  # Increased from 16 to 48
plt.ylabel('Actual Label', fontsize=42)     # Increased from 14 to 42
plt.xlabel('Predicted Label', fontsize=42)  # Increased from 14 to 42
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
# Get feature importances from XGBoost
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.to_csv(os.path.join(output_folder, 'feature_importance.csv'), index=False)

# Plot top 20 features
plt.figure(figsize=(12, 10))
top_n = min(20, len(X.columns))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
plt.title('Top Features by Importance', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_importance.png'))
plt.close()

# 4. Accuracy and Loss Comparison
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

# 5. Advanced XGBoost visualizations - SHAP values if available
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    plt.title('SHAP Feature Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'shap_summary.png'))
    plt.close()
    
    # Create a DataFrame with SHAP values
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df['prediction'] = y_test_pred
    shap_df['actual'] = y_test
    shap_df.to_csv(os.path.join(output_folder, 'shap_values.csv'), index=False)
    
    print("SHAP analysis completed and saved.")
except ImportError:
    print("SHAP package not installed. Skipping SHAP analysis.")
except Exception as e:
    print(f"Error in SHAP analysis: {e}")

print(f"\nAll results and visualizations saved to {output_folder}")