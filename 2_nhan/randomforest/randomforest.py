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
from sklearn.ensemble import RandomForestClassifier  # Changed from XGBoost to RandomForest

# Set output paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\randomforest\\bieudo_ketqua'  # Changed output folder

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

# Train Random Forest model
print("\nTraining Random Forest model...")
start_time = time.time()

# Create list to store training and validation metrics for learning curves
train_metrics = {'accuracy': [], 'loss': []}
val_metrics = {'accuracy': [], 'loss': []}

# Initialize Random Forest classifier (replaced XGBoost with Random Forest)
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees (same as XGBoost)
    max_depth=5,             # Maximum depth of trees (same as XGBoost)
    min_samples_split=2,     # Minimum samples required to split a node
    min_samples_leaf=1,      # Minimum samples required at each leaf node
    max_features='sqrt',     # Number of features to consider (similar to colsample_bytree)
    bootstrap=True,          # Use bootstrap samples (similar to subsample)
    random_state=42          # For reproducibility (same as XGBoost)
)

# Train the model - Random Forest doesn't use eval_set
model.fit(X_train_scaled, y_train)

training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Set number of trees as the "iterations" (Random Forest doesn't have boosting rounds)
best_iter = model.n_estimators
print(f"Model trained with {best_iter} trees")

# Manually create data for learning curves
n_iters = best_iter
print(f"Generating learning curve data for {n_iters} trees...")

# For loss and accuracy, we need to calculate manually at different numbers of trees
train_metrics['loss'] = []
val_metrics['loss'] = []
train_metrics['accuracy'] = []
val_metrics['accuracy'] = []

# Generate a few key points for the learning curve visualization
sample_points = list(range(0, n_iters, max(1, n_iters // 10)))
if n_iters-1 not in sample_points:
    sample_points.append(n_iters-1)

print("Calculating metrics for learning curves...")
for i in sample_points:
    # Create a model with limited number of trees
    temp_model = RandomForestClassifier(
        n_estimators=i+1,
        max_depth=model.max_depth,
        min_samples_split=model.min_samples_split,
        min_samples_leaf=model.min_samples_leaf,
        max_features=model.max_features,
        bootstrap=model.bootstrap,
        random_state=model.random_state
    )
    temp_model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    train_pred = temp_model.predict(X_train_scaled)
    val_pred = temp_model.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    # Calculate loss (log loss / cross-entropy)
    train_proba = temp_model.predict_proba(X_train_scaled)
    val_proba = temp_model.predict_proba(X_val_scaled)
    
    train_l = log_loss(y_train, train_proba)
    val_l = log_loss(y_val, val_proba)
    
    train_metrics['accuracy'].append(train_acc)
    val_metrics['accuracy'].append(val_acc)
    train_metrics['loss'].append(train_l)
    val_metrics['loss'].append(val_l)

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
plt.title('Random Forest Learning Curves - Accuracy', fontsize=16)  # Updated title
plt.xlabel('Number of Trees', fontsize=12)  # Updated label
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.ylim([0, 1.0])

# Plot loss learning curves
plt.subplot(1, 2, 2)
plt.plot(iterations, train_metrics['loss'], 'b-', label='Training Loss')
plt.plot(iterations, val_metrics['loss'], 'r-', label='Validation Loss')
plt.grid(True)
plt.title('Random Forest Learning Curves - Loss', fontsize=16)  # Updated title
plt.xlabel('Number of Trees', fontsize=12)  # Updated label
plt.ylabel('Log Loss', fontsize=12)
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curves.png'))
plt.close()

# Create a simpler loss curve (replacing XGBoost's evals_result curve)
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_metrics['loss'], 'b-', label='Train')
plt.plot(iterations, val_metrics['loss'], 'r-', label='Validation')
plt.xlabel('Number of Trees')
plt.ylabel('Log Loss')
plt.title('Random Forest Log Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'random_forest_learning_curve.png'))
plt.close()

# Create visualizations
print("\nCreating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
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
# Get feature importances from Random Forest
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

# 5. Random Forest doesn't have built-in SHAP support like XGBoost
# We can still try to use SHAP if it's available
try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # For Random Forest, shap_values might be a list with one array per class
    if isinstance(shap_values, list):
        # For binary classification, use the values for class 1 (positive class)
        plot_shap_values = shap_values[1]
        shap_df = pd.DataFrame(plot_shap_values, columns=X.columns)
    else:
        plot_shap_values = shap_values
        shap_df = pd.DataFrame(shap_values, columns=X.columns)
    
    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(plot_shap_values, X_test_scaled, feature_names=X.columns, show=False)
    plt.title('SHAP Feature Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'shap_summary.png'))
    plt.close()
    
    # Create a DataFrame with SHAP values
    shap_df['prediction'] = y_test_pred
    shap_df['actual'] = y_test
    shap_df.to_csv(os.path.join(output_folder, 'shap_values.csv'), index=False)
    
    print("SHAP analysis completed and saved.")
except ImportError:
    print("SHAP package not installed. Skipping SHAP analysis.")
except Exception as e:
    print(f"Error in SHAP analysis: {e}")

print(f"\nAll results and visualizations saved to {output_folder}")