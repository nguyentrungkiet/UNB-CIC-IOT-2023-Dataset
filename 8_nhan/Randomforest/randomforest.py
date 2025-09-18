import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, log_loss
import os
import time
# Import RandomForest instead of XGBoost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

# Set output paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\Randomforest\\bieudo_ketqua'

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

# Define the 8 attack groups
attack_groups = {
    'DDoS_Flooding': [
        'DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-PSHACK_Flood',
        'DDoS-SYN_Flood', 'DDoS-RSTFINFlood', 'DDoS-SynonymousIP_Flood'
    ],
    'DDoS_Fragmentation': [
        'DDoS-ICMP_Fragmentation', 'DDoS-UDP_Fragmentation', 'DDoS-ACK_Fragmentation'
    ],
    'DDoS_Application': [
        'DDoS-HTTP_Flood', 'DDoS-SlowLoris'
    ],
    'DoS_Flooding': [
        'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood', 'DoS-HTTP_Flood'
    ],
    'Mirai_Botnet': [
        'Mirai-greeth_flood', 'Mirai-udpplain', 'Mirai-greip_flood'
    ],
    'Recon_Spoofing': [
        'Recon-HostDiscovery', 'Recon-OSScan', 'Recon-PortScan', 'Recon-PingSweep',
        'VulnerabilityScan', 'MITM-ArpSpoofing', 'DNS_Spoofing'
    ],
    'Exploits_Injection': [
        'CommandInjection', 'BrowserHijacking', 'Backdoor_Malware',
        'DictionaryBruteForce', 'SqlInjection', 'XSS', 'Uploading_Attack'
    ],
    'Benign': ['BenignTraffic']
}

# Convert labels to 8-class classification
print("\nConverting to 8-class classification...")
def map_to_attack_group(label):
    for group_name, attacks in attack_groups.items():
        if label in attacks:
            return group_name
    # If not found in any group, return as "Other"
    return "Other"

df['attack_group'] = df['label'].apply(map_to_attack_group)
print("Attack group distribution:")
attack_group_counts = df['attack_group'].value_counts()
print(attack_group_counts)

# Calculate and print percentages
for group in attack_group_counts.index:
    count = attack_group_counts[group]
    percentage = count / len(df) * 100
    print(f"{group}: {count} ({percentage:.2f}%)")

# Save group distribution to CSV
attack_group_df = pd.DataFrame({
    'group': attack_group_counts.index,
    'count': attack_group_counts.values,
    'percentage': [(count/len(df)*100) for count in attack_group_counts.values]
})
attack_group_df.to_csv(os.path.join(output_folder, 'attack_group_counts.csv'), index=False)

# Prepare data for modeling
print("\nPreparing data for modeling...")
# Drop non-numeric columns and the original label
X = df.drop(['label', 'attack_group'], axis=1)

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

# Convert target to numeric with LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['attack_group'])

# Save the mapping between encoded labels and original labels
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
label_mapping_df = pd.DataFrame(list(label_mapping.items()), columns=['Encoded', 'Group'])
label_mapping_df.to_csv(os.path.join(output_folder, 'label_mapping.csv'), index=False)

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")

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

# Random Forest doesn't need explicit feature selection like logistic regression
print("\nSkipping explicit feature selection for Random Forest...")
# We'll use all features since Random Forest has built-in feature selection capabilities
X_train_selected = X_train_scaled
X_val_selected = X_val_scaled
X_test_selected = X_test_scaled
selected_feature_names = X.columns.tolist()

print(f"Using all {len(selected_feature_names)} features with Random Forest")

# Train Random Forest model for multi-class classification
print("\nTraining multi-class Random Forest model...")
start_time = time.time()

# Initialize Random Forest model with parameters for multi-class classification
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=6,             # Maximum depth of trees
    min_samples_split=2,     # Minimum samples required to split a node
    min_samples_leaf=1,      # Minimum samples in leaf node
    max_features='sqrt',     # Number of features to consider for best split
    bootstrap=True,          # Use bootstrap samples
    class_weight='balanced_subsample', # Adjust weights for class imbalance
    n_jobs=-1,               # Use all processors
    random_state=42          # For reproducibility
)

# Train the model (Random Forest doesn't use eval_set)
model.fit(X_train_selected, y_train)

training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# For Random Forest, there's no built-in evaluation history, so we need to calculate metrics manually
print("\nGenerating validation metrics...")

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

# Calculate additional metrics - now using weighted averaging for multi-class
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print results
print(f"Training Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
print(f"Test Precision (weighted): {test_precision:.4f}")
print(f"Test Recall (weighted): {test_recall:.4f}")
print(f"Test F1 Score (weighted): {test_f1:.4f}")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': ['Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 
              'Training Loss', 'Validation Loss', 'Test Loss',
              'Test Precision (weighted)', 'Test Recall (weighted)', 
              'Test F1 Score (weighted)', 'Training Time (seconds)'],
    'Value': [train_accuracy, val_accuracy, test_accuracy, 
             train_loss, val_loss, test_loss,
             test_precision, test_recall, test_f1,
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

# Detailed classification report - with class labels for better readability
print("\nDetailed classification report:")
class_names = list(label_mapping.values())
print(classification_report(y_test, y_test_pred, target_names=class_names))

class_report = classification_report(y_test, y_test_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv(os.path.join(output_folder, 'classification_report.csv'))

# Compute confusion matrix with descriptive labels
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f'Actual {label_mapping[i]}' for i in range(len(label_mapping))],
    columns=[f'Predicted {label_mapping[i]}' for i in range(len(label_mapping))]
)
cm_df.to_csv(os.path.join(output_folder, 'confusion_matrix.csv'))

# Generate learning curves to visualize overfitting
print("\nGenerating learning curves...")

# Create data for learning curves by varying number of trees
n_trees_range = [5, 10, 20, 30, 40, 50, 75, 100]
train_accuracy_trees = []
val_accuracy_trees = []

for n_trees in n_trees_range:
    # Create and train a model with specific number of trees
    temp_model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced_subsample',
        random_state=42
    )
    temp_model.fit(X_train_selected, y_train)
    
    # Calculate accuracy
    train_pred = temp_model.predict(X_train_selected)
    val_pred = temp_model.predict(X_val_selected)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    train_accuracy_trees.append(train_acc)
    val_accuracy_trees.append(val_acc)

# Also get standard learning curves with different training set sizes
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced_subsample',
        random_state=42
    ),
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

# Save trees learning curve data
trees_curve_df = pd.DataFrame({
    'N_Trees': n_trees_range,
    'Train_Accuracy': train_accuracy_trees,
    'Val_Accuracy': val_accuracy_trees
})
trees_curve_df.to_csv(os.path.join(output_folder, 'trees_curve_data.csv'), index=False)

# Create visualizations
print("\nCreating visualizations...")

# 1. Confusion Matrix Heatmap - modified for multi-class
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, annot_kws={"size": 30}, fmt='d', cmap='Blues',
            xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
            yticklabels=[label_mapping[i] for i in range(len(label_mapping))])
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
plt.close()

# 2. ROC Curve for multi-class (one-vs-rest)
plt.figure(figsize=(12, 10))
for i in range(len(label_mapping)):
    # Calculate ROC curve for each class (one-vs-rest)
    y_test_bin = (y_test == i).astype(int)
    y_score = y_test_proba[:, i]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    roc_auc = roc_auc_score(y_test_bin, y_score)
    plt.plot(fpr, tpr, lw=2, label=f'{label_mapping[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Multi-class ROC Curves (One-vs-Rest)', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'roc_curve.png'))
plt.close()

# 3. Feature Importance for Random Forest
plt.figure(figsize=(20, 16))
n_features = min(20, len(selected_feature_names))

# Get feature importances from Random Forest
feature_importance = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save feature importance to CSV
feature_importance.to_csv(os.path.join(output_folder, 'feature_importance.csv'), index=False)

# Plot feature importance as a single barplot
plt.figure(figsize=(14, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(n_features))
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_importance.png'))
plt.close()

# 4. Learning Curves for number of trees
plt.figure(figsize=(12, 8))
plt.plot(n_trees_range, train_accuracy_trees, 'o-', color='b', label='Training Accuracy')
plt.plot(n_trees_range, val_accuracy_trees, 'o-', color='r', label='Validation Accuracy')
plt.title('Random Forest Accuracy vs Number of Trees', fontsize=16)
plt.xlabel('Number of Trees', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'trees_learning_curve.png'))
plt.close()

# 4b. Learning Curves showing Training and Validation Accuracy
plt.figure(figsize=(12, 8))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Accuracy')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
plt.title('Learning Curves - Random Forest Accuracy', fontsize=16)
plt.xlabel('Training Examples', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'accuracy_learning_curves.png'))
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

# 6. SHAP Values for Random Forest
try:
    import shap
    explainer = shap.TreeExplainer(model)
    # Sample a subset of data for SHAP analysis
    sample_size = min(500, X_test_selected.shape[0])
    X_sample = X_test_selected[:sample_size]
    
    # Generate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Create a summary plot - for random forest, shap_values is a list with one array per class
    plt.figure(figsize=(14, 10))
    # For multi-class problem, use class 0 SHAP values as an example
    shap.summary_plot(shap_values[0], X_sample, feature_names=selected_feature_names, show=False)
    plt.title('SHAP Feature Importance Summary (Class 0)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'shap_summary.png'))
    plt.close()
    
    print("SHAP analysis completed and saved.")
except ImportError:
    print("SHAP package not installed. Skipping SHAP analysis.")
except Exception as e:
    print(f"Error in SHAP analysis: {e}")

print(f"\nAll results and visualizations saved to {output_folder}")