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
# Replace XGBoost with TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Self-contained Mamba-inspired layer implementation
class MambaLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, **kwargs):
        super(MambaLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)
        
    def build(self, input_shape):
        # Input shape should be (batch_size, seq_len, input_dim)
        seq_len = input_shape[1]
        input_dim = input_shape[2]
        
        # Layers
        self.in_proj = Dense(self.d_inner)
        self.conv1d = Conv1D(self.d_inner, kernel_size=self.d_conv, padding='same')
        self.norm = LayerNormalization()
        self.out_proj = Dense(self.d_model)
        
        # Parameters for simplified SSM
        self.A = self.add_weight(
            shape=(self.d_inner, self.d_state),
            initializer='glorot_uniform',
            name='A'
        )
        self.B = self.add_weight(
            shape=(self.d_inner, self.d_state),
            initializer='glorot_uniform',
            name='B'
        )
        self.C = self.add_weight(
            shape=(self.d_inner, self.d_state),
            initializer='glorot_uniform',
            name='C'
        )
        self.D = self.add_weight(
            shape=(self.d_inner,),
            initializer='zeros',
            name='D'
        )
        
        # Mark the layer as built
        self.built = True
        
    def call(self, inputs):
        # Get shapes without using as integers
        batch_size = tf.shape(inputs)[0]
        
        # Project input
        x = self.in_proj(inputs)
        
        # Convolutional path
        x_conv = self.conv1d(x)
        
        # For a simplified version that doesn't use loops:
        # Apply a simple transformation that approximates SSM behavior
        x_proj = tf.matmul(
            tf.reshape(x, [-1, self.d_inner]), 
            self.A
        )
        x_proj = tf.reshape(x_proj, [batch_size, -1, self.d_state])
        x_proj = tf.nn.tanh(x_proj)
        
        x_ssm = tf.matmul(
            x_proj,
            tf.transpose(self.C)
        )
        
        # Add the direct path with D
        x_direct = x * tf.reshape(self.D, [1, 1, self.d_inner])
        x_ssm = x_ssm + x_direct
        
        # Combine paths and normalize
        x = x_conv + x_ssm
        x = self.norm(x)
        
        # Project to output dimension
        return self.out_proj(x)
    
    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape but with d_model features
        return (input_shape[0], input_shape[1], self.d_model)

# Set output paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\mamba\\bieudo_ketqua'  # Updated output folder path

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

# Define the 4 attack groups
attack_groups = {
    'Benign': ['BenignTraffic'],
    'DoS_DDoS': [
        'DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-PSHACK_Flood',
        'DDoS-SYN_Flood', 'DDoS-RSTFINFlood', 'DDoS-SynonymousIP_Flood',
        'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood', 'Mirai-greeth_flood',
        'Mirai-udpplain', 'Mirai-greip_flood', 'DDoS-ICMP_Fragmentation',
        'DDoS-UDP_Fragmentation', 'DDoS-ACK_Fragmentation', 'DoS-HTTP_Flood',
        'DDoS-HTTP_Flood', 'DDoS-SlowLoris'
    ],
    'Recon_Spoofing_MITM': [
        'Recon-HostDiscovery', 'Recon-OSScan', 'Recon-PortScan', 'Recon-PingSweep',
        'VulnerabilityScan', 'MITM-ArpSpoofing', 'DNS_Spoofing'
    ],
    'Exploits_WebAttacks_Malware': [
        'CommandInjection', 'BrowserHijacking', 'Backdoor_Malware',
        'DictionaryBruteForce', 'SqlInjection', 'XSS', 'Uploading_Attack'
    ]
}

# Convert labels to 4-class classification
print("\nConverting to 4-class classification...")
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

# Reshape data for Mamba (samples, timesteps, features)
print("\nReshaping data for Mamba model...")
# For Mamba, we need 3D input similar to LSTM: (batch_size, time_steps, features)
X_train_mamba = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_val_mamba = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
X_test_mamba = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Convert labels to categorical for Keras
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# Train Mamba model for multi-class classification
print("\nTraining multi-class Mamba model...")
start_time = time.time()

# Define Mamba model architecture
inputs = Input(shape=(1, X_train_scaled.shape[1]))
x = MambaLayer(d_model=128)(inputs)
x = Dropout(0.3)(x)
x = MambaLayer(d_model=64)(x)
x = Dropout(0.3)(x)
x = Dense(units=32, activation='relu')(x)
x = Dense(units=len(label_mapping), activation='softmax')(x[:, 0, :])  # Use last timestep outputs
model = Model(inputs, outputs=x)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    os.path.join(output_folder, 'best_mamba_model.keras'),
    monitor='val_loss',
    save_best_only=True
)

# Train the model
history = model.fit(
    X_train_mamba, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_data=(X_val_mamba, y_val_cat),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Evaluate model on training, validation, and test sets
print("\nEvaluating model...")
# Calculate predictions and probabilities
y_train_proba = model.predict(X_train_mamba)
y_train_pred = np.argmax(y_train_proba, axis=1)

y_val_proba = model.predict(X_val_mamba)
y_val_pred = np.argmax(y_val_proba, axis=1)

y_test_proba = model.predict(X_test_mamba)
y_test_pred = np.argmax(y_test_proba, axis=1)

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

# Extract learning curve data from history
print("\nGenerating learning curves...")
epochs = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss_hist = history.history['loss']
val_loss_hist = history.history['val_loss']

# Save learning curve data
learning_curve_df = pd.DataFrame({
    'Epoch': epochs,
    'Train_Accuracy': train_acc,
    'Val_Accuracy': val_acc,
    'Train_Loss': train_loss_hist,
    'Val_Loss': val_loss_hist
})
learning_curve_df.to_csv(os.path.join(output_folder, 'learning_curve_data.csv'), index=False)

# Create visualizations
print("\nCreating visualizations...")

# 1. Confusion Matrix Heatmap - modified for multi-class
plt.figure(figsize=(24, 20))  # Increased from (12, 10) to accommodate larger text
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
            yticklabels=[label_mapping[i] for i in range(len(label_mapping))],
            annot_kws={"size": 36})  # Added to make annotation text 3x larger
plt.title('Mamba Model: Confusion Matrix', fontsize=48)  # Updated title
plt.ylabel('Actual Label', fontsize=42)
plt.xlabel('Predicted Label', fontsize=42)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
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
plt.title('Mamba Model: Multi-class ROC Curves (One-vs-Rest)', fontsize=16)  # Updated title
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'roc_curve.png'))
plt.close()

# 3. Learning Curves for Mamba
plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'o-', color='r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'o-', color='g', label='Validation Accuracy')
plt.title('Mamba Model Accuracy Learning Curve', fontsize=16)  # Updated title
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss_hist, 'o-', color='b', label='Training Loss')
plt.plot(epochs, val_loss_hist, 'o-', color='orange', label='Validation Loss')
plt.title('Mamba Model Loss Learning Curve', fontsize=16)  # Updated title
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Categorical Cross Entropy', fontsize=14)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'learning_curves.png'))
plt.close()

# 4. Accuracy and Loss Comparison
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.barplot(x='Dataset', y='Accuracy', data=accuracy_loss_df)
plt.title('Mamba Model: Accuracy Comparison', fontsize=16)  # Updated title
plt.xlabel('')
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1.0)

plt.subplot(1, 2, 2)
sns.barplot(x='Dataset', y='Loss', data=accuracy_loss_df)
plt.title('Mamba Model: Loss Comparison', fontsize=16)  # Updated title
plt.xlabel('')
plt.ylabel('Log Loss', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'accuracy_loss_comparison.png'))
plt.close()

# 5. Feature Importance - approximate for Mamba using permutation importance
try:
    # Attempt to calculate feature importance using a simple approach
    # Mamba doesn't have built-in feature importance like tree models
    # So we'll calculate the impact of each feature on predictions
    feature_impacts = []
    
    # Get baseline predictions
    baseline_preds = model.predict(X_test_mamba)
    baseline_acc = accuracy_score(y_test, np.argmax(baseline_preds, axis=1))
    
    # Test the impact of perturbing each feature
    for i in range(X_test_scaled.shape[1]):
        # Create a copy with one feature randomized
        X_test_perturbed = X_test_scaled.copy()
        X_test_perturbed[:, i] = np.random.permutation(X_test_perturbed[:, i])
        X_test_perturbed_mamba = X_test_perturbed.reshape(X_test_perturbed.shape[0], 1, X_test_perturbed.shape[1])
        
        # Get predictions with perturbed feature
        perturbed_preds = model.predict(X_test_perturbed_mamba)
        perturbed_acc = accuracy_score(y_test, np.argmax(perturbed_preds, axis=1))
        
        # Calculate impact
        impact = baseline_acc - perturbed_acc
        feature_impacts.append(impact)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_impacts
    }).sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(14, 10))
    n_features = min(20, len(X.columns))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(n_features))
    plt.title('Mamba Model: Feature Importance (Permutation-based)', fontsize=14)
    plt.xlabel('Importance (Accuracy Drop when Feature Permuted)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_importance.png'))
    plt.close()
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(output_folder, 'feature_importance.csv'), index=False)
    
except Exception as e:
    print(f"Error in feature importance calculation: {e}")
    print("Skipping feature importance visualization")

print(f"\nAll results and visualizations saved to {output_folder}")