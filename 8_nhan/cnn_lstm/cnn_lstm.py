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
# Import TensorFlow/Keras for CNN-LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Set output paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\cnn_lstm\\bieudo_ketqua'

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

# Reshape data for CNN-LSTM (samples, timesteps, features)
print("\nReshaping data for CNN-LSTM...")
# For CNN-LSTM, we need 3D input with multiple time steps to leverage CNN capabilities
n_features = X_train_scaled.shape[1]
# Define number of timesteps - aim for at least 4 to allow pooling operations
n_timesteps = 4  # We'll use 4 timesteps (you can experiment with other values)
n_features_per_timestep = n_features // n_timesteps

# If features aren't perfectly divisible, we'll truncate slightly
effective_features = n_features_per_timestep * n_timesteps
print(f"Using {n_timesteps} timesteps with {n_features_per_timestep} features per timestep")
print(f"Total effective features: {effective_features} out of {n_features}")

# Reshape the data: first select the usable features, then reshape
X_train_selected = X_train_scaled[:, :effective_features]
X_val_selected = X_val_scaled[:, :effective_features]
X_test_selected = X_test_scaled[:, :effective_features]

X_train_reshaped = X_train_selected.reshape(X_train_scaled.shape[0], n_timesteps, n_features_per_timestep)
X_val_reshaped = X_val_selected.reshape(X_val_scaled.shape[0], n_timesteps, n_features_per_timestep)
X_test_reshaped = X_test_selected.reshape(X_test_scaled.shape[0], n_timesteps, n_features_per_timestep)

print(f"Reshaped training data: {X_train_reshaped.shape}")
print(f"Reshaped validation data: {X_val_reshaped.shape}")
print(f"Reshaped test data: {X_test_reshaped.shape}")

# Convert labels to categorical for Keras
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# Train CNN-LSTM model for multi-class classification
print("\nTraining multi-class CNN-LSTM model...")
start_time = time.time()

# Define CNN-LSTM model architecture
model = Sequential([
    # CNN layers
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
          input_shape=(n_timesteps, n_features_per_timestep)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # LSTM layers
    LSTM(units=128, return_sequences=False),
    Dropout(0.3),
    
    # Dense output layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(label_mapping), activation='softmax')  # Output layer for 8 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    os.path.join(output_folder, 'best_cnn_lstm_model.keras'),
    monitor='val_loss',
    save_best_only=True
)

# Train the model
history = model.fit(
    X_train_reshaped, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_data=(X_val_reshaped, y_val_cat),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Save training history to CSV
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(output_folder, 'cnn_lstm_training_history.csv'), index=False)

# Evaluate model on training, validation, and test sets
print("\nEvaluating model...")
# Calculate predictions and probabilities
y_train_proba = model.predict(X_train_reshaped)
y_train_pred = np.argmax(y_train_proba, axis=1)

y_val_proba = model.predict(X_val_reshaped)
y_val_pred = np.argmax(y_val_proba, axis=1)

y_test_proba = model.predict(X_test_reshaped)
y_test_pred = np.argmax(y_test_proba, axis=1)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate loss (log loss / cross-entropy)
train_loss = log_loss(y_train_cat, y_train_proba)
val_loss = log_loss(y_val_cat, y_val_proba)
test_loss = log_loss(y_test_cat, y_test_proba)

# Calculate additional metrics - using weighted averaging for multi-class
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

# Creating visualizations
print("\nCreating visualizations...")

# 1. Confusion Matrix Heatmap - modified for multi-class
plt.figure(figsize=(18, 15))  # Increased from (12, 10) to accommodate larger text
sns.heatmap(cm, annot=True, annot_kws={"size": 60}, fmt='d', cmap='Blues',  # Doubled annotation size from 30 to 60
            xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
            yticklabels=[label_mapping[i] for i in range(len(label_mapping))])
plt.title('Confusion Matrix', fontsize=32)  # Doubled from 16 to 32
plt.ylabel('Actual Label', fontsize=28)  # Doubled from 14 to 28
plt.xlabel('Predicted Label', fontsize=28)  # Doubled from 14 to 28
plt.xticks(fontsize=24)  # Added to make tick labels 2x larger
plt.yticks(fontsize=24)  # Added to make tick labels 2x larger
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

# 3. Learning curves from CNN-LSTM training history
plt.figure(figsize=(16, 7))
# Plot accuracy learning curves
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN-LSTM Model Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)

# Plot loss learning curves
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN-LSTM Model Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'cnn_lstm_learning_curves.png'))
plt.close()

# 4. Learning Curves showing Training and Validation Accuracy from history
plt.figure(figsize=(12, 8))
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], 'o-', color='r', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'o-', color='g', label='Validation Accuracy')
plt.title('Learning Curves - CNN-LSTM Accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
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

print(f"\nAll results and visualizations saved to {output_folder}")