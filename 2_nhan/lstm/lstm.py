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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Set output paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\lstm\\bieudo_ketqua'

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

# Reshape data for LSTM (samples, timesteps, features)
# For LSTM, we need 3D input: (batch_size, time_steps, features)
# For this type of data, we'll use time_steps=1 as we don't have sequential information
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_val_lstm = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Train LSTM model
print("\nTraining LSTM model...")
start_time = time.time()

# Define LSTM model architecture - updated to use Input layer
model = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    LSTM(units=64, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(units=32, activation='relu'),
    Dropout(0.2),
    Dense(units=16, activation='relu'),
    Dense(units=1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Define callbacks - fixed file extension to .keras
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(os.path.join(output_folder, 'best_lstm_model.keras'), 
                                   save_best_only=True, monitor='val_loss')

# Fit the model
history = model.fit(
    X_train_lstm, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_lstm, y_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Save training history to CSV
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(output_folder, 'lstm_training_history.csv'), index=False)

# Evaluate model on training, validation, and test sets
print("\nEvaluating model...")
# Calculate predictions and probabilities
y_train_proba = model.predict(X_train_lstm)
y_train_pred = (y_train_proba > 0.5).astype(int).flatten()

y_val_proba = model.predict(X_val_lstm)
y_val_pred = (y_val_proba > 0.5).astype(int).flatten()

y_test_proba = model.predict(X_test_lstm)
y_test_pred = (y_test_proba > 0.5).astype(int).flatten()

# To make log_loss work with LSTM output format
y_train_proba_2d = np.column_stack((1-y_train_proba.flatten(), y_train_proba.flatten()))
y_val_proba_2d = np.column_stack((1-y_val_proba.flatten(), y_val_proba.flatten()))
y_test_proba_2d = np.column_stack((1-y_test_proba.flatten(), y_test_proba.flatten()))

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate loss (log loss / cross-entropy)
train_loss = log_loss(y_train, y_train_proba_2d)
val_loss = log_loss(y_val, y_val_proba_2d)
test_loss = log_loss(y_test, y_test_proba_2d)

# Calculate additional metrics
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba.flatten())

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

# Create learning curves from history for LSTM
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
fpr, tpr, _ = roc_curve(y_test, y_test_proba.flatten())
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

# 3. Feature importance for LSTM is not directly available
# We'll skip this part for LSTM or use a different approach if needed

# 4. Learning Curves for LSTM
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('LSTM Model Accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss_hist, 'b-', label='Training Loss')
plt.plot(epochs, val_loss_hist, 'r-', label='Validation Loss')
plt.title('LSTM Model Loss', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True)

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