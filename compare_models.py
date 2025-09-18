import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set output folder for comparison results
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\model_comparison_results'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# Define paths to all model results
model_paths = {
    # 8-class classification models
    '8-class Logistic': 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\logistic\\bieudo_ketqua',
    '8-class XGBoost': 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\XGBoost\\bieudo_ketqua',
    '8-class RandomForest': 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\Randomforest\\bieudo_ketqua',
    '8-class SVM': 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\SVM\\bieudo_ketqua',
    '8-class LSTM': 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\lstm\\bieudo_ketqua',
    '8-class CNN-LSTM': 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\cnn_lstm\\bieudo_ketqua',
    
    # 4-class classification models
    '4-class Logistic': 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\logistic\\bieudo_ketqua',
    '4-class XGBoost': 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\XGBoost\\bieudo_ketqua',
    '4-class RandomForest': 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\randomforest\\bieudo_ketqua',
    '4-class SVM': 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\SVM\\bieudo_ketqua',
    '4-class LSTM': 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\lstm\\bieudo_ketqua',
    '4-class CNN-LSTM': 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\cnn_lstm\\bieudo_ketqua',
    
    # 2-class classification models
    '2-class Logistic': 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\logistic\\bieudo_ketqua',  # Added 2-class Logistic
    '2-class XGBoost': 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\XGBoost\\bieudo_ketqua',
    '2-class RandomForest': 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\randomforest\\bieudo_ketqua',
    '2-class SVM': 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\SVM\\bieudo_ketqua',
    '2-class LSTM': 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\lstm\\bieudo_ketqua',
    '2-class CNN-LSTM': 'D:\\SourceCode\\CICIoT2023_v4\\2_nhan\\cnn_lstm\\bieudo_ketqua'
}

# Function to load accuracy and loss data from a model directory
def load_model_metrics(model_path):
    try:
        metrics_path = os.path.join(model_path, 'accuracy_loss.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            return df
        else:
            print(f"Warning: {metrics_path} not found")
            return None
    except Exception as e:
        print(f"Error loading metrics from {model_path}: {e}")
        return None

# Collect data from all models
print("Collecting accuracy and loss data from all models...")
all_data = []

for model_name, model_path in model_paths.items():
    df = load_model_metrics(model_path)
    if df is not None:
        # Add model name as a column
        df['Model'] = model_name
        
        # Extract number of classes and model type
        parts = model_name.split(' ', 1)
        df['Classification'] = parts[0]
        df['Model_Type'] = parts[1]
        
        all_data.append(df)

# Combine all data
if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_csv(os.path.join(output_folder, 'combined_metrics.csv'), index=False)
    print(f"Combined metrics saved to {output_folder}\\combined_metrics.csv")
else:
    print("Error: No data found.")
    exit(1)

# Create comparison visualizations

# 1. Compare test accuracy across all models (grouped by classification type)
print("Creating comparison visualizations...")

# Set the style for all plots
sns.set(style="whitegrid")

# Create a figure for test accuracy comparison
plt.figure(figsize=(20, 12))

# Filter for test accuracy only
test_data = combined_data[combined_data['Dataset'] == 'Test']

# Plot using seaborn
ax = sns.barplot(x='Model', y='Accuracy', hue='Classification', data=test_data)
plt.title('Test Accuracy Comparison Across All Models', fontsize=20)
plt.xlabel('Model', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title='Classification Type', fontsize=14)
plt.ylim(0, 1.0)  # Set y-axis from 0 to 1

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'test_accuracy_comparison.png'))
plt.close()

# 2. Compare test loss across all models (grouped by classification type)
plt.figure(figsize=(20, 12))

# Plot using seaborn
ax = sns.barplot(x='Model', y='Loss', hue='Classification', data=test_data)
plt.title('Test Loss Comparison Across All Models', fontsize=20)
plt.xlabel('Model', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title='Classification Type', fontsize=14)

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'test_loss_comparison.png'))
plt.close()

# 3. Create grouped comparison by model type
# Group by model type and classification, and calculate mean accuracy/loss
model_type_comparison = test_data.groupby(['Model_Type', 'Classification']).agg({
    'Accuracy': 'mean',
    'Loss': 'mean'
}).reset_index()

# Plot accuracy by model type
plt.figure(figsize=(16, 10))
sns.barplot(x='Model_Type', y='Accuracy', hue='Classification', data=model_type_comparison)
plt.title('Average Test Accuracy by Model Type', fontsize=18)
plt.xlabel('Model Type', fontsize=14)
plt.ylabel('Average Accuracy', fontsize=14)
plt.legend(title='Classification Type', fontsize=12)
plt.ylim(0, 1.0)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'model_type_accuracy_comparison.png'))
plt.close()

# Plot loss by model type
plt.figure(figsize=(16, 10))
sns.barplot(x='Model_Type', y='Loss', hue='Classification', data=model_type_comparison)
plt.title('Average Test Loss by Model Type', fontsize=18)
plt.xlabel('Model Type', fontsize=14)
plt.ylabel('Average Loss', fontsize=14)
plt.legend(title='Classification Type', fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'model_type_loss_comparison.png'))
plt.close()

# 4. Create a heatmap showing accuracy for different models and classification tasks
pivot_data = pd.pivot_table(test_data, values='Accuracy', index=['Classification'], columns='Model_Type')
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_data, annot=True, cmap='Blues', fmt=".3f", linewidths=.5)
plt.title('Accuracy Heatmap: Classification Tasks vs Model Types', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'accuracy_heatmap.png'))
plt.close()

# 5. Create a heatmap showing loss for different models and classification tasks
pivot_data = pd.pivot_table(test_data, values='Loss', index=['Classification'], columns='Model_Type')
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_data, annot=True, cmap='Reds_r', fmt=".3f", linewidths=.5)
plt.title('Loss Heatmap: Classification Tasks vs Model Types', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'loss_heatmap.png'))
plt.close()

# 6. Create side-by-side accuracy/loss comparison for each classification task
for classification in test_data['Classification'].unique():
    subset = test_data[test_data['Classification'] == classification]
    
    plt.figure(figsize=(16, 10))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    ax1 = sns.barplot(x='Model_Type', y='Accuracy', data=subset)
    plt.title(f'{classification} - Accuracy Comparison', fontsize=16)
    plt.ylim(0, 1.0)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    
    # Plot loss
    plt.subplot(2, 1, 2)
    ax2 = sns.barplot(x='Model_Type', y='Loss', data=subset)
    plt.title(f'{classification} - Loss Comparison', fontsize=16)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{classification}_comparison.png'))
    plt.close()

print(f"All comparison visualizations saved to {output_folder}")
print("Model comparison complete.")
