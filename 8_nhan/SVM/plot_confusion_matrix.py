import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set file paths
input_csv = 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\SVM\\bieudo_ketqua\\confusion_matrix.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\8_nhan\\SVM\\bieudo_ketqua'
output_file = os.path.join(output_folder, 'confusion_matrix_improved.png')

# Read the confusion matrix CSV
df = pd.read_csv(input_csv, index_col=0)

# Create short label mapping for the 8 classes
label_mapping = {
    'Actual Benign': 'Benign',
    'Predicted Benign': 'Benign',
    'Actual DDoS_Flooding': 'DDoS_Flood',
    'Predicted DDoS_Flooding': 'DDoS_Flood',
    'Actual DDoS_Fragmentation': 'DDoS_Frag',
    'Predicted DDoS_Fragmentation': 'DDoS_Frag',
    'Actual DDoS_Application': 'DDoS_App',
    'Predicted DDoS_Application': 'DDoS_App',
    'Actual DoS_Flooding': 'DoS_Flood',
    'Predicted DoS_Flooding': 'DoS_Flood',
    'Actual Mirai_Botnet': 'Mirai',
    'Predicted Mirai_Botnet': 'Mirai',
    'Actual Recon_Spoofing': 'Recon',
    'Predicted Recon_Spoofing': 'Recon',
    'Actual Exploits_Injection': 'Exploit',
    'Predicted Exploits_Injection': 'Exploit'
}

# Process row and column labels
processed_df = df.copy()
processed_df.index = [label_mapping.get(label, label) for label in df.index]
processed_df.columns = [label_mapping.get(label, label) for label in df.columns]

# Create the visualization with increased sizing
plt.figure(figsize=(32, 28))  # Doubled from (16, 14) to accommodate the larger text
cmap = sns.color_palette("Blues", as_cmap=True)

# Create heatmap with improved formatting and 3x larger text
ax = sns.heatmap(
    processed_df, 
    annot=True, 
    fmt='d',
    cmap=cmap,
    annot_kws={"size": 42},  # Increased from 14 to 42 (3x)
    linewidths=1.5  # Increased from 0.5 for better visibility with larger text
)

# Improve appearance with 3x larger text
plt.title('Confusion Matrix', fontsize=72, pad=30)  # Increased from 24 to 72 (3x)
plt.ylabel('Actual Label', fontsize=54, labelpad=20)  # Increased from 18 to 54 (3x)
plt.xlabel('Predicted Label', fontsize=54, labelpad=20)  # Increased from 18 to 54 (3x)
plt.xticks(fontsize=42, rotation=45, ha='right')  # Increased from 14 to 42 (3x)
plt.yticks(fontsize=42, rotation=0)  # Increased from 14 to 42 (3x)

# Add tight layout to ensure everything fits
plt.tight_layout()

# Save the figure with high resolution
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to: {output_file}")
