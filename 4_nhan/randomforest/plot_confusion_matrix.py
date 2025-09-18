import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set file paths
input_csv = 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\randomforest\\bieudo_ketqua\\confusion_matrix.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\4_nhan\\randomforest\\bieudo_ketqua'
output_file = os.path.join(output_folder, 'confusion_matrix_improved.png')

# Read the confusion matrix CSV
df = pd.read_csv(input_csv, index_col=0)

# Create short label mapping
label_mapping = {
    'Actual Benign': 'Benign',
    'Actual DoS_DDoS': 'DoS/DDoS',
    'Actual Recon_Spoofing_MITM': 'Recon',
    'Actual Exploits_WebAttacks_Malware': 'Exploits',
    'Predicted Benign': 'Benign',
    'Predicted DoS_DDoS': 'DoS/DDoS',
    'Predicted Recon_Spoofing_MITM': 'Recon',
    'Predicted Exploits_WebAttacks_Malware': 'Exploits'
}

# Process row and column labels
processed_df = df.copy()
processed_df.index = [label_mapping.get(label, label) for label in df.index]
processed_df.columns = [label_mapping.get(label, label) for label in df.columns]

# Create the visualization with increased sizing (doubled dimensions for 3x text)
plt.figure(figsize=(24, 18))  # Increased from (12, 9)
cmap = sns.color_palette("Blues", as_cmap=True)

# Create heatmap with improved formatting and increased text size (3x)
ax = sns.heatmap(
    processed_df, 
    annot=True, 
    fmt='d',
    cmap=cmap,
    annot_kws={"size": 48},  # Increased from 16 to 48 (3x)
    linewidths=1.5  # Increased from 0.5 for better visibility
)

# Improve appearance with 3x larger text
plt.title('Confusion Matrix', fontsize=72, pad=30)  # Increased from 24 to 72 (3x)
plt.ylabel('Actual Label', fontsize=54, labelpad=20)  # Increased from 18 to 54 (3x)
plt.xlabel('Predicted Label', fontsize=54, labelpad=20)  # Increased from 18 to 54 (3x)
plt.xticks(fontsize=42, rotation=45)  # Increased from 14 to 42 (3x)
plt.yticks(fontsize=42, rotation=0)  # Increased from 14 to 42 (3x)

# Add tight layout to ensure everything fits
plt.tight_layout()

# Save the figure with high resolution
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to: {output_file}")
