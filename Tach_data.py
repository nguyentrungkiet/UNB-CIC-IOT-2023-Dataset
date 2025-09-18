import pandas as pd
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
input_file = 'd:\\SourceCode\\CICIoT2023_v4\\combined_data_cleaned.csv.gz'
output_file = 'd:\\SourceCode\\CICIoT2023_v4\\first_50000_rows.csv'
output_folder = 'D:\\SourceCode\\CICIoT2023_v4\\bieudo_data'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# Read the first 50000 rows from the compressed CSV file
print("Reading first 50000 rows from compressed CSV...")
df = pd.read_csv(input_file, compression='gzip', nrows=50000)

# Save to a new CSV file
print(f"Saving {len(df)} rows to {output_file}...")
df.to_csv(output_file, index=False)

print("Done! Summary of extracted data:")
print(f"Shape: {df.shape}")
print("First few rows:")
print(df.head())

# List all columns in the dataset
print("\nColumns in the dataset:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")

# Display additional information about the data
print("\nData types of columns:")
print(df.dtypes)

# Display basic statistics
print("\nBasic statistics of numeric columns:")
print(df.describe())

# Count and display label distribution
if 'label' in df.columns:
    print("\nLabel distribution:")
    label_counts = df['label'].value_counts(dropna=False)
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(df)*100:.2f}%)")
    print(f"Total unique labels: {len(label_counts)}")
    
    # Save label counts to CSV
    label_csv_path = os.path.join(output_folder, 'label_counts.csv')
    label_df = pd.DataFrame({
        'label': label_counts.index,
        'count': label_counts.values,
        'percentage': [(count/len(df)*100) for count in label_counts.values]
    })
    label_df.to_csv(label_csv_path, index=False)
    print(f"\nLabel counts saved to {label_csv_path}")
    
    # Create bar chart visualization of label distribution
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Distribution of Labels', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for i, v in enumerate(label_counts.values):
        ax.text(i, v + (max(label_counts.values)*0.01), str(v), ha='center')
    
    plt.tight_layout()
    chart_path = os.path.join(output_folder, 'label_distribution.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Label distribution chart saved to {chart_path}")
    
else:
    # Try to find alternative label column
    possible_label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'category' in col.lower()]
    if possible_label_cols:
        label_col = possible_label_cols[0]
        print(f"\nLabel column '{label_col}' distribution:")
        label_counts = df[label_col].value_counts(dropna=False)
        for label, count in label_counts.items():
            print(f"{label}: {count} ({count/len(df)*100:.2f}%)")
        print(f"Total unique labels: {len(label_counts)}")
        
        # Save label counts to CSV
        label_csv_path = os.path.join(output_folder, f'{label_col}_counts.csv')
        label_df = pd.DataFrame({
            label_col: label_counts.index,
            'count': label_counts.values,
            'percentage': [(count/len(df)*100) for count in label_counts.values]
        })
        label_df.to_csv(label_csv_path, index=False)
        print(f"\nLabel counts saved to {label_csv_path}")
        
        # Create bar chart visualization of label distribution
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x=label_counts.index, y=label_counts.values)
        plt.title(f'Distribution of {label_col}', fontsize=16)
        plt.xlabel(label_col, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on top of bars
        for i, v in enumerate(label_counts.values):
            ax.text(i, v + (max(label_counts.values)*0.01), str(v), ha='center')
        
        plt.tight_layout()
        chart_path = os.path.join(output_folder, f'{label_col}_distribution.png')
        plt.savefig(chart_path)
        plt.close()
        print(f"Label distribution chart saved to {chart_path}")
    else:
        print("\nNo 'label' column found in the dataset.")