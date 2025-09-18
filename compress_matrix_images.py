import os
import glob
from PIL import Image
import shutil
import argparse

# Set source and destination paths
source_dir = "D:\\SourceCode\\CICIoT2023_v4"
dest_dir = "D:\\SourceCode\\CICIoT2023_v4\\anh"

# Function to compress image with various options
def compress_image(input_path, output_path, format='PNG', quality=75, resize_factor=0.8, optimize=True):
    """
    Compress an image with multiple techniques:
    1. Convert format (PNG/JPEG)
    2. Reduce quality
    3. Resize dimensions
    4. Optimize compression
    """
    try:
        # Open original image
        img = Image.open(input_path)
        
        # Calculate new size if resize requested
        if resize_factor < 1.0:
            new_width = int(img.width * resize_factor)
            new_height = int(img.height * resize_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert and save with compression
        if format.upper() == 'JPEG':
            # Remove alpha channel if present
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(output_path, format=format, quality=quality, optimize=optimize)
        else:  # PNG
            img.save(output_path, format=format, optimize=optimize, 
                     compress_level=9)  # Maximum compression for PNG
            
        return True
    except Exception as e:
        print(f"Error compressing image: {e}")
        return False

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    print(f"Created directory: {dest_dir}")

# Find all confusion matrix images recursively
print("Searching for confusion matrix images...")
matrix_images = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower() == "confusion_matrix.png":
            matrix_images.append(os.path.join(root, file))
        # Also include improved versions if they exist
        elif file.lower() == "confusion_matrix_improved.png":
            matrix_images.append(os.path.join(root, file))

print(f"Found {len(matrix_images)} confusion matrix images")

# Process each image with multiple compression options
total_original_size = 0
total_compressed_size = 0

# Try both PNG and JPEG formats
formats = ['PNG', 'JPEG']
best_format = {}  # Track which format worked best for each image

for i, img_path in enumerate(matrix_images, 1):
    # Get relative path to create a unique name
    rel_path = os.path.relpath(img_path, source_dir)
    base_name = rel_path.replace("\\", "_")
    
    # Original file size
    original_size = os.path.getsize(img_path) / 1024  # KB
    total_original_size += original_size
    
    best_size = float('inf')
    best_dest_path = None
    
    # Try each format
    for fmt in formats:
        # Generate output path with appropriate extension
        if fmt.upper() == 'JPEG':
            dest_path = os.path.join(dest_dir, os.path.splitext(base_name)[0] + '.jpg')
        else:
            dest_path = os.path.join(dest_dir, base_name)
        
        # Compress with stronger settings
        if compress_image(img_path, dest_path, 
                          format=fmt, 
                          quality=70,  # Lower quality (70% instead of 90%)
                          resize_factor=0.8,  # Reduce size to 80% of original
                          optimize=True):
            
            # Check resulting file size
            compressed_size = os.path.getsize(dest_path) / 1024  # KB
            
            # Keep track of the best format
            if compressed_size < best_size:
                if best_dest_path and best_dest_path != dest_path and os.path.exists(best_dest_path):
                    os.remove(best_dest_path)  # Remove previous "best" if exists
                best_size = compressed_size
                best_dest_path = dest_path
                best_format[img_path] = fmt
    
    # Add to total compressed size
    if best_dest_path:
        total_compressed_size += best_size
        reduction = (1 - best_size/original_size) * 100
        
        print(f"[{i}/{len(matrix_images)}] Processed: {rel_path}")
        print(f"  Original: {original_size:.1f} KB, Compressed ({best_format[img_path]}): {best_size:.1f} KB")
        print(f"  Reduction: {reduction:.1f}%")

# Print overall statistics
if total_original_size > 0:
    overall_reduction = (1 - total_compressed_size/total_original_size) * 100
    print(f"\nOverall reduction: {overall_reduction:.1f}%")
    print(f"Original total: {total_original_size:.1f} KB, Compressed total: {total_compressed_size:.1f} KB")

print(f"\nAll images processed and saved to {dest_dir}")
