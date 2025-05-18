import os
import shutil
from pathlib import Path

# Paths
original_dataset_path = "data/mvtec_anomaly_detection/carpet"  # Adjust if your path is different
target_ground_truth_dir = "data/carpet/ground_truth"

# Create target directory if it doesn't exist
os.makedirs(target_ground_truth_dir, exist_ok=True)

# Check if original dataset exists
if not os.path.exists(original_dataset_path):
    print(f"Error: Original dataset path {original_dataset_path} does not exist!")
    print("Please specify the correct path to the MVTec dataset.")
    exit(1)

# Path to original ground truth directory
original_gt_dir = os.path.join(original_dataset_path, "ground_truth")

if not os.path.exists(original_gt_dir):
    print(f"Error: Original ground truth directory {original_gt_dir} does not exist!")
    exit(1)

# Copy and rename ground truth files
copied_count = 0
for defect_type in os.listdir(original_gt_dir):
    defect_dir = os.path.join(original_gt_dir, defect_type)
    
    # Skip if not a directory
    if not os.path.isdir(defect_dir):
        continue
    
    print(f"Processing defect type: {defect_type}")
    
    for filename in os.listdir(defect_dir):
        src_file = os.path.join(defect_dir, filename)
        
        # Skip if not a file
        if not os.path.isfile(src_file):
            continue
        
        # Create new filename with format: ground_truth_defect_type_filename
        new_filename = f"ground_truth_{defect_type}_{filename}"
        dst_file = os.path.join(target_ground_truth_dir, new_filename)
        
        # Copy the file
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        print(f"Copied {src_file} to {dst_file}")

print(f"\nCopied {copied_count} ground truth files to {target_ground_truth_dir}")

# Now update the CSV file to reference these ground truth files
import pandas as pd

csv_path = 'data/carpet/carpet.csv'
if not os.path.exists(csv_path):
    print(f"Error: CSV file {csv_path} does not exist!")
    exit(1)

# Load the CSV file
df = pd.read_csv(csv_path)

# Get list of ground truth files
gt_files = os.listdir(target_ground_truth_dir)
print(f"Found {len(gt_files)} ground truth files in target directory")

# Update ground truth references for anomalous samples
updated_count = 0
for idx, row in df.iterrows():
    if row['label'] == 1:  # Anomalous sample
        # Extract parts from image name
        fname, fext = os.path.splitext(row['image_name'])
        parts = fname.split('_')
        
        if len(parts) < 3:
            print(f"Warning: Unexpected image name format: {row['image_name']}")
            continue
            
        class_name = parts[1]  # defect type (color, cut, etc.)
        img_id = parts[-1]  # image ID
        
        # Try different possible mask naming patterns
        possible_masks = [
            f"ground_truth_{class_name}_{img_id}_mask{fext}",
            f"ground_truth_{class_name}_{img_id}{fext}"
        ]
        
        for mask in possible_masks:
            if mask in gt_files:
                df.at[idx, 'gt_name'] = mask
                updated_count += 1
                break

# Save the modified CSV file
df.to_csv(csv_path, index=False)

print(f"Updated {updated_count} anomalous samples with ground truth masks in CSV file")
print("CSV file updated successfully!")