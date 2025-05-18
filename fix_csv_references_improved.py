import pandas as pd
import os

# Path to the CSV file
csv_path = r'C:\Users\libro\student-teacher-anomaly-detection\data\carpet\carpet.csv'
# Path to ground truth directory
gt_dir = r'C:\Users\libro\student-teacher-anomaly-detection\data\carpet\ground_truth'

# Check if files exist
if not os.path.exists(csv_path):
    print(f"Error: CSV file {csv_path} does not exist!")
    exit(1)
if not os.path.exists(gt_dir):
    print(f"Error: Ground truth directory {gt_dir} does not exist!")
    exit(1)

# Load the CSV file
df = pd.read_csv(csv_path)

# Convert gt_name column to string type to avoid warnings
df['gt_name'] = df['gt_name'].astype(str)
# Replace 'nan' strings with empty strings
df.loc[df['gt_name'] == 'nan', 'gt_name'] = ''

# Get list of ground truth files
gt_files = os.listdir(gt_dir)
print(f"Found {len(gt_files)} ground truth files")

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

# Ensure normal samples have empty gt_name
normal_count = 0
for idx, row in df.iterrows():
    if row['label'] == 0:  # Normal sample
        if row['gt_name'] != '':
            df.at[idx, 'gt_name'] = ''
            normal_count += 1

# Save the modified CSV file
df.to_csv(csv_path, index=False)

print(f"Updated {updated_count} anomalous samples with ground truth masks")
print(f"Cleared {normal_count} normal samples with incorrect ground truth references")
print("CSV file updated successfully!")