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

# Get list of ground truth files
gt_files = os.listdir(gt_dir)
print(f"Found {len(gt_files)} ground truth files")

# Update ground truth references for metal contamination samples
updated_count = 0
for idx, row in df.iterrows():
    if row['label'] == 1 and 'metal_contamination' in row['image_name']:
        # Extract parts from image name
        fname, fext = os.path.splitext(row['image_name'])
        parts = fname.split('_')
        
        if len(parts) < 3:
            print(f"Warning: Unexpected image name format: {row['image_name']}")
            continue
            
        # For metal contamination, we need the full class name
        img_id = parts[-1]  # image ID
        
        # Try the correct mask naming pattern
        mask = f"ground_truth_metal_contamination_{img_id}_mask{fext}"
        
        if mask in gt_files:
            df.at[idx, 'gt_name'] = mask
            updated_count += 1
            print(f"Updated: {row['image_name']} -> {mask}")
        else:
            print(f"Warning: Could not find ground truth for {row['image_name']}")

# Save the modified CSV file
df.to_csv(csv_path, index=False)

print(f"Updated {updated_count} metal contamination samples with ground truth masks")
print("CSV file updated successfully!")