import os
import pandas as pd

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

# Find anomalous samples without ground truth masks
missing_gt_samples = df[(df['label'] == 1) & ((df['gt_name'] == '') | (df['gt_name'] == 'nan') | pd.isna(df['gt_name']))]
print(f"Found {len(missing_gt_samples)} anomalous samples without ground truth masks")

# Investigate each missing ground truth
for idx, row in missing_gt_samples.iterrows():
    image_name = row['image_name']
    print(f"\nInvestigating: {image_name}")
    
    # Extract parts from image name
    fname, fext = os.path.splitext(image_name)
    parts = fname.split('_')
    
    if len(parts) < 3:
        print(f"  Warning: Unexpected image name format")
        continue
        
    class_name = parts[1]  # defect type (color, cut, etc.)
    img_id = parts[-1]  # image ID
    
    # List possible mask naming patterns
    possible_masks = [
        f"ground_truth_{class_name}_{img_id}_mask{fext}",
        f"ground_truth_{class_name}_{img_id}{fext}",
        f"ground_truth_{class_name}_{img_id}.png",
        f"ground_truth_{class_name}_{img_id}_mask.png"
    ]
    
    print(f"  Class: {class_name}, ID: {img_id}")
    print(f"  Looking for possible masks:")
    
    for mask in possible_masks:
        if mask in gt_files:
            print(f"  ✓ Found: {mask}")
        else:
            print(f"  ✗ Not found: {mask}")
    
    # List all ground truth files for this class
    class_gt_files = [f for f in gt_files if f.startswith(f"ground_truth_{class_name}_")]
    if class_gt_files:
        print(f"  Found {len(class_gt_files)} ground truth files for class '{class_name}':")
        for i, file in enumerate(sorted(class_gt_files)[:5]):
            print(f"  - {file}")
        if len(class_gt_files) > 5:
            print(f"  ... and {len(class_gt_files) - 5} more")
    else:
        print(f"  No ground truth files found for class '{class_name}'")