import os
import pandas as pd
import numpy as np

# Path to the CSV file
csv_path = r'C:\Users\libro\student-teacher-anomaly-detection\data\carpet\carpet.csv'

# Check if the CSV file exists
if not os.path.exists(csv_path):
    print(f"Error: CSV file {csv_path} does not exist!")
    exit(1)

# Load the CSV file
df = pd.read_csv(csv_path)

# Check if the CSV file is empty
if len(df) == 0:
    print("Error: CSV file is empty!")
    exit(1)

# Display basic information
print(f"CSV file exists and contains {len(df)} rows.")
print("\nFirst 5 rows:")
print(df.head(5))

# Display statistics
print("\nStatistics:")
print(f"Total entries: {len(df)}")
print(f"Normal samples (label=0): {len(df[df['label'] == 0])}")
print(f"Anomalous samples (label=1): {len(df[df['label'] == 1])}")
print(f"Training samples (type='train'): {len(df[df['type'] == 'train'])}")
print(f"Testing samples (type='test'): {len(df[df['type'] == 'test'])}")

# Check for missing image files
img_dir = r'C:\Users\libro\student-teacher-anomaly-detection\data\carpet\img'
missing_images = 0
for image_name in df['image_name']:
    if not os.path.exists(os.path.join(img_dir, image_name)):
        print(f"Warning: Image file not found: {image_name}")
        missing_images += 1

if missing_images > 0:
    print(f"\nWarning: {missing_images} image files referenced in CSV are missing!")
else:
    print("\nAll image files referenced in CSV exist.")

# Check for missing ground truth files
gt_dir = r'C:\Users\libro\student-teacher-anomaly-detection\data\carpet\ground_truth'
missing_gt = 0
for idx, row in df.iterrows():
    # Handle NaN values in gt_name
    if pd.notna(row['gt_name']) and row['gt_name'] != '':
        if not os.path.exists(os.path.join(gt_dir, row['gt_name'])):
            print(f"Warning: Ground truth file not found: {row['gt_name']}")
            missing_gt += 1

if missing_gt > 0:
    print(f"\nWarning: {missing_gt} ground truth files referenced in CSV are missing!")
else:
    print("\nAll ground truth files referenced in CSV exist.")

# Check for consistency
issues = []
if len(df[df['label'] == 0]) == 0:
    issues.append("No normal samples (label=0)")
if len(df[df['type'] == 'train']) == 0:
    issues.append("No training samples (type='train')")
if any(pd.notna(df.loc[df['label'] == 0, 'gt_name']) & (df.loc[df['label'] == 0, 'gt_name'] != '')):
    issues.append("Some normal samples have ground truth masks")
if any(pd.isna(df.loc[df['label'] == 1, 'gt_name']) | (df.loc[df['label'] == 1, 'gt_name'] == '') | (df.loc[df['label'] == 1, 'gt_name'] == 'nan')):
    issues.append("Some anomalous samples don't have ground truth masks")

if issues:
    print("\nIssues found:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("\nNo consistency issues found. The CSV file looks good!")

# If there are anomalous samples without ground truth, list them
if any(pd.isna(df.loc[df['label'] == 1, 'gt_name']) | (df.loc[df['label'] == 1, 'gt_name'] == '') | (df.loc[df['label'] == 1, 'gt_name'] == 'nan')):
    print("\nAnomalous samples without ground truth masks:")
    missing_gt_samples = df[(df['label'] == 1) & ((df['gt_name'] == '') | (df['gt_name'] == 'nan') | pd.isna(df['gt_name']))]
    for idx, row in missing_gt_samples.iterrows():
        print(f"- {row['image_name']}")

