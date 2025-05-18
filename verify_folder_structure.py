import os
import sys

# Path to check
base_path = r"C:\Users\libro\student-teacher-anomaly-detection\data\carpet"

# Expected structure
expected_dirs = ["ground_truth", "img"]
expected_files = ["carpet.csv"]

# Check if base path exists
if not os.path.exists(base_path):
    print(f"Error: Base path {base_path} does not exist!")
    sys.exit(1)

print(f"Checking folder structure at: {base_path}")

# Check for expected directories
for dir_name in expected_dirs:
    dir_path = os.path.join(base_path, dir_name)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # Count files in directory
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        print(f"✓ Directory {dir_name} exists and contains {len(files)} files")
        
        # Show sample files
        if files:
            print(f"  Sample files in {dir_name}:")
            for i, file in enumerate(sorted(files)[:5]):
                print(f"  - {file}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
    else:
        print(f"✗ Directory {dir_name} is missing!")

# Check for expected files
for file_name in expected_files:
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        print(f"✓ File {file_name} exists ({file_size} bytes)")
    else:
        print(f"✗ File {file_name} is missing!")

# Check ground truth file naming pattern
gt_dir = os.path.join(base_path, "ground_truth")
if os.path.exists(gt_dir) and os.path.isdir(gt_dir):
    gt_files = [f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))]
    correct_pattern = 0
    for file in gt_files:
        if file.startswith("ground_truth_"):
            correct_pattern += 1
    
    if gt_files:
        percentage = (correct_pattern / len(gt_files)) * 100
        print(f"\nGround truth naming pattern check: {correct_pattern}/{len(gt_files)} files ({percentage:.1f}%) follow the expected pattern")
    else:
        print("\nGround truth directory is empty")

# Check img file naming pattern
img_dir = os.path.join(base_path, "img")
if os.path.exists(img_dir) and os.path.isdir(img_dir):
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    test_pattern = 0
    train_pattern = 0
    
    for file in img_files:
        if file.startswith("test_"):
            test_pattern += 1
        elif file.startswith("train_"):
            train_pattern += 1
    
    if img_files:
        test_percentage = (test_pattern / len(img_files)) * 100
        train_percentage = (train_pattern / len(img_files)) * 100
        correct_percentage = ((test_pattern + train_pattern) / len(img_files)) * 100
        
        print(f"\nImage naming pattern check:")
        print(f"- {test_pattern}/{len(img_files)} files ({test_percentage:.1f}%) are test images")
        print(f"- {train_pattern}/{len(img_files)} files ({train_percentage:.1f}%) are train images")
        print(f"- {test_pattern + train_pattern}/{len(img_files)} files ({correct_percentage:.1f}%) follow the expected pattern")
    else:
        print("\nImage directory is empty")

print("\nFolder structure verification complete!")