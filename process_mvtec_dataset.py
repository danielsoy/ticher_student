#!/usr/bin/env python3
"""
Python script to process the MVTec dataset that's already downloaded.
"""

import os
import shutil
import csv
from pathlib import Path


# Configuration
DATA_DIR = "data"
MODEL_DIR = "model"
CATEGORIES = ["carpet"]  # Add more categories as needed


def log(message):
    """Print a formatted log message."""
    print(f"\033[0;35m\033[1m{message}\033[0m\033[m")


def prepare_dir():
    """Create data and model directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def move_images(dataset, src_subdir, tgt_subdir):
    """Move and rename images from source to target directory."""
    src_dir = os.path.join(DATA_DIR, dataset, src_subdir)
    tgt_dir = os.path.join(DATA_DIR, dataset, tgt_subdir)
    
    # Create target directory if it doesn't exist
    os.makedirs(tgt_dir, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(src_dir):
        print(f"Warning: Source directory {src_dir} does not exist")
        return
    
    # Process each subdirectory in the source directory
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
        
        print(f"Processing {item_path}...")
        
        # Process each file in the subdirectory
        for filename in os.listdir(item_path):
            src_file = os.path.join(item_path, filename)
            
            # Skip if not a file
            if not os.path.isfile(src_file):
                continue
            
            # Create new filename with format: datatype_class_filename
            # For example: test_good_001.png or train_good_001.png
            datatype = os.path.basename(src_dir)  # 'test' or 'train' or 'ground_truth'
            class_name = os.path.basename(item_path)  # 'good', 'color', etc.
            
            if datatype == "ground_truth":
                new_filename = f"ground_truth_{class_name}_{filename}"
            else:
                new_filename = f"{datatype}_{class_name}_{filename}"
                
            dst_file = os.path.join(tgt_dir, new_filename)
            
            # Copy the file (using copy instead of move to preserve originals)
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")


def build_csv(dataset):
    """Build CSV file for the dataset."""
    img_dir = os.path.join(DATA_DIR, dataset, "img")
    gt_dir = os.path.join(DATA_DIR, dataset, "ground_truth")
    csv_path = os.path.join(DATA_DIR, dataset, f"{dataset}.csv")
    
    # Check if directories exist
    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} does not exist!")
        return
    
    # Get list of image files
    img_files = sorted(os.listdir(img_dir))
    if not img_files:
        print(f"Error: No image files found in {img_dir}!")
        return
    
    print(f"Found {len(img_files)} image files")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'gt_name', 'label', 'type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        count = 0
        for filename in img_files:
            fname, fext = os.path.splitext(filename)
            parts = fname.split('_')
            
            if len(parts) < 3:
                print(f"Warning: Skipping file with unexpected name format: {filename}")
                continue
                
            datatype = parts[0]  # 'test' or 'train'
            class_name = parts[1]  # 'good', 'color', etc.
            img_id = parts[-1]  # The ID number
            
            # Determine if it's an anomaly
            label = 0 if class_name == 'good' else 1
            
            # For ground truth masks
            gt = ''
            if label == 1:
                # Try to find matching ground truth file
                gt_filename = f"ground_truth_{class_name}_{img_id}_mask{fext}"
                if os.path.exists(os.path.join(gt_dir, gt_filename)):
                    gt = gt_filename
                else:
                    # Try alternative naming
                    gt_filename = f"ground_truth_{class_name}_{img_id}{fext}"
                    if os.path.exists(os.path.join(gt_dir, gt_filename)):
                        gt = gt_filename
            
            row = {
                'image_name': filename,
                'gt_name': gt,
                'label': label,
                'type': datatype
            }
            
            writer.writerow(row)
            count += 1
    
    print(f"Created CSV file with {count} entries")


def process_dataset():
    """Process the dataset: move images and build CSV files."""
    log("Processing MVTec dataset...")
    
    for category in CATEGORIES:
        # Create category directory if it doesn't exist
        category_dir = os.path.join(DATA_DIR, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory {category_dir} does not exist")
            continue
            
        # Clean up existing processed data
        img_dir = os.path.join(category_dir, "img")
        gt_dir = os.path.join(category_dir, "ground_truth")
        csv_file = os.path.join(category_dir, f"{category}.csv")
        
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        if os.path.exists(csv_file):
            os.remove(csv_file)
            
        # Move images from test and train directories to img directory
        move_images(category, "test", "img")
        move_images(category, "train", "img")
        
        # Move ground truth images
        move_images(category, "ground_truth", "ground_truth")
        
        # Build CSV file
        build_csv(category)
    
    log("Done!")


def main():
    """Main function to run the script."""
    prepare_dir()
    process_dataset()


if __name__ == "__main__":
    main()