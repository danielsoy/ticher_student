#!/bin/bash

DATA_DIR="data"
DATASET="carpet"

echo "Cleaning up existing processed data..."
rm -rf "$DATA_DIR/$DATASET/img"
rm -rf "$DATA_DIR/$DATASET/ground_truth"
rm -f "$DATA_DIR/$DATASET/$DATASET.csv"

echo "Creating directories..."
mkdir -p "$DATA_DIR/$DATASET/img"
mkdir -p "$DATA_DIR/$DATASET/ground_truth"

echo "Processing test directory..."
TEST_DIR="$DATA_DIR/$DATASET/test"
if [ -d "$TEST_DIR" ]; then
    for class_dir in "$TEST_DIR"/*; do
        if [ -d "$class_dir" ]; then
            class_name=$(basename "$class_dir")
            echo "Processing class: $class_name"
            
            for img_file in "$class_dir"/*; do
                if [ -f "$img_file" ]; then
                    filename="test_${class_name}_$(basename "$img_file")"
                    cp "$img_file" "$DATA_DIR/$DATASET/img/$filename"
                    echo "Copied $img_file to $DATA_DIR/$DATASET/img/$filename"
                fi
            done
        fi
    done
else
    echo "Test directory not found: $TEST_DIR"
fi

echo "Processing train directory..."
TRAIN_DIR="$DATA_DIR/$DATASET/train"
if [ -d "$TRAIN_DIR" ]; then
    for class_dir in "$TRAIN_DIR"/*; do
        if [ -d "$class_dir" ]; then
            class_name=$(basename "$class_dir")
            echo "Processing class: $class_name"
            
            for img_file in "$class_dir"/*; do
                if [ -f "$img_file" ]; then
                    filename="train_${class_name}_$(basename "$img_file")"
                    cp "$img_file" "$DATA_DIR/$DATASET/img/$filename"
                    echo "Copied $img_file to $DATA_DIR/$DATASET/img/$filename"
                fi
            done
        fi
    done
else
    echo "Train directory not found: $TRAIN_DIR"
fi

echo "Processing ground truth directory..."
GT_DIR="$DATA_DIR/$DATASET/ground_truth"
if [ -d "$GT_DIR" ]; then
    for class_dir in "$GT_DIR"/*; do
        if [ -d "$class_dir" ]; then
            class_name=$(basename "$class_dir")
            echo "Processing class: $class_name"
            
            for gt_file in "$class_dir"/*; do
                if [ -f "$gt_file" ]; then
                    filename="ground_truth_${class_name}_$(basename "$gt_file")"
                    cp "$gt_file" "$DATA_DIR/$DATASET/ground_truth/$filename"
                    echo "Copied $gt_file to $DATA_DIR/$DATASET/ground_truth/$filename"
                fi
            done
        fi
    done
else
    echo "Ground truth directory not found: $GT_DIR"
fi

echo "Generating CSV file..."
python mvtec_dataset.py "$DATASET"

echo "Dataset rebuild complete!"