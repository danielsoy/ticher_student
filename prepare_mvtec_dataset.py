
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
CATEGORIES = ["carpet"]  # Only process carpet as in the original script

# --- Optional Download/Extract Configuration (currently disabled) ---
# DATA_URL = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz"
# DOWNLOAD_FILE_NAME = "mvtec_anomaly_detection.tar.xz"


def log(message):
    """Prints a formatted log message."""
    # Simple print, color codes are harder to make cross-platform reliably
    print(f"--- {message} ---")


def prepare_dir():
    """Creates the base data and model directories if they don't exist."""
    log("Preparing directories...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log("Directories prepared.")


# --- Download and Extract Functions (kept for reference, but not called by default) ---

# def download_dataset():
#     """Downloads the MVTec dataset."""
#     log("Downloading MVTec dataset...")
#     try:
#         # Note: Python's built-in ftplib doesn't handle passive mode well sometimes.
#         # Using a library like 'requests' or calling 'curl'/'wget' via subprocess
#         # might be more robust if needed. For simplicity, this is placeholder.
#         # Example using requests (install with: pip install requests):
#         # import requests
#         # download_path = DATA_DIR / DOWNLOAD_FILE_NAME
#         # with requests.get(DATA_URL, stream=True) as r:
#         #     r.raise_for_status()
#         #     with open(download_path, 'wb') as f:
#         #         for chunk in r.iter_content(chunk_size=8192):
#         #             f.write(chunk)
#         print("Download function needs implementation (e.g., using requests or ftplib).")
#         # Placeholder for actual download logic if uncommented
#         log("Download complete.")
#     except Exception as e:
#         log(f"Error downloading dataset: {e}")
#         sys.exit(1)

# def extract_dataset():
#     """Extracts the MVTec dataset."""
#     log("Extracting MVTec dataset...")
#     download_path = DATA_DIR / DOWNLOAD_FILE_NAME
#     try:
#         # Requires Python 3.3+ for xz support in tarfile
#         import tarfile
#         with tarfile.open(download_path, "r:xz") as tar:
#             tar.extractall(path=DATA_DIR)
#         # Remove the archive after extraction
#         download_path.unlink()
#         # Note: chmod is less relevant on Windows, skipping
#         log("Extraction complete.")
#     except ImportError:
#         log("Error: 'lzma' module needed for .tar.xz extraction is not available.")
#         sys.exit(1)
#     except FileNotFoundError:
#         log(f"Error: Archive file not found at {download_path}")
#         sys.exit(1)
#     except Exception as e:
#         log(f"Error extracting dataset: {e}")
#         sys.exit(1)

def move_images(category, source_subdir_name, target_subdir_name):
    """Moves and renames images from source subdirectories to a target directory."""
    log(f"Moving images for category '{category}': {source_subdir_name} -> {target_subdir_name}")
    src_base_dir = DATA_DIR / category / source_subdir_name
    tgt_dir = DATA_DIR / category / target_subdir_name
    tgt_dir.mkdir(parents=True, exist_ok=True)

    if not src_base_dir.exists():
        log(f"Warning: Source directory '{src_base_dir}' not found. Skipping move.")
        return

    # Iterate through subfolders (like 'good', 'broken', etc.) within train/test
    for type_folder in src_base_dir.iterdir():
        if not type_folder.is_dir():
            continue # Skip if it's not a directory

        # Check if the type_folder contains any files before iterating
        image_files = list(type_folder.glob('*')) # Get potential files
        if not any(f.is_file() for f in image_files):
            log(f"Info: Source sub-directory '{type_folder}' is empty or contains no files. Skipping.")
            continue

        for image_file in image_files:
            if image_file.is_file():
                new_filename = f"{type_folder.name}_{image_file.name}"
                dest_file = tgt_dir / new_filename
                shutil.move(str(image_file), str(dest_file))
                # print(f"Moved {image_file} to {dest_file}") # Uncomment for verbose output

def build_csv(category):
    """Calls the external Python script to build the CSV file."""
    log(f"Building CSV for category '{category}'...")
    try:
        # Use sys.executable to ensure the same Python interpreter is used
        # Adjust 'python3' if your command is just 'python'
        python_executable = sys.executable # or 'python' or 'python3' if sys.executable is wrong
        script_path = Path("mvtec_dataset.py") # Assumes it's in the same dir or PATH
        subprocess.run([python_executable, str(script_path), category], check=True, text=True)
        log("CSV build complete.")
    except FileNotFoundError:
         log(f"Error: '{python_executable}' or '{script_path}' command not found. Make sure Python is in PATH and mvtec_dataset.py exists.")
         sys.exit(1)
    except subprocess.CalledProcessError as e:
        log(f"Error running mvtec_dataset.py: {e}")
        sys.exit(1)

def process_dataset():
    """Processes the dataset categories: moves images and builds CSVs."""
    log("Processing MVTec dataset...")
    test_dir_name = "test"
    train_dir_name = "train"
    gt_dir_name = "ground_truth"
    img_dir_name = "img"  # Target for flattened train/test images

    for cat in CATEGORIES:
        # Move and flatten images from test subfolders (e.g., test/good, test/broken)
        move_images(cat, test_dir_name, img_dir_name)
        # Move and flatten images from train subfolders (e.g., train/good)
        move_images(cat, train_dir_name, img_dir_name)
        # Ground truth files might not need moving if structure is already flat
        # The original script moved ground_truth/*/* -> ground_truth/type_filename
        # If your ground_truth structure is data/carpet/ground_truth/contamination/000_mask.png etc.
        # you might need a similar move_images call:
        # move_images(cat, gt_dir_name, gt_dir_name) # This renames files within the gt folder

        # Build the CSV file using the external script
        build_csv(cat)

    log("Dataset processing finished.")


if __name__ == "__main__":
    prepare_dir()
    # download_dataset() # Uncomment if download needed
    # extract_dataset()  # Uncomment if extraction needed
    process_dataset()