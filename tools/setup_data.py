
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import os
import requests
import zipfile
import shutil

# Define paths
DATA_DIR = Path("data")
SPLIT_FILE = DATA_DIR / "splits" / "two_vs_many.csv"
ZIP_PATH = DATA_DIR / "splits.zip"
SPLITS_URL = "https://github.com/J-SNACKKB/FLIP/raw/main/splits/aav/splits.zip"

PHASE_DIR = Path("dev_phase")
INPUT_DATA_DIR = PHASE_DIR / "input_data"
REF_DATA_DIR = PHASE_DIR / "reference_data"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download data: {e}")
        sys.exit(1)

def extract_data(zip_path, extract_to, target_file):
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check if target file exists in zip
            file_list = zip_ref.namelist()
            # The file in zip might be "splits/two_vs_many.csv"
            # We want to extract it such that we can find it.
            
            found = False
            for f in file_list:
                if f.endswith(target_file.name):
                     zip_ref.extract(f, extract_to)
                     # Move to expected location if needed, but zip structure usually matches
                     # If the zip has "splits/two_vs_many.csv", and we extract to "data",
                     # it creates "data/splits/two_vs_many.csv".
                     # This matches our SPLIT_FILE path.
                     found = True
                     print(f"Extracted {f}")
                     break
            
            if not found:
                print(f"Error: {target_file.name} not found in zip.")
                sys.exit(1)
    except zipfile.BadZipFile:
        print("Error: Bad zip file.")
        sys.exit(1)

def make_csv(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"Created {filepath}")

def main():
    # 1. Check for Data
    if not SPLIT_FILE.exists():
        print(f"{SPLIT_FILE} not found. checking for zip...")
        if not ZIP_PATH.exists():
            download_file(SPLITS_URL, ZIP_PATH)
        
        extract_data(ZIP_PATH, DATA_DIR, SPLIT_FILE)
    
    if not SPLIT_FILE.exists():
        print("Error parsing data: Split file still missing after download/extraction.")
        sys.exit(1)

    print(f"Loading data from {SPLIT_FILE}...")
    df = pd.read_csv(SPLIT_FILE)

    # 2. Process Data
    # Filter for train and test sets
    train_df = df[df['set'] == 'train'].copy()
    test_full_df = df[df['set'] == 'test'].copy()

    print(f"Total Train samples: {len(train_df)}")
    print(f"Total Test samples: {len(test_full_df)}")

    # Split Test into Public Test and Private Test (50/50)
    # We use a fixed random state for reproducibility
    test_df, private_test_df = train_test_split(test_full_df, test_size=0.5, random_state=42)
    
    print(f"Public Test samples: {len(test_df)}")
    print(f"Private Test samples: {len(private_test_df)}")

    # 3. Write Data (Codabench structure)
    
    # Train Data
    # input_data/train/train_features.csv
    make_csv(train_df[['sequence']], INPUT_DATA_DIR / "train" / "train_features.csv")
    make_csv(train_df[['target']], INPUT_DATA_DIR / "train" / "train_labels.csv")

    # Public Test Data
    # input_data/test/test_features.csv
    make_csv(test_df[['sequence']], INPUT_DATA_DIR / "test" / "test_features.csv")
    # reference_data/test_labels.csv
    make_csv(test_df[['target']], REF_DATA_DIR / "test_labels.csv")

    # Private Test Data
    # input_data/private_test/private_test_features.csv
    make_csv(private_test_df[['sequence']], INPUT_DATA_DIR / "private_test" / "private_test_features.csv")
    # reference_data/private_test_labels.csv
    make_csv(private_test_df[['target']], REF_DATA_DIR / "private_test_labels.csv")

    print("Data setup complete with correct Codabench structure.")

if __name__ == "__main__":
    main()