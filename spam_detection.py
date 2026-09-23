import os
import kagglehub
import pandas as pd

# 1. Download the dataset from Kaggle
path = kagglehub.dataset_download("vishakhdapat/sms-spam-detection-dataset")
print("Dataset downloaded to:", path)
print("Files:", os.listdir(path))

# 2. Search for all CSV files in the downloaded folder
csv_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print("\nCSV files found:")
for file in csv_files:
    print(file)

# 3. Raise an error if no CSV is found, otherwise load the first one
if not csv_files:
    raise FileNotFoundError("No CSV file found in the KaggleHub dataset.")

csv_path = csv_files[0]
df = pd.read_csv(csv_path)

# --- NEW STEP: Rename columns from v1/v2 to label/message ---
df.columns = ['label', 'message']

# 4. Display dataset structure and preview rows
print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Metadata & Diagnostics ---")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values count:")
print(df.isnull().sum())

print("\nMissing value percentage:")
print(df.isnull().mean() * 100)

print("\n--- Class Balance (Ham vs Spam count) ---")
print(df['label'].value_counts())
