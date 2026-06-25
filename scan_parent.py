import os
import pandas as pd

parent_dir = r"c:\TBC SCREENING PENELITIAN"
print(f"Scanning files/directories in: {parent_dir}")
try:
    items = os.listdir(parent_dir)
    print("Items found in parent folder:")
    for item in items:
        full_path = os.path.join(parent_dir, item)
        is_dir = os.path.isdir(full_path)
        print(f" - {item} ({'Directory' if is_dir else 'File'})")
except Exception as e:
    print("Error listing parent directory:", e)

# Let's search for any CSV or Excel files in parent_dir
print("\n--- Searching for metadata files in parent_dir ---")
for root, dirs, files in os.walk(parent_dir):
    # Skip standard python folders and model binaries if any to save output size
    if ".git" in root or "RespiScan_Train_Ready" in root or "RespiScan_Features" in root:
        continue
    for f in files:
        if f.lower().endswith(('.csv', '.xlsx', '.xls')):
            file_path = os.path.join(root, f)
            print(f"Metadata file: {file_path}")
            if "metadata" in f.lower() or "dataset" in f.lower() or "key" in f.lower():
                try:
                    if f.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    print(f"   Shape: {df.shape}, Columns: {list(df.columns)}")
                except Exception as ex:
                    print(f"   Could not read: {ex}")
