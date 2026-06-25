import os

base_dirs = {
    "dataset_tbc": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_tbc',
    "bulk_audio": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\bulk_audio',
    "DKITCHEN": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\DKITCHEN',
    "NPARK": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\NPARK',
    "OHALLWAY": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\OHALLWAY'
}

print("--- Scanning Newly Added Audio Folders ---")
for name, path in base_dirs.items():
    exists = os.path.exists(path)
    print(f"\n{name}: {'Exists' if exists else 'Does NOT exist'}")
    if exists:
        try:
            items = os.listdir(path)
            print(f"  Total items: {len(items)}")
            # If there are subdirectories, let's list them
            subdirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]
            print(f"  Files count: {len(files)}, Subdirs count: {len(subdirs)}")
            if len(files) > 0:
                print(f"  File examples: {files[:5]}")
            if len(subdirs) > 0:
                print(f"  Subdir examples: {subdirs[:5]}")
                # Print count of files in the first subdirectory
                sub_path = os.path.join(path, subdirs[0])
                try:
                    sub_files = os.listdir(sub_path)
                    print(f"    * Files in '{subdirs[0]}': {len(sub_files)} (e.g. {sub_files[:3]})")
                except Exception as sub_err:
                    print(f"    * Error listing '{subdirs[0]}': {sub_err}")
        except Exception as e:
            print(f"  Error: {e}")

# Check the root directory metadata size and format
meta_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\Metadata.csv"
if os.path.exists(meta_path):
    print("\n--- Checking Root Metadata.csv ---")
    try:
        import pandas as pd
        df = pd.read_csv(meta_path)
        print("Columns:", list(df.columns))
        print("Shape:", df.shape)
        print("Class distribution:")
        print(df['Label'].value_counts() if 'Label' in df.columns else "Label column not found")
        print("First 3 rows:\n", df.head(3))
    except Exception as e:
        print("Error reading root Metadata.csv:", e)
