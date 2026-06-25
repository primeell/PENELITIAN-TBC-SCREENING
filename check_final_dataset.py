import os

final_dataset_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_Final_Dataset"

print("Checking RespiScan_Final_Dataset:")
if os.path.exists(final_dataset_dir):
    try:
        items = os.listdir(final_dataset_dir)
        print(f" - Contains {len(items)} items")
        print(f" - Examples: {items[:10]}")
        
        # Check if there are subfolders or files
        subfolders = [d for d in items if os.path.isdir(os.path.join(final_dataset_dir, d))]
        print(f" - Subfolders: {subfolders}")
        for sf in subfolders:
            sf_path = os.path.join(final_dataset_dir, sf)
            sf_files = os.listdir(sf_path)
            print(f"   * {sf}: {len(sf_files)} files (e.g. {sf_files[:3]})")
    except Exception as e:
        print(f" - Error: {e}")
else:
    print("RespiScan_Final_Dataset does NOT exist!")
