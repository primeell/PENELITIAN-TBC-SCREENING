import os

final_dataset_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_Final_Dataset"

subfolders = ['NTB', 'TB']
aug_count = 0
for sf in subfolders:
    sf_path = os.path.join(final_dataset_dir, sf)
    if os.path.exists(sf_path):
        for f in os.listdir(sf_path):
            if 'scal_' in f.lower() or 'fast' in f.lower() or 'slow' in f.lower() or 'pitch' in f.lower():
                aug_count += 1
                if aug_count <= 15:
                    print(f"Augmented file example: {sf}/{f}")

print(f"\nTotal augmented files in RespiScan_Final_Dataset: {aug_count}")
