import os

data_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_Train_Ready"

splits = ['train', 'val', 'test']
classes = ['Healthy', 'TB']

clean_count = 0
noisy_count = 0
pitch_count = 0
stretch_count = 0
total_files = 0

print(f"Inspecting files in {data_dir}:")
for split in splits:
    split_path = os.path.join(data_dir, split)
    if not os.path.exists(split_path):
        print(f" - {split} does not exist!")
        continue
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        if not os.path.exists(cls_path):
            print(f" - {split}/{cls} does not exist!")
            continue
        files = os.listdir(cls_path)
        print(f" - {split}/{cls}: {len(files)} files")
        total_files += len(files)
        
        # Count types
        for f in files:
            if f.endswith('.png'):
                if '_noisy' in f.lower():
                    noisy_count += 1
                elif 'pitch' in f.lower():
                    pitch_count += 1
                elif 'fast' in f.lower() or 'slow' in f.lower():
                    stretch_count += 1
                else:
                    clean_count += 1

print("\n--- File Type Statistics (PNG images) ---")
print(f"Total PNG files: {total_files}")
print(f" - Clean CWT images: {clean_count}")
print(f" - Noisy mixed images: {noisy_count}")
print(f" - Pitch shifted images: {pitch_count}")
print(f" - Time stretched images: {stretch_count}")

# Check if there are .pt files (MFCC)
pt_count = 0
for split in splits:
    split_path = os.path.join(data_dir, split)
    if os.path.exists(split_path):
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if os.path.exists(cls_path):
                for f in os.listdir(cls_path):
                    if f.endswith('.pt'):
                        pt_count += 1
print(f"\nTotal .pt files (MFCC): {pt_count}")

# Check RespiScan_MFCC_Only
mfcc_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_MFCC_Only"
if os.path.exists(mfcc_dir):
    print(f"\nRespiScan_MFCC_Only exists!")
    for split in splits:
        split_path = os.path.join(mfcc_dir, split)
        if os.path.exists(split_path):
            for cls in classes:
                cls_path = os.path.join(split_path, cls)
                if os.path.exists(cls_path):
                    print(f" - {split}/{cls} (MFCC): {len(os.listdir(cls_path))} files")
else:
    print(f"\nRespiScan_MFCC_Only does NOT exist!")
