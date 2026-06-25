import os

data_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_Train_Ready"

splits = ['train', 'val', 'test']
classes = ['Healthy', 'TB']

for split in splits:
    for cls in classes:
        p = os.path.join(data_dir, split, cls)
        if os.path.exists(p):
            files = os.listdir(p)
            print(f"\n=== Filenames in {split}/{cls} (First 15): ===")
            for f in sorted(files)[:15]:
                print(f" - {f}")
