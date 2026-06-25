import os

mfcc_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_MFCC_Only"

splits = ['train', 'val', 'test']
classes = ['Healthy', 'TB']

for split in splits:
    for cls in classes:
        p = os.path.join(mfcc_dir, split, cls)
        if os.path.exists(p):
            files = os.listdir(p)
            print(f"\n=== MFCC files in {split}/{cls} (First 10): ===")
            for f in sorted(files)[:10]:
                print(f" - {f}")
