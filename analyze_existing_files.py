import os
import re
from collections import defaultdict

train_ready_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_Train_Ready"
mfcc_only_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_MFCC_Only"
final_dataset_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_Final_Dataset"

splits = ['train', 'val', 'test']
classes = ['Healthy', 'TB']

subject_files = defaultdict(lambda: {'clean_cwt': [], 'noisy_cwt': [], 'aug_cwt': [], 'clean_mfcc': [], 'noisy_mfcc': [], 'raw_cwt_count': 0})

def get_subj_base(filename):
    match = re.search(r'(PID_\d+[AC])', filename)
    if match:
        return match.group(1)
    return None

# Scan RespiScan_Train_Ready
for split in splits:
    for cls in classes:
        p = os.path.join(train_ready_dir, split, cls)
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith('.png'):
                    subj = get_subj_base(f)
                    if subj:
                        full_path = os.path.join(p, f)
                        if '_noisy' in f.lower():
                            subject_files[subj]['noisy_cwt'].append(full_path)
                        elif 'fast' in f.lower() or 'slow' in f.lower() or 'pitch' in f.lower() or f.startswith('SCAL_'):
                            subject_files[subj]['aug_cwt'].append(full_path)
                        else:
                            subject_files[subj]['clean_cwt'].append(full_path)

# Scan RespiScan_MFCC_Only
for split in splits:
    for cls in classes:
        p = os.path.join(mfcc_only_dir, split, cls)
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith('.pt'):
                    subj = get_subj_base(f)
                    if subj:
                        full_path = os.path.join(p, f)
                        if '_noisy' in f.lower():
                            subject_files[subj]['noisy_mfcc'].append(full_path)
                        else:
                            subject_files[subj]['clean_mfcc'].append(full_path)

# Scan RespiScan_Final_Dataset
for cls in ['NTB', 'TB']:
    p = os.path.join(final_dataset_dir, cls)
    if os.path.exists(p):
        for f in os.listdir(p):
            subj = get_subj_base(f)
            if subj:
                subject_files[subj]['raw_cwt_count'] += 1

print("--- Subject File Coverage Summary (First 15 subjects) ---")
for subj in sorted(subject_files.keys())[:15]:
    info = subject_files[subj]
    print(f"Subject {subj}:")
    print(f"  Clean CWT: {len(info['clean_cwt'])} files")
    print(f"  Noisy CWT: {len(info['noisy_cwt'])} files")
    print(f"  Aug CWT (Time-stretch): {len(info['aug_cwt'])} files")
    print(f"  Clean MFCC: {len(info['clean_mfcc'])} files")
    print(f"  Noisy MFCC: {len(info['noisy_mfcc'])} files")
    print(f"  Final Dataset count: {info['raw_cwt_count']} files")

# Print overall stats
subjects_with_aug = [s for s, info in subject_files.items() if len(info['aug_cwt']) > 0]
print(f"\nTotal subjects in workspace: {len(subject_files)}")
print(f"Total Healthy subjects: {len([s for s in subject_files.keys() if 'C' in s])}")
print(f"Total TB subjects: {len([s for s in subject_files.keys() if 'A' in s])}")
print(f"Healthy subjects with augmented files: {len(subjects_with_aug)}")
