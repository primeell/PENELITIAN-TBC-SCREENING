import os
import re

DATA_DIR = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\RespiScan_Train_Ready"

splits = ['train', 'val', 'test']
classes = ['Healthy', 'TB']

split_real_subjects = {s: set() for s in splits}

for split in splits:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        print(f"Directory {split_path} does not exist!")
        continue
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        if os.path.exists(cls_path):
            for filename in os.listdir(cls_path):
                # Extract the base subject ID (e.g., PID_10C from PID_10C10_clean.png or similar)
                match = re.search(r'(PID_\d+[AC])', filename)
                if match:
                    subj_base = match.group(1)
                    split_real_subjects[split].add(subj_base)
                else:
                    # In case the naming format is different
                    split_real_subjects[split].add(filename)

print("--- Actual Unique Subjects (People) in Splits ---")
for split in splits:
    print(f"Split {split}: {len(split_real_subjects[split])} unique people")
    print(list(split_real_subjects[split])[:5]) # Show first 5 examples

# Check leakage of the actual people
train_subs = split_real_subjects['train']
val_subs = split_real_subjects['val']
test_subs = split_real_subjects['test']

train_val_leak = train_subs.intersection(val_subs)
train_test_leak = train_subs.intersection(test_subs)
val_test_leak = val_subs.intersection(test_subs)

print("\n--- Actual Speaker Leakage (Speaker-Level Check) ---")
print(f"Overlap Train vs Val: {len(train_val_leak)} people")
if len(train_val_leak) > 0:
    print("Leaked subjects:", list(train_val_leak))

print(f"Overlap Train vs Test: {len(train_test_leak)} people")
if len(train_test_leak) > 0:
    print("Leaked subjects:", list(train_test_leak))

print(f"Overlap Val vs Test: {len(val_test_leak)} people")
if len(val_test_leak) > 0:
    print("Leaked subjects:", list(val_test_leak))
