import pandas as pd
import os
import re

meta_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\Metadata.csv"
dataset_tbc_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_tbc"

df = pd.read_csv(meta_path)
subjects = df['subject'].tolist()

files = os.listdir(dataset_tbc_dir)

subject_file_counts = {subj: 0 for subj in subjects}
unmapped_files = []

for f in files:
    matched = False
    # Search if any of our 149 subjects matches the prefix of the file name
    # Sort subjects by length descending to match longest prefix first
    for subj in sorted(subjects, key=len, reverse=True):
        if f.startswith(subj):
            subject_file_counts[subj] += 1
            matched = True
            break
    if not matched:
        unmapped_files.append(f)

print(f"Total files in dataset_tbc: {len(files)}")
print(f"Mapped files: {sum(subject_file_counts.values())}")
print(f"Unmapped files: {len(unmapped_files)}")
if len(unmapped_files) > 0:
    print(f"Unmapped examples: {unmapped_files[:5]}")

# Print distribution of file counts per subject
counts = list(subject_file_counts.values())
print(f"\nFile counts per subject:")
print(f" - Min files per subject: {min(counts)}")
print(f" - Max files per subject: {max(counts)}")
print(f" - Mean files per subject: {sum(counts)/len(counts):.1f}")

# Check which subjects have 0 files
missing_subjects = [s for s, c in subject_file_counts.items() if c == 0]
print(f"\nSubjects with 0 files: {len(missing_subjects)}")
if len(missing_subjects) > 0:
    print(f"Missing subjects: {missing_subjects}")
