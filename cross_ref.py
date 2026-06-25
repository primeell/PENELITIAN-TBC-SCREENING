import pandas as pd
import os
import re

meta_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\Metadata.csv"
dataset_tbc_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_tbc"

df = pd.read_csv(meta_path)
print("Metadata subjects head:")
print(df['subject'].head(10))

# Unique subjects in Metadata.csv
meta_subjects = set(df['subject'].unique())
print(f"\nTotal unique subjects in Metadata.csv: {len(meta_subjects)}")

# Scan dataset_tbc files
tbc_files = os.listdir(dataset_tbc_dir)
print(f"Total files in dataset_tbc: {len(tbc_files)}")

# See how files map to metadata subjects
# Audio filenames example: PID_108A11_yeti.wav or PID_108A_0_pixel.wav
# Let's extract the subject base from filenames in dataset_tbc
mapped_subjects = set()
for f in tbc_files:
    # Match pattern like PID_ followed by digits, followed by A or C
    match = re.search(r'(PID_\d+[AC])', f)
    if match:
        mapped_subjects.add(match.group(1))

print(f"Total unique subjects mapped from dataset_tbc: {len(mapped_subjects)}")

# Intersection
matched = meta_subjects.intersection(mapped_subjects)
print(f"Subjects in Metadata.csv that have files in dataset_tbc: {len(matched)}")
print(f"Subjects in Metadata.csv NOT in dataset_tbc: {len(meta_subjects - mapped_subjects)}")
if len(meta_subjects - mapped_subjects) > 0:
    print("Missing subjects examples:", list(meta_subjects - mapped_subjects)[:10])

# Let's also check bulk_audio structure
bulk_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\bulk_audio"
for root, dirs, files in os.walk(bulk_path):
    if len(files) > 0:
        print(f"\nFolder: {root}")
        print(f"  Count: {len(files)} files")
        print(f"  Examples: {files[:3]}")
