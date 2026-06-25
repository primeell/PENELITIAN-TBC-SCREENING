import pandas as pd
import os
import re

meta_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\Metadata.csv"
dataset_tbc_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_tbc"

df = pd.read_csv(meta_path)
print(f"Total subjects in Metadata.csv: {len(df)}")

files = os.listdir(dataset_tbc_dir)

# Helper function to clean subject ID for matching
def get_clean_subject_id(subj):
    # e.g., PID_32A_now_C -> PID_32A
    # PID_117A_now_C -> PID_117A
    return re.sub(r'_now_C$', '', subj)

df['clean_subject'] = df['subject'].apply(get_clean_subject_id)

subject_file_counts = {subj: 0 for subj in df['clean_subject'].unique()}
for f in files:
    matched = False
    for clean_subj in sorted(subject_file_counts.keys(), key=len, reverse=True):
        if f.startswith(clean_subj):
            subject_file_counts[clean_subj] += 1
            matched = True
            break

# Map back to df
df['file_count'] = df['clean_subject'].map(subject_file_counts)

print("\n--- Summary of Available Subjects (People with Audio Files) ---")
print(f"Total unique people in Metadata: {df['subject'].nunique()}")
available_df = df[df['file_count'] > 0]
print(f"Total people with audio files: {len(available_df)}")
print("\nClass distribution of available people:")
print(available_df['Label'].value_counts())

print("\nFile count statistics for available people:")
print(available_df['file_count'].describe())

missing_df = df[df['file_count'] == 0]
print(f"\nSubjects with 0 files: {len(missing_df)}")
print(missing_df[['subject', 'Label']])
