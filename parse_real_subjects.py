import pandas as pd
import re

csv_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_respiscan_local\Metadata.csv"
df = pd.read_csv(csv_path)

print(f"Total rows in Metadata.csv: {len(df)}")

# Naming convention: PID_<subjectID><cohort> followed by recording number
# Let's extract the subject base: e.g., PID_108A from PID_108A0, PID_10C from PID_10C10
real_subjects = []
for subj in df['subject']:
    # Match pattern like PID_ followed by digits, followed by A or C
    match = re.match(r'(PID_\d+[AC])', subj)
    if match:
        real_subjects.append(match.group(1))
    else:
        # Fallback if it doesn't match, e.g. PID_10C, PID_18C
        # If it's already PID_10C, then the match would be PID_10C
        # Let's see if there are any that don't match the regex
        real_subjects.append(subj)

df['real_subject_id'] = real_subjects

print("\nNumber of unique real subjects:")
print(df['real_subject_id'].nunique())

# Group by real subject and check consistency of labels
grouped = df.groupby('real_subject_id')
inconsistent = 0
for name, group in grouped:
    if group['Label'].nunique() > 1:
        print(f"Inconsistent subject: {name}")
        print(group)
        inconsistent += 1

print(f"Total inconsistent subjects: {inconsistent}")

# Now let's see the unique count of Cohort A vs C based on real subject IDs
unique_subjects_df = df.drop_duplicates(subset=['real_subject_id'])
print("\nUnique Subjects Cohort distribution:")
unique_subjects_df['cohort_char'] = unique_subjects_df['real_subject_id'].apply(lambda x: 'C' if 'C' in x else 'A')
print(unique_subjects_df['cohort_char'].value_counts())

print("\nCross tab of unique subjects vs label:")
print(pd.crosstab(unique_subjects_df['cohort_char'], unique_subjects_df['Label']))
