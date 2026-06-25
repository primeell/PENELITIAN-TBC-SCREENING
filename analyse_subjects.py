import pandas as pd
import re

csv_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_respiscan_local\Metadata.csv"
df = pd.read_csv(csv_path)

print(f"Total rows in metadata: {len(df)}")

cohorts = []
for subj in df['subject']:
    # Search for 'A' or 'C' in the ID
    # Subject name format: PID_subjectID+cohort
    # Let's see if we can match the cohort letter
    match_c = re.search(r'C', subj)
    match_a = re.search(r'A', subj)
    if match_c and not match_a:
        cohorts.append('C')
    elif match_a and not match_c:
        cohorts.append('A')
    elif match_a and match_c:
        # Both letters exist, let's look closely at the end or structure
        cohorts.append('Both')
    else:
        cohorts.append('Neither')

df['cohort_detected'] = cohorts
print("\nCohort Detected distribution:")
print(df['cohort_detected'].value_counts())

print("\nCross-tabulation of cohort vs label:")
print(pd.crosstab(df['cohort_detected'], df['Label']))

# Print some examples of 'A' subjects and 'C' subjects
print("\nExamples of Cohort A:")
print(df[df['cohort_detected'] == 'A'].head(10))

print("\nExamples of Cohort C:")
print(df[df['cohort_detected'] == 'C'].head(10))
